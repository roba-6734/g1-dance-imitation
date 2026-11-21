"""Closed-loop sim-to-sim evaluation of an exported mimic policy, with readiness gates.

Runs the deploy-faithful controller (wbt.deploy_policy) against plain MuJoCo, an
engine path independent of mjlab's mujoco_warp training simulator, over:

  nominal      start on the reference pose, no noise (should be near-perfect)
  stand_entry  hardware path: settle in the deploy default pose (FixStand/Velocity),
               then switch to the mimic policy from a standstill
  latency      nominal + 20 ms command latency
  motor_pd     PD computed at 1 kHz with torque clipping (motor-board emulation)
  imu_yaw      arbitrary IMU yaw offset (real IMUs have one; init_quat must cancel it)
  random/N     domain randomisation: friction, gains, motor strength, torso mass and COM,
               pushes, sensor noise, latency, IMU yaw

    python -m wbt.sim2sim --policy policy.onnx --deploy-yaml deploy.yaml --motion motion.npz \
        [--onnx-meta <run>.onnx] [--suite quick|full] --out runs/wbt_eval/name

Writes report.json / report.md and exits non-zero unless every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from g1dance.joints import G1_JOINT_NAMES
from .deploy_policy import (DeployCfg, MimicController, Motion, OnnxPolicy, load_deploy_yaml, load_motion,
                            onnx_metadata, qaxis, qmat, qmul)
from .g1model import build_model, joint_specs

LEG = list(range(12))
WAIST = [12, 13, 14]
ARMS = list(range(15, 29))
TRACKED_BODIES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link", "left_elbow_link",
                  "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link",
                  "right_knee_link")
# FixStand gains from unitree_rl_mjlab deploy/robots/g1/config/config.yaml (holds the pre-entry stance).
FIXSTAND_KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [200] * 3 + [40] * 14, float)
FIXSTAND_KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [5] * 3 + [10] * 14, float)
TRAINING_PUSH ={"x": 0.5, "y": 0.5, "z": 0.2, "roll": 0.52, "pitch": 0.52, "yaw": 0.78}

GATES = {
    "nominal_success": 1.0,  # every deterministic scenario must finish without a fall
    "random_success_min": 0.95,
    "nominal_joint_rms_max_rad": 0.15,
    "nominal_body_err_max_m": 0.08,
    "peak_velocity_ratio_max": 0.90,  # |dq| / motor velocity limit
    "torque_saturation_frac_max": 0.05,  # per joint, share of physics steps at >= 95 % effort
    "hard_limit_frac_max": 0.0,  # share of steps within 1 % of a hard joint stop
    "self_collision_events_max": 0,  # robot-robot contacts > 10 N in deterministic runs
    "metadata_match": True,
}


@dataclass
class Scenario:
    name: str
    start: str = "reference"  # reference | stand
    friction: float = 0.6
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    effort_scale: float = 1.0
    torso_mass_add: float = 0.0
    com_offset: tuple = (0.0, 0.0, 0.0)
    latency_s: float = 0.0
    motor_pd: bool = False
    push_scale: float = 0.0  # fraction of the training push ranges, every 1-3 s
    obs_noise: float = 0.0  # fraction of the training actor noise
    imu_yaw: float = 0.0
    seed: int = 0
    deterministic: bool = True


@dataclass
class Result:
    scenario: str
    success: bool
    fall_time: float | None
    duration: float
    joint_rms: dict = field(default_factory=dict)
    body_err_m: float = 0.0
    peak_velocity_ratio: float = 0.0
    peak_velocity_joint: str = ""
    torque_saturation_frac: float = 0.0
    torque_saturation_joint: str = ""
    hard_limit_frac: float = 0.0
    self_collision_events: int = 0
    action_rate: float = 0.0
    max_tilt_rad: float = 0.0


def deterministic_suite():
    return [Scenario("nominal"), Scenario("stand_entry", start="stand"), Scenario("latency_20ms", latency_s=0.02),
            Scenario("motor_pd_1khz", motor_pd=True), Scenario("imu_yaw_offset", imu_yaw=2.0),
            Scenario("stand_entry_latency", start="stand", latency_s=0.02)]


def random_suite(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(Scenario(
            f"random_{i:03d}", start=str(rng.choice(["reference", "stand"])), friction=float(rng.uniform(0.3, 1.2)),
            kp_scale=float(rng.uniform(0.8, 1.2)), kd_scale=float(rng.uniform(0.8, 1.2)),
            effort_scale=float(rng.uniform(0.85, 1.0)), torso_mass_add=float(rng.uniform(-1.0, 2.0)),
            com_offset=tuple(rng.uniform(-0.05, 0.05, 3)), latency_s=float(rng.choice([0.0, 0.005, 0.01, 0.02])),
            motor_pd=bool(rng.random() < 0.5), push_scale=float(rng.uniform(0.0, 0.6)), obs_noise=1.0,
            imu_yaw=float(rng.uniform(-np.pi, np.pi)), seed=int(rng.integers(1 << 30)), deterministic=False))
    return out


class Sim:
    def __init__(self, cfg: DeployCfg, sc: Scenario, xml=None):
        self.cfg, self.sc = cfg, sc
        self.dt = 0.001 if sc.motor_pd else 0.005
        self.kp, self.kd = cfg.stiffness * sc.kp_scale, cfg.damping * sc.kd_scale
        self.model = build_model(xml, timestep=self.dt, foot_friction=sc.friction, kp=self.kp, kd=self.kd,
                                 effort_scale=sc.effort_scale, explicit_pd=sc.motor_pd)
        torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.model.body_mass[torso] += sc.torso_mass_add
        self.model.body_ipos[torso] += np.asarray(sc.com_offset)
        self.data = mujoco.MjData(self.model)
        self.effort = np.array([s.effort for s in joint_specs()]) * sc.effort_scale
        self.vlim = np.array([s.velocity for s in joint_specs()])
        self.hard = self.model.jnt_range[1:30].copy()
        self.robot_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] != 0}
        self.rng = np.random.default_rng(sc.seed)

    # state accessors (what lowstate provides)
    @property
    def q(self):
        return self.data.qpos[7:36].copy()

    @property
    def dq(self):
        return self.data.qvel[6:35].copy()

    def imu(self):
        quat = self.data.qpos[3:7].copy()
        gyro = self.data.qvel[3:6].copy()  # free-joint angular velocity is in the body frame
        if self.sc.imu_yaw:
            quat = qmul(qaxis("z", self.sc.imu_yaw), quat)
        return quat, gyro

    def place(self, root_pos, root_quat, q, dq=None):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3], d.qpos[3:7], d.qpos[7:36] = root_pos, root_quat, q
        if dq is not None:
            d.qvel[6:35] = dq
        mujoco.mj_forward(self.model, d)
        feet = [g for g in range(self.model.ngeom) if (self.model.geom(g).name or "").startswith(("left_foot", "right_foot"))]
        lowest = min(d.geom_xpos[g][2] - self.model.geom_size[g][0] for g in feet)
        d.qpos[2] -= lowest - 0.001  # rest the lowest foot capsule on the floor
        mujoco.mj_forward(self.model, d)

    def set_gains(self, kp, kd):
        """Switch PD gains (FSM state change: FixStand -> Mimic rewrites motor_cmd kp/kd)."""
        self.kp, self.kd = np.asarray(kp, float), np.asarray(kd, float)
        if not self.sc.motor_pd:
            self.model.actuator_gainprm[:, 0] = self.kp
            self.model.actuator_biasprm[:, 1] = -self.kp
            self.model.actuator_biasprm[:, 2] = -self.kd

    def physics_step(self, target):
        if self.sc.motor_pd:
            tau = self.kp * (target - self.q) - self.kd * self.dq
            self.data.ctrl[:] = np.clip(tau, -self.effort, self.effort)
        else:
            self.data.ctrl[:] = target
        mujoco.mj_step(self.model, self.data)

    def tilt(self):
        g = qmat(self.data.qpos[3:7]).T @ np.array([0, 0, -1.0])
        return float(np.arccos(np.clip(-g[2], -1, 1)))

    def self_collisions(self):
        n = 0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.robot_geoms and c.geom2 in self.robot_geoms:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                n += abs(f6[0]) > 10.0
        return n

    def body_local(self, ids):
        """Body positions relative to the pelvis, in the pelvis heading frame (drift and yaw invariant)."""
        p = self.data.xpos[ids] - self.data.xpos[ids[0]]
        yaw = np.arctan2(*qmat(self.data.qpos[3:7])[[1, 0], 0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def reference_body_local(sim: Sim, motion: Motion, f, ids):
    d = mujoco.MjData(sim.model)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:36] = motion.root_pos[f], motion.root_quat[f], motion.joint_pos[f]
    mujoco.mj_kinematics(sim.model, d)
    p = d.xpos[ids] - d.xpos[ids[0]]
    yaw = np.arctan2(*qmat(d.qpos[3:7])[[1, 0], 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    return p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def run_scenario(cfg, motion, policy, sc: Scenario, time_start=0.0, time_end=None, xml=None, settle_s=1.5,
                 record=False):
    sim = Sim(cfg, sc, xml)
    f0 = motion.frame(time_start)
    if sc.start == "reference":
        sim.place(motion.root_pos[f0], motion.root_quat[f0], motion.joint_pos[f0], motion.joint_vel[f0])
    else:  # standing in the deploy default pose with FixStand gains, then the FSM switches to Mimic
        yaw = sim.rng.uniform(-np.pi, np.pi) if not sc.deterministic else 0.7
        sim.place([0, 0, 0.8], qaxis("z", yaw), cfg.default_joint_pos)
        sim.set_gains(FIXSTAND_KP, FIXSTAND_KD)
        root = sim.data.qpos[:7].copy()
        for _ in range(int(settle_s / sim.dt)):  # the Velocity policy balances on hardware; here the
            sim.physics_step(cfg.default_joint_pos)  # pelvis is held so the robot is at rest at the switch
            sim.data.qpos[:7], sim.data.qvel[:6] = root, 0.0
        mujoco.mj_forward(sim.model, sim.data)
        sim.set_gains(cfg.stiffness * sc.kp_scale, cfg.damping * sc.kd_scale)
    ctrl = MimicController(cfg, motion, policy, time_start, time_end)
    quat, _ = sim.imu()
    ctrl.enter(quat, sim.q)
    sub = int(round(cfg.step_dt / sim.dt))
    delay = int(round(sc.latency_s / sim.dt))
    queue = [sim.q.copy()] * delay  # previous command is held while the new one is in flight
    body_ids = np.array([mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRACKED_BODIES])
    noise = sc.obs_noise
    next_push = sim.rng.uniform(1, 3) if sc.push_scale else None
    errs_q, errs_b, actions, frames = [], [], [], []
    peak_v = np.zeros(29)
    sat = np.zeros(29)
    near_limit = 0
    collisions = 0
    max_tilt = 0.0
    fall_time = None
    nsteps = 0
    log = []
    while not ctrl.done:
        quat, gyro = sim.imu()
        q, dq = sim.q, sim.dq
        if noise:  # training actor noise: ang vel 0.2, joint pos 0.01, joint vel 0.5 (uniform)
            gyro = gyro + sim.rng.uniform(-0.2, 0.2, 3) * noise
            q = q + sim.rng.uniform(-0.01, 0.01, 29) * noise
            dq = dq + sim.rng.uniform(-0.5, 0.5, 29) * noise
        target, action, f = ctrl.step(quat, gyro, q, dq)
        actions.append(action)
        frames.append(f)
        for _ in range(sub):
            queue.append(target)
            sim.physics_step(queue.pop(0))
            nsteps += 1
            peak_v = np.maximum(peak_v, np.abs(sim.dq))
            force = np.abs(sim.data.actuator_force)
            sat += force >= 0.95 * sim.effort
            near_limit += np.any((sim.q < sim.hard[:, 0] + 0.01 * np.ptp(sim.hard, 1)) |
                                 (sim.q > sim.hard[:, 1] - 0.01 * np.ptp(sim.hard, 1)))
        collisions += sim.self_collisions()
        tilt = sim.tilt()
        max_tilt = max(max_tilt, tilt)
        errs_q.append(sim.q - motion.joint_pos[f])
        errs_b.append(np.linalg.norm(sim.body_local(body_ids) - reference_body_local(sim, motion, f, body_ids), axis=1).mean())
        if record:
            log.append(sim.data.qpos.copy())
        if sim.data.qpos[2] < 0.45 or tilt > 1.0:
            fall_time = ctrl.time
            break
        if next_push is not None and ctrl.time >= next_push:
            r = np.array([TRAINING_PUSH[k] for k in ("x", "y", "z", "roll", "pitch", "yaw")]) * sc.push_scale
            sim.data.qvel[:6] += sim.rng.uniform(-r, r)
            next_push = ctrl.time + sim.rng.uniform(1, 3)
    e = np.array(errs_q)
    rms = lambda idx: float(np.sqrt(np.mean(e[:, idx] ** 2))) if len(e) else float("nan")
    ratio = peak_v / sim.vlim
    satf = sat / max(nsteps, 1)
    a = np.array(actions)
    res = Result(sc.name, fall_time is None, fall_time, ctrl.time - ctrl.t0,
                 {"legs": rms(LEG), "waist": rms(WAIST), "arms": rms(ARMS), "all": rms(list(range(29)))},
                 float(np.mean(errs_b)) if errs_b else float("nan"), float(ratio.max()), G1_JOINT_NAMES[int(ratio.argmax())],
                 float(satf.max()), G1_JOINT_NAMES[int(satf.argmax())], near_limit / max(nsteps, 1), int(collisions),
                 float(np.mean(np.abs(np.diff(a, axis=0)))) if len(a) > 1 else 0.0, max_tilt)
    return (res, np.array(log)) if record else res


def check_metadata(cfg: DeployCfg, meta: dict) -> list[str]:
    """Compare deploy.yaml against what training actually used (mjlab ONNX metadata)."""
    problems = []
    names = meta.get("joint_names")
    if names is not None and list(names) != list(G1_JOINT_NAMES):
        problems.append("policy joint order differs from SDK order; deploy needs a joint_ids_map")
    for key, val in (("joint_stiffness", cfg.stiffness), ("joint_damping", cfg.damping),
                     ("default_joint_pos", cfg.default_joint_pos), ("action_scale", cfg.action_scale)):
        if key in meta and not np.allclose(np.asarray(meta[key], float), val, atol=2e-3, rtol=2e-2):
            problems.append(f"deploy.yaml {key} differs from training metadata")
    if "default_joint_pos" in meta and not np.allclose(np.asarray(meta["default_joint_pos"], float),
                                                       cfg.action_offset, atol=2e-3):
        problems.append("deploy.yaml action offset differs from training default_joint_pos (use_default_offset=True)")
    obs = meta.get("observation_names")
    if obs is not None:
        want = [n if n != "command" else "motion_command" for n in obs]
        alias = {"joint_pos": "joint_pos_rel", "joint_vel": "joint_vel_rel", "actions": "last_action"}
        want = [alias.get(n, n) for n in want]
        have = [n for n, _ in cfg.obs_terms]
        if want != have:
            problems.append(f"observation order {have} differs from training {want}")
    return problems


def check_motion_fk(cfg, motion: Motion, motion_npz, xml=None):
    """The npz body poses must match MuJoCo FK of its own joint_pos (catches joint-order bugs)."""
    model = build_model(xml)
    d = mujoco.MjData(model)
    with np.load(motion_npz) as z:
        body_pos = z["body_pos_w"]
    worst = 0.0
    for f in np.linspace(0, motion.n - 1, min(motion.n, 25)).astype(int):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:36] = motion.root_pos[f], motion.root_quat[f], motion.joint_pos[f]
        mujoco.mj_kinematics(model, d)
        n = min(body_pos.shape[1], model.nbody - 1)
        worst = max(worst, float(np.abs(d.xpos[1:1 + n] - body_pos[f, :n]).max()))
    return worst


def gate(det, rnd, meta_problems, gates=GATES):
    checks = []

    def add(name, ok, detail):
        checks.append({"gate": name, "pass": bool(ok), "detail": detail})

    for r in det:
        add(f"{r.scenario}: no fall", r.success, f"fell at {r.fall_time:.2f}s" if not r.success else f"{r.duration:.1f}s")
    nom = next(r for r in det if r.scenario == "nominal")
    add("nominal joint RMS", nom.joint_rms["all"] <= gates["nominal_joint_rms_max_rad"],
        f"{nom.joint_rms['all']:.3f} rad (max {gates['nominal_joint_rms_max_rad']})")
    add("nominal body error", nom.body_err_m <= gates["nominal_body_err_max_m"],
        f"{nom.body_err_m * 100:.1f} cm (max {gates['nominal_body_err_max_m'] * 100:.0f})")
    for r in det:
        add(f"{r.scenario}: peak velocity", r.peak_velocity_ratio <= gates["peak_velocity_ratio_max"],
            f"{r.peak_velocity_ratio:.2f} of limit ({r.peak_velocity_joint})")
        add(f"{r.scenario}: torque saturation", r.torque_saturation_frac <= gates["torque_saturation_frac_max"],
            f"{r.torque_saturation_frac * 100:.1f}% of steps ({r.torque_saturation_joint})")
        add(f"{r.scenario}: hard joint stops", r.hard_limit_frac <= gates["hard_limit_frac_max"],
            f"{r.hard_limit_frac * 100:.2f}% of steps")
        add(f"{r.scenario}: self collision", r.self_collision_events <= gates["self_collision_events_max"],
            f"{r.self_collision_events} events")
    if rnd:
        rate = np.mean([r.success for r in rnd])
        add(f"randomised success ({len(rnd)} runs)", rate >= gates["random_success_min"],
            f"{rate * 100:.0f}% (min {gates['random_success_min'] * 100:.0f}%)")
    add("deploy.yaml matches training metadata", not meta_problems, "; ".join(meta_problems) or "ok / not checked")
    return checks


def write_report(out: Path, header: dict, det, rnd, checks):
    out.mkdir(parents=True, exist_ok=True)
    passed = all(c["pass"] for c in checks)
    (out / "report.json").write_text(json.dumps({**header, "ready_for_hardware_gantry_test": passed, "gates": checks,
                                                 "deterministic": [asdict(r) for r in det],
                                                 "randomised": [asdict(r) for r in rnd]}, indent=2, default=float))
    lines = [f"# Sim-to-sim readiness: {'PASS' if passed else 'FAIL'}", "",
             *(f"- {k}: `{v}`" for k, v in header.items() if k != "sha256"), "", "| gate | result | detail |", "|---|---|---|",
             *(f"| {c['gate']} | {'pass' if c['pass'] else '**FAIL**'} | {c['detail']} |" for c in checks), "",
             "| scenario | ok | joint RMS legs/waist/arms (rad) | body err (cm) | peak v | max tilt (rad) |",
             "|---|---|---|---|---|---|"]
    for r in det + rnd:
        j = r.joint_rms
        lines.append(f"| {r.scenario} | {'yes' if r.success else f'FELL {r.fall_time:.2f}s'} | "
                     f"{j['legs']:.3f}/{j['waist']:.3f}/{j['arms']:.3f} | {r.body_err_m * 100:.1f} | "
                     f"{r.peak_velocity_ratio:.2f} | {r.max_tilt_rad:.2f} |")
    lines += ["", "A pass means the policy is ready for a gantry (suspended) test, not for free standing.",
              "Follow wbt/README.md 'Hardware trial ladder'."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return passed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="deploy policy.onnx (single obs input)")
    p.add_argument("--deploy-yaml", required=True)
    p.add_argument("--motion", required=True, help="50 Hz motion npz used by the deploy (csv_to_npz output)")
    p.add_argument("--onnx-meta", help="the <run>.onnx export that carries mjlab metadata (recommended)")
    p.add_argument("--xml", help="mjlab g1.xml (default: installed mjlab or third_party)")
    p.add_argument("--time-start", type=float, default=0.0)
    p.add_argument("--time-end", type=float)
    p.add_argument("--suite", choices=["quick", "full"], default="full")
    p.add_argument("--random-runs", type=int, help="override randomised run count (quick 10, full 50)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cfg = load_deploy_yaml(a.deploy_yaml)
    motion = load_motion(a.motion)
    policy = OnnxPolicy(a.policy)
    meta_problems = check_metadata(cfg, onnx_metadata(a.onnx_meta)) if a.onnx_meta else []
    fk = check_motion_fk(cfg, motion, a.motion, a.xml)
    if fk > 0.02:
        meta_problems.append(f"motion npz body poses disagree with FK of its joint_pos by {fk:.3f} m (joint order?)")
    t = time.time()
    det = [run_scenario(cfg, motion, policy, s, a.time_start, a.time_end, a.xml) for s in deterministic_suite()]
    for r in det:
        print(f"{r.scenario:22s} {'ok' if r.success else f'FELL at {r.fall_time:.2f}s':16s} "
              f"joint RMS {r.joint_rms['all']:.3f} rad  body err {r.body_err_m * 100:.1f} cm", flush=True)
    n = a.random_runs if a.random_runs is not None else (10 if a.suite == "quick" else 50)
    rnd = []
    for s in random_suite(n, a.seed):
        rnd.append(run_scenario(cfg, motion, policy, s, a.time_start, a.time_end, a.xml))
        print(f"{s.name} {'ok' if rnd[-1].success else 'FELL'}", flush=True)
    checks = gate(det, rnd, meta_problems)
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    header = {"policy": str(a.policy), "deploy_yaml": str(a.deploy_yaml), "motion": str(a.motion),
              "sha256": {"policy": sha(a.policy), "deploy_yaml": sha(a.deploy_yaml), "motion": sha(a.motion)},
              "motion_seconds": round(motion.duration, 2), "time_range": [a.time_start, a.time_end],
              "motion_fk_max_err_m": round(fk, 5), "wall_s": round(time.time() - t, 1)}
    passed = write_report(a.out, header, det, rnd, checks)
    print(f"\n{'PASS' if passed else 'FAIL'}: {a.out / 'report.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
