"""wbt: whole-body tracking preparation and the deploy-faithful sim-to-sim tester (CPU, no GPU)."""
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest

from g1dance.joints import G1_JOINT_NAMES

pytest.importorskip("scipy")
try:
    from wbt.g1model import build_model, find_g1_xml, joint_specs
    find_g1_xml()
except FileNotFoundError:
    pytest.skip("mjlab g1.xml not available (scripts/setup_wbt.sh)", allow_module_level=True)

from wbt import deploy_policy as dp
from wbt.deploy_pkg import deploy_yaml_from_metadata
from wbt.feasibility import analyse, verdict
from wbt.npz import motion_npz
from wbt.prepare import ENTRY_POSE, FootFK, prepare, read_csv, write_csv
from wbt.sim2sim import Scenario, check_metadata, run_scenario

ROOT = Path(__file__).resolve().parents[1]
UNITREE = Path(os.environ.get("UNITREE_RL_MJLAB", ROOT / "third_party/unitree_rl_mjlab"))
SHIPPED = UNITREE / "deploy/robots/g1/config/policy/mimic/dance1_subject2"
# Unitree's shipped deploy.yaml for the G1 mimic policy (stiffness / damping).
UNITREE_KP = [40.2, 99.1, 40.2, 99.1, 28.5, 28.5] * 2 + [40.2, 28.5, 28.5] + [14.3] * 5 + [16.8] * 2 + [14.3] * 5 + [16.8] * 2
UNITREE_KD = [2.6, 6.3, 2.6, 6.3, 1.8, 1.8] * 2 + [2.6, 1.8, 1.8] + [0.9] * 5 + [1.1] * 2 + [0.9] * 5 + [1.1] * 2
UNITREE_SCALE = [0.55, 0.35, 0.55, 0.35, 0.44, 0.44] * 2 + [0.55, 0.44, 0.44] + [0.44] * 5 + [0.07] * 2 + [0.44] * 5 + [0.07] * 2


def standing_qpos(n=90, pose=ENTRY_POSE):
    fk = FootFK()
    q = np.zeros((n, 36))
    q[:, 3] = 1.0
    q[:, 7:] = pose
    q[:, 2] = fk.standing_height(q[0, 3:7], q[0, 7:])
    return q


def meta(default=ENTRY_POSE):
    return {"joint_names": list(G1_JOINT_NAMES), "joint_stiffness": np.array(UNITREE_KP),
            "joint_damping": np.array(UNITREE_KD), "default_joint_pos": np.asarray(default),
            "action_scale": np.array(UNITREE_SCALE),
            "observation_names": ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"]}


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    path = tmp_path_factory.mktemp("deploy") / "deploy.yaml"
    path.write_text(deploy_yaml_from_metadata(meta()))
    return dp.load_deploy_yaml(path)


def test_model_gains_match_unitree_deploy_yaml():
    specs = joint_specs()
    np.testing.assert_allclose([s.kp for s in specs], UNITREE_KP, atol=0.06)
    np.testing.assert_allclose([s.kd for s in specs], UNITREE_KD, atol=0.06)
    # mjlab G1_ACTION_SCALE = 0.25 * effort / stiffness (rounded to 2 decimals in the yaml)
    np.testing.assert_allclose([0.25 * s.effort / s.kp for s in specs], UNITREE_SCALE, atol=0.006)


def test_model_order_is_sdk_order():
    m = build_model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, 30)]
    assert tuple(names) == G1_JOINT_NAMES and m.nu == 29


def test_torso_quaternion_matches_mujoco_fk():
    """State_Mimic computes the anchor (torso) as pelvis * Rz(q12) Rx(q13) Ry(q14); it must equal FK."""
    m = build_model()
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    rng = np.random.default_rng(1)
    for _ in range(5):
        d.qpos[3:7] = dp.qmul(dp.qaxis("z", rng.uniform(-3, 3)), dp.qaxis("y", rng.uniform(-.3, .3)))
        d.qpos[7:] = rng.uniform(-0.4, 0.4, 29)
        mujoco.mj_kinematics(m, d)
        q = dp.torso_quat(d.qpos[3:7], *d.qpos[7 + 12:7 + 15])
        np.testing.assert_allclose(dp.qmat(q), d.xmat[torso].reshape(3, 3), atol=1e-6)


def test_generated_deploy_yaml_roundtrip(cfg):
    assert cfg.obs_dim == 154 and cfg.step_dt == 0.02
    np.testing.assert_allclose(cfg.stiffness, UNITREE_KP)
    np.testing.assert_allclose(cfg.action_offset, ENTRY_POSE)
    assert [n for n, _ in cfg.obs_terms] == ["motion_command", "motion_anchor_ori_b", "base_ang_vel",
                                              "joint_pos_rel", "joint_vel_rel", "last_action"]
    assert check_metadata(cfg, meta()) == []
    assert check_metadata(cfg, meta(default=np.zeros(29)))  # a mismatching default pose is reported
    with pytest.raises(ValueError, match="No-State-Estimation"):
        deploy_yaml_from_metadata({**meta(), "observation_names": ["command", "motion_anchor_pos_b"]})


def test_generated_deploy_yaml_clip(tmp_path):
    hard = build_model().jnt_range[1:30]
    p = tmp_path / "d.yaml"
    p.write_text(deploy_yaml_from_metadata(meta(), clip_ranges=hard))
    c = dp.load_deploy_yaml(p)
    np.testing.assert_allclose(c.action_clip, hard)


class Recorder:
    """Policy stub: remembers observations, returns zero actions."""
    obs_dim = 154

    def __init__(self):
        self.seen = []

    def __call__(self, obs):
        self.seen.append(obs)
        return np.zeros(29)


def test_observation_layout_and_frame_timing(cfg):
    q = standing_qpos(30)
    q[:, 7 + 15] = np.linspace(0, 0.5, 30)  # left shoulder pitch moves so frames are distinguishable
    motion = dp.Motion(**{k: v for k, v in _as_motion(q).items()})
    pol = Recorder()
    ctrl = dp.MimicController(cfg, motion, pol)
    pelvis = motion.root_quat[0]
    ctrl.enter(pelvis, motion.joint_pos[0])
    target, _, f = ctrl.step(pelvis, np.zeros(3), motion.joint_pos[0], np.zeros(29))
    assert f == 0  # first step uses the reference at time_start (motion->update before env->step)
    obs = pol.seen[0]
    np.testing.assert_allclose(obs[:29], motion.joint_pos[0])
    np.testing.assert_allclose(obs[58:64], [1, 0, 0, 1, 0, 0], atol=1e-6)  # robot on reference: identity
    np.testing.assert_allclose(target, cfg.action_offset)  # zero action -> default pose
    _, _, f2 = ctrl.step(pelvis, np.zeros(3), motion.joint_pos[0], np.zeros(29))
    assert f2 == 1


def test_anchor_obs_cancels_imu_yaw(cfg):
    motion = dp.Motion(**_as_motion(standing_qpos(10)))
    obs = []
    for yaw in (0.0, 1.3):
        pol = Recorder()
        ctrl = dp.MimicController(cfg, motion, pol)
        quat = dp.qmul(dp.qaxis("z", yaw), motion.root_quat[0])
        ctrl.enter(quat, motion.joint_pos[0])
        ctrl.step(quat, np.zeros(3), motion.joint_pos[0], np.zeros(29))
        obs.append(pol.seen[0])
    np.testing.assert_allclose(obs[0], obs[1], atol=1e-6)


def _as_motion(qpos30):
    z = motion_npz(qpos30)
    return {"joint_pos": z["joint_pos"].astype(float), "joint_vel": z["joint_vel"].astype(float),
            "root_pos": z["body_pos_w"][:, 0].astype(float), "root_quat": z["body_quat_w"][:, 0].astype(float),
            "fps": 50.0}


def test_sim_detects_a_fall(cfg):
    """A policy that commands nothing useful (zero gains) must be reported as a fall, not a pass."""
    motion = dp.Motion(**_as_motion(standing_qpos(150)))
    limp = {**cfg.__dict__}
    limp.update(stiffness=np.zeros(29), damping=np.full(29, 0.5))
    res = run_scenario(dp.DeployCfg(**limp), motion, Recorder(), Scenario("limp"))
    assert not res.success and res.fall_time is not None


def test_prepare_lead_in_out_and_csv_roundtrip(tmp_path):
    q = standing_qpos(60)
    q[:, 7 + 15] = 0.8  # dance pose differs from the entry stance
    q[:, :2] += [1.0, -2.0]
    out, info = prepare(q, lead_in=1.0, lead_out=1.0)
    np.testing.assert_allclose(out[0, 7:], ENTRY_POSE, atol=1e-9)
    np.testing.assert_allclose(out[-1, 7:], ENTRY_POSE, atol=1e-9)
    start = int(round(info["dance_start_s"] * 30))
    np.testing.assert_allclose(out[start, 7:], q[0, 7:], atol=1e-9)
    np.testing.assert_allclose(out[0, :2], 0, atol=1e-9)  # moved to the origin
    fk = FootFK()
    low = np.array([fk.lowest(r).min() for r in out])
    assert np.abs(low).max() < 0.01  # feet stay on the floor through the transitions
    assert np.abs(np.diff(out[:, 7:], axis=0)).max() * 30 < 2.0  # no velocity jump at the joins
    write_csv(tmp_path / "m.csv", out)
    np.testing.assert_allclose(read_csv(tmp_path / "m.csv"), out, atol=2e-6)


def test_feasibility_flags_hard_limit_and_crossed_legs():
    ok, _ = analyse(standing_qpos(30))
    assert not [c for c in verdict(ok) if c["level"] == "fail"]
    bad = standing_qpos(30)
    bad[:, 7 + 3] = 3.2  # knee far past its 2.88 rad stop
    m, _ = analyse(bad)
    assert any(c["check"] == "hard joint range" and c["level"] == "fail" for c in verdict(m))
    crossed = standing_qpos(30)
    crossed[:, 7 + 1], crossed[:, 7 + 7] = -0.5, 0.5  # hip roll both legs inward: thighs intersect
    m, _ = analyse(crossed)
    assert m["leg_collision_frac"] > 0.5
    assert any(c["check"] == "leg-leg collision" and c["level"] == "fail" for c in verdict(m))


@pytest.mark.skipif(not SHIPPED.exists(), reason="unitree_rl_mjlab checkout not present")
def test_unitree_shipped_policy_dances_in_our_sim():
    """End-to-end validation of the port: Unitree's own policy must track its own dance."""
    cfg = dp.load_deploy_yaml(SHIPPED / "params/deploy.yaml")
    motion = dp.load_motion(SHIPPED / "params/dance1_subject2.npz")
    pol = dp.OnnxPolicy(SHIPPED / "exported/policy.onnx")
    for sc in (Scenario("nominal"), Scenario("stand", start="stand")):
        r = run_scenario(cfg, motion, pol, sc, 0.0, 6.0)
        assert r.success and r.joint_rms["all"] < 0.15 and r.body_err_m < 0.06, r


@pytest.mark.skipif(not (UNITREE / ".git").exists(), reason="unitree_rl_mjlab checkout not present")
def test_safety_patch_applies():
    patch = ROOT / "wbt/patches/unitree_rl_mjlab_mimic_safety.patch"
    r = subprocess.run(["git", "-C", str(UNITREE), "apply", "--check", str(patch)], capture_output=True, text=True)
    already = subprocess.run(["git", "-C", str(UNITREE), "apply", "--check", "--reverse", str(patch)],
                             capture_output=True, text=True)
    assert r.returncode == 0 or already.returncode == 0, r.stderr
