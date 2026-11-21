"""Python port of unitree_rl_mjlab's real-robot mimic controller.

Mirrors deploy/robots/g1/src/State_Mimic.cpp + deploy/include/isaaclab (commit
1425b15f, unitree_rl_mjlab) term by term, so a policy that works here is fed
exactly what the robot's C++ controller would feed it:

* observation terms in deploy.yaml order, each ``term * scale`` (history_length 1);
* ``motion_command``  = reference joint_pos(29) + joint_vel(29) at frame floor(t / 0.02);
* ``motion_anchor_ori_b`` = first two columns of R(robot_torso)^T R(init * ref_torso),
  where torso = pelvis IMU quaternion * Rz(q12) Rx(q13) Ry(q14);
* ``init_quat`` = yaw(robot torso) * yaw(reference pelvis)^T, fixed at entry;
* ``base_ang_vel`` = pelvis gyro (body frame); ``joint_pos_rel`` = q - default;
  ``last_action`` = previous raw action;
* target q = offset + scale * action; kp/kd = deploy.yaml stiffness/damping.

The motion file is the 50 Hz npz from mjlab's csv_to_npz (joint order = SDK order).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_TERMS = ("motion_command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos_rel",
                   "joint_vel_rel", "last_action", "projected_gravity")


# ---- quaternion helpers (w, x, y, z) -------------------------------------------------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qaxis(axis, angle):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), *v])


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qyaw(q):
    """isaaclab::yawQuaternion."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def torso_quat(pelvis_quat, q12, q13, q14):
    return qmul(qmul(qmul(pelvis_quat, qaxis("z", q12)), qaxis("x", q13)), qaxis("y", q14))


def qrotate_inv(q, v):
    return qmat(q).T @ v


# ---- config ------------------------------------------------------------------------------------
@dataclass
class DeployCfg:
    step_dt: float
    stiffness: np.ndarray
    damping: np.ndarray
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    action_clip: np.ndarray | None
    obs_terms: list[tuple[str, np.ndarray]]  # (name, scale)
    joint_ids_map: np.ndarray
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def obs_dim(self):
        return sum(len(s) for _, s in self.obs_terms)


def load_deploy_yaml(path: str | Path) -> DeployCfg:
    import yaml

    d = yaml.safe_load(Path(path).read_text())
    act = d["actions"]["JointPositionAction"]
    if act.get("joint_ids") is not None:
        raise ValueError("JointPositionAction.joint_ids is not supported (the C++ ignores it too: TODO in joint_actions.h)")
    terms = []
    for name, t in d["observations"].items():
        if name not in SUPPORTED_TERMS:
            raise ValueError(f"Observation term {name!r} not supported by this port")
        if int(t.get("history_length", 1)) != 1:
            raise ValueError(f"{name}: history_length != 1 not supported")
        if t.get("clip") is not None:
            raise ValueError(f"{name}: clip not supported")
        terms.append((name, np.asarray(t["scale"], float)))
    n = len(d["joint_ids_map"])
    clip = act.get("clip")
    cfg = DeployCfg(float(d["step_dt"]), np.asarray(d["stiffness"], float), np.asarray(d["damping"], float),
                    np.asarray(d["default_joint_pos"], float), np.asarray(act["scale"], float),
                    np.asarray(act["offset"], float), None if clip is None else np.asarray(clip, float),
                    terms, np.asarray(d["joint_ids_map"], int), d)
    for k in ("stiffness", "damping", "default_joint_pos", "action_scale", "action_offset"):
        if getattr(cfg, k).shape != (n,):
            raise ValueError(f"deploy.yaml {k} must have {n} entries")
    if not np.array_equal(cfg.joint_ids_map, np.arange(29)):
        raise ValueError("This port assumes joint_ids_map == 0..28 (policy order == SDK order)")
    return cfg


# ---- motion --------------------------------------------------------------------------------------
@dataclass
class Motion:
    joint_pos: np.ndarray  # (T, 29) SDK order
    joint_vel: np.ndarray
    root_pos: np.ndarray  # (T, 3) body 0 = pelvis
    root_quat: np.ndarray  # (T, 4) wxyz
    fps: float

    @property
    def n(self):
        return len(self.joint_pos)

    @property
    def duration(self):  # MotionLoader_: num_frames * dt
        return self.n / self.fps

    def frame(self, t):
        f = int(np.floor(np.clip(t, 0.0, self.duration) * self.fps))
        return min(f, self.n - 1)


def load_motion(path: str | Path) -> Motion:
    with np.load(path) as z:
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        if abs(fps - 50) > 1e-6:
            raise ValueError(f"deploy MotionLoader_ hard-codes 50 Hz; {path} is {fps} Hz")
        m = Motion(z["joint_pos"].astype(float), z["joint_vel"].astype(float), z["body_pos_w"][:, 0].astype(float),
                   z["body_quat_w"][:, 0].astype(float), fps)
    if m.joint_pos.shape[1] != 29:
        raise ValueError("motion joint_pos must be (T, 29)")
    return m


# ---- policy ---------------------------------------------------------------------------------------
class OnnxPolicy:
    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()
        if len(inp) != 1:
            raise ValueError(f"{path}: expected the deploy policy.onnx (single 'obs' input), got {[i.name for i in inp]}")
        self.name = inp[0].name
        self.obs_dim = int(inp[0].shape[-1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.name: obs.astype(np.float32)[None]})[0][0].astype(float)


class MimicController:
    """One State_Mimic episode. Feed it robot state each 20 ms, get joint targets back."""

    def __init__(self, cfg: DeployCfg, motion: Motion, policy, time_start=0.0, time_end=None):
        if policy.obs_dim != cfg.obs_dim:
            raise ValueError(f"policy expects {policy.obs_dim} obs, deploy.yaml builds {cfg.obs_dim}")
        self.cfg, self.motion, self.policy = cfg, motion, policy
        self.t0 = float(np.clip(time_start, 0, motion.duration))
        self.t1 = motion.duration if time_end is None else float(np.clip(time_end, 0, motion.duration))
        self.episode_length = 0
        self.action = np.zeros(29)
        self.init_quat = None

    def enter(self, pelvis_quat, q):
        """State_Mimic::enter -> policy thread start: fix the yaw alignment."""
        f = self.motion.frame(self.t0)
        ref_yaw = qmat(qyaw(self.motion.root_quat[f]))
        robot_yaw = qmat(qyaw(torso_quat(pelvis_quat, q[12], q[13], q[14])))
        rot = robot_yaw @ ref_yaw.T
        from scipy.spatial.transform import Rotation

        x, y, z, w = Rotation.from_matrix(rot).as_quat()
        self.init_quat = np.array([w, x, y, z])
        self.episode_length = 0
        self.action = np.zeros(29)

    @property
    def time(self):
        return self.episode_length * self.cfg.step_dt + self.t0

    @property
    def done(self):  # registered_checks: time out -> end_state
        return self.episode_length * self.cfg.step_dt > self.t1

    def observe(self, pelvis_quat, gyro_b, q, dq, f):
        parts = []
        for name, scale in self.cfg.obs_terms:
            if name == "motion_command":
                v = np.concatenate([self.motion.joint_pos[f], self.motion.joint_vel[f]])
            elif name == "motion_anchor_ori_b":
                jp = self.motion.joint_pos[f]
                ref = torso_quat(self.motion.root_quat[f], jp[12], jp[13], jp[14])
                real = torso_quat(pelvis_quat, q[12], q[13], q[14])
                rot = qmat(qmul(qconj(qmul(self.init_quat, ref)), real)).T
                v = np.array([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])
            elif name == "base_ang_vel":
                v = np.asarray(gyro_b, float)
            elif name == "projected_gravity":
                v = qrotate_inv(pelvis_quat, np.array([0, 0, -1.0]))
            elif name == "joint_pos_rel":
                v = q - self.cfg.default_joint_pos
            elif name == "joint_vel_rel":
                v = np.asarray(dq, float)
            elif name == "last_action":
                v = self.action
            if len(v) != len(scale):
                raise ValueError(f"{name}: {len(v)} values vs {len(scale)} scales")
            parts.append(v * scale)
        return np.concatenate(parts)

    def step(self, pelvis_quat, gyro_b, q, dq):
        """motion->update(episode_length*dt + t0) then ManagerBasedRLEnv::step (which increments
        episode_length before computing obs). Returns (joint targets, raw action, reference frame)."""
        f = self.motion.frame(self.time)
        self.episode_length += 1
        obs = self.observe(pelvis_quat, gyro_b, q, dq, f)
        self.action = self.policy(obs)
        target = self.action * self.cfg.action_scale + self.cfg.action_offset
        if self.cfg.action_clip is not None:
            target = np.clip(target, self.cfg.action_clip[:, 0], self.cfg.action_clip[:, 1])
        return target, self.action.copy(), f


def onnx_metadata(path: str | Path) -> dict:
    """mjlab attaches joint names / gains / default pose / action scale to the <run>.onnx export."""
    import onnxruntime as ort

    meta = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    out = {}
    for k, v in meta.items():
        parts = v.split(",")
        try:
            out[k] = np.array([float(p) for p in parts]) if len(parts) > 1 else float(v)
        except ValueError:
            out[k] = parts if len(parts) > 1 else v
    return out
