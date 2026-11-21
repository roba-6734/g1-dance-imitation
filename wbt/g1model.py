"""MuJoCo G1 29-DoF model that mirrors mjlab 1.2.0's G1 training setup.

Mirrors ``mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`` (mjlab==1.2.0, the
version pinned by unitree_rl_mjlab): per-motor reflected-inertia armature,
built-in position actuators (kp, kv, force range), full collisions with condim 3
foot capsules (friction 0.6, priority 1) and condim 1 elsewhere, 5 ms timestep,
implicitfast integrator, 10 solver / 20 line-search iterations.

The joint order is the MJCF order, which equals the Unitree SDK motor order 0..28
(``g1dance.joints.G1_JOINT_NAMES``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from g1dance.joints import G1_JOINT_NAMES


def _two_stage(rotor, gear):
    assert gear[0] == 1
    return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


ARMATURE = {
    "5020": _two_stage((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16)),
    "7520_14": _two_stage((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
    "7520_22": _two_stage((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
    "4010": _two_stage((0.068e-4, 0.0, 0.0), (1, 5, 5)),
}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
VELOCITY = {"5020": 37.0, "7520_14": 32.0, "7520_22": 20.0, "4010": 22.0}
OMEGA = 10 * 2.0 * 3.1415926535  # mjlab NATURAL_FREQ (10 Hz)
ZETA = 2.0

# (regex, motor type, multiplier). Waist pitch/roll and ankles are two 5020s in a linkage.
GROUPS = (
    (r".*_elbow_joint|.*_shoulder_(pitch|roll|yaw)_joint|.*_wrist_roll_joint", "5020", 1),
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", "7520_14", 1),
    (r".*_hip_roll_joint|.*_knee_joint", "7520_22", 1),
    (r".*_wrist_(pitch|yaw)_joint", "4010", 1),
    (r"waist_(pitch|roll)_joint", "5020", 2),
    (r".*_ankle_(pitch|roll)_joint", "5020", 2),
)
FOOT_GEOM = re.compile(r"^(left|right)_foot[1-7]_collision$")


@dataclass(frozen=True)
class JointSpec:
    name: str
    armature: float
    kp: float
    kd: float
    effort: float
    velocity: float


def joint_specs() -> list[JointSpec]:
    specs = []
    for name in G1_JOINT_NAMES:
        hits = [(m, k) for rx, m, k in GROUPS if re.fullmatch(rx, name)]
        if len(hits) != 1:
            raise ValueError(f"{name}: expected one actuator group, got {hits}")
        motor, k = hits[0]
        a = ARMATURE[motor] * k
        specs.append(JointSpec(name, a, a * OMEGA ** 2, 2 * ZETA * a * OMEGA, EFFORT[motor] * k, VELOCITY[motor]))
    return specs


def find_g1_xml(explicit: str | Path | None = None) -> Path:
    """mjlab's asset_zoo g1.xml (what training uses): explicit path, installed mjlab, or third_party/mjlab_g1."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    try:
        import importlib.util
        spec = importlib.util.find_spec("mjlab")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml")
    except (ImportError, ValueError):
        pass
    # Not unitree_rl_mjlab/src/assets/.../g1.xml: training loads mjlab's asset_zoo model, and Unitree's
    # copy differs (collision condim/priority defaults, pelvis height).
    candidates.append(Path(__file__).resolve().parents[1] / "third_party/mjlab_g1/g1.xml")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("mjlab G1 model not found. Pass --xml, install mjlab==1.2.0, or run scripts/setup_wbt.sh. "
                            f"Tried: {[str(c) for c in candidates]}")


def build_model(xml: str | Path | None = None, *, timestep=0.005, foot_friction=0.6,
                kp=None, kd=None, effort_scale=1.0, explicit_pd=False, self_collision=True) -> mujoco.MjModel:
    """Compile the training-equivalent G1 with a floor.

    ``kp``/``kd`` (29,) override the mjlab gains (pass deploy.yaml stiffness/damping).
    ``explicit_pd`` uses torque motors so the caller applies PD itself (motor-board emulation).
    """
    spec = mujoco.MjSpec.from_file(str(find_g1_xml(xml)))
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    floor.friction = [1.0, 0.005, 0.0001]
    light = spec.worldbody.add_light()
    light.pos, light.dir = [0, 0, 4], [0, 0, -1]
    for g in spec.geoms:
        if not g.name or not g.name.endswith("_collision"):
            continue
        if FOOT_GEOM.match(g.name):
            g.condim, g.priority, g.friction = 3, 1, [foot_friction, 0.005, 0.0001]
            g.contype, g.conaffinity = 1, 1
        else:
            g.condim = 1
            g.contype, g.conaffinity = (1, 1) if self_collision else (0, 1)
    specs = joint_specs()
    kp = np.array([s.kp for s in specs]) if kp is None else np.asarray(kp, float)
    kd = np.array([s.kd for s in specs]) if kd is None else np.asarray(kd, float)
    for i, s in enumerate(specs):
        j = spec.joint(s.name)
        j.armature = s.armature
        act = spec.add_actuator()
        act.name, act.target = s.name, s.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-s.effort * effort_scale, s.effort * effort_scale]
        if explicit_pd:
            act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
            act.gainprm[0] = 1.0
        else:  # MuJoCo <position> actuator == mjlab BuiltinPositionActuator
            act.set_to_position(kp=float(kp[i]), kv=float(kd[i]))
    model = spec.compile()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0]) for a in range(model.nu)]
    if tuple(names) != G1_JOINT_NAMES or model.nq != 36:
        raise ValueError(f"Unexpected G1 joint/actuator order: {names}")
    return model


def soft_limits(model: mujoco.MjModel, factor=0.9) -> np.ndarray:
    """mjlab soft_joint_pos_limit_factor=0.9 around the range centre, (29, 2)."""
    rng = model.jnt_range[1:30].copy()
    mid, half = rng.mean(1), (rng[:, 1] - rng[:, 0]) / 2
    return np.stack([mid - factor * half, mid + factor * half], 1)
