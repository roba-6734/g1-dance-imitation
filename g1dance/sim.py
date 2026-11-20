"""MuJoCo simulation backend (offline only).

Model: MuJoCo Menagerie ``unitree_g1/g1.xml`` (29 DoF) with the floating base
REMOVED, i.e. the pelvis is welded to the world at standing height. Legs are
held at the Menagerie "stand" posture.

What this simulates: upper-body joint dynamics under the same PD law and gains
that arm_sdk applies (tau = kp*(q_cmd - q) + kd*(0 - dq)), with gravity, URDF
effort limits and Menagerie armature/friction.

What it does NOT simulate: Unitree's locomotion/balance controller, pelvis
motion, the pelvis IMU (reported as zero), firmware-side weight blending
details, or motor/transmission compliance. A previous hardware run showed the
physical waist pitching ~20.8 deg against a 6.9 deg command while a simulator
under-predicted it, so treat waist results here as optimistic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .joints import N_UPPER, UPPER_IDS, URDF_EFFORT_LIMIT
from .monitor import RobotState

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE_G1 = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
PELVIS_HEIGHT = 0.793
# Model of Unitree's own upper-body hold at arm_sdk weight 0 (unknown on the
# real robot). A real G1 in locomotion mode holds its stance still (measured
# |dq| < 0.05 rad/s, waist steady), so it is modelled as stiff: Menagerie's
# default position-actuator gain. Only affects phases with weight < 1.
UNITREE_HOLD_KP = 500.0
UNITREE_HOLD_KD = 10.0


def _mujoco():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    return mujoco


def stand_pose_29() -> np.ndarray:
    """Menagerie 'stand' keyframe joint angles (ids 0..28)."""
    q = np.zeros(29)
    q[15], q[16], q[18] = 0.2, 0.2, 1.28
    q[22], q[23], q[25] = 0.2, -0.2, 1.28
    return q


def assumed_start_upper() -> np.ndarray:
    """Offline stand-in for the measured starting pose (Menagerie stand arms)."""
    return stand_pose_29()[list(UPPER_IDS)]


def load_model(fixed_base: bool):
    mujoco = _mujoco()
    xml = MENAGERIE_G1 / ("scene.xml")
    if not xml.exists():
        raise FileNotFoundError(f"{xml} missing; run scripts/fetch_model.sh")
    spec = mujoco.MjSpec.from_file(str(xml))
    if fixed_base:
        j = spec.joint("floating_base_joint")
        spec.delete(j)
        spec.body("pelvis").pos = [0, 0, PELVIS_HEIGHT]
        for k in list(spec.keys):
            spec.delete(k)
    model = spec.compile()
    return model


class SimBackend:
    name = "mujoco_fixed_pelvis"
    data_kind = "simulated"

    def __init__(self, start_upper: np.ndarray, control_dt: float = 0.02, physics_dt: float = 0.002):
        mujoco = _mujoco()
        self.mj = mujoco
        self.model = load_model(fixed_base=True)
        self.model.opt.timestep = physics_dt
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(control_dt / physics_dt))
        self.control_dt = control_dt
        q29 = stand_pose_29()
        q29[list(UPPER_IDS)] = start_upper
        self.data.qpos[:29] = q29
        self.data.ctrl[:29] = q29
        # Upper-body actuators are replaced by our own arm_sdk-style PD torque.
        for i in UPPER_IDS:
            self.model.actuator_gainprm[i, 0] = 0.0
            self.model.actuator_biasprm[i, 1:3] = 0.0
        self.hold_upper = np.array(start_upper, float)  # what "Unitree" holds at weight 0
        self.cmd = (self.hold_upper.copy(), np.zeros(N_UPPER), np.zeros(N_UPPER), 0.0)
        self.cmd_dq = np.zeros(N_UPPER)
        self.torso_site = self.model.site("imu_in_torso").id
        mujoco.mj_forward(self.model, self.data)
        # Let gravity settle under Unitree's (stiff) hold so t=0 is at rest at the start pose.
        for _ in range(int(1.0 / physics_dt)):
            self._step()
        self.t = 0.0

    def _step(self):
        ids = list(UPPER_IDS)
        q, dq = self.data.qpos[ids], self.data.qvel[ids]
        qc, kp, kd, w = self.cmd
        tau_sdk = kp * (qc - q) + kd * (self.cmd_dq - dq)
        tau_hold = UNITREE_HOLD_KP * (self.hold_upper - q) - UNITREE_HOLD_KD * dq  # Unitree hold, weight 0
        tau = np.clip(w * tau_sdk + (1 - w) * tau_hold, -URDF_EFFORT_LIMIT, URDF_EFFORT_LIMIT)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[ids] = tau
        self.mj.mj_step(self.model, self.data)

    def read_state(self) -> RobotState:
        xm = self.data.site_xmat[self.torso_site].reshape(3, 3)
        torso = np.array([np.arctan2(xm[2, 1], xm[2, 2]), np.arcsin(-np.clip(xm[2, 0], -1, 1)),
                          np.arctan2(xm[1, 0], xm[0, 0])])
        return RobotState(stamp=self.t, q=self.data.qpos[:29].copy(), dq=self.data.qvel[:29].copy(),
                          imu_rpy=np.zeros(3), mode_machine=5, torso_rpy=torso)

    def send(self, q, kp, kd, weight, dq=None):
        self.cmd = (np.array(q, float), np.array(kp, float), np.array(kd, float), float(weight))
        self.cmd_dq = np.zeros(N_UPPER) if dq is None else np.array(dq, float)

    def wait_tick(self):
        for _ in range(self.n_sub):
            self._step()
        self.t += self.control_dt

    def now(self) -> float:
        return self.t
