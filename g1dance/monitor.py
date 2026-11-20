"""Safety monitor and the pre-playback gate.

Three waist measurements are kept apart on purpose:
  * ``waist_joint_pitch``   - encoder angle of motor 14 (waist_pitch_joint)
  * ``waist_pitch_error``   - encoder angle minus the commanded waist pitch
  * ``pelvis_imu_pitch``    - pelvis IMU pitch relative to its value at acquisition
A stop always names the check, the joint/measurement, the value and the
threshold. Thresholds come from ``config.Gates`` and are never adjusted here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Gates
from .joints import G1_JOINT_NAMES, LEG_IDS, N_UPPER, UPPER_IDS, UPPER_NAMES, WAIST_PITCH, WAIST_ROLL

MOTION_PHASES = ("acquire", "entry", "settle", "playback", "end_hold", "return", "release")


@dataclass
class RobotState:
    stamp: float  # host monotonic time the message arrived [s]
    q: np.ndarray  # (29,) rad, all motors (legs are read, never written)
    dq: np.ndarray  # (29,) rad/s
    imu_rpy: np.ndarray  # (3,) rad, LowState.imu_state.rpy
    mode_machine: int | None = None
    torso_rpy: np.ndarray | None = None  # optional second IMU, if available

    @property
    def upper_q(self) -> np.ndarray:
        return self.q[list(UPPER_IDS)]

    @property
    def upper_dq(self) -> np.ndarray:
        return self.dq[list(UPPER_IDS)]


@dataclass
class Stop:
    check: str  # e.g. "tracking_error", "waist_joint_pitch", "pelvis_imu_pitch"
    subject: str  # joint name or measurement name
    value: float
    threshold: float
    unit: str
    phase: str
    t: float  # run time [s]
    t_ref: float | None = None

    def message(self) -> str:
        v, th = self.value, self.threshold
        tr = "" if self.t_ref is None or np.isnan(self.t_ref) else f", dance t={self.t_ref:.2f}s"
        if self.unit == "-":
            return f"STOP [{self.check}] {self.subject} in phase '{self.phase}' at t={self.t:.2f}s{tr}"
        if self.unit == "rad":
            vs, ts = f"{v:+.4f} rad ({np.degrees(v):+.2f} deg)", f"{th:.4f} rad ({np.degrees(th):.2f} deg)"
        else:
            vs, ts = f"{v:+.4f} {self.unit}", f"{th:.4f} {self.unit}"
        return f"STOP [{self.check}] {self.subject}: |{vs}| > {ts} in phase '{self.phase}' at t={self.t:.2f}s{tr}"

    def as_dict(self):
        return {"check": self.check, "subject": self.subject, "value": float(self.value),
                "threshold": float(self.threshold), "unit": self.unit, "phase": self.phase,
                "t": float(self.t), "t_ref": None if self.t_ref is None or np.isnan(self.t_ref) else float(self.t_ref),
                "message": self.message()}


class Monitor:
    def __init__(self, gates: Gates, waist_roll_pitch_actuated: bool = True):
        self.g = gates
        self.track = gates.track_vector()
        self.imu0: np.ndarray | None = None
        self.waist_rp = waist_roll_pitch_actuated
        self._leg_fast_ticks = 0

    def leg_speed(self, state: RobotState) -> tuple[float, str]:
        """Largest |dq| over the leg joints (read-only) and which joint."""
        d = np.abs(np.asarray(state.dq, float)[list(LEG_IDS)])
        i = int(np.argmax(d))
        return float(np.asarray(state.dq, float)[LEG_IDS[i]]), G1_JOINT_NAMES[LEG_IDS[i]]

    def capture_reference_attitude(self, state: RobotState):
        self.imu0 = np.array(state.imu_rpy, float)

    def measurements(self, state: RobotState, cmd: np.ndarray) -> dict[str, float]:
        q = state.upper_q
        imu = np.array(state.imu_rpy, float)
        d_imu = imu - (self.imu0 if self.imu0 is not None else imu)
        return {
            "waist_joint_pitch": float(q[WAIST_PITCH]),
            "waist_pitch_error": float(q[WAIST_PITCH] - cmd[WAIST_PITCH]),
            "waist_cmd_pitch": float(cmd[WAIST_PITCH]),
            "pelvis_imu_pitch": float(d_imu[1]),
            "pelvis_imu_roll": float(d_imu[0]),
        }

    def check(self, state: RobotState, cmd: np.ndarray, weight: float, phase: str, t: float,
              now: float, t_ref: float | None = None, tracking: bool = True) -> list[Stop]:
        g, out = self.g, []
        mk = lambda c, s, v, th, u="rad": Stop(c, s, v, th, u, phase, t, t_ref)
        age = now - state.stamp
        if age > g.state_timeout_s:
            out.append(mk("state_freshness", "rt/lowstate age", age, g.state_timeout_s, "s"))
            return out  # nothing else is trustworthy
        q, dq = state.upper_q, state.upper_dq
        err = q - cmd
        if tracking and weight >= 1.0 - 1e-9:
            for j in range(N_UPPER):
                if not self.waist_rp and j in (WAIST_ROLL, WAIST_PITCH):
                    continue
                if abs(err[j]) > self.track[j]:
                    out.append(mk("tracking_error", UPPER_NAMES[j], err[j], self.track[j]))
        if self.waist_rp:
            if abs(q[WAIST_PITCH]) > g.waist_pitch_abs:
                out.append(mk("waist_joint_pitch", UPPER_NAMES[WAIST_PITCH], q[WAIST_PITCH], g.waist_pitch_abs))
            if abs(q[WAIST_ROLL]) > g.waist_roll_abs:
                out.append(mk("waist_joint_roll", UPPER_NAMES[WAIST_ROLL], q[WAIST_ROLL], g.waist_roll_abs))
        if self.imu0 is not None:
            d = np.array(state.imu_rpy, float) - self.imu0
            if abs(d[1]) > g.imu_pitch_delta:
                out.append(mk("pelvis_imu_pitch", "imu_state.rpy[1] - start", d[1], g.imu_pitch_delta))
            if abs(d[0]) > g.imu_roll_delta:
                out.append(mk("pelvis_imu_roll", "imu_state.rpy[0] - start", d[0], g.imu_roll_delta))
        for j in range(N_UPPER):
            if abs(dq[j]) > g.joint_speed:
                out.append(mk("joint_speed", UPPER_NAMES[j], dq[j], g.joint_speed, "rad/s"))
        if g.leg_step_dq > 0:
            v, name = self.leg_speed(state)
            self._leg_fast_ticks = self._leg_fast_ticks + 1 if abs(v) > g.leg_step_dq else 0
            if self._leg_fast_ticks >= max(1, int(g.leg_step_ticks)):
                out.append(mk("leg_step", name, v, g.leg_step_dq, "rad/s"))
        return out


@dataclass
class GateResult:
    passed: bool
    cmd_max_err: float
    cmd_worst_joint: str
    meas_errors: dict[str, float]
    failures: list[str]

    def as_dict(self):
        return {"passed": self.passed, "cmd_max_err_rad": self.cmd_max_err, "cmd_worst_joint": self.cmd_worst_joint,
                "meas_errors_deg": {k: round(float(np.degrees(v)), 3) for k, v in self.meas_errors.items()},
                "failures": self.failures}


def entry_gate(cmd: np.ndarray, target: np.ndarray, meas: np.ndarray, gates: Gates,
               waist_roll_pitch_actuated: bool = True) -> GateResult:
    """Both the COMMANDED and the MEASURED pose must be at the frame-0 target."""
    cerr = np.abs(cmd - target)
    jw = int(np.argmax(cerr))
    merr = meas - target
    tol = np.full(N_UPPER, gates.entry_meas_tol_arm)
    tol[:3] = gates.entry_meas_tol_waist
    fails = []
    if cerr[jw] > gates.entry_cmd_tol:
        fails.append(f"commanded {UPPER_NAMES[jw]} is {cerr[jw]:.2e} rad from frame-0 target "
                     f"(tol {gates.entry_cmd_tol:.1e})")
    for j in range(N_UPPER):
        if not waist_roll_pitch_actuated and j in (WAIST_ROLL, WAIST_PITCH):
            continue
        if abs(merr[j]) > tol[j]:
            fails.append(f"measured {UPPER_NAMES[j]} off target by {np.degrees(merr[j]):+.2f} deg "
                         f"(tol {np.degrees(tol[j]):.2f} deg)")
    return GateResult(not fails, float(cerr[jw]), UPPER_NAMES[jw],
                      {UPPER_NAMES[j]: float(merr[j]) for j in range(N_UPPER)}, fails)
