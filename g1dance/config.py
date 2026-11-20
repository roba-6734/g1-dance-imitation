"""Run configuration. Every value that shapes a run is explicit here and is
written verbatim to the run log (``RunConfig.to_dict``).

Amplitude convention (identical for every group):
    target = anchor + scale * (reference - anchor)
where ``anchor`` is the measured starting pose of that joint. So scale 0 holds
the measured starting value and scale 1 requests the recorded value. This is
what makes ``waist_pitch = 0`` hold the measured starting pitch.

Speed is a separate time-warp of the reference clock and never touches the
amplitude.
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .joints import ARM_COLS, N_UPPER, WAIST_PITCH, WAIST_ROLL, WAIST_YAW


@dataclass
class Scales:
    arms: float = 1.0
    waist_yaw: float = 1.0
    waist_roll: float = 1.0
    waist_pitch: float = 1.0

    def vector(self) -> np.ndarray:
        v = np.empty(N_UPPER)
        v[list(ARM_COLS)] = self.arms
        v[WAIST_YAW] = self.waist_yaw
        v[WAIST_ROLL] = self.waist_roll
        v[WAIST_PITCH] = self.waist_pitch
        return v


@dataclass
class Timing:
    control_dt: float = 0.02  # 50 Hz, same as Unitree's arm_sdk example
    acquire_s: float = 1.5  # weight 0 -> 1 while commanding the measured pose
    acquire_hold_s: float = 0.5
    entry_min_s: float = 4.0
    entry_arm_vel: float = 0.5  # rad/s peak, for the opening-pose entry (arms)
    entry_waist_vel: float = 0.12  # rad/s peak, for the opening-pose entry (waist)
    entry_settle_s: float = 1.0  # hold at frame-0 target before the gate check
    gate_timeout_s: float = 3.0  # how long the gate may wait for measured to converge
    end_hold_s: float = 0.5
    return_min_s: float = 4.0
    release_hold_s: float = 0.5
    release_s: float = 2.0  # weight 1 -> 0 while commanding the measured start pose


@dataclass
class Speed:
    speed: float = 1.0  # global playback rate (1 = recorded tempo)
    # Optional slow/fast passages in REFERENCE seconds: [[t_start, t_end, speed], ...]
    segments: list[list[float]] = field(default_factory=list)
    blend_s: float = 0.4  # smoothstep transition between speeds (reference seconds)
    ref_end_s: float | None = None  # truncate playback (staged trials); None = full
    # Frame 0 is mid-motion (right elbow ~1.09 rad/s). ramp_in_s > 0 eases the
    # playback clock from 0 to full speed over that many wall seconds (time warp
    # only: frame 0 and amplitudes unchanged). 0 = start at full speed (faithful).
    ramp_in_s: float = 0.0

    def at(self, t_ref: float) -> float:
        s = self.speed
        for t0, t1, sp in self.segments:
            b = max(self.blend_s, 1e-6)
            up = _smoothstep((t_ref - (t0 - b)) / b)
            down = 1.0 - _smoothstep((t_ref - t1) / b)
            w = min(up, down)
            s = s + w * (sp * self.speed - s)
        return s


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class Gains:
    # Unitree's g1_arm7_sdk_dds_example uses kp=60, kd=1.5 for all 17 joints.
    kp_arm: float = 60.0
    kd_arm: float = 1.5
    kp_waist: float = 60.0
    kd_waist: float = 1.5
    # Send the commanded joint velocity in motor_cmd[i].dq (Unitree's example
    # sends 0). With dq=0 the kd term brakes every motion, which shows up as
    # lag; with feed-forward it damps only the velocity ERROR. Positions and
    # thresholds are unchanged. Off by default until validated on hardware.
    dq_feedforward: bool = False

    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        kp = np.full(N_UPPER, float(self.kp_arm))
        kd = np.full(N_UPPER, float(self.kd_arm))
        kp[:3], kd[:3] = self.kp_waist, self.kd_waist
        return kp, kd


@dataclass
class Limits:
    pos_margin: float = 0.05  # rad inside the URDF range
    vel_arm: float = 3.0  # command rate cap, rad/s (URDF no-load is 22-37)
    vel_wrist_pitch_yaw: float = 3.0
    vel_waist: float = 0.6

    def vel_vector(self) -> np.ndarray:
        v = np.full(N_UPPER, float(self.vel_arm))
        v[:3] = self.vel_waist
        for c in (8, 9, 15, 16):  # wrist pitch/yaw columns = ids 20, 21, 27, 28 minus 12
            v[c] = self.vel_wrist_pitch_yaw
        return v


@dataclass
class Gates:
    """Stop thresholds. These are only ever changed by the operator, never by code."""
    # Before playback: commanded must equal the frame-0 target, measured must be close.
    entry_cmd_tol: float = 1e-6
    entry_meas_tol_arm: float = 0.10
    entry_meas_tol_waist: float = 0.05
    # During any motion phase: |measured - commanded| per joint group.
    track_arm: float = 0.35
    track_waist_yaw: float = 0.15
    track_waist_roll: float = 0.10
    track_waist_pitch: float = 0.10
    # Absolute waist joint position (encoder) bounds.
    waist_pitch_abs: float = 0.30
    waist_roll_abs: float = 0.25
    # Pelvis IMU tilt relative to the attitude at acquisition.
    imu_pitch_delta: float = 0.12
    imu_roll_delta: float = 0.10
    # Measured joint speed and state freshness.
    joint_speed: float = 6.0
    state_timeout_s: float = 0.10

    def track_vector(self) -> np.ndarray:
        v = np.full(N_UPPER, float(self.track_arm))
        v[WAIST_YAW], v[WAIST_ROLL], v[WAIST_PITCH] = (
            self.track_waist_yaw, self.track_waist_roll, self.track_waist_pitch)
        return v


@dataclass
class RunConfig:
    scales: Scales = field(default_factory=Scales)
    timing: Timing = field(default_factory=Timing)
    speed: Speed = field(default_factory=Speed)
    gains: Gains = field(default_factory=Gains)
    limits: Limits = field(default_factory=Limits)
    gates: Gates = field(default_factory=Gates)
    # Isolation switch: hold yaw/roll/pitch at the measured start for the WHOLE
    # run (entry and playback), regardless of the waist scales.
    hold_waist: bool = False

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["effective_scale_vector"] = self.effective_scales().tolist()
        return d

    def effective_scales(self) -> np.ndarray:
        s = self.scales.vector()
        if self.hold_waist:
            s[:3] = 0.0
        return s


def _apply(obj, data: dict, path: str = ""):
    for k, v in data.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {path}{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            _apply(cur, v, f"{path}{k}.")
        else:
            setattr(obj, k, v)


def load_config(path: str | Path | None) -> RunConfig:
    cfg = RunConfig()
    if path:
        with open(path, "rb") as f:
            _apply(cfg, tomllib.load(f))
    return cfg


def apply_overrides(cfg: RunConfig, overrides: list[str]) -> RunConfig:
    """``section.key=value`` overrides (value parsed as TOML)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        val = tomllib.loads(f"v = {raw}")["v"]
        *sections, leaf = key.strip().split(".")
        tgt = cfg
        for s in sections:
            tgt = getattr(tgt, s)
        if not hasattr(tgt, leaf):
            raise KeyError(f"unknown config key {key}")
        setattr(tgt, leaf, val)
    return cfg
