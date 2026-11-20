"""Fidelity metrics. Every number is labelled with its data kind:
  kinematic  - planned command targets (no dynamics)
  simulated  - MuJoCo fixed-pelvis response to those targets
  physical   - measured on the robot (only from hardware logs)
"""
from __future__ import annotations

import numpy as np

from .joints import N_UPPER, POS_LOWER, POS_UPPER, UPPER_NAMES
from .planner import PHASES, Plan
from .reference import UpperReference

D = np.degrees


def _worst(err: np.ndarray, t: np.ndarray):
    a = np.abs(err)
    k, j = np.unravel_index(np.nanargmax(a), a.shape)
    return {"joint": UPPER_NAMES[j], "dance_t_s": round(float(t[k]), 3), "error_deg": round(float(D(err[k, j])), 3)}


def _per_joint(err: np.ndarray) -> dict:
    return {UPPER_NAMES[j]: {"max_abs_deg": round(float(D(np.nanmax(np.abs(err[:, j])))), 3),
                             "rms_deg": round(float(D(np.sqrt(np.nanmean(err[:, j] ** 2)))), 3)}
            for j in range(N_UPPER)}


def best_lag(a: np.ndarray, b: np.ndarray, dt: float, max_lag_s: float = 0.4):
    """Lag (s) of b behind a that maximises correlation, and that correlation."""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0, float("nan")
    best = (0, -2.0)
    for L in range(0, int(max_lag_s / dt) + 1):
        x, y = (a[: len(a) - L], b[L:]) if L else (a, b)
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best[1]:
            best = (L, c)
    return best[0] * dt, best[1]


def kinematic_report(plan: Plan, ref: UpperReference) -> dict:
    pb = plan.segment("playback")
    open_err = plan.entry_target - plan.ref_frame0
    open_vs_scaled = plan.entry_target - plan.entry_target_ideal
    jw = int(np.argmax(np.abs(open_err)))
    ref_q, ref_v = ref.sample(pb.t_ref)
    headroom = np.minimum(ref.q.min(0) - POS_LOWER, POS_UPPER - ref.q.max(0))
    return {
        "data_kind": "kinematic",
        "opening_pose": {
            "commanded_vs_recorded_frame0_max_deg": round(float(D(abs(open_err[jw]))), 4),
            "worst_joint": UPPER_NAMES[jw],
            "commanded_vs_scaled_target_max_deg": round(float(D(np.max(np.abs(open_vs_scaled)))), 4),
            "entry_duration_s": plan.entry_duration,
            "entry_move_deg": {UPPER_NAMES[j]: round(float(D(plan.entry_target[j] - plan.start[j])), 2)
                               for j in range(N_UPPER)},
        },
        "playback": {
            "ticks": len(pb.q),
            "dance_duration_s": round(float(pb.t_ref[-1]), 3),
            "wall_duration_s": round(len(pb.q) * plan.dt, 3),
            "cmd_vs_scaled_reference_worst": _worst(pb.q - pb.ideal, pb.t_ref),
            "cmd_vs_recorded_reference_worst": _worst(pb.q - ref_q, pb.t_ref),
            "limit_events": [e.as_dict() for e in plan.limit_events],
        },
        "reference_headroom_to_urdf_deg": {UPPER_NAMES[j]: round(float(D(headroom[j])), 2) for j in range(N_UPPER)},
        "reference_peak_speed_rad_s": {UPPER_NAMES[j]: round(float(np.abs(ref.qd[:, j]).max()), 3)
                                       for j in range(N_UPPER)},
    }


def run_report(log: dict, plan: Plan, ref: UpperReference, data_kind: str) -> dict:
    """Response metrics from a runner log (simulated or physical)."""
    ph = log["phase"]
    qm = log["q_meas"][:, 12:29]
    qc = log["q_cmd"]
    out = {"data_kind": data_kind}
    settle = np.flatnonzero(ph == PHASES.index("settle"))
    if len(settle):
        k = settle[-1]
        e = qm[k] - plan.entry_target
        jw = int(np.argmax(np.abs(e)))
        e0 = qm[k] - plan.ref_frame0
        j0 = int(np.argmax(np.abs(e0)))
        out["opening_pose_at_gate"] = {
            "measured_vs_target_max_deg": round(float(D(abs(e[jw]))), 3), "worst_joint": UPPER_NAMES[jw],
            "measured_vs_recorded_frame0_max_deg": round(float(D(abs(e0[j0]))), 3),
            "worst_joint_vs_recorded": UPPER_NAMES[j0],
            "per_joint_deg": {UPPER_NAMES[j]: round(float(D(e[j])), 2) for j in range(N_UPPER)},
        }
    pbm = ph == PHASES.index("playback")
    if pbm.any():
        t_ref = log["t_ref"][pbm]
        pb = plan.segment("playback")
        n = min(pbm.sum(), len(pb.q))
        ideal = pb.ideal[:n]
        meas = qm[pbm][:n]
        out["playback"] = {
            "completed_fraction": round(n / len(pb.q), 4),
            "measured_vs_scaled_reference_worst": _worst(meas - ideal, t_ref[:n]),
            "measured_vs_command_worst": _worst(meas - qc[pbm][:n], t_ref[:n]),
            "per_joint_vs_reference": _per_joint(meas - ideal),
        }
        lags = {}
        for j in range(N_UPPER):
            L, c = best_lag(ideal[:, j], meas[:, j], plan.dt)
            lags[UPPER_NAMES[j]] = {"lag_s": round(L, 3), "corr": None if np.isnan(c) else round(c, 4)}
        out["playback"]["shape_lag_corr"] = lags
    from .joints import WAIST_PITCH
    wp_err = qm[:, WAIST_PITCH] - qc[:, WAIST_PITCH]
    full = log["weight"] >= 1 - 1e-9
    out["waist_pitch_measurements"] = {
        "waist_joint_pitch_max_deg": round(float(D(np.max(np.abs(qm[:, WAIST_PITCH])))), 3),
        "waist_pitch_error_max_deg(weight=1)": round(float(D(np.max(np.abs(wp_err[full])))) if full.any() else 0.0, 3),
        "pelvis_imu_pitch_delta_max_deg": round(float(D(np.max(np.abs(log["imu_rpy"][:, 1] - log["imu_rpy"][0, 1])))), 3),
        "torso_imu_pitch_delta_max_deg": (None if np.all(np.isnan(log["torso_rpy"])) else
                                          round(float(D(np.nanmax(np.abs(log["torso_rpy"][:, 1] - log["torso_rpy"][0, 1])))), 3)),
    }
    return out
