"""Kinematic and quasi-static screen of a G1 reference motion before training.

    python -m wbt.feasibility --csv wbt_motions/ayyala_gmr.csv [--out report.json]
    python -m wbt.feasibility --qpos outputs/my_dance/qpos.npy

A failing motion will not train into a safe policy, so fix the source (video, retargeter,
trim, filter) first. Checks:
  joint position    vs the 0.9 soft range mjlab penalises (and the hard range)
  joint velocity    vs motor limits (fail > 80 %, warn > 50 %); acceleration and jerk are reported
  feet              penetration below the floor, float of the lower foot, skate of a planted foot
  balance proxy     CoM ground projection vs the support polygon of planted feet (Ayyala is
                    quasi-static, so a CoM far outside support means a bad root or retarget)
  self collision    penetrating robot-robot geom pairs
  root              tilt and height range (a lying or jumping root is a retarget failure here)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from g1dance.joints import G1_JOINT_NAMES
from .g1model import build_model, joint_specs, soft_limits
from .prepare import FPS, read_csv

# Calibrated so Unitree's shipped dance1_subject2 (trains and runs on hardware) is WARN, not FAIL:
# it uses ankle roll near its range, claps (hand-hand contact) and steps with the CoM outside support.
LIMITS = {
    "soft_limit_frac_warn": 0.05, "hard_limit_violation_fail_rad": 0.05, "velocity_ratio_fail": 0.8,
    "velocity_ratio_warn": 0.5, "penetration_fail_m": 0.03, "float_warn_m": 0.05, "skate_warn_mps": 0.15,
    "com_outside_frac_warn": 0.05, "com_outside_dist_warn_m": 0.10, "leg_collision_frac_fail": 0.01,
    "arm_collision_frac_warn": 0.05, "tilt_fail_rad": 0.6, "root_height_min_fail_m": 0.45,
}
LEG_GEOM = ("hip", "thigh", "shin", "linkage", "foot", "ankle", "knee")


def _is_leg(name):
    return any(k in name for k in LEG_GEOM) and name.startswith(("left_", "right_"))


def analyse(qpos, fps=FPS, xml=None, contact_tol=0.02):
    model = build_model(xml)
    d = mujoco.MjData(model)
    specs = joint_specs()
    T = len(qpos)
    q = qpos[:, 7:]
    dq = np.gradient(q, 1 / fps, axis=0)
    ddq = np.gradient(dq, 1 / fps, axis=0)
    jerk = np.gradient(ddq, 1 / fps, axis=0)
    soft, hard = soft_limits(model), model.jnt_range[1:30]
    vlim = np.array([s.velocity for s in specs])
    feet = {side: [g for g in range(model.ngeom) if (model.geom(g).name or "").startswith(f"{side}_foot")]
            for side in ("left", "right")}
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and model.geom_contype[g]]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    low = np.zeros((T, 2))
    foot_xy = np.zeros((T, 2, 2))
    com_out = np.full(T, np.nan)
    leg_hits = np.zeros(T, bool)
    arm_hits = np.zeros(T, bool)
    pairs = {}
    tilt = np.zeros(T)
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(model, d)
        pts = []
        for k, side in enumerate(("left", "right")):
            ends = []
            for g in feet[side]:
                axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
                half, r = model.geom_size[g][1], model.geom_size[g][0]
                for sgn in (-1, 1):
                    e = d.geom_xpos[g] + sgn * half * axis
                    ends.append(np.array([e[0], e[1], e[2] - r]))
            ends = np.array(ends)
            low[t, k] = ends[:, 2].min()
            foot_xy[t, k] = ends[:, :2].mean(0)
            pts.append(ends[ends[:, 2] < ends[:, 2].min() + contact_tol])
        planted = [p for k, p in enumerate(pts) if low[t, k] < low[t].min() + contact_tol]
        support = np.concatenate(planted)[:, :2]
        com = d.subtree_com[1][:2]
        com_out[t] = _outside_distance(com, support)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != floor and c.geom2 != floor and c.geom1 in robot and c.geom2 in robot and c.dist < -0.01:
                a, b = sorted((model.geom(c.geom1).name, model.geom(c.geom2).name))
                pairs[f"{a}|{b}"] = pairs.get(f"{a}|{b}", 0) + 1
                # left leg against right leg can trip the robot; hand/arm touches are normal in dance
                if _is_leg(a) and _is_leg(b) and a.split("_")[0] != b.split("_")[0]:
                    leg_hits[t] = True
                else:
                    arm_hits[t] = True
        g = d.xmat[1].reshape(3, 3)[:, 2]
        tilt[t] = np.arccos(np.clip(g[2], -1, 1))
    lower = low.argmin(1)
    planted_speed = np.linalg.norm(np.gradient(foot_xy, 1 / fps, axis=0), axis=2)
    skate = np.array([planted_speed[t, lower[t]] for t in range(T)])
    vel_ratio = np.abs(dq).max(0) / vlim
    soft_frac = ((q < soft[:, 0]) | (q > soft[:, 1])).mean(0)
    hard_over = np.maximum(hard[:, 0] - q, q - hard[:, 1]).max(0).clip(0)
    per_joint = {n: {"soft_limit_frac": round(float(soft_frac[j]), 4), "hard_over_rad": round(float(hard_over[j]), 4),
                     "peak_vel_ratio": round(float(vel_ratio[j]), 3),
                     "peak_acc": round(float(np.abs(ddq[:, j]).max()), 1),
                     "rms_jerk": round(float(np.sqrt(np.mean(jerk[:, j] ** 2))), 1)}
                 for j, n in enumerate(G1_JOINT_NAMES)}
    m = {
        "frames": T, "seconds": round((T - 1) / fps, 2),
        "max_soft_limit_frac": float(soft_frac.max()), "max_soft_limit_joint": G1_JOINT_NAMES[int(soft_frac.argmax())],
        "max_hard_over_rad": float(hard_over.max()), "max_hard_over_joint": G1_JOINT_NAMES[int(hard_over.argmax())],
        "peak_velocity_ratio": float(vel_ratio.max()), "peak_velocity_joint": G1_JOINT_NAMES[int(vel_ratio.argmax())],
        "rms_jerk_all": float(np.sqrt(np.mean(jerk ** 2))),
        "penetration_max_m": float(max(0.0, -low.min())), "lower_foot_float_p95_m": float(np.percentile(low.min(1), 95)),
        "skate_p95_mps": float(np.percentile(skate, 95)),
        "com_outside_frac": float(np.mean(com_out > 0.0)), "com_outside_max_m": float(np.nanmax(com_out).clip(0)),
        "leg_collision_frac": float(leg_hits.mean()), "other_self_collision_frac": float(arm_hits.mean()),
        "self_collision_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:8]),
        "root_tilt_max_rad": float(tilt.max()),
        "root_height_min_m": float(qpos[:, 2].min()), "root_height_max_m": float(qpos[:, 2].max()),
        "root_travel_m": float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])),
        "root_path_m": float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).sum()),
    }
    return m, per_joint


def _outside_distance(p, pts):
    """Signed distance of p outside the convex hull of pts (<= 0 inside)."""
    if len(pts) < 3:
        return float(np.linalg.norm(p - pts.mean(0)))
    try:
        hull = ConvexHull(pts)
    except Exception:  # degenerate (collinear) support
        return float(np.linalg.norm(p - pts.mean(0)))
    return float(np.max(hull.equations[:, :2] @ p + hull.equations[:, 2]))


def verdict(m, lim=LIMITS):
    out = []

    def add(level, name, ok, detail):
        out.append({"check": name, "level": "pass" if ok else level, "detail": detail})

    add("warn", "soft joint range", m["max_soft_limit_frac"] <= lim["soft_limit_frac_warn"],
        f"{m['max_soft_limit_joint']} outside the 0.9 range {m['max_soft_limit_frac'] * 100:.1f}% of frames")
    add("fail", "hard joint range", m["max_hard_over_rad"] <= lim["hard_limit_violation_fail_rad"],
        f"{m['max_hard_over_joint']} {m['max_hard_over_rad']:.3f} rad past its stop")
    add("fail", "joint velocity", m["peak_velocity_ratio"] <= lim["velocity_ratio_fail"],
        f"{m['peak_velocity_joint']} at {m['peak_velocity_ratio'] * 100:.0f}% of motor limit")
    add("warn", "joint velocity margin", m["peak_velocity_ratio"] <= lim["velocity_ratio_warn"],
        f"{m['peak_velocity_ratio'] * 100:.0f}% (warn above {lim['velocity_ratio_warn'] * 100:.0f}%)")
    add("fail", "foot penetration", m["penetration_max_m"] <= lim["penetration_fail_m"], f"{m['penetration_max_m'] * 100:.1f} cm")
    add("warn", "foot float", m["lower_foot_float_p95_m"] <= lim["float_warn_m"],
        f"lower foot p95 {m['lower_foot_float_p95_m'] * 100:.1f} cm above floor")
    add("warn", "foot skate", m["skate_p95_mps"] <= lim["skate_warn_mps"], f"planted foot p95 {m['skate_p95_mps']:.2f} m/s")
    add("warn", "CoM over support", m["com_outside_frac"] <= lim["com_outside_frac_warn"],
        f"outside {m['com_outside_frac'] * 100:.1f}% of frames")
    add("warn", "CoM distance outside support", m["com_outside_max_m"] <= lim["com_outside_dist_warn_m"],
        f"max {m['com_outside_max_m'] * 100:.1f} cm (quasi-static proxy; steps and jumps exceed it legitimately)")
    add("fail", "leg-leg collision", m["leg_collision_frac"] <= lim["leg_collision_frac_fail"],
        f"{m['leg_collision_frac'] * 100:.1f}% of frames")
    top = ", ".join(f"{k} x{v}" for k, v in list(m["self_collision_pairs"].items())[:3]) or "none"
    add("warn", "arm/body self contact", m["other_self_collision_frac"] <= lim["arm_collision_frac_warn"],
        f"{m['other_self_collision_frac'] * 100:.1f}% of frames ({top})")
    add("fail", "root tilt", m["root_tilt_max_rad"] <= lim["tilt_fail_rad"], f"max {m['root_tilt_max_rad']:.2f} rad")
    add("fail", "root height", m["root_height_min_m"] >= lim["root_height_min_fail_m"], f"min {m['root_height_min_m']:.2f} m")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="prepared Unitree CSV (30 Hz)")
    src.add_argument("--qpos", type=Path, help="(T,36) qpos.npy (30 Hz, wxyz)")
    p.add_argument("--xml")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    qpos = read_csv(a.csv) if a.csv else np.load(a.qpos)
    m, per_joint = analyse(qpos, xml=a.xml)
    checks = verdict(m)
    status = "fail" if any(c["level"] == "fail" for c in checks) else "warn" if any(c["level"] == "warn" for c in checks) else "pass"
    for c in checks:
        print(f"[{c['level'].upper():4s}] {c['check']}: {c['detail']}")
    print(f"\nfeasibility: {status.upper()}")
    out = a.out or (a.csv or a.qpos).with_suffix(".feasibility.json")
    out.write_text(json.dumps({"status": status, "checks": checks, "metrics": m, "per_joint": per_joint}, indent=2))
    print(f"report: {out}")
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
