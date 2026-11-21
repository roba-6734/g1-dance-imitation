"""A/B comparison of two retargeted references (e.g. GMR vs NMR) and of two g1dance runs.

Reference level (no robot), from two pipeline output directories:
    python -m video2ref.compare refs outputs/ayyala outputs/ayyala_nmr --names GMR NMR --out runs/ab_ref

Robot level, from two ``g1dance trial`` (or ``preview``) run directories made with the two asset folders
under identical settings (stage, speed, scales, gains):
    python -m video2ref.compare runs runs/2026..._trial_full_unitree runs/2026..._trial_full_unitree --names GMR NMR

Smoothness is measured, not inferred: RMS jerk / acceleration of the 50 Hz upper-body reference and,
on the robot, of the measured joint angles during playback plus IMU angular-rate RMS (body shake).
Lower is smoother. Agreement (correlation, lag) between the two references shows whether they encode the
same dance; a retargeter that is smoother only because it moves less shows up as lower range of motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1dance.joints import UPPER_NAMES

GROUPS = {"waist": list(range(0, 3)), "left_arm": list(range(3, 10)), "right_arm": list(range(10, 17))}


def smoothness(q: np.ndarray, dt: float) -> dict:
    """q: (T, J) joint angles sampled every dt seconds."""
    v = np.gradient(q, dt, axis=0)
    acc = np.gradient(v, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    out = {}
    for g, idx in GROUPS.items():
        out[g] = {"rms_jerk": float(np.sqrt(np.mean(jerk[:, idx] ** 2))),
                  "rms_acc": float(np.sqrt(np.mean(acc[:, idx] ** 2))),
                  "peak_vel": float(np.abs(v[:, idx]).max()),
                  "range_of_motion": float(np.ptp(q[:, idx], axis=0).mean())}
    out["all"] = {"rms_jerk": float(np.sqrt(np.mean(jerk ** 2))), "rms_acc": float(np.sqrt(np.mean(acc ** 2))),
                  "peak_vel": float(np.abs(v).max()), "range_of_motion": float(np.ptp(q, axis=0).mean())}
    # spectral share above 4 Hz: dance content is mostly below; above is mostly estimation noise
    spec = np.abs(np.fft.rfft(q - q.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(q), dt)
    out["all"]["power_above_4hz_frac"] = float(spec[f > 4].sum() / max(spec.sum(), 1e-12))
    return out


def agreement(a: np.ndarray, b: np.ndarray, dt: float, max_lag_s=0.3) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    per = {}
    for j, name in enumerate(UPPER_NAMES):
        x, y = a[:, j] - a[:, j].mean(), b[:, j] - b[:, j].mean()
        if x.std() < 1e-6 or y.std() < 1e-6:
            per[name] = {"corr": None, "lag_s": None, "mean_abs_diff_deg": float(np.degrees(np.abs(a[:, j] - b[:, j]).mean()))}
            continue
        lags = range(-int(max_lag_s / dt), int(max_lag_s / dt) + 1)
        cs = [np.corrcoef(x[max(0, -L):n - max(0, L)], y[max(0, L):n - max(0, -L)])[0, 1] for L in lags]
        k = int(np.nanargmax(cs))
        per[name] = {"corr": round(float(cs[k]), 4), "lag_s": round(list(lags)[k] * dt, 3),
                     "mean_abs_diff_deg": round(float(np.degrees(np.abs(a[:, j] - b[:, j]).mean())), 2)}
    corrs = [v["corr"] for v in per.values() if v["corr"] is not None]
    return {"median_corr": float(np.median(corrs)) if corrs else None, "per_joint": per}


def load_ref(run: Path):
    with np.load(run / "reference_upper_50hz.npz") as z:
        return z["joint_pos"].astype(float), 1 / 50.0


def compare_refs(a: Path, b: Path, names):
    (qa, dt), (qb, _) = load_ref(a), load_ref(b)
    rep = {"kind": "reference", "inputs": {names[0]: str(a), names[1]: str(b)},
           "smoothness": {names[0]: smoothness(qa, dt), names[1]: smoothness(qb, dt)},
           "agreement": agreement(qa, qb, dt)}
    try:  # whole-body feasibility of the 30 Hz full-body qpos (feet, balance proxy, collisions)
        from wbt.feasibility import analyse

        rep["feasibility"] = {n: analyse(np.load(p / "qpos.npy"))[0] for n, p in zip(names, (a, b))}
    except Exception as exc:  # model not available etc.: reference smoothness is still valid
        rep["feasibility_error"] = f"{type(exc).__name__}: {exc}"
    return rep


def load_run(run: Path):
    with np.load(run / "log.npz", allow_pickle=False) as z:
        phases = [str(p) for p in z["phases"]]
        pb = z["phase"] == phases.index("playback")
        t = z["t"][pb]
        q = z["q_meas"][pb][:, 12:29].astype(float)
        imu = z["imu_rpy"][pb].astype(float) if "imu_rpy" in z.files else None
    summary = json.loads((run / "summary.json").read_text())
    return t, q, imu, summary


def compare_runs(a: Path, b: Path, names):
    rep = {"kind": "robot_runs", "inputs": {names[0]: str(a), names[1]: str(b)}, "measured": {}}
    for n, run in zip(names, (a, b)):
        t, q, imu, summary = load_run(run)
        if len(t) < 10:
            rep["measured"][n] = {"error": "no playback samples"}
            continue
        dt = float(np.median(np.diff(t)))
        entry = {"data_kind": summary.get("data_kind"), "completed": summary.get("completed"),
                 "stop": summary.get("stop"), "playback_seconds": round(float(t[-1] - t[0]), 2),
                 "measured_smoothness": smoothness(q, dt)}
        if imu is not None:
            rate = np.gradient(np.unwrap(imu, axis=0), dt, axis=0)
            entry["imu_rate_rms_rad_s"] = {"roll": float(np.sqrt(np.mean(rate[:, 0] ** 2))),
                                           "pitch": float(np.sqrt(np.mean(rate[:, 1] ** 2)))}
        resp = summary.get("response") or summary.get("simulated") or {}
        pbk = resp.get("playback", {})
        per = pbk.get("per_joint_vs_reference", {})
        if per:
            entry["tracking_rms_deg_mean"] = float(np.mean([v["rms_deg"] for v in per.values()]))
            entry["tracking_worst"] = pbk.get("measured_vs_scaled_reference_worst")
        lags = [v["lag_s"] for v in pbk.get("shape_lag_corr", {}).values() if v.get("lag_s") is not None]
        if lags:
            entry["median_lag_s"] = float(np.median(lags))
        rep["measured"][n] = entry
    return rep


def to_markdown(rep, names):
    lines = [f"# A/B: {names[0]} vs {names[1]} ({rep['kind']})", ""]
    if rep["kind"] == "reference":
        s = rep["smoothness"]
        lines += ["Upper-body 50 Hz reference (lower jerk = smoother; check range of motion did not collapse).", "",
                  f"| group | metric | {names[0]} | {names[1]} |", "|---|---|---|---|"]
        for g in ("all", *GROUPS):
            for k in ("rms_jerk", "rms_acc", "peak_vel", "range_of_motion"):
                lines.append(f"| {g} | {k} | {s[names[0]][g][k]:.3g} | {s[names[1]][g][k]:.3g} |")
        lines.append(f"| all | power_above_4hz_frac | {s[names[0]]['all']['power_above_4hz_frac']:.3g} | "
                     f"{s[names[1]]['all']['power_above_4hz_frac']:.3g} |")
        lines += ["", f"Median per-joint correlation between the two: {rep['agreement']['median_corr']}"]
        if "feasibility" in rep:
            f = rep["feasibility"]
            lines += ["", f"| whole-body feasibility | {names[0]} | {names[1]} |", "|---|---|---|"]
            for k in ("peak_velocity_ratio", "penetration_max_m", "lower_foot_float_p95_m", "skate_p95_mps",
                      "com_outside_frac", "leg_collision_frac", "other_self_collision_frac", "rms_jerk_all"):
                lines.append(f"| {k} | {f[names[0]][k]:.3g} | {f[names[1]][k]:.3g} |")
    else:
        lines += [f"| measured on {rep['measured'][names[0]].get('data_kind', '?')} | {names[0]} | {names[1]} |",
                  "|---|---|---|"]
        m = rep["measured"]
        rows = [("completed", lambda e: e.get("completed")), ("playback s", lambda e: e.get("playback_seconds")),
                ("tracking RMS (deg, mean of joints)", lambda e: e.get("tracking_rms_deg_mean")),
                ("median lag (s)", lambda e: e.get("median_lag_s")),
                ("measured RMS jerk (rad/s^3)", lambda e: e["measured_smoothness"]["all"]["rms_jerk"]),
                ("measured RMS acc (rad/s^2)", lambda e: e["measured_smoothness"]["all"]["rms_acc"]),
                ("measured power > 4 Hz", lambda e: e["measured_smoothness"]["all"]["power_above_4hz_frac"]),
                ("IMU pitch-rate RMS (rad/s)", lambda e: e.get("imu_rate_rms_rad_s", {}).get("pitch")),
                ("IMU roll-rate RMS (rad/s)", lambda e: e.get("imu_rate_rms_rad_s", {}).get("roll"))]
        for label, get in rows:
            vals = []
            for n in names:
                try:
                    v = get(m[n])
                except (KeyError, TypeError):
                    v = None
                vals.append("-" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v))
            lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["refs", "runs"])
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--names", nargs=2, default=["A", "B"])
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    rep = compare_refs(a.a, a.b, a.names) if a.kind == "refs" else compare_runs(a.a, a.b, a.names)
    md = to_markdown(rep, a.names)
    print(md)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "compare.json").write_text(json.dumps(rep, indent=2, default=float))
        (a.out / "compare.md").write_text(md)


if __name__ == "__main__":
    main()
