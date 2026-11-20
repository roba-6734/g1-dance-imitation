"""READ-ONLY hardware check. Subscribes to rt/lowstate (and listens on
rt/arm_sdk to detect other publishers). Never creates a publisher.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..joints import G1_JOINT_NAMES, N_UPPER, POS_LOWER, POS_UPPER, UPPER_IDS, UPPER_NAMES, classify_variant

D = np.degrees

# Morning stances (10 runs, no stepping) agreed with runs/stance.json within 1.8 deg per leg joint;
# the post-recharge stance that stepped (FINDINGS 4k) differed by up to 5.4 deg.
LEG_TOL_DEG = 2.0
PITCH_CHAIN = ((0, 6, "hip_pitch"), (3, 9, "knee"), (4, 10, "ankle_pitch"))


def add_parser(sub):
    p = sub.add_parser("check", help="READ-ONLY: read robot state, classify variant, save stance")
    p.add_argument("--iface", required=True, help="network interface connected to the robot, e.g. enp3s0")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--save-stance", default=None, help="write measured upper-body stance JSON here")
    p.add_argument("--assets", default=None)
    p.add_argument("--leg-reference", default=None,
                   help="stance JSON from a run where Unitree did not step (e.g. runs/stance.json); "
                        "compares the current leg posture against it")
    p.add_argument("--leg-tol-deg", type=float, default=LEG_TOL_DEG)
    p.set_defaults(fn=run)


def stance_asymmetry(leg_deg) -> dict[str, float]:
    """left - right (deg) for hip pitch, knee and ankle pitch (SDK leg order, 12 values)."""
    leg_deg = np.asarray(leg_deg, float)
    return {n: round(float(leg_deg[l] - leg_deg[r]), 2) for l, r, n in PITCH_CHAIN}


def stance_gate(leg_deg, limit_deg: float) -> tuple[bool, str]:
    """Pre-run gate on the staggered-feet signature (hip and ankle pitch)."""
    asym = stance_asymmetry(leg_deg)
    worst = max(abs(asym["hip_pitch"]), abs(asym["ankle_pitch"]))
    ok = limit_deg <= 0 or worst <= limit_deg
    return ok, (f"leg stance left-minus-right (deg): hip {asym['hip_pitch']:+.1f}, knee {asym['knee']:+.1f}, "
                f"ankle {asym['ankle_pitch']:+.1f}; limit |hip|,|ankle| <= {limit_deg:.1f}")


def leg_stance_checks(leg_deg, leg_ref_deg=None, tol_deg=LEG_TOL_DEG):
    """Read-only comparison of the leg posture (12 values, deg, SDK order). Returns check dicts."""
    leg_deg = np.asarray(leg_deg, float)
    out = []
    asym = {n: round(float(leg_deg[l] - leg_deg[r]), 1) for l, r, n in PITCH_CHAIN}
    out.append({"level": "INFO", "msg": f"leg stance left-minus-right (deg): {asym}"})
    if leg_ref_deg is not None:
        diff = leg_deg - np.asarray(leg_ref_deg, float)
        j = int(np.argmax(np.abs(diff)))
        bad = {G1_JOINT_NAMES[i]: round(float(diff[i]), 1) for i in range(12) if abs(diff[i]) > tol_deg}
        if bad:
            out.append({"level": "WARN", "msg": f"leg stance differs from reference by > {tol_deg:.1f} deg: {bad}. "
                        "Feet may be staggered; Unitree stepped in a run like this (FINDINGS 4k). Re-stand the robot."})
        else:
            out.append({"level": "PASS", "msg": f"leg stance matches reference within {tol_deg:.1f} deg "
                        f"(max {G1_JOINT_NAMES[j]} {diff[j]:+.1f})"})
    return out


def summarize(states, stamps, variant, probe_count, probe_weight, ref_frame0=None,
              leg_ref_deg=None, leg_tol_deg=LEG_TOL_DEG) -> dict:
    q = np.array([s.q for s in states])
    rpy = np.array([s.imu_rpy for s in states])
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    gaps = np.diff(stamps) if len(stamps) > 1 else np.array([0.0])
    upper = np.median(q[:, list(UPPER_IDS)], axis=0)
    checks = []
    add = lambda level, msg: checks.append({"level": level, "msg": msg})
    add("PASS" if rate > 100 else "FAIL", f"rt/lowstate rate {rate:.0f} Hz, max gap {gaps.max() * 1000:.1f} ms")
    add("PASS" if variant.supported else "FAIL", variant.describe())
    if variant.waist_roll_pitch_actuated is False:
        add("WARN", "waist roll/pitch locked: runs must use --waist-roll 0 --waist-pitch 0 (or --hold-waist)")
    add("PASS" if probe_count == 0 else "FAIL",
        f"rt/arm_sdk messages from other publishers during check: {probe_count}"
        + ("" if probe_weight is None else f" (last weight {probe_weight:.2f})"))
    out_of_range = [UPPER_NAMES[j] for j in range(N_UPPER) if not (POS_LOWER[j] <= upper[j] <= POS_UPPER[j])]
    add("PASS" if not out_of_range else "FAIL", f"upper joints within URDF range: {out_of_range or 'all'}")
    still = np.max(np.abs(np.array([s.dq for s in states])[:, list(UPPER_IDS)]))
    add("PASS" if still < 0.5 else "WARN", f"upper-body max |dq| during check {still:.3f} rad/s")
    tilt = D(np.ptp(rpy[:, :2], axis=0))
    add("INFO", f"pelvis IMU rpy (deg) median {np.round(D(np.median(rpy, 0)), 2).tolist()}, "
                f"roll/pitch spread {np.round(tilt, 2).tolist()}")
    checks += leg_stance_checks(D(np.median(q[:, :12], axis=0)), leg_ref_deg, leg_tol_deg)
    rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "data_kind": "physical",
           "mode_machine": variant.mode_machine, "variant": variant.describe(),
           "lowstate_rate_hz": rate, "upper_q": upper.tolist(),
           "upper_q_deg": {UPPER_NAMES[j]: round(float(D(upper[j])), 2) for j in range(N_UPPER)},
           "leg_q_deg": {G1_JOINT_NAMES[i]: round(float(D(np.median(q[:, i]))), 2) for i in range(12)},
           "imu_rpy_median": np.median(rpy, 0).tolist(), "checks": checks}
    if ref_frame0 is not None:
        rep["entry_move_to_frame0_deg"] = {UPPER_NAMES[j]: round(float(D(ref_frame0[j] - upper[j])), 2)
                                           for j in range(N_UPPER)}
    return rep


def run(a):
    from .unitree import LowStateReader, TopicActivityProbe, init_dds

    print("READ-ONLY check: subscribing to rt/lowstate and rt/arm_sdk; nothing is published.")
    init_dds(a.iface)
    reader = LowStateReader()
    probe = TopicActivityProbe()
    reader.wait_first()
    states, stamps = [], []
    t_end = time.monotonic() + a.seconds
    last = None
    while time.monotonic() < t_end:
        st = reader.latest()
        if st is not None and st is not last:
            states.append(st)
            last = st
        time.sleep(0.002)
    stamps = list(reader.stamps)
    variant = classify_variant(states[-1].mode_machine)
    f0 = None
    if a.assets:
        from ..reference import UPPER_FILE, load_upper
        f0 = load_upper(Path(a.assets) / UPPER_FILE).q[0]
    leg_ref = None
    if a.leg_reference:
        legs = json.loads(Path(a.leg_reference).read_text())["leg_q_deg"]
        leg_ref = [legs[G1_JOINT_NAMES[i]] for i in range(12)]
    rep = summarize(states, stamps, variant, probe.count, probe.last_weight, f0, leg_ref, a.leg_tol_deg)
    for c in rep["checks"]:
        print(f"  [{c['level']:4s}] {c['msg']}")
    print("upper-body stance (deg):")
    for k, v in rep["upper_q_deg"].items():
        mv = rep.get("entry_move_to_frame0_deg", {}).get(k)
        print(f"   {k:28s} {v:+8.2f}" + ("" if mv is None else f"   -> frame0 move {mv:+7.2f}"))
    print("operator verification still required: see docs/HARDWARE.md section 'Operator checklist'")
    if a.save_stance:
        Path(a.save_stance).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save_stance).write_text(json.dumps(rep, indent=2))
        print(f"stance written to {a.save_stance}")
    return 0 if all(c["level"] != "FAIL" for c in rep["checks"]) else 2
