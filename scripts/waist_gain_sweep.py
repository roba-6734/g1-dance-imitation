"""SIMULATION-ONLY analysis: waist pitch response vs waist kp during entry/playback.

Gates are widened inside this script ONLY so that each simulated run finishes
and the full error can be measured. This is an analysis tool, not a run
configuration; the real gates live in configs/*.toml and are unchanged.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from g1dance.config import load_config  # noqa: E402
from g1dance.monitor import Monitor  # noqa: E402
from g1dance.planner import PHASES, build_plan  # noqa: E402
from g1dance.reference import load_upper  # noqa: E402
from g1dance.runner import execute  # noqa: E402
from g1dance.sim import SimBackend, assumed_start_upper  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ref = load_upper(ROOT / "motion_assets" / "gmr_reference_full_960_2804_upper_50hz.npz")
D = np.degrees
print(f"{'kp_waist':>8} {'kd':>5} | {'settle: cmd':>11} {'sim':>7} {'err':>6} | {'playback max err':>16} | "
      f"{'max |waist pitch|':>17} | hold-waist arm-only entry err")
for kp in (60, 100, 150, 200, 300, 400):
    row = []
    for hold in (False, True):
        cfg = load_config(ROOT / "configs" / "default.toml")
        cfg.gains.kp_waist, cfg.gains.kd_waist = float(kp), max(1.5, kp / 40)
        cfg.hold_waist = hold
        g = cfg.gates
        g.track_waist_pitch = g.track_waist_roll = g.waist_pitch_abs = g.waist_roll_abs = 1.0
        g.entry_meas_tol_waist = g.entry_meas_tol_arm = 1.0
        b = SimBackend(assumed_start_upper())
        p = build_plan(ref, b.read_state().upper_q, cfg)
        r = execute(p, b, cfg, Monitor(cfg.gates))
        a = r.log.arrays()
        ph, qm, qc = a["phase"], a["q_meas"][:, 14], a["q_cmd"][:, 2]
        st = np.flatnonzero(ph == PHASES.index("settle"))[-1]
        pb = ph == PHASES.index("playback")
        en = ph == PHASES.index("entry")
        row.append((qc[st], qm[st], np.abs(qm - qc)[pb].max(), np.abs(qm).max(), np.abs(qm - qc)[en | (ph == 2)].max()))
    (c, m, pe, mx, _), (_, _, _, _, he) = row
    print(f"{kp:8d} {max(1.5, kp / 40):5.2f} | {D(c):+10.2f}d {D(m):+6.2f}d {D(m - c):+5.2f}d | {D(pe):15.2f}d | "
          f"{D(mx):16.2f}d | {D(he):6.2f}d")
