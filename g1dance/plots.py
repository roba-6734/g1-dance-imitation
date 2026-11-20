"""Static joint-trace figure: recorded reference vs command vs response."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .joints import UPPER_NAMES
from .planner import PHASES
from .reference import UpperReference

SHOW = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint", "right_elbow_joint", "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "left_elbow_joint", "right_shoulder_yaw_joint"]
PHASE_COLORS = ["#c7d2fe", "#fde68a", "#fef08a", "#ffffff", "#e5e7eb", "#fecaca", "#e9d5ff"]


def plot_traces(out: Path, log: dict, ref: UpperReference, resp_label: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, ph = log["t"], log["phase"]
    fig, axes = plt.subplots(len(SHOW), 1, figsize=(13, 2.0 * len(SHOW)), sharex=True)
    pb = ph == PHASES.index("playback")
    ref_t = np.full_like(t, np.nan)
    ref_t[pb] = log["t_ref"][pb]
    ref_q = np.full((len(t), len(UPPER_NAMES)), np.nan)
    if pb.any():
        ref_q[pb] = ref.sample(ref_t[pb])[0]
    for ax, name in zip(axes, SHOW):
        j = UPPER_NAMES.index(name)
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or ph[k] != ph[start]:
                if ph[start] >= 0:
                    ax.axvspan(t[start], t[k - 1], color=PHASE_COLORS[ph[start]], lw=0)
                start = k
        ax.plot(t, np.degrees(ref_q[:, j]), color="#111827", lw=2.2, label="recorded reference (raw)")
        ax.plot(t, np.degrees(log["q_cmd"][:, j]), color="#2563eb", lw=1.2, label="command (kinematic)")
        ax.plot(t, np.degrees(log["q_meas"][:, 12 + j]), color="#dc2626", lw=1.0, ls="--", label=resp_label)
        ax.axhline(np.degrees(ref.q[0, j]), color="#6b7280", lw=0.8, ls=":")
        ax.set_ylabel(name.replace("_joint", "").replace("_", " ") + "\n[deg]", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title(title + "   (dotted: recorded frame 0; shading = phase: "
                      + ", ".join(PHASES) + ")", fontsize=9)
    axes[-1].set_xlabel("run time [s]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
