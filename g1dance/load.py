"""Arm-induced load on the body, used to cap the playback tempo.

Moving the arms and waist pushes and twists the torso; Unitree's balance
controller has to absorb that through the feet. On hardware, steps followed
the largest of these loads, and the loads grow with tempo squared
(FINDINGS 4r). This module computes, with the Menagerie G1 model (legs at the
nominal stand pose, pelvis level, kinematic only):

  * torque [N m]  - rate of change of whole-body angular momentum about the
                    CoM, (roll, pitch, yaw) axes of the pelvis frame
  * force  [N]    - rate of change of whole-body linear momentum, horizontal (x, y)

Only the upper-body joints move, so both are produced by our commands. The
numbers are a model estimate used to compare passages and tempos, not a
measurement of foot forces.
"""
from __future__ import annotations

import numpy as np

SMOOTH = 5  # samples, moving average applied to velocities and derivatives


def _smooth(x: np.ndarray, n: int = SMOOTH) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, "valid"), 0, xp)


class ArmLoadModel:
    def __init__(self):
        import mujoco

        from .sim import load_model, stand_pose_29
        self._mj = mujoco
        self.m = load_model(fixed_base=False)
        self.d = mujoco.MjData(self.m)
        self.q29 = stand_pose_29()
        self.mass = float(self.m.body_subtreemass[1])

    def momentum(self, q: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Angular (n,3) and linear (n,3) momentum for upper-body q, qd (n,17)."""
        mj, m, d = self._mj, self.m, self.d
        n = len(q)
        L, P = np.zeros((n, 3)), np.zeros((n, 3))
        for k in range(n):
            d.qpos[:] = 0
            d.qpos[3] = 1.0
            d.qpos[7:36] = self.q29
            d.qpos[19:36] = q[k]
            d.qvel[:] = 0
            d.qvel[18:35] = qd[k]
            mj.mj_forward(m, d)
            mj.mj_subtreeVel(m, d)
            L[k] = d.subtree_angmom[1]
            P[k] = d.subtree_linvel[1] * self.mass
        return L, P

    def loads(self, q: np.ndarray, t: np.ndarray, qd: np.ndarray | None = None):
        """Torque (n,3) and horizontal force (n,2) along a trajectory sampled at times t.
        qd defaults to a smoothed finite difference of q."""
        q = np.asarray(q, float)
        t = np.asarray(t, float)
        if qd is None:
            qd = _smooth(np.gradient(q, t, axis=0))
        L, P = self.momentum(q, qd)
        tau = _smooth(np.gradient(L, t, axis=0))
        F = _smooth(np.gradient(P, t, axis=0))[:, :2]
        return tau, F


def peaks(tau: np.ndarray, F: np.ndarray) -> dict:
    i_tau = int(np.argmax(np.abs(tau[:, :2]).max(1)))
    fh = np.hypot(F[:, 0], F[:, 1])
    return {"roll_nm": float(np.abs(tau[:, 0]).max()), "pitch_nm": float(np.abs(tau[:, 1]).max()),
            "yaw_nm": float(np.abs(tau[:, 2]).max()), "force_n": float(fh.max()),
            "_i_tau": i_tau, "_i_force": int(np.argmax(fh))}


def tempo_cap(t: np.ndarray, tau: np.ndarray, F: np.ndarray, max_tau: float | None,
              max_force: float | None, window_s: float) -> np.ndarray:
    """Largest tempo factor (<= 1) per reference sample so that the 1.0x loads,
    which scale with tempo^2, stay below the limits. The cap is spread over
    +-window_s so the tempo changes ahead of a demanding passage, then smoothed."""
    ratio = np.ones(len(t))
    if max_tau:
        ratio = np.minimum(ratio, max_tau / np.maximum(np.abs(tau[:, :2]).max(1), 1e-9))
    if max_force:
        ratio = np.minimum(ratio, max_force / np.maximum(np.hypot(F[:, 0], F[:, 1]), 1e-9))
    s = np.sqrt(np.minimum(ratio, 1.0))
    return _spread_min(t, s, window_s)


def _spread_min(t: np.ndarray, s: np.ndarray, window_s: float) -> np.ndarray:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    w = max(1, int(round(window_s / dt)))
    sp = np.pad(s, (w, w), mode="edge")
    m = np.array([sp[k:k + 2 * w + 1].min() for k in range(len(s))])
    # smooth, then never exceed the spread minimum's neighbourhood by more than the smoothing
    k = np.ones(w) / w
    mp = np.pad(m, (w, w), mode="edge")
    sm = np.convolve(mp, k, "same")[w:-w]
    return np.minimum(sm, 1.0)
