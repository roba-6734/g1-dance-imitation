"""Load and validate the motion assets.

* ``UpperReference``: the 50 Hz upper-body playback target (the ONLY motion
  source that reaches the command path).
* ``FullBodyRetarget``: the 30 Hz full-body GMR qpos. Used for visual
  comparison and consistency checks only. Its root and leg columns are split
  off into separately named fields and are never returned as a command.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .joints import G1_JOINT_NAMES, N_UPPER, UPPER_IDS, UPPER_NAMES

UPPER_FILE = "gmr_reference_full_960_2804_upper_50hz.npz"
FULL_FILE = "retargeted_g1_qpos.npy"
VIDEO_FILE = "human_motion.mov"
FULL_FPS = 30.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class UpperReference:
    fps: float
    q: np.ndarray  # (N, 17) rad, columns ordered by motor id 12..28
    qd: np.ndarray  # (N, 17) rad/s
    path: str

    @property
    def n(self) -> int:
        return self.q.shape[0]

    @property
    def duration(self) -> float:
        return (self.n - 1) / self.fps

    def times(self) -> np.ndarray:
        return np.arange(self.n) / self.fps

    def sample(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Cubic-Hermite sample (positions, velocities) at reference time(s) t [s].

        Uses the file's own joint_vel as tangents (they equal np.gradient of
        joint_pos, verified in tests), so frames are hit exactly and slow
        playback stays C1-smooth instead of piecewise-linear.
        """
        t = np.clip(np.atleast_1d(np.asarray(t, float)), 0.0, self.duration)
        dt = 1.0 / self.fps
        k = np.minimum((t * self.fps).astype(int), self.n - 2)
        s = (t - k * dt) / dt
        p0, p1 = self.q[k], self.q[k + 1]
        m0, m1 = self.qd[k] * dt, self.qd[k + 1] * dt
        s = s[:, None]
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        dh00 = 6 * s**2 - 6 * s
        dh10 = 3 * s**2 - 4 * s + 1
        dh01 = -6 * s**2 + 6 * s
        dh11 = 3 * s**2 - 2 * s
        vel = (dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1) / dt
        return pos, vel


def load_upper(path: str | Path) -> UpperReference:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        fps = float(z["fps"][0])
        ids = tuple(int(i) for i in z["motor_ids"])
        names = tuple(str(n) for n in z["joint_names"])
        q = z["joint_pos"].astype(np.float64)
        qd = z["joint_vel"].astype(np.float64)
        version = int(z["schema_version"][0])
    if version != 1:
        raise ValueError(f"unsupported schema_version {version}")
    if ids != UPPER_IDS:
        raise ValueError(f"motor_ids {ids} != expected upper-body ids {UPPER_IDS}")
    if names != UPPER_NAMES:
        raise ValueError(f"joint_names {names} != SDK names {UPPER_NAMES}")
    for i, n in zip(ids, names):
        if G1_JOINT_NAMES[i] != n:
            raise ValueError(f"id {i} is {G1_JOINT_NAMES[i]} in the SDK, file says {n}")
    if q.ndim != 2 or q.shape[1] != N_UPPER or qd.shape != q.shape:
        raise ValueError(f"bad shapes q{q.shape} qd{qd.shape}")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise ValueError("non-finite values in reference")
    return UpperReference(fps=fps, q=q, qd=qd, path=str(path))


@dataclass(frozen=True)
class FullBodyRetarget:
    """Full-body MuJoCo qpos (36 = 7 root + 29 joints). Visualisation only."""
    fps: float
    root_pos: np.ndarray  # (N, 3) m
    root_quat_wxyz: np.ndarray  # (N, 4)
    legs: np.ndarray  # (N, 12) rad, ids 0..11 -- NEVER commanded
    upper: np.ndarray  # (N, 17) rad, ids 12..28
    path: str

    @property
    def n(self) -> int:
        return self.upper.shape[0]

    @property
    def duration(self) -> float:
        return (self.n - 1) / self.fps

    def mujoco_qpos(self) -> np.ndarray:
        return np.concatenate([self.root_pos, self.root_quat_wxyz, self.legs, self.upper], axis=1)


def load_full(path: str | Path, fps: float = FULL_FPS) -> FullBodyRetarget:
    path = Path(path)
    a = np.load(path, allow_pickle=False).astype(np.float64)
    if a.ndim != 2 or a.shape[1] != 36:
        raise ValueError(f"expected (N, 36) MuJoCo qpos, got {a.shape}")
    quat = a[:, 3:7]
    if np.max(np.abs(np.linalg.norm(quat, axis=1) - 1.0)) > 1e-3:
        raise ValueError("root quaternion not unit norm")
    return FullBodyRetarget(fps=fps, root_pos=a[:, 0:3], root_quat_wxyz=quat,
                            legs=a[:, 7:19], upper=a[:, 19:36], path=str(path))


def resample_full_upper(full: FullBodyRetarget, t: np.ndarray) -> np.ndarray:
    tf = np.arange(full.n) / full.fps
    return np.stack([np.interp(t, tf, full.upper[:, j]) for j in range(N_UPPER)], axis=1)


def quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(q)
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)
