import csv
import pickle
from pathlib import Path

import numpy as np


def wxyz_to_xyzw(quat):
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion last dimension must be 4")
    return quat[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(quat):
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError("Quaternion last dimension must be 4")
    return quat[..., [3, 0, 1, 2]]


def check_qpos(qpos):
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] < 1:
        raise ValueError(f"Expected nonempty (T,36) G1 qpos, got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("Nonfinite values in qpos")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1) > 1e-3):
        raise ValueError("Root quaternions are not normalized (expected wxyz)")
    return qpos


def save_formats(qpos, fps, joint_names, output: Path, csv_enabled=False):
    qpos = check_qpos(qpos)
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("Expected 29 distinct joint names in MuJoCo qpos order")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "qpos.npy", qpos)
    motion = {"fps": float(fps), "root_pos": qpos[:, :3].copy(),
              "root_rot": wxyz_to_xyzw(qpos[:, 3:7]), "dof_pos": qpos[:, 7:].copy(),
              "local_body_pos": None, "link_body_list": None}
    with (output / "motion.pkl").open("wb") as handle:
        pickle.dump(motion, handle)
    if csv_enabled:
        with (output / "qpos.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz", *joint_names])
            writer.writerows(qpos)
    return motion


def smooth_qpos(qpos, window):
    qpos = check_qpos(qpos)
    if window < 1 or window % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd integer")
    if window == 1:
        return qpos.copy()
    pad = window // 2
    padded = np.pad(qpos, ((pad, pad), (0, 0)), mode="edge")
    result = np.stack([padded[i:i + window].mean(axis=0) for i in range(len(qpos))])
    # Hemisphere-align quaternions before averaging, then restore unit norm.
    for i in range(len(qpos)):
        segment = padded[i:i + window, 3:7].copy()
        segment[np.sum(segment * qpos[i, 3:7], axis=1) < 0] *= -1
        quat = segment.mean(axis=0)
        result[i, 3:7] = quat / np.linalg.norm(quat)
    return check_qpos(result)
