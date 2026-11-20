"""Stage 3: full-body GMR qpos (30 Hz) -> 50 Hz upper-body reference for g1dance.

Numpy only (no GMR, torch or MuJoCo needed), so it also runs in the g1dance venv:

    python -m video2ref.upper --qpos outputs/my_video/qpos.npy \
        --output outputs/my_video/reference_upper_50hz.npz

Steps (identical to the procedure that produced
gmr_reference_full_960_2804_upper_50hz.npz):
  1. take motor ids 12..28 (waist + both arms) from the (T, 36) MuJoCo qpos;
     the root and the 12 leg joints are dropped and never written;
  2. centred moving average over ``smoothing_window`` source frames (default 3,
     edge padded);
  3. linear interpolation from 30 Hz to 50 Hz (last sample <= last source time);
  4. joint_vel = np.gradient(joint_pos, 0.02);
  5. refuse the result if a sample leaves the soft limits or exceeds the
     release velocity / acceleration screen below.

The limits are project screening values, not certified hardware limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

UPPER_IDS = tuple(range(12, 29))
UPPER_NAMES = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# Soft limits (rad) used for the original reference (G1 29-DoF URDF ranges, as float32).
SOFT_LIMITS_RAD = np.array([
    [-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52],
    [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.618, 2.618], [-1.0472, 2.0944],
    [-1.97222, 1.97222], [-1.61443, 1.61443], [-1.61443, 1.61443],
    [-3.0892, 2.6704], [-2.2515, 1.5882], [-2.618, 2.618], [-1.0472, 2.0944],
    [-1.97222, 1.97222], [-1.61443, 1.61443], [-1.61443, 1.61443],
], dtype=np.float32).astype(np.float64)
# Release screen: waist 1.1 rad/s, 30 rad/s^2; arms 5.1 rad/s, 83 rad/s^2.
RELEASE_VELOCITY_RAD_S = np.r_[np.full(3, 1.1), np.full(14, 5.1)]
RELEASE_ACCELERATION_RAD_S2 = np.r_[np.full(3, 30.0), np.full(14, 83.0)]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def motion_metrics(positions, fps) -> dict:
    positions = np.asarray(positions, dtype=np.float64)
    velocity = np.abs(np.diff(positions, axis=0)) * fps
    acceleration = np.abs(np.diff(positions, n=2, axis=0)) * fps * fps
    return {
        "peak_abs_velocity_rad_s": velocity.max(axis=0).tolist(),
        "peak_abs_acceleration_rad_s2": acceleration.max(axis=0).tolist(),
        "release_velocity_exceedance_samples": int(np.count_nonzero(velocity > RELEASE_VELOCITY_RAD_S + 1e-6)),
        "release_acceleration_exceedance_samples": int(
            np.count_nonzero(acceleration > RELEASE_ACCELERATION_RAD_S2 + 1e-6)),
    }


def prepare(qpos, source_fps=30.0, smoothing_window=3, output_fps=50.0, limits=SOFT_LIMITS_RAD):
    """Return (joint_pos float32 (N,17), metrics). Raises ValueError on a limit breach."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 3 or not np.isfinite(qpos).all():
        raise ValueError(f"expected finite (T>=3, 36) MuJoCo qpos, got {qpos.shape}")
    if smoothing_window < 1 or smoothing_window % 2 != 1:
        raise ValueError("smoothing_window must be a positive odd integer")
    upper = qpos[:, 7 + UPPER_IDS[0]:7 + UPPER_IDS[-1] + 1]
    pad = smoothing_window // 2
    padded = np.pad(upper, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([padded[i:i + smoothing_window].mean(axis=0) for i in range(len(upper))])
    source_time = np.arange(len(upper)) / source_fps
    target_time = np.arange(int(np.floor(source_time[-1] * output_fps)) + 1) / output_fps
    sampled = np.stack([np.interp(target_time, source_time, smoothed[:, j]) for j in range(len(UPPER_IDS))],
                       axis=1).astype(np.float32)
    margin = np.minimum(sampled - limits[:, 0], limits[:, 1] - sampled)
    if np.any(margin < -1e-6):
        frame, joint = np.unravel_index(np.argmin(margin), margin.shape)
        raise ValueError(f"frame {frame}, motor {joint + 12} ({UPPER_NAMES[joint]}) exceeds its soft limit")
    metrics = motion_metrics(sampled, output_fps)
    if metrics["release_velocity_exceedance_samples"] or metrics["release_acceleration_exceedance_samples"]:
        raise ValueError("upper reference exceeds a release velocity/acceleration screen; "
                         "try a larger --smoothing-window or inspect the GMR output")
    metrics.update({
        "frames": len(sampled), "fps": output_fps, "last_sample_seconds": float(target_time[-1]),
        "smoothing_window_source_frames": smoothing_window,
        "minimum_soft_limit_margin_rad": float(margin.min()),
        "maximum_upper_displacement_from_first_rad": float(np.max(np.abs(sampled - sampled[0]))),
    })
    return sampled, metrics


def write_reference(path: Path, positions: np.ndarray, fps: float = 50.0) -> None:
    """Write the schema-1 npz that g1dance.reference.load_upper reads."""
    velocity = np.gradient(positions.astype(np.float64), 1.0 / fps, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream,
                            schema_version=np.array([1], dtype=np.int64),
                            fps=np.array([fps], dtype=np.float64),
                            motor_ids=np.asarray(UPPER_IDS, dtype=np.int64),
                            joint_names=np.asarray(UPPER_NAMES),
                            joint_pos=positions,
                            joint_vel=velocity)


def export(qpos_path: Path, output: Path, smoothing_window: int = 3, source_fps: float = 30.0,
           overwrite: bool = False) -> dict:
    summary_path = output.with_suffix(".json")
    if not overwrite and (output.exists() or summary_path.exists()):
        raise FileExistsError(f"{output} or {summary_path} exists; choose a new name or pass --overwrite")
    qpos = np.load(qpos_path, allow_pickle=False)
    positions, metrics = prepare(qpos, source_fps, smoothing_window)
    write_reference(output, positions)
    report = {
        "kind": "gmr_direct_upper_reference_50hz",
        "reference_path": str(output.resolve()), "reference_sha256": sha256(output),
        "source_qpos_path": str(Path(qpos_path).resolve()), "source_qpos_sha256": sha256(qpos_path),
        "source_fps": source_fps, "motor_ids": list(UPPER_IDS), "joint_names": list(UPPER_NAMES),
        "root_and_leg_commands_included": False, "physical_playback_validated": False,
        **metrics,
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qpos", type=Path, required=True, help="(T, 36) MuJoCo qpos from the GMR stage (qpos.npy)")
    p.add_argument("--output", type=Path, required=True, help="output .npz (a .json summary is written next to it)")
    p.add_argument("--smoothing-window", type=int, default=3, help="odd moving-average window in 30 Hz frames")
    p.add_argument("--source-fps", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    r = export(a.qpos, a.output, a.smoothing_window, a.source_fps, a.overwrite)
    print(f"wrote {r['frames']} upper-body frames at 50 Hz ({r['last_sample_seconds']:.2f} s): {a.output}")
    print(f"min soft-limit margin {r['minimum_soft_limit_margin_rad']:.3f} rad; summary {a.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
