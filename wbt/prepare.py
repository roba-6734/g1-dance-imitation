"""Turn a retargeted G1 qpos (GMR or NMR, 30 Hz) into a training CSV for unitree_rl_mjlab.

    python -m wbt.prepare --qpos outputs/my_dance/qpos.npy --out wbt_motions/ayyala_gmr.csv \
        [--trim 0 22.8] [--lead-in 2.0] [--lead-out 2.0] [--lowpass-hz 6] [--root-xy keep|lock]

Output CSV (no header, one row per 30 Hz frame): root x y z, root quaternion x y z w,
29 joint angles in SDK order. This is the convention of Unitree's motion CSVs and of
``scripts/csv_to_npz.py`` (which converts xyzw -> wxyz itself).

Steps, in order:
  1. optional trim, then optional zero-phase Butterworth low-pass (joints, root position, root quaternion)
  2. optional root-xy lock (in-place dance; removes GVHMR moving-camera drift; adds foot skate if the
     dancer really steps, so feasibility reports it)
  3. yaw so the dance starts facing +x and xy so it starts at the origin (cosmetic: deploy re-aligns yaw)
  4. one constant height offset so the feet touch the floor (5th percentile of per-frame lowest foot point)
  5. lead-in: hold the Velocity-policy stance, then a min-jerk blend into frame 0; lead-out: the reverse.
     The real robot enters the mimic state from that stance, so the policy never has to jump to a mid-dance pose.
A JSON sidecar records every choice and where the dance starts and ends in the new timeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from video2ref.motion import check_qpos
from .g1model import build_model

FPS = 30.0
# Velocity-policy stance (unitree_rl_mjlab deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml):
# the pose the robot is in when the FSM switches to the mimic state.
ENTRY_POSE = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0,
                       0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0])


def minjerk(s):
    s = np.clip(s, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def lowpass(qpos, cutoff_hz, fps=FPS):
    if cutoff_hz <= 0:
        return qpos.copy()
    if len(qpos) < 16:
        raise ValueError("too few frames to filter")
    b, a = butter(4, cutoff_hz / (fps / 2), btype="low")
    out = qpos.copy()
    out[:, :3] = filtfilt(b, a, qpos[:, :3], axis=0)
    out[:, 7:] = filtfilt(b, a, qpos[:, 7:], axis=0)
    quat = qpos[:, 3:7].copy()
    for i in range(1, len(quat)):  # hemisphere-continuous before filtering components
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    quat = filtfilt(b, a, quat, axis=0)
    out[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return out


def heading(quat_wxyz):
    return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("ZYX")[0]


def set_yaw_origin(qpos):
    out = qpos.copy()
    rz = Rotation.from_euler("z", -heading(qpos[0, 3:7]))
    out[:, :3] = rz.apply(qpos[:, :3] - [qpos[0, 0], qpos[0, 1], 0])
    out[:, 3:7] = (rz * Rotation.from_quat(qpos[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    return out


class FootFK:
    def __init__(self, xml=None):
        self.model = build_model(xml)
        self.data = mujoco.MjData(self.model)
        self.feet = [g for g in range(self.model.ngeom)
                     if (self.model.geom(g).name or "").startswith(("left_foot", "right_foot"))]
        self.side = np.array([self.model.geom(g).name.startswith("left") for g in self.feet])

    def lowest(self, q):
        """Lowest foot-capsule surface point (m) per foot: (left, right)."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        out = []
        for left in (True, False):
            zs = []
            for g in np.array(self.feet)[self.side == left]:
                half = self.model.geom_size[g][1]
                axis = self.data.geom_xmat[g].reshape(3, 3)[:, 2]
                c = self.data.geom_xpos[g]
                zs.append(min(c[2] - half * axis[2], c[2] + half * axis[2]) - self.model.geom_size[g][0])
            out.append(min(zs))
        return np.array(out)

    def standing_height(self, root_quat, q):
        probe = np.concatenate([[0, 0, 1.0], root_quat, q])
        return 1.0 - self.lowest(probe).min()


def transition(a, b, seconds, fk: FootFK, hold=0.5):
    """Frames from qpos a to qpos b (exclusive): hold a, then min-jerk joints/root, slerp root orientation."""
    n_hold, n_move = int(round(hold * FPS)), int(round(seconds * FPS))
    s = minjerk(np.arange(n_move) / max(n_move, 1))
    rots = Rotation.from_quat(np.stack([a[[4, 5, 6, 3]], b[[4, 5, 6, 3]]]))
    quat = Slerp([0, 1], rots)(s).as_quat()[:, [3, 0, 1, 2]] if n_move else np.zeros((0, 4))
    moving = np.zeros((n_move, 36))
    moving[:, :3] = a[:3] + s[:, None] * (b[:3] - a[:3])
    moving[:, 3:7] = quat
    moving[:, 7:] = a[7:] + s[:, None] * (b[7:] - a[7:])
    for i in range(n_move):  # keep the feet on the floor through the blend (shift root height only)
        moving[i, 2] -= fk.lowest(moving[i]).min()
    return np.concatenate([np.repeat(a[None], n_hold, 0), moving])


def stance_like(frame, fk: FootFK, pose=ENTRY_POSE):
    """The entry stance placed under the dance's first/last frame (same xy and heading, upright)."""
    q = np.zeros(36)
    yaw = heading(frame[3:7])
    q[3:7] = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    q[7:] = pose
    q[0:2] = frame[0:2]
    q[2] = fk.standing_height(q[3:7], q[7:])
    return q


def prepare(qpos, *, trim=None, lowpass_hz=0.0, root_xy="keep", lead_in=2.0, lead_out=2.0, xml=None):
    qpos = check_qpos(np.asarray(qpos, float))
    info = {"source_frames": len(qpos), "fps": FPS}
    if trim:
        a, b = int(round(trim[0] * FPS)), int(round(trim[1] * FPS)) + 1
        qpos = qpos[a:b]
        info["trim_s"] = list(trim)
    qpos = lowpass(qpos, lowpass_hz)
    info["lowpass_hz"] = lowpass_hz
    if root_xy == "lock":
        qpos[:, :2] = qpos[0, :2]
    info["root_xy"] = root_xy
    qpos = set_yaw_origin(qpos)
    fk = FootFK(xml)
    low = np.array([fk.lowest(q).min() for q in qpos])
    offset = float(np.percentile(low, 5))
    qpos[:, 2] -= offset
    info["height_offset_m"] = -offset
    stance_in, stance_out = stance_like(qpos[0], fk), stance_like(qpos[-1], fk)
    head = transition(stance_in, qpos[0], lead_in, fk) if lead_in > 0 else np.zeros((0, 36))
    tail = transition(qpos[-1], stance_out, lead_out, fk, hold=0.0) if lead_out > 0 else np.zeros((0, 36))
    if lead_out > 0:
        tail = np.concatenate([tail, np.repeat(stance_out[None], int(FPS), 0)])  # 1 s settled stance at the end
    out = check_qpos(np.concatenate([head, qpos, tail]))
    info.update({"frames": len(out), "seconds": round((len(out) - 1) / FPS, 3),
                 "dance_start_s": round(len(head) / FPS, 3), "dance_end_s": round((len(head) + len(qpos) - 1) / FPS, 3),
                 "entry_pose": ENTRY_POSE.tolist()})
    return out, info


def write_csv(path: Path, qpos):
    rows = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)  # wxyz -> xyzw
    np.savetxt(path, rows, delimiter=",", fmt="%.6f")


def read_csv(path: Path):
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim != 2 or rows.shape[1] != 36:
        raise ValueError(f"{path}: expected 36 columns")
    return np.concatenate([rows[:, :3], rows[:, [6, 3, 4, 5]], rows[:, 7:]], axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qpos", type=Path, required=True, help="(T,36) qpos.npy at 30 Hz: root xyz, quat wxyz, 29 joints")
    p.add_argument("--out", type=Path, required=True, help="output .csv")
    p.add_argument("--trim", type=float, nargs=2, metavar=("START_S", "END_S"))
    p.add_argument("--lowpass-hz", type=float, default=0.0, help="zero-phase Butterworth cut-off; 0 = off")
    p.add_argument("--root-xy", choices=["keep", "lock"], default="keep")
    p.add_argument("--lead-in", type=float, default=2.0)
    p.add_argument("--lead-out", type=float, default=2.0)
    p.add_argument("--xml", help="mjlab g1.xml")
    a = p.parse_args(argv)
    src = np.load(a.qpos)
    out, info = prepare(src, trim=a.trim, lowpass_hz=a.lowpass_hz, root_xy=a.root_xy, lead_in=a.lead_in,
                        lead_out=a.lead_out, xml=a.xml)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(a.out, out)
    info["source"] = str(a.qpos)
    info["source_sha256"] = hashlib.sha256(a.qpos.read_bytes()).hexdigest()
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{a.out}: {info['frames']} frames ({info['seconds']} s); dance {info['dance_start_s']}-{info['dance_end_s']} s")


if __name__ == "__main__":
    main()
