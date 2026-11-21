"""CPU stand-in for unitree_rl_mjlab's scripts/csv_to_npz.py (MuJoCo FK instead of mjlab/mujoco_warp).

    python -m wbt.npz --csv wbt_motions/ayyala_gmr.csv --out wbt_motions/ayyala_gmr_cpu.npz

Same resampling as csv_to_npz (times = arange(0, duration, 1/50); lerp joints/root, slerp root;
np.gradient velocities; SO(3) central-difference angular velocity) and the same body set/order
(all 30 G1 bodies, MJCF order). Use it for tests and CPU previews; train and deploy with the mjlab
output, which is what the policy was optimised against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .g1model import build_model
from .prepare import read_csv


def resample(qpos30, in_fps=30.0, out_fps=50.0):
    n = len(qpos30)
    duration = (n - 1) / in_fps
    times = np.arange(0, duration, 1 / out_fps)
    phase = times / duration
    i0 = np.floor(phase * (n - 1)).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (phase * (n - 1) - i0)[:, None]
    out = np.zeros((len(times), 36))
    out[:, :3] = qpos30[i0, :3] * (1 - w) + qpos30[i1, :3] * w
    out[:, 7:] = qpos30[i0, 7:] * (1 - w) + qpos30[i1, 7:] * w
    rots = Rotation.from_quat(qpos30[:, [4, 5, 6, 3]])
    out[:, 3:7] = Slerp(np.arange(n) / in_fps, rots)(np.clip(times, 0, duration)).as_quat()[:, [3, 0, 1, 2]]
    return out


def motion_npz(qpos30, in_fps=30.0, out_fps=50.0, xml=None) -> dict:
    q = resample(np.asarray(qpos30, float), in_fps, out_fps)
    dt = 1 / out_fps
    # velocities exactly as csv_to_npz: gradient for root position / joints, SO(3) central difference
    # (world frame) for the root, then the simulator's own body velocities for that state
    root_lin = np.gradient(q[:, :3], dt, axis=0)
    rots = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    w = (rots[2:] * rots[:-2].inv()).as_rotvec() / (2 * dt)
    root_ang = np.concatenate([w[:1], w, w[-1:]])
    joint_vel = np.gradient(q[:, 7:], dt, axis=0)
    model = build_model(xml)
    d = mujoco.MjData(model)
    nb = model.nbody - 1
    pos, quat = np.zeros((len(q), nb, 3)), np.zeros((len(q), nb, 4))
    lin, ang = np.zeros((len(q), nb, 3)), np.zeros((len(q), nb, 3))
    v6 = np.zeros(6)
    for t in range(len(q)):
        d.qpos[:] = q[t]
        d.qvel[:3] = root_lin[t]
        d.qvel[3:6] = rots[t].inv().apply(root_ang[t])  # MuJoCo free joint: angular velocity in body frame
        d.qvel[6:] = joint_vel[t]
        mujoco.mj_forward(model, d)
        pos[t], quat[t] = d.xpos[1:], d.xquat[1:]
        for b in range(nb):
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_XBODY, b + 1, v6, 0)
            ang[t, b], lin[t, b] = v6[:3], v6[3:]
    return {"fps": np.array([out_fps]), "joint_pos": q[:, 7:].astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32), "body_pos_w": pos.astype(np.float32),
            "body_quat_w": quat.astype(np.float32), "body_lin_vel_w": lin.astype(np.float32),
            "body_ang_vel_w": ang.astype(np.float32)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--xml")
    a = p.parse_args(argv)
    data = motion_npz(read_csv(a.csv), xml=a.xml)
    np.savez(a.out, **data)
    print(f"{a.out}: {len(data['joint_pos'])} frames at 50 Hz")


if __name__ == "__main__":
    main()
