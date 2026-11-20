"""Offline balance analysis of hardware run logs (no robot involved).

For every logged tick, place the Menagerie G1 model at the measured joint angles
(legs included, read-only data) with the pelvis attitude from the IMU (roll,
pitch), and compute in a frame centred between the feet (x forward, y left):
  * com_xy  - whole-body centre of mass
  * zmp_xy  - zero-moment point from the cart-table approximation
              zmp = com - (com_z - foot_z)/g * com_ddot
and the upper-body-only share of the CoM acceleration (legs frozen at their
start angles, pelvis level) - the part our commands control.

Usage: python scripts/balance_analysis.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1dance.sim import load_model  # noqa: E402

G = 9.81
STEP_DQ = 1.5  # rad/s, same as gates.leg_step_dq


def rpy_to_quat(r, p, y=0.0):
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class Kin:
    def __init__(self):
        self.m = load_model(fixed_base=False)
        self.d = mujoco.MjData(self.m)
        self.lf = self.m.site("left_foot").id
        self.rf = self.m.site("right_foot").id

    def com_and_feet(self, q29, rpy):
        d = self.d
        d.qpos[:] = 0
        d.qpos[3:7] = rpy_to_quat(rpy[0], rpy[1])
        d.qpos[7:36] = q29
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.subtree_com[1].copy(), d.site_xpos[self.lf].copy(), d.site_xpos[self.rf].copy()


def smooth_ddot(t, x, win=7):
    k = np.ones(win) / win
    xs = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, x)
    v = np.gradient(xs, t, axis=0)
    vs = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, v)
    return np.gradient(vs, t, axis=0)


def analyse(run_dir: Path, kin: Kin):
    a = np.load(run_dir / "log.npz")
    t, q, rpy = a["t"], a["q_meas"], a["imu_rpy"]
    n = len(t)
    com = np.zeros((n, 3)); mid = np.zeros((n, 3)); yaw = np.zeros(n); com_up = np.zeros((n, 3))
    q_leg0 = q[0, :12]
    for k in range(n):
        c, lf, rf = kin.com_and_feet(q[k], rpy[k])
        com[k], mid[k] = c, (lf + rf) / 2
        dv = lf - rf
        yaw[k] = np.arctan2(dv[1], dv[0]) - np.pi / 2  # feet line is the lateral axis
        qu = q[k].copy(); qu[:12] = q_leg0
        com_up[k] = kin.com_and_feet(qu, (0.0, 0.0))[0]

    def to_feet(v):  # horizontal vector into the feet frame
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)

    rel = com - mid
    acc = smooth_ddot(t, com)
    acc_up = smooth_ddot(t, com_up)
    h = rel[:, 2:3]
    zmp = to_feet(rel[:, :2] - h / G * acc[:, :2])
    com_f = to_feet(rel[:, :2])
    acc_up_f = to_feet(acc_up[:, :2])
    leg = np.abs(a["dq_meas"][:, :12]).max(1)
    fast = (leg[1:] > STEP_DQ) & (leg[:-1] > STEP_DQ)
    onset = int(np.argmax(fast)) + 1 if fast.any() else None
    feet_w = np.linalg.norm((lambda c: c)(kin.com_and_feet(q[0], rpy[0])[1] - kin.com_and_feet(q[0], rpy[0])[2]))
    return dict(t=t, t_ref=a["t_ref"], phase=a["phase"], com=com_f, zmp=zmp, acc_up=acc_up_f,
                onset=onset, feet_width=feet_w)


def main(dirs):
    kin = Kin()
    for d in dirs:
        r = analyse(Path(d), kin)
        k0 = r["onset"]
        lim = slice(0, k0) if k0 is not None else slice(None)
        z, c, au = r["zmp"][lim], r["com"][lim], r["acc_up"][lim]
        rel0 = r["com"][0]
        print(f"{Path(d).name}: feet width {r['feet_width']:.3f} m; CoM at start x={rel0[0]*100:+.1f} y={rel0[1]*100:+.1f} cm")
        tag = "before first step" if k0 is not None else "whole run (no step)"
        print(f"  {tag}: ZMP x [{z[:,0].min()*100:+.1f}, {z[:,0].max()*100:+.1f}] cm, "
              f"y [{z[:,1].min()*100:+.1f}, {z[:,1].max()*100:+.1f}] cm; "
              f"upper-body CoM accel max |x| {np.abs(au[:,0]).max():.2f} |y| {np.abs(au[:,1]).max():.2f} m/s^2")
        if k0 is not None:
            w = slice(max(0, k0 - 25), k0)
            print(f"  0.5 s before step (dance t={r['t_ref'][k0]:.2f}): ZMP x {r['zmp'][w,0].min()*100:+.1f}..{r['zmp'][w,0].max()*100:+.1f} cm, "
                  f"y {r['zmp'][w,1].min()*100:+.1f}..{r['zmp'][w,1].max()*100:+.1f} cm")


if __name__ == "__main__":
    main(sys.argv[1:])
