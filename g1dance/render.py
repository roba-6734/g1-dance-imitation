"""Side-by-side comparison video.

Columns: human recording | retargeted source (full body, 30 Hz) |
planned upper-body COMMAND (kinematic targets) | SIMULATED response.
Rows: front view and side view. During acquire/entry/settle the source
columns are frozen on dance frame 0; after playback they freeze on the last
frame, so the opening-pose match and the return/handback are visible.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .joints import UPPER_IDS, WAIST_PITCH
from .planner import PHASES
from .reference import FullBodyRetarget, quat_wxyz_to_rpy
from .sim import PELVIS_HEIGHT, _mujoco, load_model, stand_pose_29

PW, PH = 360, 480  # panel size
FPS = 30


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


class HumanVideo:
    """Sequential reader of the human recording at 30 fps, panel-sized."""

    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
             f"fps={FPS},scale=-2:{PH},pad={PW}:{PH}:(ow-iw)/2:0:black", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.idx = -1
        self.frame = np.zeros((PH, PW, 3), np.uint8)

    def at(self, t: float) -> np.ndarray:
        want = int(round(t * FPS))
        while self.idx < want:
            buf = self.proc.stdout.read(PW * PH * 3)
            if len(buf) < PW * PH * 3:
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(PH, PW, 3)
            self.idx += 1
        return self.frame

    def close(self):
        self.proc.kill()


def _aac_stream(path) -> int | None:
    """Index of the first decodable (AAC) audio stream; iPhone files also carry APAC."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index,codec_name", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        for line in r.stdout.split():
            idx, name = line.split(",")[:2]
            if name == "aac":
                return int(idx)
    except Exception:
        pass
    return None


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def source_qpos(full: FullBodyRetarget, t_ref: float) -> np.ndarray:
    """Full-body source pose at t_ref, re-headed so frame 0 faces +x at the origin."""
    qp = full.mujoco_qpos()
    x = np.clip(t_ref * full.fps, 0, full.n - 1)
    k = min(int(x), full.n - 2)
    a = x - k
    q = (1 - a) * qp[k] + a * qp[k + 1]
    quat = q[3:7] / np.linalg.norm(q[3:7])
    yaw0 = quat_wxyz_to_rpy(full.root_quat_wxyz[0])[0, 2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    d = q[0:2] - full.root_pos[0, 0:2]
    q[0:2] = [c * d[0] - s * d[1], s * d[0] + c * d[1]]
    q[3:7] = _qmul(_yaw_quat(-yaw0), quat)
    return q


def robot_qpos(upper: np.ndarray) -> np.ndarray:
    q = np.zeros(36)
    q[0:3] = [0, 0, PELVIS_HEIGHT]
    q[3] = 1.0
    j = stand_pose_29()
    j[list(UPPER_IDS)] = upper
    q[7:] = j
    return q


def render_video(out: Path, full: FullBodyRetarget, log: dict, human: Path | None,
                 title: str, notes: list[str], audio_offset_s: float | None = None,
                 sim_label: str = "SIMULATED (MuJoCo, fixed pelvis)"):
    mujoco = _mujoco()
    model = load_model(fixed_base=False)
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, PH, PW)
    cams = []
    for az in (180.0, 90.0):  # front, side (robot faces +x)
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = [0, 0, 1.0]
        c.distance, c.elevation, c.azimuth = 2.3, -8.0, az
        cams.append(c)

    def shot(qpos, cam, lookat_xy=(0, 0)):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        cam.lookat[0], cam.lookat[1] = lookat_xy
        rend.update_scene(data, cam)
        return rend.render()

    t_log = log["t"]
    dt = float(np.median(np.diff(t_log)))
    T = t_log[-1]
    pb_mask = log["phase"] == PHASES.index("playback")
    pb_idx = np.flatnonzero(pb_mask)
    t_last = float(np.nanmax(log["t_ref"])) if pb_mask.any() else 0.0
    hv = HumanVideo(human) if human and Path(human).exists() else None
    f_big, f_small, f_lab = _font(22), _font(15), _font(17)
    W, H = PW * 4, PH * 2
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    aac = _aac_stream(human) if (audio_offset_s is not None and human) else None
    if aac is not None:
        # Music starts when the dance timeline starts (after acquire/entry/settle).
        cmd += ["-itsoffset", f"{audio_offset_s:.3f}", "-i", str(human), "-map", "0:v", "-map", f"1:{aac}",
                "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    labels = ["HUMAN recording", "RETARGETED source (full body)", "PLANNED command (kinematic)", sim_label]
    n_frames = int(T * FPS) + 1
    for fi in range(n_frames):
        t = fi / FPS
        k = min(int(round(t / dt)), len(t_log) - 1)
        phase = PHASES[log["phase"][k]] if log["phase"][k] >= 0 else "?"
        if pb_mask.any() and k >= pb_idx[0]:
            tr = log["t_ref"][k] if not np.isnan(log["t_ref"][k]) else t_last
        else:
            tr = 0.0
        src = source_qpos(full, tr)
        plan_q = robot_qpos(log["q_cmd"][k])
        sim_q = robot_qpos(log["q_meas"][k, 12:29])
        canvas = Image.new("RGB", (W, H), (18, 18, 22))
        human_img = hv.at(tr) if hv else np.zeros((PH, PW, 3), np.uint8)
        canvas.paste(Image.fromarray(human_img), (0, 0))
        for ci, q in enumerate((src, plan_q, sim_q)):
            xy = q[0:2] if ci == 0 else (0, 0)
            for ri, cam in enumerate(cams):
                canvas.paste(Image.fromarray(shot(q, cam, xy)), (PW * (ci + 1), PH * ri))
        d = ImageDraw.Draw(canvas)
        for ci, lab in enumerate(labels):
            d.rectangle([PW * ci, 0, PW * (ci + 1), 24], fill=(0, 0, 0))
            d.text((PW * ci + 6, 3), lab, font=f_lab, fill=(255, 255, 255))
        for ci in (1, 2, 3):
            d.text((PW * ci + 6, PH + 4), "side view", font=f_small, fill=(230, 230, 230))
        # info panel
        x0, y0 = 10, PH + 10
        d.text((x0, y0), title, font=f_big, fill=(255, 220, 120))
        w = log["weight"][k]
        qm, qc = log["q_meas"][k, 12:29], log["q_cmd"][k]
        lines = [f"run t = {t:6.2f} s    phase: {phase.upper()}",
                 f"dance t = {tr:6.2f} s   arm_sdk weight = {w:.2f}",
                 f"waist pitch  cmd {np.degrees(qc[WAIST_PITCH]):+6.2f} deg",
                 f"             sim {np.degrees(qm[WAIST_PITCH]):+6.2f} deg",
                 f"max |sim-cmd| {np.degrees(np.max(np.abs(qm - qc))):5.2f} deg",
                 ""] + notes
        for i, ln in enumerate(lines):
            d.text((x0, y0 + 34 + 20 * i), ln, font=f_small, fill=(235, 235, 235))
        # phase timeline
        bar_y = H - 30
        cols = {0: (90, 90, 200), 1: (220, 160, 40), 2: (200, 200, 60), 3: (60, 180, 90),
                4: (120, 120, 120), 5: (200, 90, 60), 6: (150, 80, 160)}
        for i in range(0, len(t_log), 5):
            xx = x0 + int((PW - 20) * t_log[i] / T)
            d.line([xx, bar_y, xx, bar_y + 14], fill=cols.get(int(log["phase"][i]), (80, 80, 80)), width=2)
        xx = x0 + int((PW - 20) * t / T)
        d.polygon([(xx - 5, bar_y - 8), (xx + 5, bar_y - 8), (xx, bar_y - 1)], fill=(255, 255, 255))
        ff.stdin.write(np.asarray(canvas).tobytes())
    ff.stdin.close()
    ff.wait()
    if hv:
        hv.close()
    return out
