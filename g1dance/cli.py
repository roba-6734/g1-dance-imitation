"""Command line entry points.

  python -m g1dance inspect                       # asset report
  python -m g1dance preview [run options]         # offline: plan + MuJoCo sim + video
  python -m g1dance check --iface IFACE           # hardware, READ-ONLY
  python -m g1dance trial --stage S [--backend unitree --iface IFACE --enable-motion]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import RunConfig, apply_overrides, load_config
from .joints import (G1_JOINT_NAMES, N_UPPER, POS_LOWER, POS_UPPER, UPPER_IDS, UPPER_NAMES,
                     classify_variant)
from .reference import (FULL_FILE, UPPER_FILE, VIDEO_FILE, load_full, load_upper, quat_wxyz_to_rpy,
                        resample_full_upper, sha256)

ROOT = Path(__file__).resolve().parent.parent
D = np.degrees


# --------------------------------------------------------------------- options
def add_run_args(p: argparse.ArgumentParser):
    p.add_argument("--assets", default=str(ROOT / "motion_assets"))
    p.add_argument("--config", default=str(ROOT / "configs" / "default.toml"),
                   help="TOML run config (default: configs/default.toml)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any config value, e.g. --set gates.track_arm=0.3")
    p.add_argument("--arms", type=float, help="arm amplitude scale (0=hold start, 1=recorded)")
    p.add_argument("--waist-yaw", type=float, help="waist yaw amplitude scale")
    p.add_argument("--waist-roll", type=float, help="waist roll amplitude scale")
    p.add_argument("--waist-pitch", type=float, help="waist pitch scale: 0=hold measured start pitch, 1=recorded")
    p.add_argument("--hold-waist", action="store_true",
                   help="hold all three waist joints at the measured start for the whole run")
    p.add_argument("--speed", type=float, help="global playback speed (1=recorded tempo)")
    p.add_argument("--slow", action="append", default=[], metavar="T0:T1:FACTOR",
                   help="extra speed factor for dance seconds T0..T1 (e.g. 8:11:0.5); amplitude unchanged")
    p.add_argument("--ref-end", type=float, help="stop playback at this dance time [s]")
    p.add_argument("--max-load", metavar="TORQUE_NM:FORCE_N",
                   help="load-limited tempo: slow down only where the arm-induced torque/force on the body "
                        "would exceed these (e.g. 1.2:4.1 = level of the clean 0.5x hardware runs); "
                        "amplitude unchanged")
    p.add_argument("--constant-tempo", action="store_true",
                   help="with --max-load: one constant tempo for the whole dance (needed for --music)")
    p.add_argument("--music-file", metavar="PATH",
                   help="song for --music / the preview video (default: the dance video's own soundtrack)")
    p.add_argument("--music-start", type=float, default=0.0, metavar="S",
                   help="start this many seconds into the song")
    p.add_argument("--music-stretch", action="store_true",
                   help="time-stretch the song to the dance tempo (needs a constant tempo); default: natural speed")
    p.add_argument("--beat-align", action="store_true",
                   help="with --max-load and --music-file: warp the timing so the dance's key moves land on the "
                        "song's beats (within the load limit); the song plays at natural speed")
    p.add_argument("--smooth", type=float, metavar="HZ",
                   help="low-pass the dance before planning (e.g. 2 or 3 Hz); removes jitter, lowers the arm load")
    p.add_argument("--out", default=str(ROOT / "runs"))


def resolve_config(a) -> RunConfig:
    cfg = load_config(a.config if a.config and Path(a.config).exists() else None)
    apply_overrides(cfg, a.set)
    for attr, key in (("arms", "arms"), ("waist_yaw", "waist_yaw"), ("waist_roll", "waist_roll"),
                      ("waist_pitch", "waist_pitch")):
        v = getattr(a, attr)
        if v is not None:
            setattr(cfg.scales, key, float(v))
    if a.hold_waist:
        cfg.hold_waist = True
    if a.speed is not None:
        cfg.speed.speed = float(a.speed)
    for s in a.slow:
        t0, t1, f = (float(x) for x in s.split(":"))
        cfg.speed.segments.append([t0, t1, f])
    if a.ref_end is not None:
        cfg.speed.ref_end_s = float(a.ref_end)
    if getattr(a, "max_load", None):
        tq, f = (float(x) for x in a.max_load.split(":"))
        cfg.speed.max_torque_nm, cfg.speed.max_force_n = tq, f
    if getattr(a, "constant_tempo", False):
        cfg.speed.constant_tempo = True
    if getattr(a, "smooth", None):
        cfg.speed.smooth_hz = float(a.smooth)
    return cfg


def setup_beat_grid(cfg: RunConfig, a, ref_duration: float):
    """For --beat-align: track the song's beats and give the period to the planner."""
    if not getattr(a, "beat_align", False):
        return None
    if not a.music_file:
        raise SystemExit("--beat-align needs --music-file (the song to align to)")
    if not (cfg.speed.max_torque_nm or cfg.speed.max_force_n):
        raise SystemExit("--beat-align needs --max-load")
    from .beats import beat_grid
    grid = beat_grid(Path(a.music_file), a.music_start, ref_duration / 0.5 + 5.0)
    cfg.speed.beat_period_s = grid.period_s
    print(f"beat grid: {grid.bpm:.2f} BPM, first beat {grid.first_beat_s:.3f}s after --music-start, "
          f"strength x{grid.strength:.2f}")
    return grid


def print_load_limit(plan):
    L = plan.load_limit
    if not L:
        return
    pk, p1 = L["peak_load_planned"], L["peak_load_recorded_tempo"]
    print(f"load-limited tempo ({L['data_kind']}): {L['recorded_s']} s of dance in {L['playback_s']} s, "
          f"min tempo {L['min_tempo']} at dance t={L['min_tempo_dance_t']} s, within limits: {L['within_limits']}")
    print(f"  peak load planned:  pitch {pk['pitch_nm']} roll {pk['roll_nm']} N m, force {pk['force_n']} N "
          f"(at recorded tempo: pitch {p1['pitch_nm']} roll {p1['roll_nm']} N m, force {p1['force_n']} N)")
    if L.get("beat_aligned"):
        print(f"  beat-aligned: {L['accents_on_grid']}/{L['accents']} key moves on the grid "
              f"({L['accents_on_full_beat']} on full beats), tempo {L['tempo_range'][0]}-{L['tempo_range'][1]}, "
              f"timing residual {L['accent_timing_residual_max_s']}s")
    elif L.get("constant_tempo"):
        print(f"  constant tempo {L['music_tempo']} for the whole dance")
    else:
        print("  slowed passages [dance t0, t1, min tempo]: " + ", ".join(str(x) for x in L["slowed_passages"]))


def print_settings(cfg: RunConfig):
    s = cfg.scales
    print("effective settings:")
    print(f"  scales: arms={s.arms} waist_yaw={s.waist_yaw} waist_roll={s.waist_roll} "
          f"waist_pitch={s.waist_pitch} hold_waist={cfg.hold_waist}")
    print(f"  effective per-joint scale: {np.round(cfg.effective_scales(), 3).tolist()}")
    print(f"  speed={cfg.speed.speed} segments={cfg.speed.segments} ref_end={cfg.speed.ref_end_s}")
    if cfg.speed.max_torque_nm or cfg.speed.max_force_n:
        print(f"  load-limited tempo: max torque {cfg.speed.max_torque_nm} N m, max force {cfg.speed.max_force_n} N "
              f"(window {cfg.speed.load_window_s} s, constant tempo {cfg.speed.constant_tempo})")
    if cfg.speed.smooth_hz:
        print(f"  reference low-pass: {cfg.speed.smooth_hz} Hz")
    g = cfg.gains
    print(f"  gains: arm kp={g.kp_arm} kd={g.kd_arm}  waist kp={g.kp_waist} kd={g.kd_waist}")
    print(f"  gates: {cfg.gates}")
    print(f"  limits: {cfg.limits}")


# --------------------------------------------------------------------- inspect
def cmd_inspect(a):
    assets = Path(a.assets)
    ref = load_upper(assets / UPPER_FILE)
    full = load_full(assets / FULL_FILE)
    rpy0 = quat_wxyz_to_rpy(full.root_quat_wxyz[0])[0]
    rpy = quat_wxyz_to_rpy(full.root_quat_wxyz)
    up_from_full = resample_full_upper(full, ref.times())
    diff = up_from_full - ref.q
    rep = {
        "upper_reference": {
            "file": UPPER_FILE, "sha256": sha256(assets / UPPER_FILE), "fps": ref.fps,
            "joint_pos_shape": list(ref.q.shape), "joint_vel_shape": list(ref.qd.shape),
            "duration_s": round(ref.duration, 3), "units": "rad, rad/s (joint_vel == np.gradient(joint_pos))",
            "motor_ids": list(UPPER_IDS), "joint_names": list(UPPER_NAMES),
        },
        "full_body_retarget": {
            "file": FULL_FILE, "sha256": sha256(assets / FULL_FILE), "fps_assumed": full.fps,
            "shape": [full.n, 36], "duration_s": round(full.duration, 3),
            "layout": "MuJoCo qpos: [0:3] root pos m, [3:7] root quat wxyz, [7:36] joints ids 0..28 rad",
            "root_height_m_range": [round(float(full.root_pos[:, 2].min()), 4), round(float(full.root_pos[:, 2].max()), 4)],
            "root_rpy0_deg": [round(float(x), 2) for x in D(rpy0)],
            "root_roll_pitch_range_deg": [round(float(D(np.ptp(rpy[:, 0]))), 2), round(float(D(np.ptp(rpy[:, 1]))), 2)],
            "leg_joint_range_deg": round(float(D(np.ptp(full.legs, axis=0).max())), 2),
            "used_for": "visual comparison only; root and legs never enter the command path",
        },
        "consistency": {
            "upper_npz_vs_full_resampled_rms_deg": round(float(D(np.sqrt((diff ** 2).mean()))), 4),
            "max_deg": round(float(D(np.abs(diff).max())), 4),
            "time_offset_s": 0.0,
            "note": "npz == full-body joints 12..28 linearly resampled 30->50 Hz, same t=0",
        },
    }
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type,width,height,r_frame_rate:stream_side_data=rotation",
                                "-of", "json", str(assets / VIDEO_FILE)], capture_output=True, text=True, timeout=30)
        rep["video"] = json.loads(probe.stdout)
    except Exception as e:  # pragma: no cover
        rep["video"] = {"error": str(e)}
    frame0 = []
    for j in range(N_UPPER):
        frame0.append({"motor_id": UPPER_IDS[j], "joint": UPPER_NAMES[j],
                       "frame0_rad": round(float(ref.q[0, j]), 4), "frame0_deg": round(float(D(ref.q[0, j])), 2),
                       "range_deg": [round(float(D(ref.q[:, j].min())), 1), round(float(D(ref.q[:, j].max())), 1)],
                       "urdf_deg": [round(float(D(POS_LOWER[j])), 1), round(float(D(POS_UPPER[j])), 1)],
                       "peak_speed_rad_s": round(float(np.abs(ref.qd[:, j]).max()), 3)})
    rep["frame0_mapping"] = frame0
    print(json.dumps({k: v for k, v in rep.items() if k != "frame0_mapping"}, indent=2))
    print("\nframe 0 -> G1 motor mapping (upper body):")
    print(f"  {'id':>3} {'joint':28s} {'frame0':>8} {'min..max deg':>16} {'URDF deg':>16} {'peak rad/s':>10}")
    for r in frame0:
        print(f"  {r['motor_id']:3d} {r['joint']:28s} {r['frame0_deg']:+8.2f} "
              f"{r['range_deg'][0]:+7.1f}..{r['range_deg'][1]:+6.1f} "
              f"{r['urdf_deg'][0]:+7.1f}..{r['urdf_deg'][1]:+6.1f} {r['peak_speed_rad_s']:10.3f}")
    out = Path(a.out) / "asset_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwritten {out}")


# --------------------------------------------------------------------- preview
def cmd_preview(a):
    from .metrics import kinematic_report, run_report
    from .monitor import Monitor
    from .planner import build_plan
    from .runner import execute, new_run_dir, write_run
    from .sim import SimBackend, assumed_start_upper

    cfg = resolve_config(a)
    print_settings(cfg)
    assets = Path(a.assets)
    ref = load_upper(assets / UPPER_FILE)
    full = load_full(assets / FULL_FILE)
    if a.start_pose:
        start_in = np.array(json.loads(Path(a.start_pose).read_text())["upper_q"], float)
        start_src = f"measured stance from {a.start_pose}"
    else:
        start_in = assumed_start_upper()
        start_src = "ASSUMED stance (Menagerie 'stand' arms); live runs use the measured pose"
    backend = SimBackend(start_in, cfg.timing.control_dt)
    start = backend.read_state().upper_q
    beat = setup_beat_grid(cfg, a, ref.duration)
    plan = build_plan(ref, start, cfg)
    result = execute(plan, backend, cfg, Monitor(cfg.gates))
    tag = a.tag or "preview"
    run_dir = new_run_dir(Path(a.out), tag)
    log = result.log.arrays()
    kin = kinematic_report(plan, ref)
    sim = run_report(log, plan, ref, "simulated")
    extra = {"backend": backend.name, "start_pose_source": start_src, "start_pose_rad": start.tolist(),
             "assets": {UPPER_FILE: sha256(assets / UPPER_FILE)},
             "kinematic": kin, "simulated": sim}
    if plan.load_limit:
        extra["load_limit"] = plan.load_limit
    print_load_limit(plan)
    write_run(run_dir, result, cfg, extra)
    np.savez_compressed(run_dir / "plan.npz", **plan.flat(), start=plan.start, entry_target=plan.entry_target)
    _print_summary(kin, sim, result)
    from .plots import plot_traces
    plot_traces(run_dir / "traces.png", log, ref, "simulated response",
                f"{tag}: kinematic command vs MuJoCo fixed-pelvis simulation")
    if not a.no_video:
        from .render import render_video
        pb = np.flatnonzero(log["phase"] == 3)
        audio, wav = None, None
        if len(pb) and not a.no_audio:
            from .music import plan_music
            try:
                wav, delay, minfo = plan_music(plan, a.music_file, assets / VIDEO_FILE, run_dir / "music",
                                               a.music_start, a.music_stretch, beat)
                audio = float(log["t"][pb[0]]) + delay
                print(f"music: {minfo['source']} (stretched to {minfo['stretched_to_tempo']}), "
                      f"from {a.music_start:.1f}s into the song")
            except SystemExit as e:
                print(f"no music in the video: {e}")
        g = cfg.gains
        notes = [f"scales arms={cfg.scales.arms} yaw={cfg.scales.waist_yaw}",
                 f"  roll={cfg.scales.waist_roll} pitch={cfg.scales.waist_pitch}"
                 + (" HOLD-WAIST" if cfg.hold_waist else ""),
                 f"speed={cfg.speed.speed} {cfg.speed.segments or ''}",
                 *( [f"max-load {cfg.speed.max_torque_nm}:{cfg.speed.max_force_n}"
                     + (" constant" if cfg.speed.constant_tempo else "")] if cfg.speed.max_torque_nm else []),
                 *( [f"smooth {cfg.speed.smooth_hz} Hz"] if cfg.speed.smooth_hz else []),
                 f"kp arm/waist = {g.kp_arm:g}/{g.kp_waist:g}",
                 "legs: Unitree controller (not modelled;",
                 "  shown at nominal stand)",
                 "source: re-headed to face camera"]
        if result.stop:
            notes += ["", "STOPPED: " + result.stop.check, "  " + result.stop.subject[:40]]
        vid = render_video(run_dir / "comparison.mp4", full, log, assets / VIDEO_FILE,
                           f"G1 upper-body dance: {tag}", notes, audio, wav)
        print(f"video: {vid}")
    print(f"run dir: {run_dir}")
    return run_dir


def _print_summary(kin, sim, result):
    o = kin["opening_pose"]
    print("\n=== KINEMATIC (planned command targets) ===")
    print(f"opening pose: command vs recorded frame 0 max {o['commanded_vs_recorded_frame0_max_deg']} deg "
          f"({o['worst_joint']}); entry {o['entry_duration_s']:.2f} s")
    p = kin["playback"]
    print(f"playback: {p['dance_duration_s']} s of dance in {p['wall_duration_s']} s; worst cmd-vs-recorded "
          f"{p['cmd_vs_recorded_reference_worst']}")
    print(f"limit events: {len(p['limit_events'])}")
    for e in p["limit_events"][:12]:
        print(f"   {e}")
    print("=== SIMULATED (MuJoCo fixed pelvis; not hardware) ===")
    if "opening_pose_at_gate" in sim:
        g = sim["opening_pose_at_gate"]
        print(f"opening pose at gate: measured vs target max {g['measured_vs_target_max_deg']} deg "
              f"({g['worst_joint']}); vs recorded frame0 {g['measured_vs_recorded_frame0_max_deg']} deg")
    if "playback" in sim:
        print(f"playback worst vs reference: {sim['playback']['measured_vs_scaled_reference_worst']}  "
              f"completed {sim['playback']['completed_fraction'] * 100:.1f}%")
    print(f"waist pitch measurements: {sim['waist_pitch_measurements']}")
    print("RESULT:", "completed + handed back" if result.completed else result.stop.message())


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1dance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="report asset shapes, units, joint mapping, frame 0")
    p.add_argument("--assets", default=str(ROOT / "motion_assets"))
    p.add_argument("--out", default=str(ROOT / "runs"))
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("preview", help="offline plan + simulation + comparison video")
    add_run_args(p)
    p.add_argument("--start-pose", help="JSON from `check --save-stance` to plan from a measured stance")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(fn=cmd_preview)

    from .hw import check as hwcheck, trial as hwtrial
    hwcheck.add_parser(sub)
    hwtrial.add_parser(sub, add_run_args)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main() and 0)
