"""Operator-initiated staged trials.

Stages (each one ends by returning to the measured start and handing back):
  acquire   take arm_sdk control (weight 0->1) holding the measured pose, hold, release
  entry     acquire -> opening-pose entry -> gate -> return -> release
  playback  full sequence, truncated at --ref-end
  full      full sequence, whole dance

``--backend sim`` (default) runs the identical code path against MuJoCo.
``--backend unitree`` publishes to rt/arm_sdk ONLY with --enable-motion AND a
typed confirmation at an interactive terminal.
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import numpy as np

from ..joints import UPPER_NAMES, classify_variant

D = np.degrees


def add_parser(sub, add_run_args):
    p = sub.add_parser("trial", help="staged trial (sim by default; live needs --enable-motion)")
    add_run_args(p)
    p.add_argument("--music", action="store_true",
                   help="live only: play the dance's song (time-stretched to the constant tempo) from this "
                        "computer, starting with playback")
    p.add_argument("--music-offset", type=float, default=0.0, metavar="S",
                   help="shift the music start: positive = later, negative = earlier (audio latency)")
    p.add_argument("--allow-stance-asym", action="store_true",
                   help="run even if the feet start staggered (stance gate); logged")
    p.add_argument("--stage", required=True, choices=["acquire", "entry", "playback", "full", "recover"])
    p.add_argument("--backend", choices=["sim", "unitree"], default="sim")
    p.add_argument("--iface", help="robot network interface (unitree backend)")
    p.add_argument("--enable-motion", action="store_true",
                   help="REQUIRED to publish motor commands on the unitree backend")
    p.add_argument("--auto-advance", action="store_true",
                   help="do not pause for confirmation between the gate and playback")
    p.add_argument("--assume-mode-machine", type=int, default=None,
                   help="only if the robot reports an unknown mode_machine and you have verified the variant")
    p.add_argument("--sim-start-pose", default=None, help="sim backend: stance JSON from `check`")
    p.add_argument("--to-stance", default=None,
                   help="recover: stance JSON (from `check`) to return to before handing back")
    p.set_defaults(fn=run)


def _ask(prompt: str, expect: str) -> bool:
    if not sys.stdin.isatty():
        print("refusing: confirmation needs an interactive terminal")
        return False
    try:
        return input(prompt).strip() == expect
    except EOFError:
        return False


def run(a):
    from ..cli import print_load_limit, print_settings, resolve_config, setup_beat_grid
    from ..metrics import kinematic_report, run_report
    from ..monitor import Monitor
    from ..planner import build_acquire_only_plan, build_entry_only_plan, build_plan, build_recover_plan
    from ..reference import UPPER_FILE, load_upper, sha256
    from ..runner import execute, new_run_dir, write_run

    cfg = resolve_config(a)
    if a.stage == "playback" and cfg.speed.ref_end_s is None:
        raise SystemExit("--stage playback needs --ref-end SECONDS (use --stage full for the whole dance)")
    if a.stage == "full" and cfg.speed.ref_end_s is not None:
        raise SystemExit("--stage full plays the whole dance; drop --ref-end or use --stage playback")
    ref = load_upper(Path(a.assets) / UPPER_FILE)

    live = a.backend == "unitree"
    if live:
        if not a.iface:
            raise SystemExit("--iface is required for the unitree backend")
        if not a.enable_motion:
            raise SystemExit("refusing to command motors: pass --enable-motion (and run `check` first)")
        from .unitree import TopicActivityProbe, UnitreeBackend
        backend = UnitreeBackend(a.iface, cfg.timing.control_dt)
        probe = TopicActivityProbe()
        import time
        time.sleep(0.5)
        if probe.count:
            raise SystemExit(f"another process is publishing rt/arm_sdk ({probe.count} msgs); stop it first")
    else:
        from ..sim import SimBackend, assumed_start_upper
        s0 = (np.array(json.loads(Path(a.sim_start_pose).read_text())["upper_q"])
              if a.sim_start_pose else assumed_start_upper())
        backend = SimBackend(s0, cfg.timing.control_dt)

    st = backend.read_state()
    mm = st.mode_machine if a.assume_mode_machine is None else a.assume_mode_machine
    variant = classify_variant(mm)
    print(f"robot: {variant.describe()}" + ("  (ASSUMED by operator)" if a.assume_mode_machine is not None else ""))
    if not variant.supported:
        raise SystemExit("unsupported or unknown G1 variant; verify mode_machine (docs/HARDWARE.md)")
    if variant.waist_roll_pitch_actuated is False and not cfg.hold_waist and (
            cfg.scales.waist_roll != 0 or cfg.scales.waist_pitch != 0):
        raise SystemExit("waist roll/pitch are locked on this variant: use --waist-roll 0 --waist-pitch 0 or --hold-waist")

    stance_info = None
    if a.stage in ("entry", "playback", "full"):
        from .check import stance_gate
        import time as _t
        legs = []
        for _ in range(25):  # median over ~0.5 s of state
            legs.append(np.degrees(backend.read_state().q[:12]))
            if live:
                _t.sleep(0.02)
        ok, msg = stance_gate(np.median(legs, axis=0), cfg.gates.stance_asym_deg)
        stance_info = {"passed": ok, "message": msg, "overridden": bool(a.allow_stance_asym and not ok)}
        print(("stance gate PASS: " if ok else "stance gate FAIL: ") + msg)
        if not ok and not a.allow_stance_asym:
            raise SystemExit("feet look staggered: Unitree stepped in runs that started like this. Re-stand the "
                             "robot from the remote and try again (or pass --allow-stance-asym to override).")

    start = st.upper_q.copy()
    beat = setup_beat_grid(cfg, a, ref.duration) if a.stage in ("playback", "full") else None
    if a.stage == "recover":
        if not a.to_stance:
            raise SystemExit("--stage recover needs --to-stance runs/stance.json")
        stance = np.array(json.loads(Path(a.to_stance).read_text())["upper_q"], float)
        plan = build_recover_plan(start, stance, cfg)
    elif a.stage == "acquire":
        plan = build_acquire_only_plan(start, cfg)
    elif a.stage == "entry":
        plan = build_entry_only_plan(ref, start, cfg)
    else:
        plan = build_plan(ref, start, cfg)

    print_settings(cfg)
    print(f"stage={a.stage} backend={backend.name} data_kind={backend.data_kind}")
    print("measured start -> opening target (deg):")
    for j, n in enumerate(UPPER_NAMES):
        print(f"   {n:28s} {D(start[j]):+8.2f} -> {D(plan.entry_target[j]):+8.2f}")
    total = sum(len(s.q) for s in plan.segments) * plan.dt
    print_load_limit(plan)
    print(f"planned duration {total:.1f}s (entry {plan.entry_duration:.1f}s)")

    player, music_info = None, None
    if a.music:
        if a.stage not in ("playback", "full"):
            raise SystemExit("--music is for --stage playback/full")
        if not live:
            raise SystemExit("--music needs the live backend (the sim does not run in real time); "
                             "use `preview` with the same options to see and hear the result")
        from ..music import MusicPlayer, plan_music
        from ..reference import VIDEO_FILE
        wav, delay, music_info = plan_music(plan, a.music_file, Path(a.assets) / VIDEO_FILE, Path(a.out) / "music",
                                            a.music_start, a.music_stretch, beat)
        player = MusicPlayer(wav, delay + a.music_offset)
        music_info["operator_offset_s"] = a.music_offset
        print(f"music: {Path(music_info['source']).name} -> {wav.name}, starts {delay + a.music_offset:+.3f}s "
              f"after playback starts")

    if live:
        if not _ask(f"Type '{a.stage}' to take upper-body control and start: ", a.stage):
            print("aborted by operator before any command was sent")
            return 1
        backend.arm(confirmed=True)

    confirm = None
    if a.stage in ("playback", "full") and not a.auto_advance:
        if live:
            confirm = lambda: _ask("Opening pose reached and gate passed. Type 'go' to play: ", "go")
        # sim: no pause

    from ..runner import OperatorStop
    op_stop = OperatorStop()
    old = signal.signal(signal.SIGINT, op_stop.press)
    try:
        result = execute(plan, backend, cfg, Monitor(cfg.gates, variant.waist_roll_pitch_actuated is not False),
                         confirm_playback=confirm,
                         waist_roll_pitch_actuated=variant.waist_roll_pitch_actuated is not False,
                         operator_stop=op_stop,
                         on_phase=player.on_phase if player else None)
    finally:
        signal.signal(signal.SIGINT, old)
        if player is not None and sys.exc_info()[0] is not None:
            player.stop()  # crashed: silence the music

    run_dir = new_run_dir(Path(a.out), f"trial_{a.stage}_{a.backend}")
    log = result.log.arrays()
    extra = {"backend": backend.name, "stage": a.stage, "variant": variant.describe(),
             "mode_machine_reported": st.mode_machine, "mode_machine_assumed": a.assume_mode_machine,
             "start_pose_rad": start.tolist(), "assets": {UPPER_FILE: sha256(Path(a.assets) / UPPER_FILE)},
             "response": run_report(log, plan, ref, backend.data_kind)}
    if a.stage in ("playback", "full"):
        extra["kinematic"] = kinematic_report(plan, ref)
        if plan.load_limit:
            extra["load_limit"] = plan.load_limit
    if live:
        extra["control_overruns"] = backend.overruns
    if music_info:
        extra["music"] = music_info
    if stance_info:
        extra["stance_gate"] = stance_info
    write_run(run_dir, result, cfg, extra)
    bar = "=" * 78
    if not result.handed_back:
        bang = "!" * 78
        print(f"{bang}\nRESULT: LINK LOST - CONTROL WAS NOT HANDED BACK\n"
              f"  {result.stop.message() if result.stop else ''}\n"
              "  The robot stopped reporting its state and did not come back in time.\n"
              "  The firmware keeps holding the LAST arm command: the arms/waist may still be\n"
              "  under arm_sdk control (frozen). Do NOT switch modes or damp while frozen.\n"
              "  Restore the cable/network, then run:\n"
              f"    .venv/bin/python -m g1dance trial --stage recover --to-stance runs/stance.json "
              f"--backend unitree --iface {a.iface} --enable-motion\n{bang}")
        print(f"run dir: {run_dir}")
        return 4
    if result.completed:
        print(f"{bar}\nRESULT: COMPLETED + handed back\n{bar}")
    else:
        print(f"{bar}\nRESULT: STOPPED (safe abort: returned to start, handed back)\n"
              f"  {result.stop.message()}\n{bar}")
    print(f"run dir: {run_dir}")
    return 0 if result.completed else 3
