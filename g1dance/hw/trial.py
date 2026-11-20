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
    from ..cli import print_settings, resolve_config
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

    start = st.upper_q.copy()
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
    print(f"planned duration {total:.1f}s (entry {plan.entry_duration:.1f}s)")

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

    stop_flag = {"v": False}
    old = signal.signal(signal.SIGINT, lambda *_: stop_flag.__setitem__("v", True))
    try:
        result = execute(plan, backend, cfg, Monitor(cfg.gates, variant.waist_roll_pitch_actuated is not False),
                         confirm_playback=confirm,
                         waist_roll_pitch_actuated=variant.waist_roll_pitch_actuated is not False,
                         operator_stop=lambda: stop_flag["v"])
    finally:
        signal.signal(signal.SIGINT, old)

    run_dir = new_run_dir(Path(a.out), f"trial_{a.stage}_{a.backend}")
    log = result.log.arrays()
    extra = {"backend": backend.name, "stage": a.stage, "variant": variant.describe(),
             "mode_machine_reported": st.mode_machine, "mode_machine_assumed": a.assume_mode_machine,
             "start_pose_rad": start.tolist(), "assets": {UPPER_FILE: sha256(Path(a.assets) / UPPER_FILE)},
             "response": run_report(log, plan, ref, backend.data_kind)}
    if a.stage in ("playback", "full"):
        extra["kinematic"] = kinematic_report(plan, ref)
    if live:
        extra["control_overruns"] = backend.overruns
    write_run(run_dir, result, cfg, extra)
    bar = "=" * 78
    if result.completed:
        print(f"{bar}\nRESULT: COMPLETED + handed back\n{bar}")
    else:
        print(f"{bar}\nRESULT: STOPPED (safe abort: returned to start, handed back)\n"
              f"  {result.stop.message()}\n{bar}")
    print(f"run dir: {run_dir}")
    return 0 if result.completed else 3
