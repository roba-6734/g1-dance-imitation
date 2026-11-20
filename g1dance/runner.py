"""Closed-loop executor shared by the simulator and the real robot.

The exact same code path (gate, monitor, abort, logging) runs against
``SimBackend`` offline and ``UnitreeBackend`` on hardware.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import RunConfig
from .joints import N_UPPER
from .monitor import Monitor, RobotState, Stop, entry_gate
from .planner import PHASES, Plan, min_jerk, min_jerk_duration


class Backend(Protocol):
    name: str
    data_kind: str  # "simulated" | "physical"

    def read_state(self) -> RobotState: ...
    def send(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray, weight: float,
             dq: np.ndarray | None = None) -> None: ...
    def wait_tick(self) -> None: ...
    def now(self) -> float: ...


@dataclass
class RunLog:
    data_kind: str
    t: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    t_ref: list = field(default_factory=list)
    q_cmd: list = field(default_factory=list)
    weight: list = field(default_factory=list)
    q_meas: list = field(default_factory=list)  # (29,)
    dq_meas: list = field(default_factory=list)
    imu_rpy: list = field(default_factory=list)
    torso_rpy: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def event(self, kind: str, **kw):
        e = {"kind": kind, "t": round(self.t[-1], 4) if self.t else 0.0, **kw}
        self.events.append(e)
        print(f"[{e['t']:7.2f}s] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k != "detail"))

    def record(self, t, phase, t_ref, q_cmd, weight, st: RobotState):
        self.t.append(t)
        self.phase.append(PHASES.index(phase) if phase in PHASES else -1)
        self.t_ref.append(np.nan if t_ref is None else t_ref)
        self.q_cmd.append(np.array(q_cmd))
        self.weight.append(weight)
        self.q_meas.append(np.array(st.q))
        self.dq_meas.append(np.array(st.dq))
        self.imu_rpy.append(np.array(st.imu_rpy))
        self.torso_rpy.append(np.full(3, np.nan) if st.torso_rpy is None else np.array(st.torso_rpy))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(getattr(self, k)) for k in
                ("t", "phase", "t_ref", "q_cmd", "weight", "q_meas", "dq_meas", "imu_rpy", "torso_rpy")}

    def save(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_dir / "log.npz", data_kind=self.data_kind, phases=np.array(PHASES), **self.arrays())
        with open(run_dir / "events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=float) + "\n")


class OperatorStop:
    """Ctrl-C counter: truthy once pressed; ``presses()`` lets a wait notice a NEW press."""

    def __init__(self):
        self.n = 0

    def press(self, *_):
        self.n += 1

    def presses(self) -> int:
        return self.n

    def __call__(self) -> bool:
        return self.n > 0


@dataclass
class RunResult:
    completed: bool
    stop: Stop | None
    log: RunLog
    gate: dict | None
    handed_back: bool = True  # False: link lost, the robot may still be under arm control


def execute(plan: Plan, backend: Backend, cfg: RunConfig, monitor: Monitor,
            confirm_playback: Callable[[], bool] | None = None,
            waist_roll_pitch_actuated: bool = True,
            operator_stop: Callable[[], bool] | None = None,
            on_phase: Callable[[str], None] | None = None) -> RunResult:
    """``on_phase(name)`` is called when each phase starts and with "abort" when a
    stop is handled (used to start/stop music; it must not block)."""
    kp, kd = cfg.gains.vectors()
    ff = _Feedforward(cfg)
    log = RunLog(backend.data_kind)
    t0 = backend.now()
    gate_info = None
    last_cmd = plan.start.copy()

    def tick(q, w, phase, t_ref=None, tracking=True) -> list[Stop]:
        nonlocal last_cmd
        st = backend.read_state()
        now = backend.now()
        t = now - t0
        log.record(t, phase, t_ref, q, w, st)
        stops = monitor.check(st, q, w, phase, t, now, t_ref, tracking=tracking)
        if operator_stop is not None and operator_stop():
            stops = [Stop("operator", "Ctrl-C", 0.0, 0.0, "-", phase, t, t_ref)] + stops
        if stops:
            return stops
        backend.send(q, kp, kd, w, ff(q))
        last_cmd = np.array(q)
        backend.wait_tick()
        return []

    first = backend.read_state()
    monitor.capture_reference_attitude(first)
    log.event("start", backend=backend.name, data_kind=backend.data_kind,
              mode_machine=first.mode_machine)

    def _abort_notify(*args, **kw):
        if on_phase is not None:
            on_phase("abort")
        return _abort(*args, operator_stop=operator_stop, **kw)

    names = [s.name for s in plan.segments]
    for si, seg in enumerate(plan.segments):
        log.event("phase", name=seg.name, ticks=len(seg.q))
        if on_phase is not None:
            on_phase(seg.name)
        for k in range(len(seg.q)):
            tr = None if np.isnan(seg.t_ref[k]) else float(seg.t_ref[k])
            stops = tick(seg.q[k], float(seg.weight[k]), seg.name, tr)
            if stops:
                return _abort_notify(stops, plan, backend, cfg, monitor, log, kp, kd, t0, gate_info)
        last_settle = seg.name == "settle" and (si + 1 == len(names) or names[si + 1] != "settle")
        if last_settle:
            # Gate: commanded AND measured pose must be at frame 0 before anything else.
            ok, gate_info, stops = _gate(plan, backend, cfg, tick, waist_roll_pitch_actuated, log)
            if stops:
                return _abort_notify(stops, plan, backend, cfg, monitor, log, kp, kd, t0, gate_info)
            if not ok:
                stop = Stop("entry_gate", "; ".join(gate_info["failures"])[:300], 0.0, 0.0, "-",
                            "settle", log.t[-1])
                return _abort_notify([stop], plan, backend, cfg, monitor, log, kp, kd, t0, gate_info)
            if confirm_playback is not None and "playback" in names[si + 1:]:
                ok2, stops = _hold_until(confirm_playback, plan.entry_target, tick)
                if stops:
                    return _abort_notify(stops, plan, backend, cfg, monitor, log, kp, kd, t0, gate_info)
                if not ok2:
                    log.event("operator_declined_playback")
                    return _abort_notify([Stop("operator", "declined playback", 0, 0, "-", "settle", log.t[-1])],
                                  plan, backend, cfg, monitor, log, kp, kd, t0, gate_info)
    log.event("done", handed_back=True)
    return RunResult(True, None, log, gate_info)


class _Feedforward:
    """Commanded velocity for motor_cmd.dq: finite difference of successive
    commands, clipped to the command-rate caps. Returns None when disabled."""

    def __init__(self, cfg: RunConfig):
        self.on = bool(cfg.gains.dq_feedforward)
        self.dt = cfg.timing.control_dt
        self.cap = cfg.limits.vel_vector()
        self.prev: np.ndarray | None = None

    def reset(self, q):
        self.prev = np.array(q, float)

    def __call__(self, q):
        q = np.array(q, float)
        if not self.on:
            return None
        v = np.zeros_like(q) if self.prev is None else np.clip((q - self.prev) / self.dt, -self.cap, self.cap)
        self.prev = q
        return v


def _gate(plan, backend, cfg, tick, wrp, log):
    """Hold the frame-0 target until measured converges (or timeout)."""
    deadline_ticks = int(round(cfg.timing.gate_timeout_s / cfg.timing.control_dt))
    res = None
    for i in range(deadline_ticks + 1):
        st = backend.read_state()
        # The command actually sent last must equal the target, and measured must be close.
        res_cmd = entry_gate(np.array(log.q_cmd[-1]), plan.entry_target, st.upper_q, cfg.gates, wrp)
        if res_cmd.passed:
            log.event("entry_gate", passed=True, waited_s=round(i * cfg.timing.control_dt, 3),
                      cmd_max_err_rad=res_cmd.cmd_max_err, detail=res_cmd.as_dict())
            return True, res_cmd.as_dict(), []
        res = res_cmd
        stops = tick(plan.entry_target, 1.0, "settle")
        if stops:
            return False, res.as_dict(), stops
    log.event("entry_gate", passed=False, failures=res.failures, detail=res.as_dict())
    return False, res.as_dict(), []


def _hold_until(confirm, q, tick):
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("ok", bool(confirm())), daemon=True)
    th.start()
    while th.is_alive():
        stops = tick(q, 1.0, "settle")
        if stops:
            return False, stops
    return box.get("ok", False), []


def _abort(stops, plan, backend, cfg, monitor, log, kp, kd, t0, gate_info, operator_stop=None) -> RunResult:
    """Hold where the robot is, return slowly to the start pose, release.

    During the abort return ONLY state freshness and joint speed are checked
    (tracking/waist/IMU gates would re-trigger on the very fault being
    handled). This is logged explicitly. A joint-speed fault during the return
    releases immediately. A stale state (link loss) at any point holds the last
    command and waits for the link (``_await_link``); the result says whether
    control was actually handed back.
    """
    for s in stops:
        log.event("stop", **s.as_dict())
    first = stops[0]
    ctx = (plan, backend, cfg, monitor, log, kp, kd, t0, operator_stop)
    if first.check == "state_freshness":
        if not _await_link(backend, cfg, log, kp, kd, t0, np.array(log.q_cmd[-1]), float(log.weight[-1]),
                           operator_stop):
            return RunResult(False, first, log, gate_info, handed_back=False)
    log.event("abort", action="hold measured pose, return to start (tracking/waist/IMU gates off; "
                              "freshness+speed gates on), then release")
    slowdown, pre_hold = 1.0, True
    if any(s.check == "leg_step" for s in stops):
        r = _hold_while_stepping(backend.read_state().upper_q.copy(), *ctx)
        if r is not None:
            return RunResult(False, first, log, gate_info, handed_back=(r == "released"))
        slowdown, pre_hold = cfg.timing.step_return_slowdown, False
    ok = _return_and_release(*ctx, slowdown=slowdown, pre_hold=pre_hold)
    return RunResult(False, first, log, gate_info, handed_back=ok)


def _return_and_release(plan, backend, cfg, monitor, log, kp, kd, t0, operator_stop,
                        slowdown=1.0, pre_hold=True) -> bool:
    """Min-jerk from the MEASURED pose to the start, then release. True if handed back."""
    dt = cfg.timing.control_dt
    q_here = backend.read_state().upper_q.copy()
    v = np.full(N_UPPER, cfg.timing.entry_arm_vel)
    v[:3] = cfg.timing.entry_waist_vel
    T = min_jerk_duration(plan.start - q_here, v, cfg.timing.return_min_s) * slowdown
    if slowdown != 1.0:
        log.event("abort", action=f"returning to start over {T:.1f}s (slowed x{slowdown})")
    path = min_jerk(q_here, plan.start, max(1, int(round(T / dt))))
    if pre_hold:
        path = np.concatenate([np.repeat(q_here[None], int(0.5 / dt), 0), path])
    for q in path:
        s2 = backend.read_state()
        now = backend.now()
        log.record(now - t0, "return", None, q, 1.0, s2)
        sec = [x for x in monitor.check(s2, q, 1.0, "return", now - t0, now, tracking=False)
               if x.check in ("state_freshness", "joint_speed")]
        for s in sec:
            log.event("stop", **s.as_dict())
        if any(x.check == "state_freshness" for x in sec):
            if not _await_link(backend, cfg, log, kp, kd, t0, q, 1.0, operator_stop):
                return False
            return _return_and_release(plan, backend, cfg, monitor, log, kp, kd, t0, operator_stop, slowdown)
        if sec:
            log.event("abort", action="second fault: immediate release")
            return _release(backend, log, s2.upper_q.copy(), cfg, kp, kd, t0, operator_stop)
        backend.send(q, kp, kd, 1.0)
        backend.wait_tick()
    return _release(backend, log, plan.start, cfg, kp, kd, t0, operator_stop)


def _await_link(backend, cfg, log, kp, kd, t0, q_hold, w_hold, operator_stop=None) -> bool:
    """rt/lowstate went stale. Keep publishing the last command (harmless if the
    link is down; keeps the robot held where it is if it comes back) and wait for
    the state to be fresh again for link_restore_s. Returns True when restored.
    On timeout or operator stop, stops publishing and returns False: the firmware
    keeps holding the last command, so the arms stay under arm_sdk control."""
    tm = cfg.timing
    dt = tm.control_dt
    need = max(1, int(round(tm.link_restore_s / dt)))
    max_ticks = max(1, int(round(tm.link_wait_s / dt)))
    log.event("link_wait", action=f"robot state stale: holding last command (weight {w_hold:.2f}) and waiting up to "
                                  f"{tm.link_wait_s:.0f}s for the link; Ctrl-C gives up")
    presses = getattr(operator_stop, "presses", None)
    p0 = presses() if presses else (bool(operator_stop()) if operator_stop else False)
    fresh = 0
    for i in range(max_ticks):
        st = backend.read_state()
        now = backend.now()
        log.record(now - t0, "link_wait", None, q_hold, w_hold, st)
        fresh = fresh + 1 if (now - st.stamp) <= cfg.gates.state_timeout_s else 0
        if fresh >= need:
            log.event("link_restored", after_s=round((i + 1) * dt, 2))
            return True
        if operator_stop is not None:
            # only a Ctrl-C pressed AFTER waiting began gives up (the one that started an abort is already counted)
            if (presses() > p0) if presses else (operator_stop() and not p0):
                log.event("link_wait", action="operator gave up waiting")
                break
        if i and i % int(round(5.0 / dt)) == 0:
            log.event("link_wait", action=f"still no robot state after {i * dt:.0f}s")
        try:
            backend.send(q_hold, kp, kd, w_hold)
        except Exception as e:  # pragma: no cover - publisher errors while the link is down
            log.event("link_wait", action=f"send failed: {e}")
        backend.wait_tick()
    log.event("link_lost", handed_back=False,
              action="gave up waiting; publishing stopped. The firmware holds the last command: "
                     "the robot may still be under arm control. Run the recover stage.")
    return False


def _hold_while_stepping(q_hold, plan, backend, cfg, monitor, log, kp, kd, t0, operator_stop):
    """Hold the arms still while Unitree's balance controller finishes stepping.
    Returns None when the legs are quiet (or the hold times out), "released" if a
    joint-speed fault forced an immediate release, or "lost" if the link was lost."""
    dt = cfg.timing.control_dt
    quiet_need = max(1, int(round(cfg.timing.step_quiet_s / dt)))
    max_ticks = max(1, int(round(cfg.timing.step_hold_max_s / dt)))
    log.event("abort", action=f"leg step: holding measured pose until legs quiet for {cfg.timing.step_quiet_s}s "
                              f"(max {cfg.timing.step_hold_max_s}s)")
    quiet = 0
    for i in range(max_ticks):
        s2 = backend.read_state()
        now = backend.now()
        log.record(now - t0, "return", None, q_hold, 1.0, s2)
        sec = [x for x in monitor.check(s2, q_hold, 1.0, "return", now - t0, now, tracking=False)
               if x.check in ("state_freshness", "joint_speed")]
        for s in sec:
            log.event("stop", **s.as_dict())
        if any(x.check == "state_freshness" for x in sec):
            if not _await_link(backend, cfg, log, kp, kd, t0, q_hold, 1.0, operator_stop):
                return "lost"
            continue
        if sec:
            log.event("abort", action="second fault: immediate release")
            _release(backend, log, s2.upper_q.copy(), cfg, kp, kd, t0, operator_stop)
            return "released"
        v, _ = monitor.leg_speed(s2)
        quiet = quiet + 1 if abs(v) < 0.5 * cfg.gates.leg_step_dq else 0
        backend.send(q_hold, kp, kd, 1.0)
        backend.wait_tick()
        if quiet >= quiet_need:
            log.event("abort", action=f"legs quiet after {(i + 1) * dt:.2f}s")
            return None
    log.event("abort", action=f"legs not quiet after {cfg.timing.step_hold_max_s}s: returning anyway")
    return None


def _release(backend, log, q, cfg, kp, kd, t0, operator_stop=None) -> bool:
    """Ramp the weight 1 -> 0 commanding q. Waits for the link if the state goes
    stale. True if the full ramp was sent with a live link."""
    n = max(1, int(round(cfg.timing.release_s / cfg.timing.control_dt)))
    for w in np.linspace(1.0, 0.0, n + 1)[1:]:
        st = backend.read_state()
        now = backend.now()
        if now - st.stamp > cfg.gates.state_timeout_s:
            log.event("stop", check="state_freshness", subject="rt/lowstate age", value=float(now - st.stamp),
                      threshold=cfg.gates.state_timeout_s, unit="s", phase="release")
            if not _await_link(backend, cfg, log, kp, kd, t0, q, float(w), operator_stop):
                return False
        log.record(backend.now() - t0, "release", None, q, float(w), st)
        backend.send(q, kp, kd, float(w))
        backend.wait_tick()
    log.event("released", handed_back=True)
    return True


def write_run(run_dir: Path, result: RunResult, cfg: RunConfig, extra: dict):
    result.log.save(run_dir)
    summary = {"completed": result.completed, "handed_back": result.handed_back, "data_kind": result.log.data_kind,
               "stop": None if result.stop is None else result.stop.as_dict(),
               "entry_gate": result.gate, **extra}
    with open(run_dir / "settings.json", "w") as f:
        json.dump({"config": cfg.to_dict(), **extra}, f, indent=2, default=float)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


def new_run_dir(root: Path, tag: str) -> Path:
    return root / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
