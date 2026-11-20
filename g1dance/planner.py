"""Trajectory generation: acquire -> entry -> settle/gate -> playback -> return -> release.

All outputs are KINEMATIC TARGETS (what we would send as motor_cmd[i].q for
ids 12..28). They are not simulated or measured motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RunConfig
from .joints import N_UPPER, POS_LOWER, POS_UPPER, UPPER_NAMES
from .reference import UpperReference

PHASES = ("acquire", "entry", "settle", "playback", "end_hold", "return", "release")


def min_jerk(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """n samples from q0 (exclusive) to q1 (inclusive) along a quintic min-jerk path."""
    s = np.arange(1, n + 1) / n
    b = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0[None, :] + b[:, None] * (q1 - q0)[None, :]


def min_jerk_duration(delta: np.ndarray, vmax: np.ndarray, t_min: float) -> float:
    # Peak speed of a min-jerk move is 1.875 * |delta| / T.
    return float(max(t_min, np.max(1.875 * np.abs(delta) / vmax)))


def clip_limits(q: np.ndarray, margin: float) -> np.ndarray:
    return np.clip(q, POS_LOWER + margin, POS_UPPER - margin)


@dataclass
class Segment:
    name: str
    q: np.ndarray  # (K, 17) commanded targets
    weight: np.ndarray  # (K,) arm_sdk blend weight
    t_ref: np.ndarray  # (K,) reference clock [s], NaN outside playback
    ideal: np.ndarray  # (K, 17) unconstrained scaled reference, NaN outside playback


@dataclass
class LimitEvent:
    joint: str
    cause: str  # "position" | "rate"
    t_ref_start: float
    t_ref_end: float
    max_dev_rad: float

    def as_dict(self):
        return {"joint": self.joint, "cause": self.cause, "t_ref_start": round(self.t_ref_start, 3),
                "t_ref_end": round(self.t_ref_end, 3), "max_dev_deg": round(float(np.degrees(self.max_dev_rad)), 3)}


@dataclass
class Plan:
    dt: float
    start: np.ndarray  # measured (or assumed) starting pose, anchor for scales
    ref_frame0: np.ndarray  # raw reference frame 0
    entry_target_ideal: np.ndarray  # anchor + scale*(frame0 - anchor), before position clip
    entry_target: np.ndarray  # what we command at the end of entry
    segments: list[Segment]
    limit_events: list[LimitEvent] = field(default_factory=list)
    load_limit: dict | None = None  # load-limited tempo report (None when off)
    entry_duration: float = 0.0
    return_duration: float = 0.0

    def segment(self, name: str) -> Segment:
        return next(s for s in self.segments if s.name == name)

    def flat(self) -> dict[str, np.ndarray]:
        q = np.concatenate([s.q for s in self.segments])
        phase = np.concatenate([np.full(len(s.q), PHASES.index(s.name)) for s in self.segments])
        return {
            "t": np.arange(len(q)) * self.dt,
            "phase": phase,
            "q": q,
            "weight": np.concatenate([s.weight for s in self.segments]),
            "t_ref": np.concatenate([s.t_ref for s in self.segments]),
            "ideal": np.concatenate([s.ideal for s in self.segments]),
        }


def scaled(ref_q: np.ndarray, anchor: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return anchor + scales * (ref_q - anchor)


def playback_clock(cfg: RunConfig, duration: float, cap=None) -> np.ndarray:
    """Reference-time samples at each control tick, integrating the speed profile.
    ``cap(t_ref)``, if given, is an upper bound on the tempo (load-limited tempo)."""
    end = duration if cfg.speed.ref_end_s is None else min(duration, cfg.speed.ref_end_s)
    dt = cfg.timing.control_dt
    ts = [0.0]
    ramp = cfg.speed.ramp_in_s
    while ts[-1] < end - 1e-9:
        sp = cfg.speed.at(ts[-1])
        if cap is not None:
            sp = min(sp, float(cap(ts[-1])))
        if sp <= 0:
            raise ValueError("playback speed must be > 0")
        if ramp > 0:
            x = min(1.0, len(ts) * dt / ramp)
            sp *= max(x * x * (3 - 2 * x), 0.05)
        ts.append(min(end, ts[-1] + sp * dt))
    return np.array(ts)


def rate_limit(q_start: np.ndarray, targets: np.ndarray, vmax: np.ndarray, dt: float) -> np.ndarray:
    out = np.empty_like(targets)
    prev = q_start.copy()
    step = vmax * dt
    for k in range(len(targets)):
        prev = prev + np.clip(targets[k] - prev, -step, step)
        out[k] = prev
    return out


def _hold(q: np.ndarray, n: int, w0: float = 1.0, w1: float = 1.0) -> Segment:
    n = max(n, 1)
    w = np.linspace(w0, w1, n + 1)[1:] if w0 != w1 else np.full(n, w1)
    nan = np.full((n, N_UPPER), np.nan)
    return Segment("", np.repeat(q[None], n, 0), w, np.full(n, np.nan), nan)


def _named(seg: Segment, name: str) -> Segment:
    seg.name = name
    return seg


def _limit_events(t_ref, ideal, clipped, cmd, tol=np.radians(0.25)) -> list[LimitEvent]:
    events = []
    pos_dev = np.abs(ideal - clipped)
    dev = np.abs(ideal - cmd)
    for j in range(N_UPPER):
        on = dev[:, j] > tol
        if not on.any():
            continue
        idx = np.flatnonzero(on)
        splits = np.flatnonzero(np.diff(idx) > 1)
        for run in np.split(idx, splits + 1):
            cause = "position" if np.any(pos_dev[run, j] > tol) else "rate"
            events.append(LimitEvent(UPPER_NAMES[j], cause, float(t_ref[run[0]]), float(t_ref[run[-1]]),
                                     float(dev[run, j].max())))
    return events


def build_plan(ref: UpperReference, start: np.ndarray, cfg: RunConfig) -> Plan:
    start = np.asarray(start, float).copy()
    if start.shape != (N_UPPER,):
        raise ValueError(f"start pose must have {N_UPPER} entries")
    tm, dt = cfg.timing, cfg.timing.control_dt
    scales = cfg.effective_scales()
    margin = cfg.limits.pos_margin
    if cfg.speed.smooth_hz:
        from .reference import smooth_reference
        ref = smooth_reference(ref, cfg.speed.smooth_hz)
    vcap = cfg.limits.vel_vector()
    steps = lambda sec: max(1, int(round(sec / dt)))

    segs: list[Segment] = []
    # 1. acquire: ramp weight 0->1 while commanding exactly the measured pose.
    segs.append(_named(_hold(start, steps(tm.acquire_s), 0.0, 1.0), "acquire"))
    acq_hold = _hold(start, steps(tm.acquire_hold_s))
    acq_hold.name = "acquire"
    segs.append(acq_hold)

    # 2. entry: min-jerk to the (scaled, limit-clipped) frame-0 target.
    f0 = ref.q[0]
    target_ideal = scaled(f0, start, scales)
    target = clip_limits(target_ideal, margin)
    entry_v = np.full(N_UPPER, tm.entry_arm_vel)
    entry_v[:3] = tm.entry_waist_vel
    T_entry = min_jerk_duration(target - start, entry_v, tm.entry_min_s)
    n_entry = steps(T_entry)
    nan = lambda n: np.full((n, N_UPPER), np.nan)
    segs.append(Segment("entry", min_jerk(start, target, n_entry), np.ones(n_entry),
                        np.full(n_entry, np.nan), nan(n_entry)))
    # 3. settle at the exact frame-0 target (the live gate runs at the end of this).
    segs.append(_named(_hold(target, steps(tm.entry_settle_s)), "settle"))

    # 4. playback from frame 0.
    def playback(cap=None):
        t_ref = playback_clock(cfg, ref.duration, cap)
        ref_q, _ = ref.sample(t_ref)
        ideal = scaled(ref_q, start, scales)
        clipped = clip_limits(ideal, margin)
        cmd = rate_limit(target, clipped, vcap, dt)
        cmd[0] = target  # frame 0 == entry target by construction
        return t_ref, ideal, clipped, cmd

    load_info = None
    if cfg.speed.max_torque_nm or cfg.speed.max_force_n:
        t_ref, ideal, clipped, cmd, load_info = _load_limited_playback(ref, start, scales, cfg, playback)
    else:
        t_ref, ideal, clipped, cmd = playback()
    events = _limit_events(t_ref, ideal, clipped, cmd)
    segs.append(Segment("playback", cmd, np.ones(len(cmd)), t_ref, ideal))

    # 5. end hold, 6. return to the measured start, 7. release (weight 1->0).
    last = cmd[-1]
    segs.append(_named(_hold(last, steps(tm.end_hold_s)), "end_hold"))
    T_ret = min_jerk_duration(start - last, entry_v, tm.return_min_s)
    n_ret = steps(T_ret)
    segs.append(Segment("return", min_jerk(last, start, n_ret), np.ones(n_ret),
                        np.full(n_ret, np.nan), nan(n_ret)))
    rel_hold = _named(_hold(start, steps(tm.release_hold_s)), "release")
    segs.append(rel_hold)
    segs.append(_named(_hold(start, steps(tm.release_s), 1.0, 0.0), "release"))

    return Plan(dt=dt, start=start, ref_frame0=f0.copy(), entry_target_ideal=target_ideal,
                entry_target=target, segments=segs, limit_events=events, load_limit=load_info,
                entry_duration=n_entry * dt, return_duration=n_ret * dt)


def _load_limited_playback(ref, start, scales, cfg, playback, max_iter: int = 8):
    """Cap the tempo where the arm load would exceed cfg.speed.max_torque_nm /
    max_force_n, then verify on the actual playback command (which includes the
    speed profile, ramp-in, clipping and rate limits) and tighten until it holds."""
    from .load import ArmLoadModel, _spread_min, peaks, tempo_cap
    sp, dt = cfg.speed, cfg.timing.control_dt
    model = ArmLoadModel()
    tg = np.arange(ref.n) / ref.fps
    qg = scaled(ref.q, start, scales)
    tau1, F1 = model.loads(qg, tg, ref.qd * scales)
    cap = tempo_cap(tg, tau1, F1, sp.max_torque_nm, sp.max_force_n, sp.load_window_s)
    if sp.constant_tempo:
        cap = np.full(len(tg), float(cap.min()))

    def ratio(tau, F):
        r = np.zeros(len(tau))
        if sp.max_torque_nm:
            r = np.maximum(r, np.abs(tau[:, :2]).max(1) / sp.max_torque_nm)
        if sp.max_force_n:
            r = np.maximum(r, np.hypot(F[:, 0], F[:, 1]) / sp.max_force_n)
        return r

    for it in range(1, max_iter + 1):
        out = playback(lambda tr: np.interp(tr, tg, cap))
        t_ref, cmd = out[0], out[3]
        tau, F = model.loads(cmd, np.arange(len(cmd)) * dt)
        r = ratio(tau, F)
        if r.max() <= 1.0 or it == max_iter:
            break
        # tighten the cap where the verified load is over the limit
        fac = np.ones(len(tg))
        for k in np.where(r > 1.0)[0]:
            j = int(np.clip(np.searchsorted(tg, t_ref[k]), 0, len(tg) - 1))
            fac[j] = min(fac[j], 0.97 / np.sqrt(r[k]))
        if sp.constant_tempo:
            cap = cap * float(fac.min())
        else:
            cap = np.minimum(cap, _spread_min(tg, np.minimum(cap * fac, 1.0), sp.load_window_s))
    p = peaks(tau, F)
    pk_tau, pk_f = p.pop("_i_tau"), p.pop("_i_force")
    p1 = peaks(*model.loads(qg, tg, ref.qd * scales))
    p1.pop("_i_tau"), p1.pop("_i_force")
    slowed = cap < 0.99
    info = {
        "data_kind": "kinematic model estimate (g1dance/load.py)",
        "max_torque_nm": sp.max_torque_nm, "max_force_n": sp.max_force_n,
        "load_window_s": sp.load_window_s, "iterations": it, "within_limits": bool(r.max() <= 1.0 + 1e-6),
        "peak_load_planned": {k: round(v, 2) for k, v in p.items()},
        "peak_load_at": {"torque_dance_t": round(float(t_ref[pk_tau]), 2), "force_dance_t": round(float(t_ref[pk_f]), 2)},
        "peak_load_recorded_tempo": {k: round(v, 2) for k, v in p1.items()},
        "playback_s": round(len(cmd) * dt, 2), "recorded_s": round(ref.duration, 2),
        "min_tempo": round(float(cap.min()), 3), "min_tempo_dance_t": round(float(tg[int(cap.argmin())]), 2),
        "slowed_fraction_of_dance": round(float(slowed.mean()), 3),
        "slowed_passages": _passages(tg, cap),
        "constant_tempo": bool(sp.constant_tempo),
    }
    if sp.constant_tempo:
        s = float(min(cap.min(), sp.speed))
        # music runs at tempo s; it starts this long after playback starts (ramp-in lag)
        info["music_tempo"] = round(s, 4)
        info["music_delay_s"] = round(float((len(cmd) - 1) * dt - t_ref[-1] / s), 3)
    return (*out, info)


def _passages(t: np.ndarray, cap: np.ndarray, thr: float = 0.99) -> list[list[float]]:
    on = cap < thr
    out, k = [], 0
    while k < len(t):
        if on[k]:
            j = k
            while j + 1 < len(t) and on[j + 1]:
                j += 1
            out.append([round(float(t[k]), 2), round(float(t[j]), 2), round(float(cap[k:j + 1].min()), 2)])
            k = j + 1
        else:
            k += 1
    return out


def build_entry_only_plan(ref: UpperReference, start: np.ndarray, cfg: RunConfig) -> Plan:
    """Staged trial: acquire -> entry -> settle -> return -> release (no playback)."""
    full = build_plan(ref, start, cfg)
    tm, dt = cfg.timing, cfg.timing.control_dt
    entry_v = np.full(N_UPPER, tm.entry_arm_vel)
    entry_v[:3] = tm.entry_waist_vel
    T_ret = min_jerk_duration(start - full.entry_target, entry_v, tm.return_min_s)
    n = max(1, int(round(T_ret / dt)))
    keep = [s for s in full.segments if s.name in ("acquire", "entry", "settle")]
    ret = Segment("return", min_jerk(full.entry_target, np.asarray(start, float), n), np.ones(n),
                  np.full(n, np.nan), np.full((n, N_UPPER), np.nan))
    rel = [s for s in full.segments if s.name == "release"]
    full.segments = keep + [ret] + rel
    full.limit_events = []
    full.return_duration = n * dt
    return full


def build_acquire_only_plan(start: np.ndarray, cfg: RunConfig, hold_s: float = 3.0) -> Plan:
    """Staged trial: take control and hold the measured pose, then hand back."""
    tm, dt = cfg.timing, cfg.timing.control_dt
    start = np.asarray(start, float)
    steps = lambda sec: max(1, int(round(sec / dt)))
    segs = [_named(_hold(start, steps(tm.acquire_s), 0.0, 1.0), "acquire"),
            _named(_hold(start, steps(hold_s)), "acquire"),
            _named(_hold(start, steps(tm.release_s), 1.0, 0.0), "release")]
    return Plan(dt=dt, start=start, ref_frame0=np.full(N_UPPER, np.nan),
                entry_target_ideal=start, entry_target=start, segments=segs)


def build_recover_plan(current: np.ndarray, stance: np.ndarray, cfg: RunConfig) -> Plan:
    """Recovery after a crash left arm_sdk holding its last command.

    The firmware keeps the last arm_sdk command at weight 1 when our process
    dies (observed 2026-11-20). So recovery must start AT WEIGHT 1 on the
    currently measured pose: starting the weight near 0 would hand the arm to
    Unitree in one step and snap it toward Unitree's pose. Then move slowly
    to the saved stance and only then ramp the weight down.
    """
    tm, dt = cfg.timing, cfg.timing.control_dt
    current, stance = np.asarray(current, float), np.asarray(stance, float)
    steps = lambda sec: max(1, int(round(sec / dt)))
    v = np.full(N_UPPER, tm.entry_arm_vel)
    v[:3] = tm.entry_waist_vel
    T = min_jerk_duration(stance - current, v, tm.return_min_s)
    n = steps(T)
    nan = lambda k: np.full((k, N_UPPER), np.nan)
    segs = [_named(_hold(current, steps(1.0)), "acquire"),  # weight 1 from the first tick
            Segment("return", min_jerk(current, stance, n), np.ones(n), np.full(n, np.nan), nan(n)),
            _named(_hold(stance, steps(tm.release_hold_s)), "release"),
            _named(_hold(stance, steps(tm.release_s), 1.0, 0.0), "release")]
    # start = stance: an abort during recovery must also head DOWN to the stance.
    return Plan(dt=dt, start=stance, ref_frame0=np.full(N_UPPER, np.nan), entry_target_ideal=stance,
                entry_target=stance, segments=segs, return_duration=n * dt)
