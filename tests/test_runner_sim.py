"""End-to-end runner against the MuJoCo backend (same code path as hardware)."""
import numpy as np
import pytest

from g1dance.config import RunConfig, load_config
from g1dance.monitor import Monitor
from g1dance.planner import PHASES, build_entry_only_plan, build_plan
from tests.conftest import ROOT, needs_assets, needs_model

pytestmark = [needs_assets, needs_model]


def run(ref, cfg, entry_only=False, operator_stop=None):
    from g1dance.runner import execute
    from g1dance.sim import SimBackend, assumed_start_upper
    b = SimBackend(assumed_start_upper(), cfg.timing.control_dt)
    start = b.read_state().upper_q
    p = (build_entry_only_plan if entry_only else build_plan)(ref, start, cfg)
    return p, execute(p, b, cfg, Monitor(cfg.gates), operator_stop=operator_stop), start


def test_entry_stage_completes_and_hands_back(ref):
    cfg = load_config(ROOT / "configs" / "default.toml")
    p, r, start = run(ref, cfg, entry_only=True)
    assert r.completed and r.gate["passed"]
    log = r.log.arrays()
    assert log["weight"][-1] == 0.0
    assert np.allclose(log["q_cmd"][-1], start)


def test_sdk_example_waist_gain_stops_on_waist_pitch_tracking(ref):
    cfg = load_config(ROOT / "configs" / "sdk_example_gains.toml")
    p, r, start = run(ref, cfg, entry_only=True)
    assert not r.completed
    assert r.stop.check == "tracking_error" and r.stop.subject == "waist_pitch_joint"
    assert r.stop.phase == "entry" and r.stop.threshold == cfg.gates.track_waist_pitch
    log = r.log.arrays()
    assert log["weight"][-1] == 0.0  # abort still returns and hands back
    assert np.allclose(log["q_cmd"][-1], start, atol=1e-9)


def test_operator_stop_mid_playback_returns_and_releases(ref):
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ref_end_s = 3.0
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 600  # ~12 s in: during playback

    p, r, start = run(ref, cfg, operator_stop=stop)
    assert not r.completed and r.stop.check == "operator"
    log = r.log.arrays()
    assert log["weight"][-1] == 0.0 and np.allclose(log["q_cmd"][-1], start)


def test_dq_feedforward_reduces_waist_lag_in_sim(ref):
    errs = []
    for ff in (False, True):
        cfg = load_config(ROOT / "configs" / "default.toml")
        cfg.gains.dq_feedforward = ff
        cfg.speed.ref_end_s = 5.0
        p, r, start = run(ref, cfg)
        a = r.log.arrays()
        m = a["phase"] == PHASES.index("playback")
        errs.append(np.abs(a["q_meas"][m, 14] - a["q_cmd"][m, 2]).max())
        assert r.completed
    assert errs[1] < errs[0]


def test_leg_step_stops_holds_until_quiet_then_returns_slowly(ref):
    """Inject a Unitree 'step' (fast leg joint) into the sim state mid-playback."""
    from g1dance.runner import execute
    from g1dance.sim import SimBackend, assumed_start_upper
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ref_end_s = 4.0
    b = SimBackend(assumed_start_upper(), cfg.timing.control_dt)
    real_read, n = b.read_state, {"k": 0, "first": None}

    def read_state():
        st = real_read()
        n["k"] += 1
        if 500 <= n["k"] < 560:  # ~1.2 s of stepping during playback
            st.dq = st.dq.copy()
            st.dq[9] = 4.0
        return st
    b.read_state = read_state
    start = real_read().upper_q
    p = build_plan(ref, start, cfg)
    r = execute(p, b, cfg, Monitor(cfg.gates))
    assert not r.completed and r.stop.check == "leg_step" and r.stop.subject == "right_knee_joint"
    assert r.stop.phase == "playback"
    acts = [e["action"] for e in r.log.events if e["kind"] == "abort"]
    assert any("leg step: holding" in a for a in acts) and any("legs quiet after" in a for a in acts)
    assert any("slowed" in a for a in acts)
    log = r.log.arrays()
    assert log["weight"][-1] == 0.0 and np.allclose(log["q_cmd"][-1], start)


def _stale_backend(cfg, window):
    """Sim backend whose rt/lowstate goes stale (frozen stamp) for ticks in `window`."""
    from g1dance.sim import SimBackend, assumed_start_upper
    b = SimBackend(assumed_start_upper(), cfg.timing.control_dt)
    real, n, frozen = b.read_state, {"k": 0}, {}

    def read_state():
        n["k"] += 1
        st = real()
        if n["k"] in window:
            if "st" not in frozen:
                frozen["st"] = st
            return frozen["st"]  # old message: stamp stops advancing
        frozen.pop("st", None)
        return st
    b.read_state = read_state
    return b, real().upper_q


def test_link_lost_then_restored_returns_and_hands_back(ref):
    from g1dance.runner import execute
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ref_end_s = 3.0
    b, start = _stale_backend(cfg, range(500, 600))  # ~2 s outage during playback
    r = execute(build_plan(ref, start, cfg), b, cfg, Monitor(cfg.gates))
    assert not r.completed and r.stop.check == "state_freshness" and r.handed_back
    kinds = [e["kind"] for e in r.log.events]
    assert "link_wait" in kinds and "link_restored" in kinds and kinds[-1] == "released"
    log = r.log.arrays()
    lw = log["phase"] == PHASES.index("link_wait")
    assert lw.any() and np.all(log["weight"][lw] == 1.0)  # kept holding, did not release into the void
    assert np.ptp(log["q_cmd"][lw], axis=0).max() == 0.0  # held the last command
    assert log["weight"][-1] == 0.0 and np.allclose(log["q_cmd"][-1], start)


def test_link_lost_for_good_reports_not_handed_back(ref):
    from g1dance.runner import execute
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ref_end_s = 3.0
    cfg.timing.link_wait_s = 2.0
    b, start = _stale_backend(cfg, range(500, 10 ** 9))
    sent = []
    real_send = b.send
    b.send = lambda *a, **k: (sent.append(a[3]), real_send(*a, **k))
    r = execute(build_plan(ref, start, cfg), b, cfg, Monitor(cfg.gates))
    assert not r.completed and not r.handed_back
    assert r.log.events[-1]["kind"] == "link_lost"
    assert sent[-1] == 1.0  # never ramped the weight down into a dead link


def test_link_lost_during_handback_phase_waits_and_finishes_handback():
    from g1dance.planner import build_acquire_only_plan
    from g1dance.runner import execute
    from g1dance.sim import assumed_start_upper
    cfg = load_config(ROOT / "configs" / "default.toml")
    p = build_acquire_only_plan(assumed_start_upper(), cfg)
    k = sum(len(s.q) for s in p.segments if s.name != "release") + 30  # inside the weight ramp-down
    b, start = _stale_backend(cfg, range(k, k + 40))
    r = execute(build_acquire_only_plan(start, cfg), b, cfg, Monitor(cfg.gates))
    assert r.handed_back and r.log.events[-1]["kind"] == "released"
    assert r.log.arrays()["weight"][-1] == 0.0


def test_link_lost_inside_abort_release_ramp_waits(ref):
    """Stale state during the abort's own release ramp (the cable-pull case, FINDINGS 4v)."""
    from g1dance.planner import PHASES
    from g1dance.runner import execute
    from g1dance.sim import SimBackend, assumed_start_upper
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ref_end_s = 3.0
    cfg.timing.link_wait_s = 1.0

    def make_stop():
        c = {"n": 0}

        def stop():
            c["n"] += 1
            return c["n"] > 450  # during playback
        return stop
    start = SimBackend(assumed_start_upper(), cfg.timing.control_dt).read_state().upper_q
    p = build_plan(ref, start, cfg)
    # dry run without outage: find where the abort's release ramp starts (read_state calls ~ log ticks)
    dry = execute(p, SimBackend(assumed_start_upper(), cfg.timing.control_dt), cfg, Monitor(cfg.gates),
                  operator_stop=make_stop())
    ph = dry.log.arrays()["phase"]
    k_rel = int(np.flatnonzero(ph == PHASES.index("release"))[0])
    b, _ = _stale_backend(cfg, range(k_rel + 15, 10 ** 9))
    r = execute(p, b, cfg, Monitor(cfg.gates), operator_stop=make_stop())
    stops = [e for e in r.log.events if e["kind"] == "stop"]
    assert r.stop.check == "operator" and stops[-1]["phase"] == "release"
    assert not r.handed_back and r.log.events[-1]["kind"] == "link_lost"
    assert "released" not in [e["kind"] for e in r.log.events]


def test_operator_stop_counter_only_new_press_gives_up():
    from g1dance.runner import OperatorStop
    s = OperatorStop()
    assert not s() and s.presses() == 0
    s.press()
    assert s() and s.presses() == 1
