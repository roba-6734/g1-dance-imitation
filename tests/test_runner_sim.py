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
