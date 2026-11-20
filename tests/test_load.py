"""Load-limited tempo: arm-induced load model and planner integration."""
import numpy as np

from g1dance.config import load_config
from g1dance.planner import build_plan
from tests.conftest import ROOT, needs_assets, needs_model

pytestmark = [needs_assets, needs_model]


def _start():
    from g1dance.sim import assumed_start_upper
    return assumed_start_upper()


def test_load_scales_with_tempo_squared(ref):
    from g1dance.load import ArmLoadModel
    m = ArmLoadModel()
    q = ref.q[200:400]
    t1 = np.arange(len(q)) / ref.fps
    tau1, F1 = m.loads(q, t1)
    tau2, F2 = m.loads(q, 2 * t1)  # same motion at half tempo
    r_tau = np.abs(tau2[10:-10, :2]).max() / np.abs(tau1[10:-10, :2]).max()
    r_f = np.hypot(*F2[10:-10].T).max() / np.hypot(*F1[10:-10].T).max()
    assert abs(r_tau - 0.25) < 0.02 and abs(r_f - 0.25) < 0.02


def test_off_by_default(ref):
    cfg = load_config(ROOT / "configs" / "default.toml")
    assert cfg.speed.max_torque_nm is None and cfg.speed.max_force_n is None
    assert build_plan(ref, _start(), cfg).load_limit is None


def test_load_limited_plan_within_limits_full_dance_amplitude_unchanged(ref):
    from g1dance.load import ArmLoadModel, peaks
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ramp_in_s = 0.3
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1
    p = build_plan(ref, _start(), cfg)
    L = p.load_limit
    assert L["within_limits"]
    pb = p.segment("playback")
    # independent re-check of the load on the playback command actually planned
    tau, F = ArmLoadModel().loads(pb.q, np.arange(len(pb.q)) * p.dt)
    pk = peaks(tau, F)
    assert max(pk["pitch_nm"], pk["roll_nm"]) <= 1.2 + 1e-6 and pk["force_n"] <= 4.1 + 1e-6
    # whole dance, same targets (time warp only), between 1.0x and 0.5x in length
    assert np.all(np.diff(pb.t_ref) > 0) and abs(pb.t_ref[-1] - ref.duration) < 1e-9
    q_ref, _ = ref.sample(pb.t_ref)
    s = cfg.effective_scales()
    assert np.allclose(pb.ideal, p.start + s * (q_ref - p.start))
    assert ref.duration < len(pb.q) * p.dt < 2 * ref.duration
    assert L["slowed_passages"] and L["min_tempo"] < 1.0


def test_load_limit_respects_explicit_slower_speed(ref):
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.speed = 0.4  # already below what the cap needs: cap must not speed it up
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1
    p = build_plan(ref, _start(), cfg)
    n = len(p.segment("playback").q)
    assert n * p.dt >= ref.duration / 0.4 - 0.05
