import numpy as np
import pytest

from g1dance.config import RunConfig
from g1dance.joints import ARM_COLS, POS_LOWER, POS_UPPER, WAIST_PITCH, WAIST_ROLL, WAIST_YAW
from g1dance.planner import build_entry_only_plan, build_plan, min_jerk
from tests.conftest import needs_assets

pytestmark = needs_assets

START = np.zeros(17)
START[[3, 4, 6]] = [0.2, 0.2, 1.28]
START[[10, 11, 13]] = [0.2, -0.2, 1.28]
START[WAIST_PITCH] = -0.02


def plan(ref, **kw):
    cfg = RunConfig()
    for k, v in kw.items():
        sec, _, key = k.partition("__")
        setattr(getattr(cfg, sec), key, v) if key else setattr(cfg, sec, v)
    return build_plan(ref, START, cfg), cfg


def test_phase_order_and_handback(ref):
    p, _ = plan(ref)
    names = [s.name for s in p.segments]
    assert names[0] == "acquire" and names[-1] == "release"
    order = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
    assert order == ["acquire", "entry", "settle", "playback", "end_hold", "return", "release"]
    f = p.flat()
    acq = p.segments[0]
    assert np.allclose(acq.q, START)  # acquire holds the measured pose
    assert np.all(np.diff(acq.weight) > 0) and acq.weight[-1] == 1.0
    assert f["weight"][-1] == 0.0 and np.allclose(f["q"][-1], START)  # returned + released
    assert np.all(f["weight"][(f["phase"] >= 1) & (f["phase"] <= 5)] == 1.0)


def test_entry_reaches_frame0_exactly(ref):
    p, _ = plan(ref)
    e = p.segment("entry")
    assert np.allclose(e.q[-1], ref.q[0], atol=1e-12)
    assert np.allclose(p.entry_target, ref.q[0])
    pb = p.segment("playback")
    assert np.allclose(pb.q[0], ref.q[0], atol=1e-12) and pb.t_ref[0] == 0.0


def test_entry_is_smooth_and_speed_bounded(ref):
    p, cfg = plan(ref)
    q = np.vstack([START, p.segment("entry").q])
    v = np.diff(q, axis=0) / cfg.timing.control_dt
    assert np.all(np.abs(v[:, list(ARM_COLS)]) <= cfg.timing.entry_arm_vel + 1e-6)
    assert np.all(np.abs(v[:, :3]) <= cfg.timing.entry_waist_vel + 1e-6)
    assert np.abs(v[0]).max() < 0.01 and np.abs(v[-1]).max() < 0.01  # starts and ends at rest


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_waist_pitch_scale(ref, s):
    p, _ = plan(ref, scales__waist_pitch=s)
    f = p.flat()
    expected0 = START[WAIST_PITCH] + s * (ref.q[0, WAIST_PITCH] - START[WAIST_PITCH])
    assert p.entry_target[WAIST_PITCH] == pytest.approx(expected0)
    if s == 0.0:
        assert np.allclose(f["q"][:, WAIST_PITCH], START[WAIST_PITCH])  # held for the whole run
    pb = p.segment("playback")
    rq, _ = ref.sample(pb.t_ref)
    want = START[WAIST_PITCH] + s * (rq[:, WAIST_PITCH] - START[WAIST_PITCH])
    assert np.allclose(pb.q[:, WAIST_PITCH], want, atol=1e-9)
    # pitch scale must not touch yaw/roll/arms
    assert np.allclose(pb.q[:, WAIST_YAW], rq[:, WAIST_YAW], atol=1e-9)
    assert np.allclose(pb.q[:, list(ARM_COLS)], rq[:, list(ARM_COLS)], atol=1e-9)


def test_hold_waist_isolates_arm_entry(ref):
    p, cfg = plan(ref, hold_waist=True)
    f = p.flat()
    for c in (WAIST_YAW, WAIST_ROLL, WAIST_PITCH):
        assert np.allclose(f["q"][:, c], START[c])
    assert np.allclose(p.entry_target[list(ARM_COLS)], ref.q[0, list(ARM_COLS)])
    assert cfg.to_dict()["effective_scale_vector"][:3] == [0.0, 0.0, 0.0]


def test_speed_changes_time_not_amplitude(ref):
    p1, _ = plan(ref)
    p2, _ = plan(ref, speed__speed=0.5)
    a, b = p1.segment("playback"), p2.segment("playback")
    assert len(b.q) == pytest.approx(2 * len(a.q), abs=2)
    assert np.allclose(a.q.max(0), b.q.max(0), atol=np.radians(0.3))
    assert np.allclose(a.q.min(0), b.q.min(0), atol=np.radians(0.3))
    assert b.t_ref[-1] == pytest.approx(ref.duration)


def test_slow_segment(ref):
    p1, _ = plan(ref)
    p2, _ = plan(ref, speed__segments=[[8.0, 11.0, 0.5]])
    a, b = p1.segment("playback"), p2.segment("playback")
    assert len(b.q) - len(a.q) == pytest.approx(150, abs=25)  # +3 s of wall time
    m = (b.t_ref > 8.5) & (b.t_ref < 10.5)
    assert np.allclose(np.diff(b.t_ref[m]), 0.01, atol=1e-6)
    assert np.allclose(a.q.max(0), b.q.max(0), atol=np.radians(0.3))


def test_amplitude_scale_arms(ref):
    p, _ = plan(ref, scales__arms=0.5)
    pb = p.segment("playback")
    rq, _ = ref.sample(pb.t_ref)
    want = START + 0.5 * (rq - START)
    assert np.allclose(pb.q[:, list(ARM_COLS)], want[:, list(ARM_COLS)], atol=1e-9)


def test_rate_limit_events_reported(ref):
    p, _ = plan(ref, limits__vel_arm=1.0)
    ev = [e for e in p.limit_events if e.cause == "rate"]
    assert any(e.joint == "right_elbow_joint" for e in ev)
    step = np.abs(np.diff(p.segment("playback").q[:, list(ARM_COLS)], axis=0)).max()
    assert step <= 1.0 * 0.02 + 1e-9


def test_position_limits_respected_and_reported(ref):
    p, cfg = plan(ref, limits__pos_margin=0.45)  # large margin forces clipping
    f = p.flat()
    lo, hi = POS_LOWER + 0.45, POS_UPPER - 0.45
    inside = (f["q"] >= np.minimum(lo, START) - 1e-9) & (f["q"] <= np.maximum(hi, START) + 1e-9)
    assert inside.all()
    assert any(e.cause == "position" for e in p.limit_events)


def test_wrist_rate_cap_columns():
    from g1dance.config import Limits
    from g1dance.joints import UPPER_NAMES
    v = Limits(vel_arm=3.0, vel_wrist_pitch_yaw=1.5, vel_waist=0.5).vel_vector()
    for c, n in enumerate(UPPER_NAMES):
        want = 0.5 if n.startswith("waist") else 1.5 if ("wrist_pitch" in n or "wrist_yaw" in n) else 3.0
        assert v[c] == want, n


def test_default_plan_within_urdf_and_no_limit_events(ref):
    p, cfg = plan(ref)
    f = p.flat()
    assert np.all(f["q"] >= POS_LOWER) and np.all(f["q"] <= POS_UPPER)
    assert p.limit_events == []
    assert f["q"].shape[1] == 17  # no leg columns exist in the command


def test_entry_only_plan(ref):
    cfg = RunConfig()
    p = build_entry_only_plan(ref, START, cfg)
    names = [s.name for s in p.segments]
    assert "playback" not in names
    f = p.flat()
    assert f["weight"][-1] == 0.0 and np.allclose(f["q"][-1], START)


def test_min_jerk_endpoints():
    q = min_jerk(np.zeros(2), np.ones(2), 100)
    assert np.allclose(q[-1], 1.0) and q[0, 0] < 1e-3


def test_ramp_in_softens_start_without_changing_frame0(ref):
    p0, cfg = plan(ref)
    p1, _ = plan(ref, speed__ramp_in_s=0.4)
    a, b = p0.segment("playback"), p1.segment("playback")
    assert np.allclose(b.q[0], ref.q[0], atol=1e-12)
    dv = lambda s: np.abs(np.diff(s.q[:3], axis=0)).max() / cfg.timing.control_dt
    assert dv(b) < 0.5 * dv(a)
    assert b.t_ref[-1] == pytest.approx(ref.duration)
