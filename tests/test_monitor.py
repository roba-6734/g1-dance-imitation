import numpy as np

from g1dance.config import Gates
from g1dance.joints import WAIST_PITCH
from g1dance.monitor import Monitor, RobotState, entry_gate


def state(upper=None, imu=(0, 0, 0), stamp=0.0):
    q = np.zeros(29)
    if upper is not None:
        q[12:29] = upper
    return RobotState(stamp=stamp, q=q, dq=np.zeros(29), imu_rpy=np.array(imu, float), mode_machine=5)


def test_previous_hardware_observation_is_named_precisely():
    """Physical waist 20.8 deg forward while commanding 6.9 deg."""
    g = Gates()
    m = Monitor(g)
    m.capture_reference_attitude(state())
    cmd = np.zeros(17)
    cmd[WAIST_PITCH] = np.radians(6.9)
    meas = cmd.copy()
    meas[WAIST_PITCH] = np.radians(20.8)
    stops = m.check(state(meas), cmd, 1.0, "entry", 3.2, now=0.0)
    checks = {s.check: s for s in stops}
    assert set(checks) == {"tracking_error", "waist_joint_pitch"}
    te = checks["tracking_error"]
    assert te.subject == "waist_pitch_joint"
    assert np.degrees(te.value) == np.float64(np.degrees(te.value))
    assert abs(np.degrees(te.value) - 13.9) < 1e-6 and te.threshold == g.track_waist_pitch
    assert checks["waist_joint_pitch"].threshold == g.waist_pitch_abs
    assert "phase 'entry'" in te.message() and "waist_pitch_joint" in te.message()
    meas_d = m.measurements(state(meas), cmd)
    assert abs(np.degrees(meas_d["waist_joint_pitch"]) - 20.8) < 1e-9
    assert abs(np.degrees(meas_d["waist_pitch_error"]) - 13.9) < 1e-9
    assert meas_d["pelvis_imu_pitch"] == 0.0  # separate measurement, unaffected


def test_pelvis_imu_tilt_is_its_own_check():
    m = Monitor(Gates())
    m.capture_reference_attitude(state(imu=(0, 0.02, 0)))
    stops = m.check(state(imu=(0, 0.02 + 0.2, 0)), np.zeros(17), 1.0, "playback", 5.0, now=0.0, t_ref=1.5)
    assert [s.check for s in stops] == ["pelvis_imu_pitch"]
    assert "dance t=1.50s" in stops[0].message()


def test_stale_state_stops_everything_else():
    m = Monitor(Gates())
    stops = m.check(state(stamp=0.0), np.ones(17), 1.0, "playback", 1.0, now=0.5)
    assert [s.check for s in stops] == ["state_freshness"]


def test_tracking_ignored_while_weight_below_one():
    m = Monitor(Gates())
    stops = m.check(state(), np.full(17, 0.5), 0.5, "release", 1.0, now=0.0)
    assert stops == []


def test_locked_waist_skips_roll_pitch():
    m = Monitor(Gates(), waist_roll_pitch_actuated=False)
    cmd = np.zeros(17)
    cmd[1:3] = 0.3
    assert m.check(state(), cmd, 1.0, "entry", 0.0, now=0.0) == []


def test_thresholds_are_not_modified_by_monitor():
    g = Gates()
    before = repr(g)
    m = Monitor(g)
    m.check(state(np.full(17, 1.0)), np.zeros(17), 1.0, "entry", 0.0, now=0.0)
    assert repr(g) == before


def test_entry_gate_checks_command_and_measurement():
    g = Gates()
    tgt = np.linspace(-0.3, 0.3, 17)
    ok = entry_gate(tgt.copy(), tgt, tgt + 0.01, g)
    assert ok.passed
    bad_cmd = tgt.copy()
    bad_cmd[5] += 0.01
    r = entry_gate(bad_cmd, tgt, tgt, g)
    assert not r.passed and "commanded left_shoulder_yaw_joint" in r.failures[0]
    meas = tgt.copy()
    meas[WAIST_PITCH] += np.radians(13.9)
    r = entry_gate(tgt, tgt, meas, g)
    assert not r.passed and any("measured waist_pitch_joint" in f for f in r.failures)


def _leg_state(leg_dq_val, joint=9):
    s = state()
    s.dq = np.zeros(29)
    s.dq[joint] = leg_dq_val
    return s


def test_leg_step_needs_consecutive_ticks_and_names_joint():
    g = Gates()
    m = Monitor(g)
    cmd = np.zeros(17)
    fast = g.leg_step_dq + 0.5
    assert not [s for s in m.check(_leg_state(fast), cmd, 1.0, "playback", 1.0, now=0.0) if s.check == "leg_step"]
    assert not [s for s in m.check(_leg_state(0.0), cmd, 1.0, "playback", 1.02, now=0.0) if s.check == "leg_step"]
    m.check(_leg_state(fast), cmd, 1.0, "playback", 1.04, now=0.0)
    st = [s for s in m.check(_leg_state(-fast), cmd, 1.0, "playback", 1.06, now=0.0) if s.check == "leg_step"]
    assert len(st) == 1 and st[0].subject == "right_knee_joint" and st[0].unit == "rad/s"
    assert st[0].value == -fast and st[0].threshold == g.leg_step_dq
    assert "leg_step" in st[0].message() and "right_knee_joint" in st[0].message()


def test_leg_step_below_threshold_and_disabled():
    g = Gates()
    m = Monitor(g)
    for _ in range(10):  # the clean 0.5x run peaked at 1.03 rad/s
        assert not [s for s in m.check(_leg_state(1.03), np.zeros(17), 1.0, "playback", 0, now=0.0) if s.check == "leg_step"]
    g0 = Gates(leg_step_dq=0.0)
    m0 = Monitor(g0)
    for _ in range(10):
        assert not [s for s in m0.check(_leg_state(20.0), np.zeros(17), 1.0, "playback", 0, now=0.0) if s.check == "leg_step"]
