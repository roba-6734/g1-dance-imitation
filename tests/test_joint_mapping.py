from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from g1dance import joints as J
from tests.conftest import MENAGERIE, needs_assets, needs_model


def test_sdk_index_table():
    assert len(J.G1_JOINT_NAMES) == 29
    assert J.G1_JOINT_NAMES[12:15] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    assert J.G1_JOINT_NAMES[15] == "left_shoulder_pitch_joint"
    assert J.G1_JOINT_NAMES[22] == "right_shoulder_pitch_joint"
    assert J.G1_JOINT_NAMES[28] == "right_wrist_yaw_joint"
    assert J.UPPER_IDS == tuple(range(12, 29))
    assert set(J.LEG_IDS).isdisjoint(J.UPPER_IDS)
    assert J.WEIGHT_INDEX == 29
    assert all("hip" in n or "knee" in n or "ankle" in n for n in (J.G1_JOINT_NAMES[i] for i in J.LEG_IDS))


@needs_assets
def test_reference_ids_and_names_match_sdk(ref):
    # load_upper validates; check again explicitly against the SDK table
    for col, mid in enumerate(J.UPPER_IDS):
        assert J.UPPER_NAMES[col] == J.G1_JOINT_NAMES[mid]


@needs_model
def test_menagerie_joint_order_and_limits_match_sdk_table():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
    names = [m.joint(i).name for i in range(1, m.njnt)]  # skip floating base
    assert tuple(names) == J.G1_JOINT_NAMES
    for col, mid in enumerate(J.UPPER_IDS):
        lo, hi = m.jnt_range[mid + 1]
        assert lo == pytest.approx(J.POS_LOWER[col], abs=1e-4)
        assert hi == pytest.approx(J.POS_UPPER[col], abs=1e-4)
        assert m.jnt_qposadr[mid + 1] == 7 + mid  # qpos layout used for the full-body file


def test_variant_classification():
    assert J.classify_variant(5).waist_roll_pitch_actuated is True
    assert J.classify_variant(6).waist_roll_pitch_actuated is False
    assert J.classify_variant(12).waist_roll_pitch_actuated is False
    assert J.classify_variant(4).supported is False  # 23 DoF
    v = J.classify_variant(99)
    assert v.supported is False and v.waist_roll_pitch_actuated is None
    assert J.classify_variant(None).supported is False


def test_assert_upper_only_rejects_legs():
    J.assert_upper_only(range(12, 29))
    with pytest.raises(ValueError):
        J.assert_upper_only([11, 12])
    with pytest.raises(ValueError):
        J.assert_upper_only([29])


def test_readonly_check_never_publishes():
    src = (Path(J.__file__).parent / "hw" / "check.py").read_text()
    for forbidden in ("ChannelPublisher", "UnitreeBackend", ".arm(", ".Write(", "send("):
        assert forbidden not in src, forbidden


def test_unitree_backend_writes_only_upper_body_and_weight():
    from g1dance.hw.unitree import UnitreeBackend

    sentinel = -999.0
    msg = SimpleNamespace(motor_cmd=[SimpleNamespace(q=sentinel, dq=sentinel, kp=sentinel, kd=sentinel,
                                                     tau=sentinel) for _ in range(35)], crc=0)
    written = []
    b = object.__new__(UnitreeBackend)
    b._pub = SimpleNamespace(Write=lambda m: written.append(m))
    b._msg = msg
    b._crc = SimpleNamespace(Crc=lambda m: 1234)
    q = np.linspace(-0.5, 0.5, 17)
    b.send(q, np.full(17, 60.0), np.full(17, 1.5), 0.7)
    assert written and msg.crc == 1234
    for i in J.LEG_IDS:
        mc = msg.motor_cmd[i]
        assert (mc.q, mc.kp, mc.kd, mc.tau) == (sentinel,) * 4, f"leg motor {i} was written"
    for c, i in enumerate(J.UPPER_IDS):
        assert msg.motor_cmd[i].q == pytest.approx(q[c])
        assert msg.motor_cmd[i].kp == 60.0 and msg.motor_cmd[i].dq == 0.0 and msg.motor_cmd[i].tau == 0.0
    assert msg.motor_cmd[29].q == pytest.approx(0.7)
    for i in range(30, 35):
        assert msg.motor_cmd[i].q == sentinel


def test_unitree_backend_refuses_without_arming():
    from g1dance.hw.unitree import UnitreeBackend
    b = object.__new__(UnitreeBackend)
    b._pub = None
    with pytest.raises(PermissionError):
        b.send(np.zeros(17), np.zeros(17), np.zeros(17), 1.0)
    with pytest.raises(PermissionError):
        b.arm(confirmed=False)


def test_unitree_backend_dq_feedforward_field():
    from g1dance.hw.unitree import UnitreeBackend
    msg = SimpleNamespace(motor_cmd=[SimpleNamespace(q=0.0, dq=-9.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(35)], crc=0)
    b = object.__new__(UnitreeBackend)
    b._pub, b._msg, b._crc = SimpleNamespace(Write=lambda m: None), msg, SimpleNamespace(Crc=lambda m: 1)
    b.send(np.zeros(17), np.ones(17), np.ones(17), 1.0)  # default: dq = 0 like Unitree's example
    assert all(msg.motor_cmd[i].dq == 0.0 for i in J.UPPER_IDS)
    dq = np.linspace(-1, 1, 17)
    b.send(np.zeros(17), np.ones(17), np.ones(17), 1.0, dq)
    assert all(msg.motor_cmd[i].dq == pytest.approx(dq[c]) for c, i in enumerate(J.UPPER_IDS))
    assert all(msg.motor_cmd[i].dq == -9.0 for i in J.LEG_IDS)


def test_leg_stance_check_flags_staggered_feet():
    import json
    from g1dance.hw.check import leg_stance_checks
    from tests.conftest import ROOT
    ref = ROOT / "runs" / "stance.json"
    good = [-21.9, -1.5, 1.0, 35.7, -15.9, 1.9, -19.5, 1.2, -0.7, 35.1, -19.3, 0.3]     # 121046: no steps
    bad = [-18.5, -1.5, 0.8, 36.4, -19.6, 1.1, -22.4, 1.2, -1.0, 33.1, -14.3, -0.8]     # 134151: stepped
    refv = ([json.loads(ref.read_text())["leg_q_deg"][J.G1_JOINT_NAMES[i]] for i in range(12)]
            if ref.exists() else good)
    lv = lambda c: [x["level"] for x in c]
    assert lv(leg_stance_checks(good, refv)) == ["INFO", "PASS"]
    c = leg_stance_checks(bad, refv)
    assert lv(c) == ["INFO", "WARN"] and "right_ankle_pitch_joint" in c[1]["msg"]
    assert lv(leg_stance_checks(bad)) == ["INFO"]  # no reference: report only


def test_stance_gate_on_hardware_starts():
    from g1dance.hw.check import stance_gate

    def legs(hip_lr, ankle_lr, knee_lr=1.0):
        q = np.array([-20.0, -1.5, 1.0, 35.0, -17.0, 1.5, -20.0, 1.2, -0.8, 35.0, -17.0, -0.3])
        q[0] += hip_lr
        q[3] += knee_lr
        q[4] += ankle_lr
        return q
    assert stance_gate(legs(0.6, -1.7, 2.4), 1.8)[0]  # 162841: passed dance t=8 s cleanly
    assert stance_gate(legs(0.9, -1.0), 1.8)[0]  # 144014: clean at 1.5:5.0
    assert not stance_gate(legs(2.8, -3.5, 2.5), 1.8)[0]  # 173855: stepped at 8.2 s
    assert not stance_gate(legs(2.0, -1.8), 1.8)[0]  # 144601: stepped at the lean
    assert stance_gate(legs(5.0, 5.0), 0.0)[0]  # 0 disables
    assert "hip +0.6" in stance_gate(legs(0.6, -1.7), 1.8)[1]
