"""video2ref: the numpy-only stages (no GVHMR/GMR install needed)."""
import pickle

import numpy as np
import pytest

from g1dance.joints import UPPER_IDS, UPPER_NAMES
from g1dance.reference import load_upper
from video2ref.motion import check_qpos, save_formats, smooth_qpos, wxyz_to_xyzw, xyzw_to_wxyz
from video2ref.upper import UPPER_IDS as V_IDS, UPPER_NAMES as V_NAMES, export, prepare
from video2ref.validate import validate

from conftest import ASSETS


def standing(n=60):
    q = np.zeros((n, 36))
    q[:, 2], q[:, 3] = 0.8, 1.0
    return q


def test_joint_tables_match_g1dance():
    assert V_IDS == UPPER_IDS and V_NAMES == UPPER_NAMES


def test_quaternion_order_roundtrip():
    q = np.array([[1., 0., 0., 0.], [0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
    np.testing.assert_allclose(wxyz_to_xyzw(q)[0], [0, 0, 0, 1])


def test_full_body_exports_and_validation(tmp_path):
    names = [f"j{i}" for i in range(29)]
    save_formats(standing(4), 30, names, tmp_path, csv_enabled=True)
    with (tmp_path / "motion.pkl").open("rb") as f:
        m = pickle.load(f)
    assert m["dof_pos"].shape == (4, 29) and m["fps"] == 30
    np.testing.assert_allclose(m["root_rot"][0], [0, 0, 0, 1])  # pickle stores xyzw
    assert validate(standing(4), 30, names, np.tile([-2, 2], (29, 1)))["passed_structural_checks"]
    with pytest.raises(ValueError):
        check_qpos(standing(4)[:, :-1])


def test_full_body_smoothing_handles_quaternion_sign():
    q = standing(5)
    q[1, 7], q[2, 3] = 1.0, -1.0
    s = smooth_qpos(q, 3)
    assert np.isclose(np.linalg.norm(s[2, 3:7]), 1) and s[1, 7] < 1


def test_upper_export_drops_legs_and_loads_in_g1dance(tmp_path):
    q = standing(91)  # 3 s at 30 Hz
    t = np.arange(91) / 30
    q[:, 7:19] = 0.3  # legs: must not appear in the output
    q[:, 7 + 15] = 0.4 * np.sin(2 * np.pi * 0.5 * t)  # left shoulder pitch
    np.save(tmp_path / "qpos.npy", q)
    rep = export(tmp_path / "qpos.npy", tmp_path / "ref.npz")
    ref = load_upper(tmp_path / "ref.npz")
    assert ref.fps == 50 and ref.n == 151 and rep["frames"] == 151
    assert not rep["root_and_leg_commands_included"]
    assert np.all(ref.q[:, [0, 1, 2] + list(range(4, 17))] == 0)  # leg value 0.3 never leaks in
    assert np.max(np.abs(ref.q[:, 3] - 0.4 * np.sin(np.pi * np.arange(151) / 50))) < 0.02  # 3-frame mean + edge pad
    with pytest.raises(FileExistsError):
        export(tmp_path / "qpos.npy", tmp_path / "ref.npz")


def test_upper_export_refuses_limit_breach():
    q = standing(30)
    q[:, 7 + 13] = 0.9  # waist roll beyond its 0.52 rad soft limit
    with pytest.raises(ValueError, match="soft limit"):
        prepare(q)


def test_upper_export_reproduces_shipped_reference():
    full, upper = ASSETS / "retargeted_g1_qpos.npy", ASSETS / "gmr_reference_full_960_2804_upper_50hz.npz"
    if not (full.exists() and upper.exists()):
        pytest.skip("motion_assets not present")
    pos, _ = prepare(np.load(full))
    with np.load(upper) as z:
        assert np.array_equal(pos, z["joint_pos"])
        assert np.array_equal(np.gradient(pos.astype(np.float64), 0.02, axis=0).astype(np.float32), z["joint_vel"])
