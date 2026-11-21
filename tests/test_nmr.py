"""NMR retargeter adapter and the GMR/NMR A/B comparison (numpy-only parts)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from g1dance.joints import G1_JOINT_NAMES
from video2ref.compare import agreement, compare_runs, smoothness
from video2ref.nmr import NMR_JOINT_ORDER, NMR_TO_SDK, nmr_to_qpos

ROOT = Path(__file__).resolve().parents[1]
NMR = ROOT / "third_party/NMR"
# JOINT_MAPPING from NMR convert_bmimic.py at ba87ad65 (network DoF order -> MuJoCo XML order).
NMR_JOINT_MAPPING = (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
                     20, 27, 21, 28)


def test_mapping_matches_nmr_convert_bmimic():
    assert NMR_TO_SDK == NMR_JOINT_MAPPING
    assert sorted(NMR_TO_SDK) == list(range(29)) and set(NMR_JOINT_ORDER) == set(G1_JOINT_NAMES)


@pytest.mark.skipif(not (NMR / "convert_bmimic.py").exists(), reason="NMR checkout not present")
def test_mapping_against_checkout():
    src = (NMR / "convert_bmimic.py").read_text()
    assert tuple(eval(re.search(r"JOINT_MAPPING = (\[.*?\])", src, re.S).group(1))) == NMR_TO_SDK
    bodies = eval(re.search(r"G1_BMIMIC_BODY_NAMES = (\[.*?\])", src, re.S).group(1))
    # body k+1 is the child of joint k; the torso hangs off waist_pitch
    child = {"torso_link": "waist_pitch_joint"}
    assert [child.get(b, b.replace("_link", "_joint")) for b in bodies[1:]] == list(NMR_JOINT_ORDER)


def test_nmr_to_qpos_reorders_and_normalises():
    rng = np.random.default_rng(0)
    sdk = rng.uniform(-0.5, 0.5, (8, 29))
    raw = {"dof": sdk[:, list(NMR_TO_SDK)], "root_trans": np.tile([0, 0, 0.78], (8, 1)),
           "root_rot_quat": np.tile([2.0, 0, 0, 0], (8, 1))}  # unnormalised on purpose
    q = nmr_to_qpos(raw)
    np.testing.assert_allclose(q[:, 7:], sdk)
    np.testing.assert_allclose(q[:, 3:7], np.tile([1, 0, 0, 0], (8, 1)))
    left_knee_in_nmr = NMR_JOINT_ORDER.index("left_knee_joint")
    np.testing.assert_allclose(q[:, 7 + 3], raw["dof"][:, left_knee_in_nmr])
    with pytest.raises(ValueError):
        nmr_to_qpos({**raw, "dof": raw["dof"][:, :28]})


def test_smoothness_ranks_noise_and_agreement_finds_lag():
    t = np.arange(500) / 50
    clean = np.tile(0.4 * np.sin(2 * np.pi * 0.8 * t)[:, None], (1, 17))
    noisy = clean + np.random.default_rng(1).normal(0, 0.01, clean.shape)
    s_clean, s_noisy = smoothness(clean, 0.02), smoothness(noisy, 0.02)
    assert s_noisy["all"]["rms_jerk"] > 10 * s_clean["all"]["rms_jerk"]
    assert s_noisy["all"]["power_above_4hz_frac"] > s_clean["all"]["power_above_4hz_frac"]
    shifted = np.roll(clean, 5, axis=0)  # 100 ms later
    ag = agreement(clean[20:], shifted[20:], 0.02)
    assert ag["median_corr"] > 0.99
    assert all(abs(v["lag_s"] - 0.1) < 1e-9 for v in ag["per_joint"].values())


def _fake_run(path: Path, q_upper, completed=True):
    path.mkdir(parents=True)
    n = len(q_upper)
    phases = np.array(["acquire", "entry", "settle", "playback", "return", "release"])
    q = np.zeros((n, 29))
    q[:, 12:29] = q_upper
    np.savez_compressed(path / "log.npz", data_kind="physical", phases=phases, t=np.arange(n) * 0.002,
                        phase=np.full(n, 3), t_ref=np.arange(n) * 0.002, q_cmd=q_upper, weight=np.ones(n),
                        q_meas=q, dq_meas=np.zeros_like(q), imu_rpy=np.zeros((n, 3)), torso_rpy=np.zeros((n, 3)))
    (path / "summary.json").write_text(json.dumps({"completed": completed, "data_kind": "physical", "response": {
        "playback": {"per_joint_vs_reference": {"j": {"rms_deg": 1.5, "max_abs_deg": 3}},
                     "shape_lag_corr": {"j": {"lag_s": 0.05, "corr": 0.99}}}}}))


def test_compare_runs_reads_g1dance_run_dirs(tmp_path):
    t = np.arange(2000) * 0.002
    smooth = np.tile(0.3 * np.sin(2 * np.pi * t)[:, None], (1, 17))
    shaky = smooth + 0.005 * np.sin(2 * np.pi * 15 * t)[:, None]
    _fake_run(tmp_path / "a", shaky)
    _fake_run(tmp_path / "b", smooth)
    rep = compare_runs(tmp_path / "a", tmp_path / "b", ["GMR", "NMR"])
    a, b = rep["measured"]["GMR"], rep["measured"]["NMR"]
    assert a["measured_smoothness"]["all"]["rms_jerk"] > b["measured_smoothness"]["all"]["rms_jerk"]
    assert a["tracking_rms_deg_mean"] == 1.5 and a["median_lag_s"] == 0.05


def test_clamp_to_limits_reports_and_bounds():
    from video2ref.nmr import clamp_to_limits

    q = np.zeros((4, 36))
    q[:, 3] = 1
    q[2, 7 + 5] = 0.35  # left ankle roll past its 0.2618 rad stop
    limits = np.tile([-0.2618, 0.2618], (29, 1))
    out, rep = clamp_to_limits(q, limits)
    assert out[2, 7 + 5] == 0.2618 and rep == {"left_ankle_roll_joint": {"max_over_rad": 0.0882, "frames": 1}}
    assert q[2, 7 + 5] == 0.35  # input untouched
