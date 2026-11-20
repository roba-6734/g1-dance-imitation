import numpy as np
import pytest

from g1dance.reference import resample_full_upper
from tests.conftest import MENAGERIE, needs_assets, needs_model

pytestmark = needs_assets


def test_shapes_and_timing(ref, full):
    assert ref.q.shape == (1137, 17) and ref.fps == 50.0
    assert ref.duration == pytest.approx(22.72)
    assert full.n == 683 and full.fps == 30.0
    assert full.duration == pytest.approx(22.733, abs=1e-3)


def test_velocity_is_gradient(ref):
    assert np.allclose(np.gradient(ref.q, 1 / ref.fps, axis=0), ref.qd, atol=1e-4)


def test_upper_file_is_resampled_full_body_upper(ref, full):
    d = resample_full_upper(full, ref.times()) - ref.q
    assert np.degrees(np.sqrt((d ** 2).mean())) < 0.2
    assert np.allclose(ref.q[0], full.upper[0], atol=np.radians(1.5))


def test_full_body_split_keeps_root_and_legs_apart(full):
    qp = full.mujoco_qpos()
    assert np.array_equal(qp[:, 7:19], full.legs)
    assert np.array_equal(qp[:, 19:36], full.upper)
    assert full.legs.shape[1] == 12 and full.upper.shape[1] == 17


def test_hermite_sampling_hits_frames_and_is_smooth(ref):
    pos, vel = ref.sample(ref.times())
    assert np.allclose(pos, ref.q, atol=1e-9)
    assert np.allclose(vel, ref.qd, atol=1e-6)
    t = np.linspace(0, ref.duration, 20000)
    p, _ = ref.sample(t)
    step = np.abs(np.diff(p, axis=0)).max()
    assert step < 0.01  # no jumps between frames


@needs_model
def test_root_quaternion_is_wxyz(full):
    """With wxyz the retargeted feet rest on the floor; xyzw would tilt the robot ~45 deg."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
    d = mujoco.MjData(m)
    d.qpos[:] = full.mujoco_qpos()[0]
    mujoco.mj_kinematics(m, d)
    z = [d.site_xpos[m.site(s).id][2] for s in ("left_foot", "right_foot")]
    assert all(abs(v) < 0.08 for v in z), z
    q = full.root_quat_wxyz[0]
    d.qpos[3:7] = [q[3], q[0], q[1], q[2]]  # reinterpret as xyzw
    mujoco.mj_kinematics(m, d)
    z2 = [d.site_xpos[m.site(s).id][2] for s in ("left_foot", "right_foot")]
    assert max(abs(v) for v in z2) > 0.1
