import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(os.environ.get("G1DANCE_ASSETS", ROOT / "motion_assets"))
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1.xml"

needs_assets = pytest.mark.skipif(not (ASSETS / "gmr_reference_full_960_2804_upper_50hz.npz").exists(),
                                  reason="motion_assets not present")
needs_model = pytest.mark.skipif(not MENAGERIE.exists(), reason="Menagerie G1 model not fetched")


@pytest.fixture(scope="session")
def ref():
    from g1dance.reference import load_upper
    return load_upper(ASSETS / "gmr_reference_full_960_2804_upper_50hz.npz")


@pytest.fixture(scope="session")
def full():
    from g1dance.reference import load_full
    return load_full(ASSETS / "retargeted_g1_qpos.npy")
