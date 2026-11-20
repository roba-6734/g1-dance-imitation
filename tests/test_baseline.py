"""The baseline commands used on hardware on 2026-11-20 must keep producing
exactly the same plan when new (opt-in) options are added."""
import hashlib

import numpy as np

from g1dance.config import load_config
from g1dance.planner import build_plan
from tests.conftest import ROOT, needs_assets, needs_model

pytestmark = [needs_assets, needs_model]


def _fingerprint(ref, cfg):
    from g1dance.sim import assumed_start_upper
    f = build_plan(ref, assumed_start_upper(), cfg).flat()
    data = np.round(np.concatenate([f["q"].ravel(), np.nan_to_num(f["t_ref"], nan=-1)]), 9)
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def test_baseline_load_limited_1p2_4p1_unchanged(ref):
    # trial --stage full --max-load 1.2:4.1 --set speed.ramp_in_s=0.3   (FINDINGS 4t, 4w)
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ramp_in_s = 0.3
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1
    assert _fingerprint(ref, cfg) == "94d2e31e03196193"


def test_baseline_half_speed_unchanged(ref):
    # trial --stage full --speed 0.5 --set speed.ramp_in_s=0.3   (FINDINGS 4l)
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ramp_in_s = 0.3
    cfg.speed.speed = 0.5
    assert _fingerprint(ref, cfg) == "06a7adeb66545ec8"
