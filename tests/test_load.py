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


def test_smoothing_keeps_shape_and_lowers_load(ref):
    from g1dance.load import ArmLoadModel, peaks
    from g1dance.reference import smooth_reference
    s = smooth_reference(ref, 2.0)
    d = np.degrees(np.abs(s.q - ref.q))
    assert d.max() < 4.0 and np.degrees(np.abs(s.q[0] - ref.q[0])).max() < 1.0  # opening pose ~kept
    assert smooth_reference(ref, None) is ref
    m, t = ArmLoadModel(), ref.times()
    assert peaks(*m.loads(s.q, t, s.qd))["pitch_nm"] < 0.7 * peaks(*m.loads(ref.q, t, ref.qd))["pitch_nm"]


def test_constant_tempo_is_constant_within_limits_and_music_syncs(ref):
    from g1dance.music import music_sync
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.ramp_in_s = 0.3
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1
    cfg.speed.constant_tempo, cfg.speed.smooth_hz = True, 2.0
    p = build_plan(ref, _start(), cfg)
    L = p.load_limit
    assert L["within_limits"] and L["constant_tempo"]
    tempo, delay = music_sync(p)
    assert abs(tempo - L["music_tempo"]) < 1e-3 and 0 < delay < 0.3  # ramp-in lag only
    tr = p.segment("playback").t_ref
    rate = np.diff(tr) / p.dt
    assert np.ptp(rate[len(rate) // 4:-2]) < 1e-9


def test_music_refuses_varying_tempo(ref):
    import pytest
    from g1dance.music import music_sync
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1
    with pytest.raises(SystemExit):
        music_sync(build_plan(ref, _start(), cfg))
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.speed = 0.5
    assert abs(music_sync(build_plan(ref, _start(), cfg))[0] - 0.5) < 1e-9


def test_music_player_starts_on_playback_and_stops_on_abort(tmp_path, monkeypatch):
    import g1dance.music as M
    launched, killed = [], []

    class P:
        def poll(self): return None
        def terminate(self): killed.append(1)
    monkeypatch.setattr(M.shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(M.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd) or P())
    mp = M.MusicPlayer(tmp_path / "x.wav", -0.2)
    mp.on_phase("entry")
    assert not launched
    mp.on_phase("playback")
    assert launched and "-ss" in launched[0] and "0.200" in launched[0]
    mp.on_phase("abort")
    assert killed


def test_music_file_natural_speed_trimmed_normalised(ref, tmp_path):
    import subprocess
    from g1dance.music import plan_music
    song = tmp_path / "song.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=90,volume=0.02",
                    str(song)], check=True)
    cfg = load_config(ROOT / "configs" / "default.toml")
    cfg.speed.max_torque_nm, cfg.speed.max_force_n = 1.2, 4.1  # varying tempo is fine for a natural-speed song
    p = build_plan(ref, _start(), cfg)
    wav, delay, info = plan_music(p, str(song), ROOT / "motion_assets" / "human_motion.mov", tmp_path, start_s=10)
    assert delay == 0.0 and info["stretched_to_tempo"] is None
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(wav)], capture_output=True, text=True).stdout)
    play_s = len(p.segment("playback").q) * p.dt
    assert abs(dur - play_s) < 0.1
    vol = subprocess.run(["ffmpeg", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    mean = float(vol.split("mean_volume:")[1].split("dB")[0])
    assert mean > -25  # normalised up from a -40 dB source
