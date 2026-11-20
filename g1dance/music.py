"""Music in sync with a constant-tempo playback.

The dance timeline and the human video share t = 0 (human_motion.mov is
22.78 s; the reference is 22.72 s). At a constant tempo s the song is
time-stretched by s (ffmpeg atempo, which keeps the pitch) and started when
playback starts, plus the plan's ramp-in delay, plus an operator offset that
compensates the audio output latency.

Playback uses ffplay on this computer. The audio is only output; nothing here
talks to the robot.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path


def _aac_stream(path: Path) -> int | None:
    from .render import _aac_stream as f
    return f(path)


def _atempo_chain(tempo: float) -> str:
    stages, t = [], tempo
    while t < 0.5:  # atempo accepts 0.5..2.0 per stage; chain stages for slower tempos
        stages.append(0.5)
        t /= 0.5
    stages.append(t)
    return ",".join(f"atempo={x:.6f}" for x in stages)


def prepare_music(src: Path, out_dir: Path, *, tempo: float = 1.0, start_s: float = 0.0,
                  duration_s: float | None = None, fade_s: float = 2.0) -> Path:
    """Loudness-normalised WAV for the dance, cached.

    * src: a song file (mp3, m4a, wav, ...) or a video whose AAC soundtrack is used
    * tempo: time-stretch factor (pitch kept); 1.0 = the song's natural speed
    * start_s: where in the song to begin; duration_s: length to keep (in the
      output, after stretching), ending with a fade-out of fade_s
    """
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found; needed for music")
    src = Path(src)
    if not src.exists():
        raise SystemExit(f"music file not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.sha256(f"{src.resolve()}|{src.stat().st_size}|{tempo:.5f}|{start_s:.3f}|{duration_s}|{fade_s}"
                         .encode()).hexdigest()[:12]
    out = out_dir / f"music_{src.stem[:30].replace(' ', '_')}_{key}.wav"
    if out.exists():
        return out
    a = _aac_stream(src)
    amap = f"0:{a}" if a is not None else "0:a:0"
    filt = []
    if abs(tempo - 1.0) > 1e-6:
        filt.append(_atempo_chain(tempo))
    if duration_s:
        filt.append(f"atrim=0:{duration_s:.3f}")
        if fade_s > 0:
            filt.append(f"afade=t=out:st={max(0.0, duration_s - fade_s):.3f}:d={fade_s:.3f}")
    filt.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src), "-map", amap, "-vn", "-ar", "48000", "-filter:a", ",".join(filt), str(out)]
    subprocess.run(cmd, check=True)
    return out


def stretched_audio(video: Path, tempo: float, cache_dir: Path) -> Path:
    """The video's own soundtrack at `tempo`, normalised (kept for compatibility)."""
    return prepare_music(video, cache_dir, tempo=tempo)


def plan_music(plan, cfg_music_file: str | None, video: Path, out_dir: Path,
               start_s: float = 0.0, stretch: bool | None = None, beat=None) -> tuple[Path, float, dict]:
    """Prepare the audio for this plan. Returns (wav, delay after playback start [s], info).

    A song file plays at its natural speed by default (the dance was recorded
    without music, so there is no beat to match), trimmed to the playback length
    with a fade-out. The dance video's own soundtrack, or a song with stretch=True,
    is time-stretched to the plan's constant tempo, which must then be constant."""
    pb = plan.segment("playback")
    play_s = len(pb.q) * plan.dt
    if cfg_music_file:
        src, do_stretch = Path(cfg_music_file), bool(stretch)
    else:
        src, do_stretch = Path(video), True
    L = plan.load_limit or {}
    if L.get("beat_aligned"):
        if beat is None or do_stretch:
            raise SystemExit("a beat-aligned plan needs the song's beat grid and natural-speed music")
        # start the song so that its beats fall on the planned grid
        tempo, delay = 1.0, (L["beat_grid_phase_s"] - beat.first_beat_s) % beat.period_s
    elif do_stretch:
        tempo, delay = music_sync(plan)
    else:
        tempo, delay = 1.0, 0.0
    wav = prepare_music(src, out_dir, tempo=tempo, start_s=start_s, duration_s=play_s - delay)
    info = {"source": str(src), "stretched_to_tempo": tempo if do_stretch else None, "song_start_s": start_s,
            **({"beat_bpm": round(beat.bpm, 2), "song_first_beat_s": round(beat.first_beat_s, 3),
                "beat_strength": round(beat.strength, 2)} if beat is not None else {}),
            "plan_delay_s": round(delay, 3), "file": str(wav)}
    return wav, delay, info


class MusicPlayer:
    """Start the song at playback start (+ delay); stop on abort or at the end."""

    def __init__(self, wav: Path, delay_s: float):
        if not shutil.which("ffplay"):
            raise SystemExit("ffplay not found; needed for --music")
        self.wav, self.delay = Path(wav), float(delay_s)
        self.proc: subprocess.Popen | None = None
        self._timer: threading.Timer | None = None
        self.started_at: float | None = None

    def _launch(self, skip: float):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if skip > 0:
            cmd += ["-ss", f"{skip:.3f}"]
        self.proc = subprocess.Popen(cmd + [str(self.wav)], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.started_at = time.monotonic()

    def on_phase(self, name: str):
        if name == "playback" and self.proc is None and self._timer is None:
            if self.delay >= 0:
                self._timer = threading.Timer(self.delay, self._launch, args=(0.0,))
                self._timer.daemon = True
                self._timer.start()
            else:  # negative delay: start now, skipping into the song
                self._launch(-self.delay)
        elif name in ("end_hold", "abort"):
            self.stop(fade=name == "end_hold")

    def stop(self, fade: bool = False):
        if self._timer is not None:
            self._timer.cancel()
        if self.proc is not None and self.proc.poll() is None:
            if not fade:
                self.proc.terminate()
            # on a normal end the song is already over (same length as the dance); let it finish


def music_sync(plan) -> tuple[float, float]:
    """(tempo, delay_s) for music that follows this plan's playback. Refuses a
    playback whose tempo varies (after the ramp-in): music cannot follow that."""
    import numpy as np
    pb = plan.segment("playback")
    tr, dt = pb.t_ref, plan.dt
    rate = np.diff(tr) / dt
    steady = rate[len(rate) // 4: -2]  # past any ramp-in
    s = float(np.median(steady))
    if s <= 0 or np.ptp(steady) > 0.01 * s:
        raise SystemExit("--music needs a constant tempo: use --speed S (no --slow), or --max-load with "
                         "--constant-tempo")
    delay = (len(tr) - 1) * dt - tr[-1] / s
    return s, float(delay)
