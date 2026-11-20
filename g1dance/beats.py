"""Beat alignment: make the dance's key moves land on the song's beats.

1. ``beat_grid``: beat period and phase of a song section, from an onset-strength
   envelope (spectral flux), its autocorrelation (period) and a comb search
   (phase). Offline, numpy only; ffmpeg decodes the audio.
2. ``dance_accents``: "hits" in the reference, i.e. moments where a fast movement
   lands (a pronounced minimum of the summed joint speed after a peak).
3. ``align``: dynamic programming that assigns accents (in order) to beats or
   half-beats, so that the average tempo between consecutive anchors stays
   between ``lo_factor * base`` and the load-limited tempo cap of that stretch.
   Accents may be skipped. The result is a monotone time map
   dance-time -> wall-time (piecewise cubic, Fritsch-Carlson) through the
   anchors, which the planner turns into a tempo profile.

Amplitude and poses are never changed; only the timing.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22050
HOP, WIN = 512, 2048
# A frame's onset strength rises once a sharp onset enters its (Hann) window; frame i
# starts at i*HOP. Measured on a synthetic click track: onsets register 0.065 s after the
# frame start (~0.7 WIN), so frame times are shifted by this lag (tests/test_load.py).
ONSET_LAG_S = 0.065


def _decode(src: Path, start_s: float, dur_s: float) -> np.ndarray:
    from .music import _aac_stream
    a = _aac_stream(Path(src))
    amap = f"0:{a}" if a is not None else "0:a:0"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur_s:.3f}", "-i", str(src),
           "-map", amap, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32)


def onset_envelope(x: np.ndarray, hop: int = HOP, win: int = WIN) -> tuple[np.ndarray, float]:
    n = (len(x) - win) // hop
    if n < 10:
        raise ValueError("audio too short for beat tracking")
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n] * np.hanning(win)
    logm = np.log1p(10 * np.abs(np.fft.rfft(fr, axis=1)))
    flux = np.concatenate([[0.0], np.maximum(np.diff(logm, axis=0), 0).sum(1)])
    o = np.maximum(flux - np.convolve(flux, np.ones(32) / 32, "same"), 0)
    return o, SR / hop


@dataclass
class BeatGrid:
    period_s: float
    first_beat_s: float  # first beat at or after the section start, relative to it [s]
    strength: float  # mean onset strength on the grid / mean overall (1 = no beat)

    @property
    def bpm(self) -> float:
        return 60.0 / self.period_s


def beat_grid(src: Path, start_s: float, dur_s: float, bpm_range=(60.0, 180.0)) -> BeatGrid:
    """Beat grid of the song section [start_s, start_s + dur_s]. A margin of
    up to 10 s either side is analysed for a more stable period."""
    pre = min(10.0, start_s)
    x = _decode(src, start_s - pre, dur_s + pre + 10.0)
    o, fr = onset_envelope(x)
    ac = np.correlate(o, o, "full")[len(o) - 1:]
    lags = np.arange(len(ac)) / fr
    m = (lags > 60 / bpm_range[1]) & (lags < 60 / bpm_range[0])
    p0 = float(lags[m][np.argmax(ac[m])])
    t = np.arange(len(o)) / fr
    best = (-1.0, p0, 0.0)
    for p in np.linspace(p0 * 0.98, p0 * 1.02, 81):
        for ph in np.arange(0.0, p, 0.005):
            idx = np.round((np.arange(ph, t[-1], p)) * fr).astype(int)
            idx = idx[idx < len(o)]
            sc = float(o[idx].mean())
            if sc > best[0]:
                best = (sc, p, ph)
    sc, p, ph = best
    first = (ph + ONSET_LAG_S - pre) % p  # relative to start_s
    return BeatGrid(period_s=float(p), first_beat_s=float(first), strength=sc / float(o.mean() + 1e-12))


def dance_accents(q: np.ndarray, qd: np.ndarray, fps: float, window_s: float = 0.6,
                  merge_s: float = 0.3) -> np.ndarray:
    """(n, 2) array of [dance time, strength] for movement "hits"."""
    v = np.convolve(np.abs(qd).sum(1), np.ones(5) / 5, "same")
    W = int(window_s * fps)
    med = float(np.median(v))
    t = np.arange(len(v)) / fps
    out = []
    for i in range(W, len(v) - W):
        if v[i] == v[max(0, i - 8):i + 9].min():
            pre, post = v[i - W:i].max(), v[i:i + W].max()
            if pre - v[i] > 0.5 * med and min(pre, post) - v[i] > 0.25 * med:
                out.append([t[i], pre - v[i]])
    out = np.array(out).reshape(-1, 2)
    merged = []  # keep the stronger of accents closer than merge_s
    for a in out:
        if merged and a[0] - merged[-1][0] < merge_s:
            if a[1] > merged[-1][1]:
                merged[-1] = a
        else:
            merged.append(a)
    return np.array(merged).reshape(-1, 2)


@dataclass
class Alignment:
    anchors: np.ndarray  # (k, 2) [dance time, wall time from playback start], incl. (0, 0) and the end
    on_beat: list[bool]  # per aligned accent: full beat (True) or half beat (False)
    grid_phase_s: float  # wall time (from playback start) of a beat in the planned grid
    n_accents: int


def align(accents: np.ndarray, cap_t: np.ndarray, cap: np.ndarray, base: float, duration: float,
          period: float, lo_factor: float = 0.8, tempo_penalty: float = 2.0, window_s: float = 1.2) -> Alignment:
    """Assign accents to beats/half-beats of a grid with free phase (the music
    start is chosen to match). cap(t) is the per-dance-time tempo cap."""
    A = accents[:, 0]
    S = accents[:, 1] / max(accents[:, 1].mean(), 1e-9)

    def capmin(a, b):
        i0 = int(np.searchsorted(cap_t, a))
        i1 = max(int(np.searchsorted(cap_t, b)), i0 + 1)
        return float(cap[i0:i1].min())

    best = None
    for phase in np.arange(0.0, period, 0.02):
        grid = phase + np.arange(-2, int(duration / base / (period / 2)) + 8) * (period / 2)
        on = (np.round((grid - phase) / (period / 2)).astype(int) % 2) == 0
        states = {(-1, -1): (0.0, None)}
        for i, a in enumerate(A):
            for j in np.flatnonzero(np.abs(grid - a / base) < window_s):
                tb = grid[j]
                if tb <= 0:
                    continue
                bs, bp = -1e18, None
                for (pi, pj), (sc, _) in states.items():
                    ta, taua = (0.0, 0.0) if pi < 0 else (grid[pj], A[pi])
                    if tb <= ta or pi >= i:
                        continue
                    sp = (a - taua) / (tb - ta)
                    if sp < lo_factor * base or sp > capmin(taua, a):
                        continue
                    s = sc + (1.0 if on[j] else 0.6) * S[i] - tempo_penalty * abs(np.log(sp / base))
                    if s > bs:
                        bs, bp = s, (pi, pj)
                if bp is not None:
                    states[(i, j)] = (bs, bp)
        k = max(states, key=lambda x: states[x][0])
        if best is None or states[k][0] > best[0]:
            best = (states[k][0], phase, states, k, grid, on)
    _, phase, states, k, grid, on = best
    path = []
    while k != (-1, -1):
        path.append(k)
        k = states[k][1]
    path = path[::-1]
    anchors = [(0.0, 0.0)] + [(A[i], grid[j]) for i, j in path]
    last_tau, last_t = anchors[-1]
    anchors.append((duration, last_t + (duration - last_tau) / base))
    return Alignment(np.array(anchors), [bool(on[j]) for _, j in path], float(phase), len(A))


def monotone_map(x: np.ndarray, y: np.ndarray):
    """Fritsch-Carlson monotone cubic through (x, y). Returns (f, df)."""
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(len(x))
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(x) - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1]
                                                                          + (h[i] + 2 * h[i - 1]) / d[i])

    def seg(t):
        t = np.clip(np.atleast_1d(t), x[0], x[-1])
        i = np.clip(np.searchsorted(x, t) - 1, 0, len(x) - 2)
        s = (t - x[i]) / h[i]
        return i, s

    def f(t):
        i, s = seg(t)
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]

    def df(t):
        i, s = seg(t)
        d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        return (d00 * y[i] + d10 * h[i] * m[i] + d01 * y[i + 1] + d11 * h[i] * m[i + 1]) / h[i]

    return f, df
