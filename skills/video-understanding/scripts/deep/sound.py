"""sound: the soundtrack as an editor hears it. Loudness, the music bed (where, how loud under the voice, its
tempo and beat grid, how it feels and where it changes), every sound effect with its kind, and the mix.

How it measures (numpy only, on 16 kHz mono; a second stereo read for width):
  - A short-time spectrum (64 ms window, 10 ms hop) folded into 48 log-spaced bands.
  - Speech is where the transcript has words. The music bed is judged in the gaps between words (only the bed is
    left there) and by a beat in the 40-150 Hz band, where a voice has little energy.
  - Tempo: an onset-strength envelope (positive change of the log bands) autocorrelated over 60-190 BPM with a
    mild preference for 120, then a beat grid placed by dynamic programming (Ellis 2007).
  - Sound effects: onsets the voice does not explain (outside words, or a broadband burst above the voice's own
    highs), not on the music's beat unless far louder. Each is classified by its measured shape: length,
    noisiness (spectral flatness), centroid sweep, low-end share and tonality.
  - Loudness: integrated LUFS, loudness range and true peak from ffmpeg's EBU R128 meter.

    python3 sound.py demo
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Ctx, eprint, near, pct, r3  # noqa: E402

VERSION = 1
SR = 16000
NFFT = 1024
HOP = 160                 # 10 ms
FPS_A = SR / HOP          # spectrum frames per second
N_BANDS = 48
# a frame lights up as soon as its window starts to cover a hit, so frame k is "at" its window's far end
# (less the taper): measured on clicks, onsets land 60 ms after k / FPS_A
OFFSET = (NFFT - 64) / SR


def t_of(k):
    """Spectrum frame index -> seconds."""
    return np.asarray(k) / FPS_A + OFFSET


def k_of(t: float) -> int:
    """Seconds -> spectrum frame index."""
    return int(round((t - OFFSET) * FPS_A))


# ─── spectrum ────────────────────────────────────────────────────────────────────────────────────

def stft_mag(x: np.ndarray) -> np.ndarray:
    """Magnitude spectrum (frames, NFFT//2+1)."""
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    n = 1 + (len(x) - NFFT) // HOP
    win = np.hanning(NFFT).astype(np.float32)
    out = np.empty((n, NFFT // 2 + 1), np.float32)
    for k in range(0, n, 2048):   # in blocks, to keep memory small on long videos
        idx = np.arange(k, min(n, k + 2048))[:, None] * HOP + np.arange(NFFT)[None, :]
        out[k:k + len(idx)] = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    return out


def band_edges() -> np.ndarray:
    return np.geomspace(40, SR / 2 - 100, N_BANDS + 1)


def bands(mag: np.ndarray) -> np.ndarray:
    """Power in log-spaced bands (frames, N_BANDS)."""
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    edges = band_edges()
    out = np.zeros((mag.shape[0], N_BANDS), np.float32)
    p = mag.astype(np.float32) ** 2
    for b in range(N_BANDS):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not sel.any():
            sel = np.argmin(np.abs(freqs - (edges[b] + edges[b + 1]) / 2))
            out[:, b] = p[:, sel]
        else:
            out[:, b] = p[:, sel].mean(axis=1)
    return out


def db(x, floor=1e-10):
    return 10 * np.log10(np.maximum(x, floor))


def onset_env(bp: np.ndarray) -> np.ndarray:
    """Spectral flux: summed positive change of the log band power, normalised."""
    lb = db(bp)
    flux = np.maximum(0, np.diff(lb, axis=0)).sum(axis=1)
    flux = np.r_[0.0, flux]
    flux -= np.convolve(flux, np.ones(31) / 31, mode="same")   # take out the slow trend
    flux = np.maximum(flux, 0)
    return flux / (flux.std() + 1e-9)


def voicing(mag: np.ndarray) -> np.ndarray:
    """Pitch strength per frame: the cepstral peak over 80-400 Hz against its median (voiced speech is high)."""
    lm = np.log(mag + 1e-6)
    cep = np.abs(np.fft.irfft(lm, axis=1))
    q = np.arange(cep.shape[1]) / SR
    sel = (q >= 1 / 400) & (q <= 1 / 80)
    return cep[:, sel].max(axis=1) / (np.median(cep[:, sel], axis=1) + 1e-9)


# ─── tempo and beats ─────────────────────────────────────────────────────────────────────────────

def tempo(env: np.ndarray) -> tuple[float | None, float]:
    """BPM and a 0-1 confidence from the onset envelope's autocorrelation (60-190 BPM, mild prior at 120)."""
    if len(env) < FPS_A * 4:
        return None, 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac /= ac[0] + 1e-9
    lags = np.arange(len(ac))
    lo, hi = int(FPS_A * 60 / 190), int(FPS_A * 60 / 60)
    bpm = 60 * FPS_A / np.maximum(lags[lo:hi], 1)
    prior = np.exp(-0.5 * (np.log2(bpm / 120) / 0.9) ** 2)
    score = ac[lo:hi] * prior
    k = int(np.argmax(score))
    lag = lags[lo + k]
    # refine with the neighbours (parabolic peak)
    if 0 < lo + k < len(ac) - 1:
        y0, y1, y2 = ac[lo + k - 1], ac[lo + k], ac[lo + k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            lag = lag + 0.5 * (y0 - y2) / den
    conf = float(max(0.0, min(1.0, ac[lo + k] / 0.35)))
    return round(60 * FPS_A / lag, 1), round(conf, 2)


def beat_track(env: np.ndarray, bpm: float, tight: float = 100.0) -> list[float]:
    """Beat times by dynamic programming: each beat rewards onset strength and punishes straying from the
    period (Ellis 2007)."""
    period = 60 * FPS_A / bpm
    n = len(env)
    score = env.astype(np.float64).copy()
    back = np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for i in range(lo, n):
        prev = np.arange(max(0, i - hi), i - lo + 1)
        if not len(prev):
            continue
        cost = -tight * (np.log((i - prev) / period)) ** 2
        cand = score[prev] + cost
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[i] += cand[j]
            back[i] = prev[j]
    # start from the best score in the last period and walk back
    tail = np.arange(max(0, n - int(period)), n)
    i = int(tail[np.argmax(score[tail])])
    beats = []
    while i >= 0:
        beats.append(float(t_of(i)))
        i = back[i]
    return sorted(round(b, 3) for b in beats)


# ─── loudness ────────────────────────────────────────────────────────────────────────────────────

def loudness(path: Path) -> dict:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true",
                        "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = p.stderr.decode(errors="replace")
    tail = err[err.rfind("Summary:"):] if "Summary:" in err else err

    def grab(pat):
        m = re.search(pat, tail)
        return float(m.group(1)) if m else None
    return {"integrated_lufs": grab(r"I:\s+(-?[\d.]+) LUFS"), "lra": grab(r"LRA:\s+(-?[\d.]+) LU"),
            "true_peak_db": grab(r"Peak:\s+(-?[\d.]+) dBFS")}


def stereo_width(path: Path) -> float | None:
    """Side/mid energy ratio in dB (very negative = mono)."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", "8000", "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    x = np.frombuffer(p.stdout, np.float32)
    if len(x) < 16000:
        return None
    x = x[: len(x) // 2 * 2].reshape(-1, 2)
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return round(float(db((side ** 2).mean()) - db((mid ** 2).mean())), 1)


# ─── sound effects ───────────────────────────────────────────────────────────────────────────────

def _flatness(p: np.ndarray) -> float:
    p = np.maximum(p, 1e-12)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def classify_sfx(bp: np.ndarray, i: int, dur_hint: float | None = None) -> dict:
    """Shape of the event starting at spectrum frame i, measured only on what rises above the sound just before
    it (so a music bed or a drum hit nearby doesn't colour it): length, attack, noisiness, centroid sweep, the
    low end at its onset, tonality -> its kind."""
    n = bp.shape[0]
    edges = band_edges()
    centers = np.sqrt(edges[:-1] * edges[1:])
    i = max(0, i - 4)   # look from 40 ms before the detected moment: a hit may already peak there
    bg = np.median(bp[max(0, i - 40):max(1, i - 2)], axis=0)
    j1 = min(n, i + int(1.6 * FPS_A))
    ex = np.maximum(bp[i:j1] - 2 * bg, 0)
    if not len(ex) or ex.sum() <= 0:
        return {"kind": "other"}
    total = ex.sum(axis=1)
    # a sharp onset peaks at once (a later, louder peak is the next sound); a swell takes its time
    head = int(0.15 * FPS_A) if dur_hint is None else int(0.6 * FPS_A) + 3
    pk = int(np.argmax(total[:head]))
    if dur_hint is not None and pk < 0.06 * FPS_A:
        # found as a swell but it peaks at once: a hit after all (its smear only looked like a swell)
        dur_hint = None
        pk = int(np.argmax(total[:int(0.15 * FPS_A)]))
    peak_db = float(db(total[pk]))
    rise = db(total[:pk + 1]) > peak_db - 15
    start = int(np.argmax(rise)) if rise.any() else pk        # where the event clears the background
    if dur_hint is not None:
        dur = dur_hint
    else:
        # from the peak until it falls 15 dB, or until it dips and rises again (the next hit, not this one)
        lv = db(total[pk:])
        k, lo = 0, lv[0]
        while k < len(lv) and lv[k] > peak_db - 15 and not (lo < peak_db - 8 and lv[k] > lo + 6):
            lo = min(lo, lv[k])
            k += 1
        dur = (k + pk - start) / FPS_A
    attack = (pk - start) / FPS_A
    end = max(2, min(len(ex), int(max(dur, 0.05) * FPS_A)))
    span = ex[:end]
    cents = (span * centers).sum(axis=1) / (span.sum(axis=1) + 1e-12)
    keep = span.sum(axis=1) > 0.05 * span.sum(axis=1).max()
    cents = cents[keep]
    sweep = float(np.log2((cents[-1] + 1) / (cents[0] + 1))) if len(cents) > 2 else 0.0
    onset = ex[:3].mean(axis=0)
    low = float(onset[centers < 150].sum() / (onset.sum() + 1e-12))
    prof = span.mean(axis=0)
    upper = prof[centers >= 150]
    active = upper[upper > 0.02 * upper.max()] if upper.max() > 0 else upper
    flat = _flatness(active)                        # noise is flat across the bands it uses
    top2 = np.sort(upper)[-2:].sum() / (upper.sum() + 1e-12)
    tonal = float(top2)                             # a tone puts most of its energy in one or two bands
    peak_hz = float(centers[centers >= 150][int(np.argmax(upper))]) if len(upper) else 0.0
    if dur < 0.1 and low < 0.45:
        kind = "click" if centers[int(np.argmax(prof))] > 2500 else "pop"
    elif low > 0.45 and dur < 1.2:
        kind = "impact"
    elif tonal > 0.6 and dur >= 0.15 and peak_hz >= 1000:
        kind = "ding"     # a bell or chime rings above the voice's strong harmonics
    elif dur >= 1.0 and sweep > 0.5:
        kind = "riser"
    elif 0.12 <= dur < 1.5 and flat > 0.3:
        kind = "whoosh"
    else:
        kind = "other"
    return {"kind": kind, "dur_s": r3(dur), "attack_s": r3(attack), "sweep_oct": round(sweep, 2),
            "low_share": round(low, 2), "flatness": round(flat, 3), "peak_hz": round(peak_hz), "level_db": round(float(db(bp[min(n - 1, i + pk)].sum())), 1)}


def find_sfx(bp: np.ndarray, env: np.ndarray, words: list[dict], beats: list[float], music_conf: float,
             active: np.ndarray | None = None, voiced: np.ndarray | None = None) -> list[dict]:
    """Sound effects: onsets and swells the voice does not explain. `active` marks frames where someone is
    speaking (from the sound; word times alone stretch over pauses)."""
    n = len(env)
    t = t_of(np.arange(n))
    in_word = np.zeros(n, bool)
    starts = []
    for w in words:
        a, b = k_of(w["a"] - 0.03), k_of(w["b"] + 0.03)
        in_word[max(0, a):min(n, b)] = True
        starts.append(w["a"])
    if active is not None:
        in_word &= active[:n]
    edges = band_edges()
    centers_ = np.sqrt(edges[:-1] * edges[1:])
    hi = db(bp[:, centers_ > 4000].sum(axis=1))
    lo_b = db(bp[:, centers_ < 120].sum(axis=1))
    speech_hi = np.percentile(hi[in_word], 90) if in_word.any() else -999
    speech_lo = np.percentile(lo_b[in_word], 90) if in_word.any() else -999
    total = db(bp.sum(axis=1))
    loc = np.array([np.median(total[max(0, k - 150):k + 150]) for k in range(0, n, 25)])
    floor = np.interp(np.arange(n), np.arange(0, n, 25), loc)
    cands = []
    thr = np.percentile(env, 97) if n else 0
    on_beat = [env[k_of(b)] for b in beats if 0 <= k_of(b) < n]
    beat_level = float(np.median(on_beat)) if on_beat else 0.0
    for k in range(2, n - 2):
        if env[k] < max(2.5, thr) or env[k] < env[k - 1] or env[k] < env[k + 1]:
            continue
        if total[k + 2 if k + 2 < n else k] < floor[k] + 6:
            continue
        if in_word[k]:
            # inside speech only a burst above the voice's own highs or a boom under its lows, and not just a
            # word starting
            kk = min(n - 1, k + 2)
            if (hi[kk] < speech_hi + 6 and lo_b[kk] < speech_lo + 8) or near(t[k], starts, 0.04):
                continue
        if beats and music_conf > 0.4 and near(t[k], beats, 0.04) and env[k] < 2 * beat_level:
            continue   # a drum hit of the music bed (an effect on the beat has to be far louder than the drums)
        if cands and k - cands[-1] < int(0.15 * FPS_A):
            continue
        cands.append(k)
    # swells: whooshes and risers fade in, so they make no sharp onset; find noise in the upper bands rising
    # well above its own floor for 0.12-1.5 s, away from speech
    up = db(bp[:, np.sqrt(edges[:-1] * edges[1:]) > 1200].sum(axis=1))
    up_s = np.convolve(up, np.ones(3) / 3, mode="same")
    fl = np.array([np.percentile(up_s[max(0, k - 200):k + 200], 20) for k in range(0, n, 25)])
    up_floor = np.interp(np.arange(n), np.arange(0, n, 25), fl)
    hot = (up_s > up_floor + 12) & ~in_word
    hints: dict[int, float] = {}
    k = 0
    while k < n:
        if not hot[k]:
            k += 1
            continue
        j = k
        while j < n and hot[j]:
            j += 1
        if 0.12 * FPS_A <= j - k <= 1.5 * FPS_A and not any(abs(k - c) < 0.15 * FPS_A for c in cands):
            # start where it first clears the floor by 3 dB
            s0 = k
            while s0 > 0 and up_s[s0 - 1] > up_floor[s0 - 1] + 3:
                s0 -= 1
            cands.append(s0)
            hints[s0] = (j - s0) / FPS_A
        k = j
    cands = sorted(set(cands))
    # an effect is mixed to be heard: within 10 dB of the voice (a breath between phrases is far quieter)
    loud_ref = float(np.percentile(total[in_word], 90)) if in_word.any() else float(np.percentile(total, 90))
    out = []
    speech_voicing = float(np.median(voiced[in_word])) if voiced is not None and in_word.any() else None
    in_span = np.zeros(n, bool)          # inside a word's time span, however the sound looks
    for w in words:
        in_span[max(0, k_of(w["a"])):min(n, k_of(w["b"]))] = True
    mid_e = db(bp[:, (centers_ > 300) & (centers_ < 2000)].sum(axis=1))
    for k in cands:
        c = classify_sfx(bp, k, hints.get(k))
        if c.get("level_db") is None or c["level_db"] < loud_ref - 10:
            continue
        if out and abs(t[k] - out[-1]["t"]) < 0.12:
            continue
        span_k = slice(k, min(n, k + max(2, int(min(c.get("dur_s") or 0.1, 1.0) * FPS_A))))
        if speech_voicing is not None and c["kind"] != "ding" and float(np.median(voiced[span_k])) >= speech_voicing:
            continue   # it has a pitch like the voice: speech, not an effect
        if in_span[k] and (c.get("dur_s") or 0) < 0.3 and float(np.median(hi[span_k] - mid_e[span_k])) > 10:
            continue   # an "s" or "sch": short, high-only, inside a word
        if in_span[k] and not (c["kind"] == "impact" and c.get("low_share", 0) > 0.6) and c["level_db"] < loud_ref + 3:
            continue   # over speech only a low boom or something louder than the voice is surely an effect
        conf = 0.5 if in_span[k] else 0.7
        out.append({"t": r3(t[k]), **c, "in_voice": bool(in_word[k]), "conf": conf})
    return out


# ─── music ───────────────────────────────────────────────────────────────────────────────────────

def voice_and_pauses(bp: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """The speech-band level per frame, the voice's running level (90th percentile over 2 s) and the real pauses:
    runs of 150 ms or more where the speech band falls 12 dB under the voice. Found in the sound itself,
    because a transcript's word times stretch over short pauses."""
    c = np.sqrt(band_edges()[:-1] * band_edges()[1:])
    L = db(bp[:, (c > 100) & (c < 4000)].sum(axis=1))
    n = len(L)
    vl = np.array([np.percentile(L[max(0, k - 100):k + 100], 90) for k in range(0, n, 20)])
    vl = np.interp(np.arange(n), np.arange(0, n, 20), vl)
    dip = L < vl - 12
    runs, k = [], 0
    while k < n:
        if dip[k]:
            j = k
            while j < n and dip[j]:
                j += 1
            if j - k >= int(0.15 * FPS_A):
                runs.append((k, j))
            k = j
        else:
            k += 1
    return L, vl, runs


def low_beat(bp: np.ndarray, a: int, b: int) -> float:
    """How periodic the 40-150 Hz band is over frames a..b (0..1): a drum beat, which a voice rarely makes."""
    c = np.sqrt(band_edges()[:-1] * band_edges()[1:])
    lw = np.maximum(0, np.diff(db(bp[a:b, c < 150].sum(axis=1))))
    if len(lw) <= FPS_A:
        return 0.0
    e = lw - lw.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = int(FPS_A * 60 / 190), min(len(ac) - 1, int(FPS_A * 60 / 60))
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def music_presence(bp: np.ndarray, duration: float) -> dict:
    """Is there a bed under the voice? Judged where it can be heard: at the deepest point of each real pause.
    A pause that falls to silence has no bed; one that holds 20-30 dB under the voice has one. Tonal or
    periodic content there says music; a flat, noisy floor says room tone or hiss. Too few pauses: unknown."""
    L, vl, runs = voice_and_pauses(bp)
    c = np.sqrt(band_edges()[:-1] * band_edges()[1:])
    floors = [(a, b, float(np.min(L[a:b]) - np.median(vl[a:b])), _flatness(bp[a:b].mean(axis=0)[c > 150])) for a, b in runs]
    voice_db = round(float(np.median(vl)), 1) if len(vl) else None
    out = {"voice_db": voice_db, "pauses": len(runs), "pause_s": round(sum(b - a for a, b in runs) / FPS_A, 1)}
    if len(floors) < 3:
        return out | {"verdict": "unknown", "why": "fewer than 3 pauses of 150 ms+ to listen in", "present_pct": None,
                      "level_db": None, "rows": []}
    level = float(np.median([f[2] for f in floors]))
    flat = float(np.median([f[3] for f in floors]))
    beat = float(np.median([low_beat(bp, k, min(len(L), k + int(6 * FPS_A))) for k in range(0, max(1, len(L) - int(3 * FPS_A)), int(6 * FPS_A))]))
    if level < -40:
        verdict, why = "none", f"pauses fall to near silence ({level:.0f} dB under the voice)"
    elif level < -32:
        verdict, why = "room tone", f"pauses hold a quiet floor {abs(level):.0f} dB under the voice"
    elif flat >= 0.35 and beat < 0.45:
        verdict, why = "noise-like floor", (f"pauses hold a flat, noise-like floor {abs(level):.0f} dB under the voice: "
                                            "room tone, hiss or a soft pad; no tones or beat to call it music")
    else:
        verdict, why = "music bed", f"pauses hold {'tonal' if flat < 0.35 else 'rhythmic'} sound {abs(level):.0f} dB under the voice"
    rows = []
    for a, b, f, fl in floors:
        rows.append({"t": r3(t_of(a)), "dur_s": r3((b - a) / FPS_A), "floor_db": round(f, 1), "flatness": round(fl, 2)})
    # where it plays: pauses whose floor holds up (a bed can come and go)
    held = [r for r in rows if r["floor_db"] > -32]
    if verdict == "music bed":
        span_ = (min(r["t"] for r in held), max(r["t"] + r["dur_s"] for r in held)) if held else (0, 0)
        present = pct(span_[1] - span_[0], duration)
    else:
        present = 0.0
    return out | {"verdict": verdict, "why": why, "level_db": round(level, 1), "flatness": round(flat, 2),
                  "low_beat": round(beat, 2), "present_pct": present, "rows": rows}


def music_changes(rows: list[dict]) -> list[dict]:
    """Where the bed comes in or drops out: pauses whose floor flips between holding up and falling away."""
    out, prev = [], None
    for r in rows:
        on = r["floor_db"] > -32
        if prev is not None and on != prev:
            out.append({"t": r["t"], "what": "bed audible in pauses" if on else "bed gone from pauses"})
        prev = on
    return out


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def run(ctx: Ctx) -> dict:
    x = ctx.audio(SR)
    if not len(x) or np.abs(x).max() < 1e-4:
        return {"summary": ["No sound."], "loudness": {}, "music": {"verdict": "none", "present_pct": 0}, "sfx": [],
                "sfx_per_min": 0, "kinds": {}, "mix": {}, "evidence": []}
    eprint("sound: spectrum …")
    mag = stft_mag(x)
    bp = bands(mag)
    env = onset_env(bp)
    words = ctx.words()
    d = ctx.duration
    centers = np.sqrt(band_edges()[:-1] * band_edges()[1:])
    pres = music_presence(bp, d)
    L, vl, pauses = voice_and_pauses(bp)
    active = L >= vl - 12
    music_on = pres["verdict"] == "music bed"
    # tempo from the low band (drums), else the whole mix; only reported when there is a bed to have one
    low_env = onset_env(bp[:, centers < 200])
    bpm, conf = tempo(low_env)
    bpm2, conf2 = tempo(env)
    if conf2 > conf + 0.15:
        bpm, conf, use = bpm2, conf2, env
    else:
        use = low_env
    beats = beat_track(use, bpm) if bpm and music_on and conf >= 0.3 else []
    pause_spec = np.concatenate([bp[a:b] for a, b in pauses]).mean(axis=0) if pauses else None
    brightness = float((pause_spec * centers).sum() / (pause_spec.sum() + 1e-12)) if pause_spec is not None else None
    rms = db(bp.sum(axis=1))
    energy_curve = [[r3(t_of(k)), round(float(np.median(rms[k:k + int(FPS_A)])), 1)] for k in range(0, len(rms), int(FPS_A))]
    music = {
        "verdict": pres["verdict"], "why": pres["why"], "present_pct": pres["present_pct"],
        "level_db": pres["level_db"] if music_on else None, "pause_floor_db": pres["level_db"],
        "bpm": bpm if music_on and conf >= 0.3 else None, "bpm_conf": conf if music_on else None, "beats": beats,
        "feel": {"tempo": None if not (music_on and bpm and conf >= 0.3) else ("slow" if bpm < 90 else "mid" if bpm < 125 else "fast"),
                 "brightness_hz": None if brightness is None or not music_on else round(brightness),
                 "low_beat": pres.get("low_beat")},
        "changes": music_changes(pres["rows"]) if music_on else [],
        "pauses": pres["rows"],
    }
    eprint("sound: effects …")
    sfx = find_sfx(bp, env, words, beats, conf if music_on else 0.0, active, voicing(mag))
    kinds: dict[str, int] = {}
    for s_ in sfx:
        kinds[s_["kind"]] = kinds.get(s_["kind"], 0) + 1
    loud = loudness(ctx.v.path)
    width = stereo_width(ctx.v.path)
    crest = None
    if active.any():
        # how compressed the voice is: peak against average power while someone speaks
        sm = np.repeat(active, HOP)
        sm = np.r_[np.zeros(NFFT - 64, bool), sm][:len(x)]
        seg = x[:len(sm)][sm]
        crest = round(float(db(np.max(seg ** 2)) - db(np.mean(seg ** 2))), 1) if len(seg) > SR // 10 else None
    mix = {"voice_db": pres["voice_db"], "pause_floor_db": pres["level_db"], "pauses": pres["pauses"],
           "pause_s": pres["pause_s"], "stereo_side_db": width, "voice_crest_db": crest}
    minutes = d / 60 if d else 1
    summary = []
    if loud.get("integrated_lufs") is not None:
        summary.append(f"Loudness {loud['integrated_lufs']} LUFS integrated, range {loud['lra']} LU, true peak {loud['true_peak_db']} dBFS.")
    if music_on:
        summary.append(f"Music bed: {pres['why']}"
                       + (f"; about {bpm:.0f} BPM (confidence {conf:.2f})" if music["bpm"] else "; no steady beat found") + ".")
    else:
        summary.append({"none": "No music bed", "room tone": "No music bed", "noise-like floor": "No clear music bed",
                        "unknown": "Music bed unknown"}[pres["verdict"]] + f": {pres['why']}.")
    if sfx:
        summary.append(f"{len(sfx)} sound effects ({len(sfx) / minutes:.1f}/min): "
                       + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
                       + f"; {sum(1 for s_ in sfx if s_['in_voice'])} of them over the voice.")
    else:
        summary.append("No sound effects found.")
    if music["changes"]:
        summary.append("Bed: " + ", ".join(f"{c['what']} at {c['t']:.1f} s" for c in music["changes"][:6]) + ".")
    if width is not None:
        summary.append(f"Mix: {'mono' if width < -30 else 'narrow stereo' if width < -15 else 'wide stereo'} "
                       f"(side {width} dB under mid)" + (f", voice crest {crest} dB" if crest is not None else "")
                       + (" (heavily compressed)" if crest is not None and crest < 14 else "") + ".")
    evidence = [{"t": 0.0, "what": "spectrogram with words (blue), beats (ticks) and sound effects (red)",
                 "img": spectrogram_img(ctx, bp, words, beats, sfx)}]
    return {"summary": summary, "loudness": loud, "music": music, "sfx": sfx, "sfx_per_min": round(len(sfx) / minutes, 1),
            "kinds": {k: pct(v, len(sfx)) for k, v in kinds.items()}, "mix": mix, "energy_curve": energy_curve,
            "evidence": evidence}


def spectrogram_img(ctx: Ctx, bp: np.ndarray, words: list[dict], beats: list[float], sfx: list[dict]) -> str:
    W, H = 1200, 200
    lb = db(bp).T[::-1]                       # bands high to low
    lo, hi = np.percentile(lb, 5), np.percentile(lb, 99.5)
    v = np.clip((lb - lo) / (hi - lo + 1e-9), 0, 1)
    cols = np.linspace(0, v.shape[1] - 1, W).astype(int)
    v = v[:, cols]
    rows = np.linspace(0, v.shape[0] - 1, H).astype(int)
    v = v[rows]
    img = np.zeros((H + 30, W, 3), np.uint8)
    img[:H] = (np.stack([v * 40 + 10, v * 200 + 10, v * 255], axis=2)).astype(np.uint8)
    d = max(ctx.duration, 1e-6)
    for w in words:
        a, b = int(w["a"] / d * W), max(int(w["a"] / d * W) + 1, int(w["b"] / d * W))
        img[H + 4:H + 12, a:b] = (10, 132, 255)
    for bt in beats:
        x = min(W - 1, int(bt / d * W))
        img[H + 14:H + 20, x] = (255, 255, 255)
    for s in sfx:
        x = min(W - 2, int(s["t"] / d * W))
        img[:H, x:x + 2] = (255, 59, 48)
        img[H + 22:H + 30, max(0, x - 2):x + 4] = (255, 59, 48)
    return ctx.img("sound-spectrogram.png", img)


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    rng = np.random.default_rng(1)
    d = 12.0
    t = np.arange(int(d * SR)) / SR
    x = np.zeros_like(t, dtype=np.float32)
    # a kick drum every 0.5 s (120 BPM)
    for k in np.arange(0.25, d, 0.5):
        i = int(k * SR)
        n = int(0.12 * SR)
        x[i:i + n] += 0.5 * np.sin(2 * np.pi * 60 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.03 * SR))
    # a noise whoosh at 6.0 s: band-passed noise swelling over 0.5 s with a rising centre
    i, n = int(6.0 * SR), int(0.5 * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= ((f > 1500) & (f < 7000))
    w = np.fft.irfft(spec, n).astype(np.float32)
    w *= np.sin(np.pi * np.arange(n) / n) ** 2
    x[i:i + n] += 0.8 * w / (np.abs(w).max() + 1e-9)
    # a pop at 9.13 s
    i = int(9.13 * SR)
    x[i:i + 400] += 0.9 * rng.standard_normal(400).astype(np.float32) * np.exp(-np.arange(400) / 80)
    bp = bands(stft_mag(x))
    centers = np.sqrt(band_edges()[:-1] * band_edges()[1:])
    low_env = onset_env(bp[:, centers < 200])
    bpm, conf = tempo(low_env)
    assert bpm and abs(bpm - 120) < 3, (bpm, conf)
    beats = beat_track(low_env, bpm)
    off = [min(abs(b - k) for k in np.arange(0.25, d, 0.5)) for b in beats]
    assert np.median(off) < 0.03, np.median(off)
    sfx = find_sfx(bp, onset_env(bp), [], beats, conf)
    got = {round(s["t"], 1): s["kind"] for s in sfx}
    wh = [s for s in sfx if 5.9 <= s["t"] <= 6.4]
    pop = [s for s in sfx if abs(s["t"] - 9.13) < 0.05]
    assert wh and wh[0]["kind"] == "whoosh", got
    assert pop and pop[0]["kind"] in ("pop", "click"), got
    assert not [s for s in sfx if s["t"] < 5.8], got   # the kicks are the beat, not effects
    print(f"  bpm {bpm} conf {conf}, sfx {got}")
    print("sound demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
