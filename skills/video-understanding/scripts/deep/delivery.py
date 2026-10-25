"""delivery: how the person talks and presents. Pace, pauses, fillers, which words they stress, how their pitch
and energy move, the shape of the script, and their body language.

How it measures:
  - Transcript: vu's whisper run, checked. When it is poor (low word confidence, characters repeating, or text
    that doesn't match the captions on screen) whisper runs again in English and the better one is kept and
    written to teardown/words.json for every later lens. Mixed Hindi-English speech comes back from an English
    run as English: the words then stand in for what was said, and pace from them is marked approximate.
  - Pauses come from the sound (the speech band falling 12 dB under the voice), because whisper stretches
    word times over short pauses. max_pause_s is the 90th percentile pause, as creator-teardown measures it.
  - Pitch: YIN on 40 ms frames every 10 ms (70-400 Hz); loudness: RMS in dB per frame.
  - A stressed word stands out in pitch peak, loudness and length against the speaker's own norm (z-scores),
    with a bonus when a pause comes right before it.
  - Body language (proxies, from Apple Vision): face on screen, looking at the camera (face turned under 15°),
    head movement, hands in frame and gestures (fast hand movement, in face widths per second).

    python3 delivery.py demo
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Ctx, eprint, near, pct, r3  # noqa: E402  (common puts vu on the path)
import vu  # noqa: E402

VERSION = 1
SR = 16000
HOP = 160
WIN = 640

FILLERS = {
    "en": {"um", "uh", "erm", "hmm", "like", "basically", "actually", "literally", "so", "right", "okay"},
    "en2": {"you know", "i mean", "kind of", "sort of"},
    "de": {"äh", "ähm", "öh", "hm", "also", "halt", "quasi", "sozusagen", "eigentlich", "irgendwie", "ne", "genau", "ja"},
    "de2": {"und so"},
    "hi": {"matlab", "toh", "to", "yaani", "na", "haan", "accha", "acha", "basically", "bas", "woh", "मतलब", "तो", "यानी", "ना", "हाँ", "अच्छा", "बस"},
    "hi2": {"kya hai", "kya bolte"},
}
YOU = {"you", "your", "yours", "you're", "du", "dich", "dir", "dein", "deine", "deinen", "deinem", "deiner", "ihr", "euch",
       "euer", "eure", "sie", "ihnen", "aap", "aapko", "aapka", "aapki", "tum", "tumhe", "tumhara", "tumhari", "tu", "tera",
       "teri", "आप", "आपको", "तुम", "तुम्हें", "तू", "तेरा"}
CTA = re.compile(r"comment|follow|link|save|share|subscribe|bio|dm\b|kommentier|folg|speicher|teil|abonnier|schreib|"
                 r"karo\b|kijiye|bhejo|likho", re.I)
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand",
                "million", "billion", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
                "hundert", "tausend", "million", "milliarde", "ek", "do", "teen", "char", "paanch", "das", "sau", "hazaar",
                "lakh", "crore"}


# ─── words ───────────────────────────────────────────────────────────────────────────────────────

def token(w: str) -> str:
    """A word reduced to its letters, marks and digits (Devanagari vowel signs are marks, so they stay)."""
    w = unicodedata.normalize("NFC", w).lower()
    return "".join(ch for ch in w if unicodedata.category(ch)[0] in "LMN" or ch == "'")


def is_word(w: str) -> bool:
    return bool(token(w))


def script_of(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        if unicodedata.category(ch)[0] == "L":
            name = unicodedata.name(ch, "")
            sc = "devanagari" if "DEVANAGARI" in name else "latin" if "LATIN" in name else "other"
            counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def junk_share(words: list[dict]) -> float:
    """Share of words that are whisper noise: a character or two repeating many times."""
    bad = sum(1 for w in words if re.search(r"(.{1,3})\1{4,}", w["w"]))
    return bad / max(1, len(words))


def flat_words(tr: dict | None) -> list[dict]:
    return [w for s in (tr or {}).get("segments", []) for w in s.get("words", []) if is_word(w["w"])]


def caption_words(ctx: Ctx) -> list[str]:
    """Words shown in the lower half at 1 fps (most reels caption what is said): to check a transcript against."""
    times = list(np.arange(0.5, ctx.duration, 1.0))[:120]
    frames = ctx.frames_at(times, long_edge=720)
    out = []
    for r in ctx.vision("text", frames):
        for line in r.get("lines", []) or []:
            box = line.get("box") or [0, 0, 0, 0]
            if box[1] > 0.4 and len(line["text"]) <= 40:
                out.extend(token(t) for t in line["text"].split() if token(t))
    return out


def agreement(words: list[dict], caps: list[str]) -> float | None:
    if not caps:
        return None
    vocab = {token(w["w"]) for w in words}
    return round(sum(1 for c in caps if c in vocab) / len(caps), 2)


def choose_transcript(ctx: Ctx) -> tuple[list[dict], dict]:
    """vu's transcript, or a better one; returns (words, info). Writes teardown/words.json when it replaces."""
    tr = ctx.v.transcript()
    words = flat_words(tr)
    info = {"source": f"whisper {tr.get('model', 'small') if tr else '-'} ({(tr or {}).get('language') or 'auto'})",
            "language": (tr or {}).get("language"), "replaced": False}
    if not words:
        return words, info | {"mean_conf": None, "why": "no speech found"}
    conf = statistics.mean(w["p"] for w in words)
    caps = caption_words(ctx)
    agree = agreement(words, caps)
    junk = junk_share(words)
    info |= {"mean_conf": round(conf, 2), "caption_agreement": agree, "junk_share": round(junk, 2)}
    cap_script = script_of(" ".join(caps))
    mismatch = bool(caps) and cap_script != "none" and script_of(" ".join(w["w"] for w in words)) != cap_script
    poor = conf < 0.55 or junk > 0.05 or mismatch
    if not poor:
        return words, info | {"why": "kept: confident and consistent with the captions"}
    eprint("delivery: the transcript looks poor; trying whisper in English …")
    alt_f = ctx.out / "transcript-small-en.json"
    alt = json.loads(alt_f.read_text()) if alt_f.exists() else vu.whisper(ctx.v.path, "small", "en", name=ctx.v.meta["name"])
    if alt:
        alt_f.write_text(json.dumps(alt))
    aw = flat_words(alt)
    if aw:
        a_conf = statistics.mean(w["p"] for w in aw)
        a_agree = agreement(aw, caps)
        better = a_conf > conf + 0.1 and junk_share(aw) <= junk and (a_agree is None or agree is None or a_agree >= agree)
        if better:
            translated = info["language"] not in (None, "en")
            ctx.save("words", {"source": "whisper small (en)", "language": "en", "words": aw,
                               "spoken_language": info["language"], "translated": translated})
            why = (f"replaced: the {info['language']} run was poor (confidence {conf:.2f}, caption agreement {agree}); "
                   f"English run confidence {a_conf:.2f}, caption agreement {a_agree}")
            if translated:
                why += ". The speech is not English, so these words are whisper's English rendering: counts and pace are approximate"
            return aw, info | {"source": "whisper small (en)", "replaced": True, "translated": translated,
                               "mean_conf": round(a_conf, 2), "caption_agreement": a_agree, "why": why}
    return words, info | {"why": "poor, and no better run found: read the words with care"}


# ─── pitch and loudness ──────────────────────────────────────────────────────────────────────────

def frames_of(x: np.ndarray) -> np.ndarray:
    n = max(0, 1 + (len(x) - WIN) // HOP)
    idx = np.arange(n)[:, None] * HOP + np.arange(WIN)[None, :]
    return x[idx] if n else np.zeros((0, WIN), np.float32)


def yin(x: np.ndarray, fmin: float = 70, fmax: float = 400, thresh: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Pitch in Hz per 10 ms frame (0 when unvoiced) and the aperiodicity (lower = clearer pitch)."""
    fr = frames_of(x).astype(np.float64)
    if not len(fr):
        return np.zeros(0), np.zeros(0)
    tmax, tmin = int(SR / fmin), int(SR / fmax)
    F = np.fft.rfft(fr, 2 * WIN, axis=1)
    r = np.fft.irfft(F * np.conj(F), axis=1)[:, :tmax + 2]
    cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(tmax + 2)
    e1 = cs[:, WIN - taus]                       # sum of x_j^2 for j < W - tau
    e2 = cs[:, WIN:WIN + 1] - cs[:, taus]        # sum of x_{j+tau}^2 for j < W - tau
    d = e1 + e2 - 2 * r
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    dp = np.ones_like(d)
    dp[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cm, 1e-12)
    f0 = np.zeros(len(fr))
    ap = np.ones(len(fr))
    seg = dp[:, tmin:tmax]
    for i in range(len(fr)):
        row = seg[i]
        below = np.nonzero(row < thresh)[0]
        if len(below):
            k = below[0]
            while k + 1 < len(row) and row[k + 1] < row[k]:
                k += 1
        else:
            k = int(np.argmin(row))
        if 0 < k < len(row) - 1:
            a, b, c = row[k - 1], row[k], row[k + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        else:
            off = 0.0
        ap[i] = row[k]
        f0[i] = SR / (tmin + k + off)
    energy = (fr ** 2).mean(axis=1)
    quiet = energy < 1e-5 * max(1e-12, energy.max())
    f0[(ap > 0.3) | quiet] = 0.0
    return f0, ap


def loudness_db(x: np.ndarray) -> np.ndarray:
    fr = frames_of(x)
    return 10 * np.log10(np.maximum((fr.astype(np.float64) ** 2).mean(axis=1), 1e-12)) if len(fr) else np.zeros(0)


def ftime(i) -> np.ndarray:
    """Frame index -> the time at its centre."""
    return (np.asarray(i) * HOP + WIN / 2) / SR


def fidx(t: float) -> int:
    return int(round((t * SR - WIN / 2) / HOP))


def pauses_from_sound(loud: np.ndarray, words: list[dict]) -> list[list[float]]:
    """[start, length] of each pause of 80 ms+ between the first and last word: frames 12 dB under the voice."""
    if not len(loud) or not words:
        return []
    n = len(loud)
    vl = np.array([np.percentile(loud[max(0, k - 100):k + 100], 90) for k in range(0, n, 20)])
    vl = np.interp(np.arange(n), np.arange(0, n, 20), vl)
    dip = loud < vl - 12
    a0, b0 = fidx(words[0]["a"]), fidx(words[-1]["b"])
    out, k = [], max(0, a0)
    while k < min(n, b0):
        if dip[k]:
            j = k
            while j < n and dip[j]:
                j += 1
            if j - k >= 8:
                out.append([r3(ftime(k)), r3((j - k) * HOP / SR)])
            k = j
        else:
            k += 1
    return out


# ─── structure ───────────────────────────────────────────────────────────────────────────────────

def sentences(words: list[dict]) -> list[dict]:
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        end = w["w"].rstrip().endswith((".", "?", "!", "।")) or nxt is None or nxt["a"] - w["b"] >= 0.6
        if end:
            a, b = cur[0]["a"], cur[-1]["b"]
            n = len(cur)
            out.append({"a": r3(a), "b": r3(b), "text": " ".join(x["w"] for x in cur).strip(), "words": n,
                        "wpm": round(n / max(0.3, b - a) * 60), "question": cur[-1]["w"].rstrip().endswith("?")})
            cur = []
    return out


def _mm_lang():
    """motionmaxxing's language packs (lib/mm/lang.py) when this runs inside it: their curated fillers win."""
    for up in Path(__file__).resolve().parents[:6]:
        if (up / "lib" / "mm" / "lang.py").exists():
            sys.path.insert(0, str(up / "lib"))
            try:
                from mm import lang  # type: ignore
                return lang
            except Exception:
                return None
    return None


def filler_sets() -> tuple[set[str], set[str]]:
    singles = set().union(FILLERS["en"], FILLERS["de"], FILLERS["hi"])
    doubles = set().union(FILLERS["en2"], FILLERS["de2"], FILLERS["hi2"])
    lang = _mm_lang()
    if lang is not None:
        for p in getattr(lang, "PACKS", {}).values():
            for f in list(getattr(p, "hard_fillers", []) or []) + list(getattr(p, "soft_fillers", []) or []):
                f = unicodedata.normalize("NFC", f).lower()
                (doubles if " " in f else singles).add(f)
    return singles, doubles


def filler_counts(words: list[dict]) -> tuple[dict, int]:
    toks = [token(w["w"]) for w in words]
    singles, doubles = filler_sets()
    counts: dict[str, int] = {}
    # "so", "also", "to", "ja" are fillers only at the start of a phrase; elsewhere they are plain words
    weak = {"so", "also", "to", "toh", "ja", "right", "okay", "like", "na", "ne", "genau", "bas", "eigentlich"}
    for i, tk in enumerate(toks):
        starts = (i == 0 or words[i]["a"] - words[i - 1]["b"] > 0.25 or words[i - 1]["w"].rstrip().endswith((".", "?", "!", ","))
                  or toks[i - 1] in singles)   # right after a filler is still the start of the phrase
        if tk in singles and (tk not in weak or starts):
            counts[tk] = counts.get(tk, 0) + 1
        if i + 1 < len(toks) and f"{tk} {toks[i + 1]}" in doubles:
            counts[f"{tk} {toks[i + 1]}"] = counts.get(f"{tk} {toks[i + 1]}", 0) + 1
    restarts = sum(1 for i in range(1, len(toks)) if toks[i] and toks[i] == toks[i - 1] and len(toks[i]) > 1)
    return counts, restarts


def repeats(words: list[dict], n: int = 3) -> list[dict]:
    toks = [token(w["w"]) for w in words]
    seen: dict[tuple, list[float]] = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        if all(g):
            seen.setdefault(g, []).append(words[i]["a"])
    return [{"phrase": " ".join(g), "times": ts} for g, ts in seen.items() if len(ts) >= 2][:10]


# ─── body language ───────────────────────────────────────────────────────────────────────────────

def body(ctx: Ctx, emphasis: list[dict]) -> tuple[dict, list[dict]]:
    faces = ctx.faces_track()
    with_face = [f for f in faces if f["faces"]]
    out: dict = {"face_pct": pct(len(with_face), len(faces))}
    if with_face:
        look = [f for f in with_face if abs(f["faces"][0].get("yaw") or 0) < 15 and abs(f["faces"][0].get("pitch") or 0) < 20]
        out["eye_contact_pct"] = pct(len(look), len(with_face))
        sizes = [f["faces"][0]["box"][2] for f in with_face]
        out["face_width_pct"] = {"median": round(100 * float(np.median(sizes)), 1), "p10": round(100 * float(np.percentile(sizes, 10)), 1),
                                 "p90": round(100 * float(np.percentile(sizes, 90)), 1)}
        moves = []
        for f0, f1 in zip(faces, faces[1:]):
            if f0["faces"] and f1["faces"] and f1["t"] - f0["t"] < 1.0:
                b0, b1 = f0["faces"][0]["box"], f1["faces"][0]["box"]
                c0, c1 = (b0[0] + b0[2] / 2, b0[1] + b0[3] / 2), (b1[0] + b1[2] / 2, b1[1] + b1[3] / 2)
                moves.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]) / max(1e-3, b0[2]) / (f1["t"] - f0["t"])))
        if moves:
            # cuts move the face too; the median keeps the speaker's own movement
            out["head_motion"] = round(float(np.median(moves)), 2)
    # hands at 4 fps inside the sections
    times = [t for s in ctx.sections for t in np.arange(s["a"] + 0.1, s["b"], 0.25)][:600]
    if not times:
        return out, []
    eprint(f"delivery: hands and gestures at {len(times)} moments …")
    poses = ctx.vision("pose", ctx.frames_at(times, long_edge=640))
    face_w = float(np.median([f["faces"][0]["box"][2] for f in with_face])) if with_face else 0.25
    cents: list[tuple[float, float] | None] = []
    for p in poses:
        pts = [(j[0], j[1]) for h in p.get("hands", []) or [] for j in (h.get("joints") or {}).values() if j[2] > 0.3]
        cents.append((float(np.mean([q[0] for q in pts])), float(np.mean([q[1] for q in pts]))) if len(pts) >= 4 else None)
    out["hands_pct"] = pct(sum(1 for c in cents if c), len(cents))
    speed = np.zeros(len(times))
    for i in range(1, len(times)):
        if cents[i] and cents[i - 1] and times[i] - times[i - 1] < 0.6:
            speed[i] = np.hypot(cents[i][0] - cents[i - 1][0], cents[i][1] - cents[i - 1][1]) / face_w / (times[i] - times[i - 1])
    gestures = []
    for i in range(1, len(times) - 1):
        if speed[i] > 1.5 and speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1]:
            if not gestures or times[i] - gestures[-1]["t"] > 0.5:
                gestures.append({"t": r3(times[i]), "speed": round(float(speed[i]), 2)})
    seen_s = len(times) * 0.25
    out["gestures_per_min"] = round(len(gestures) / max(1e-6, seen_s / 60), 1)
    if emphasis:
        out["gesture_on_emphasis_pct"] = pct(sum(1 for e in emphasis if near(e["t"], [g["t"] for g in gestures], 0.3)), len(emphasis))
    return out, gestures


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def run(ctx: Ctx) -> dict:
    words, tinfo = choose_transcript(ctx)
    d = ctx.duration
    if not words:
        return {"summary": ["No speech."], "transcript": tinfo, "language": None, "pace": {}, "pauses": {}, "fillers": {},
                "prosody": {}, "emphasis": [], "sentences": [], "hook": {}, "script": {}, "body": body(ctx, [])[0], "evidence": []}
    x = ctx.audio(SR)
    eprint("delivery: pitch and loudness …")
    f0, _ = yin(x)
    loud = loudness_db(x)
    voiced = f0 > 0
    ref = float(np.median(f0[voiced])) if voiced.any() else 150.0
    st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1) / ref), np.nan)

    # pace
    n = len(words)
    speak = sum(w["b"] - w["a"] for w in words)
    span = words[-1]["b"] - words[0]["a"]
    rolling = []
    for t in np.arange(0, max(0.0, d - 5) + 1e-6, 1.0):
        k = sum(1 for w in words if t <= w["a"] < t + 5)
        rolling.append([round(t + 2.5, 1), round(k * 12)])
    rv = [r for r in rolling if r[1] > 0]
    pace = {"wpm": round(n / d * 60) if d else 0, "wpm_speaking": round(n / max(1e-6, span) * 60),
            "speaking_pct": pct(speak, d), "rolling": rolling,
            "fastest": max(rv, key=lambda r: r[1]) if rv else None, "slowest": min(rv, key=lambda r: r[1]) if rv else None,
            "approximate": bool(tinfo.get("translated"))}

    # pauses (from the sound)
    plist = pauses_from_sound(loud, words)
    lens = sorted(p[1] for p in plist)
    pauses = {"n": len(plist), "per_min": round(len(plist) / (span / 60), 1) if span else 0,
              "median_s": r3(np.median(lens)) if lens else None,
              "max_pause_s": r3(lens[int(0.9 * (len(lens) - 1))]) if lens else None,
              "longest_s": r3(lens[-1]) if lens else None, "list": plist}

    # per-word prosody and stress
    feats = []
    for i, w in enumerate(words):
        a, b = max(0, fidx(w["a"])), max(fidx(w["a"]) + 1, fidx(w["b"]))
        seg_st, seg_ld = st[a:b], loud[a:b]
        tk = token(w["w"])
        if not len(seg_ld) or len(tk) < 2:
            continue
        p_peak = float(np.nanmax(seg_st)) if np.isfinite(seg_st).any() else np.nan
        gap = w["a"] - words[i - 1]["b"] if i else 0.0
        feats.append({"i": i, "t": w["a"], "w": w["w"], "pitch": p_peak, "loud": float(np.percentile(seg_ld, 90)),
                      "dur": w["b"] - w["a"], "chars": len(tk), "gap": gap})
    emphasis = []
    if len(feats) >= 8:
        per_char = np.median([f["dur"] / f["chars"] for f in feats])
        for f in feats:
            f["stretch"] = f["dur"] / max(1e-3, per_char * f["chars"])

        def z(key):
            v = np.array([f[key] for f in feats], float)
            m, s_ = np.nanmean(v), np.nanstd(v) or 1.0
            return np.nan_to_num((v - m) / s_)
        score = z("pitch") + z("loud") + 0.7 * z("stretch") + np.array([0.5 if f["gap"] >= 0.25 else 0.0 for f in feats])
        fill = set().union(FILLERS["en"], FILLERS["de"], FILLERS["hi"])
        order = np.argsort(-score)
        k = max(3, len(feats) // 10)
        for j in order[:k]:
            f = feats[j]
            if score[j] < 1.0 or token(f["w"]) in fill:
                continue
            emphasis.append({"t": r3(f["t"]), "w": f["w"], "score": round(float(score[j]), 2),
                             "pitch_st": None if np.isnan(f["pitch"]) else round(f["pitch"], 1),
                             "loud_db": round(f["loud"], 1), "stretch": round(f["stretch"], 2), "pause_before": f["gap"] >= 0.25})
        emphasis.sort(key=lambda e: e["t"])
    vs = st[np.isfinite(st)]
    energy = [[round(t, 1), round(float(np.percentile(loud[max(0, fidx(t)):fidx(t + 1)], 90)), 1)]
              for t in np.arange(0, d - 0.5, 1.0) if fidx(t + 1) <= len(loud) and fidx(t + 1) > max(0, fidx(t))]
    sents = sentences(words)
    rises = 0
    for s_ in sents:
        a, b = fidx(s_["b"] - 0.4), fidx(s_["b"])
        seg = st[max(0, a):max(0, b)]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 6 and np.polyfit(np.arange(len(seg)), seg, 1)[0] > 0.08:
            rises += 1
    hook_e = [e[1] for e in energy if e[0] < 3]
    rest_e = [e[1] for e in energy if e[0] >= 3]
    prosody = {"pitch_median_hz": round(ref), "range_st": round(float(np.percentile(vs, 95) - np.percentile(vs, 5)), 1) if len(vs) else None,
               "variability_st": round(float(np.std(vs)), 2) if len(vs) else None,
               "energy_curve": energy,
               "hook_energy_vs_rest_db": round(float(np.mean(hook_e) - np.mean(rest_e)), 1) if hook_e and rest_e else None,
               "sentence_end_rises": rises}
    fillers, restarts = filler_counts(words)
    toks = [token(w["w"]) for w in words]
    nums = [w["w"] for w in words if re.search(r"\d", w["w"]) or token(w["w"]) in NUMBER_WORDS]
    cta = [s_ for s_ in sents if CTA.search(s_["text"])]
    beats = [{"t": p[0], "why": f"pause {p[1]:.2f} s"} for p in plist if p[1] >= 0.4]
    first3 = [w for w in words if w["a"] < words[0]["a"] + 3]
    hook = {"first_word_s": r3(words[0]["a"]), "text": sents[0]["text"] if sents else "",
            "wpm_first_3s": round(len(first3) / 3 * 60), "energy_db": round(float(np.mean(hook_e)), 1) if hook_e else None}
    script = {"questions": sum(1 for s_ in sents if s_["question"]),
              "address_per_100": round(100 * sum(1 for t in toks if t in YOU) / max(1, n), 1),
              "numbers": nums[:20], "repeats": repeats(words), "cta": cta[-1] if cta else None, "beats": beats}
    bd, gestures = body(ctx, emphasis)
    minutes = d / 60 if d else 1
    summary = [
        f"Transcript: {tinfo['source']}, confidence {tinfo.get('mean_conf')}"
        + (f"; {tinfo['why']}" if tinfo.get("replaced") or "poor" in tinfo.get("why", "") else "") + ".",
        f"Pace {pace['wpm']} words/min over the whole video ({pace['wpm_speaking']} while speaking)"
        + (" — approximate, from a translation" if pace["approximate"] else "")
        + (f"; fastest {pace['fastest'][1]} around {pace['fastest'][0]} s, slowest {pace['slowest'][1]} around {pace['slowest'][0]} s" if rv else "") + ".",
        f"Pauses: {pauses['n']} ({pauses['per_min']}/min while talking), median {pauses['median_s']} s, "
        f"typical longest (p90) {pauses['max_pause_s']} s, longest {pauses['longest_s']} s.",
        f"Voice: median pitch {prosody['pitch_median_hz']} Hz, range {prosody['range_st']} semitones, "
        f"variability {prosody['variability_st']} st" + (f"; the first 3 s are {prosody['hook_energy_vs_rest_db']:+.1f} dB against the rest"
                                                         if prosody["hook_energy_vs_rest_db"] is not None else "") + ".",
    ]
    if emphasis:
        summary.append("Stressed words: " + ", ".join(f"“{e['w']}” {e['t']:.1f} s" for e in emphasis[:8]) + ".")
    fl = sum(fillers.values())
    summary.append(f"Fillers {fl / minutes:.1f}/min" + (f" ({', '.join(f'{k} ×{v}' for k, v in sorted(fillers.items(), key=lambda kv: -kv[1])[:5])})" if fl else "")
                   + f", restarts {restarts}; {script['questions']} questions; you-address {script['address_per_100']} per 100 words.")
    summary.append(f"Hook: first word at {hook['first_word_s']} s, {hook['wpm_first_3s']} words/min in the first 3 s: “{hook['text'][:90]}”.")
    if script["cta"]:
        summary.append(f"Call to action at {script['cta']['a']:.1f} s: “{script['cta']['text'][:90]}”.")
    if bd.get("face_pct") is not None:
        summary.append(f"Body (proxies): face on screen {bd['face_pct']}%"
                       + (f", looking at the camera {bd['eye_contact_pct']}%" if "eye_contact_pct" in bd else "")
                       + (f", hands in frame {bd['hands_pct']}%, {bd['gestures_per_min']} gestures/min" if "hands_pct" in bd else "")
                       + (f", {bd['gesture_on_emphasis_pct']}% of stressed words with a gesture" if "gesture_on_emphasis_pct" in bd else "") + ".")
    evidence = [{"t": 0.0, "what": "pitch (dots), loudness (line) and pace (bars), stressed words in red", "img": plot(ctx, st, loud, rolling, emphasis)}]
    if gestures:
        g = max(gestures, key=lambda g: g["speed"])
        evidence.append({"t": g["t"], "what": "the strongest gesture", "img": ctx.strip("delivery-gesture.jpg", [g["t"] - 0.25, g["t"], g["t"] + 0.25], long_edge=320)})
    return {"summary": summary, "transcript": tinfo, "language": tinfo.get("language"), "pace": pace, "pauses": pauses,
            "fillers": {"per_min": round(fl / minutes, 1), "words": fillers, "restarts": restarts}, "prosody": prosody,
            "emphasis": emphasis, "sentences": sents, "hook": hook, "script": script, "body": bd, "gestures": gestures,
            "evidence": evidence}


def plot(ctx: Ctx, st: np.ndarray, loud: np.ndarray, rolling: list, emphasis: list[dict]) -> str:
    from PIL import Image, ImageDraw
    W, H = 1200, 260
    img = Image.new("RGB", (W, H), (252, 251, 248))
    dr = ImageDraw.Draw(img)
    d = max(ctx.duration, 1e-6)
    # pace bars along the bottom
    mx = max([r[1] for r in rolling] + [1])
    for t, wpm in rolling:
        x = int(t / d * W)
        dr.rectangle([x - 2, H - 4 - int(wpm / mx * 50), x + 2, H - 4], fill=(200, 220, 255))
    # loudness line
    if len(loud):
        lo, hi = np.percentile(loud, 5), np.percentile(loud, 99)
        pts = [(int(ftime(i) / d * W), int(150 - (min(max(v, lo), hi) - lo) / (hi - lo + 1e-9) * 70)) for i, v in enumerate(loud[::5] if len(loud) > 5 else loud) for i in [i * 5]]
        dr.line(pts, fill=(160, 160, 165), width=1)
    # pitch dots
    for i in range(0, len(st), 2):
        if np.isfinite(st[i]):
            x, y = int(ftime(i) / d * W), int(60 - np.clip(st[i], -12, 12) * 4)
            dr.point((x, y), fill=(10, 132, 255))
    for e in emphasis:
        x = int(e["t"] / d * W)
        dr.line([(x, 8), (x, H - 60)], fill=(255, 59, 48), width=1)
        dr.text((x + 2, 8), e["w"][:14], fill=(255, 59, 48))
    return ctx.img("delivery-voice.png", np.asarray(img))


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    t = np.arange(int(3 * SR)) / SR
    f = 120 + 60 * t / 3                      # a glide 120 -> 180 Hz
    x = (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)
    x[int(1.4 * SR):int(1.7 * SR)] = 0        # a 0.3 s pause
    f0, _ = yin(x)
    got = f0[fidx(0.5)], f0[fidx(2.5)]
    assert abs(got[0] - 130) < 4 and abs(got[1] - 170) < 4, got
    assert f0[fidx(1.55)] == 0
    loud = loudness_db(x)
    words = [{"w": "eins", "a": 0.1, "b": 1.3}, {"w": "zwei", "a": 1.75, "b": 2.9}]
    pz = pauses_from_sound(loud, words)
    assert len(pz) == 1 and abs(pz[0][0] - 1.4) < 0.05 and abs(pz[0][1] - 0.3) < 0.06, pz
    ws = [{"w": w, "a": i * 0.3, "b": i * 0.3 + 0.25} for i, w in enumerate("Ähm also ich ich glaube, so you know das ist es.".split())]
    fl, rs = filler_counts(ws)
    assert fl == {"ähm": 1, "also": 1, "so": 1, "you know": 1} and rs == 1, (fl, rs)
    assert token("की") == "की" and token("Rock-n-Roll!") == "rocknroll"
    assert script_of("तो मेंने") == "devanagari" and script_of("Poori skill") == "latin"
    assert junk_share([{"w": "अपिस्ट्ट्ट्ट्ट्ट्ट्ट्"}, {"w": "ok"}]) == 0.5
    s = sentences([{"w": "Hi.", "a": 0, "b": .3}, {"w": "Was", "a": .4, "b": .6}, {"w": "geht?", "a": .7, "b": 1.0}])
    assert len(s) == 2 and s[1]["question"]
    print("delivery demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
