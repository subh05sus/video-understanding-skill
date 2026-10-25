"""typography: every piece of text on screen and how it is set. Captions above all (the style.json "captions"
block), then titles, labels, calls to action, and the text inside screenshots.

How it measures:
  - Text track: Apple Vision reads 4 frames a second; lines with the same words in the same place are one
    text item with a start, an end and a box.
  - Roles: captions are short-lived lines in one band that say what is being said (matched against the words
    with their times) or, when the transcript can't be matched (a translation), that keep changing in the same
    band; UI text is small or comes in blocks; a title is large and holds; a call to action asks for something.
  - Style, on full-resolution crops: the fill colour (the colour farthest from the surroundings), an outline (a
    ring of one colour hugging the letters), a drop shadow (a darker copy of the letters at an offset), a box
    (a plain area of one colour around the text), a highlight colour (a word in another colour), case, size
    (ink height turned into a font size with the matched font's metrics) and the font (fonts.py, by rendering).
  - Entrance: a caption's first frames at the video's own rate: when it appears, and whether it pops (scale),
    fades (contrast) or slides (position), fitted with ease.fit. Sync: when it appears against when its word
    is spoken.

    python3 typography.py demo
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ease  # noqa: E402
import fonts  # noqa: E402
from common import Ctx, eprint, hexcolor, pct, r3  # noqa: E402

VERSION = 1
FPS_TRACK = 4.0
CTA = re.compile(r"comment|kommentier|follow|folg|link|save|speicher|share|teil|subscribe|abonnier|dm\b|bio\b", re.I)


def tok(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower()
    out, cur = [], []
    for ch in s:
        if unicodedata.category(ch)[0] in "LMN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


# ─── the text track ──────────────────────────────────────────────────────────────────────────────

def text_track(ctx: Ctx) -> list[dict]:
    got = ctx.load("ocr_track")
    if got and got.get("fps") == FPS_TRACK and got.get("sections") == ctx.sections:
        return got["track"]
    times = [round(float(t), 3) for s in ctx.sections for t in np.arange(s["a"] + 0.05, s["b"], 1 / FPS_TRACK)]
    eprint(f"typography: reading text at {len(times)} moments …")
    track = []
    for k in range(0, len(times), 120):
        chunk = times[k:k + 120]
        frames = ctx.frames_at(chunk, long_edge=960)
        for t, r in zip(chunk, ctx.vision("text", frames)):
            track.append({"t": t, "lines": [l for l in (r.get("lines") or []) if l.get("box") and l["text"].strip()]})
    ctx.save("ocr_track", {"fps": FPS_TRACK, "sections": ctx.sections, "track": track})
    return track


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def items_of(track: list[dict]) -> list[dict]:
    """Lines that persist (same words, same place) become items with a start, an end, a box."""
    open_: list[dict] = []
    done: list[dict] = []
    step = 1 / FPS_TRACK
    for fr in track:
        t = fr["t"]
        seen = set()
        for l in fr["lines"]:
            key = " ".join(tok(l["text"]))
            if not key:
                continue
            hit = next((it for it in open_ if id(it) not in seen and it["key"] == key and iou(it["box"], l["box"]) > 0.3), None)
            if hit:
                hit["b"] = t
                hit["boxes"].append(l["box"])
                hit["n"] += 1
                seen.add(id(hit))
            else:
                it = {"key": key, "text": l["text"], "a": t, "b": t, "box": l["box"], "boxes": [l["box"]], "n": 1,
                      "conf": l.get("conf")}
                open_.append(it)
                seen.add(id(it))
        still = []
        for it in open_:
            if id(it) in seen or t - it["b"] <= step * 1.5:
                still.append(it)
            else:
                done.append(it)
        open_ = still
    done += open_
    for it in done:
        bx = np.median(np.array(it["boxes"]), axis=0)
        it["box"] = [round(float(v), 4) for v in bx]
        it["b"] = round(it["b"] + step, 3)          # it was seen until the next sample
        it["life"] = round(it["b"] - it["a"], 3)
        del it["boxes"]
    return sorted(done, key=lambda x: x["a"])


def classify_roles(items: list[dict], words: list[dict], duration: float, concurrent: dict) -> None:
    spoken = [(w["a"], set(tok(w["w"]))) for w in words]

    def said_near(it) -> float:
        ts = tok(it["text"])
        if not ts:
            return 0.0
        near = set().union(*[s for a, s in spoken if it["a"] - 1.2 <= a <= it["b"] + 0.6]) if spoken else set()
        return sum(1 for x in ts if x in near) / len(ts)

    for it in items:
        it["said"] = round(said_near(it), 2)
        h = it["box"][3]
        it["role"] = "ui" if h < 0.016 or concurrent.get(id(it), 0) >= 5 else None
    # the caption band: where short, mostly spoken lines keep appearing
    cand = [it for it in items if it["role"] is None and it["life"] <= 2.5 and len(tok(it["text"])) <= 7]
    band = None
    if cand:
        ys = np.array([it["box"][1] + it["box"][3] / 2 for it in cand])
        best = None
        for y0 in np.arange(0.05, 0.95, 0.01):
            sel = [it for it, y in zip(cand, ys) if abs(y - y0) < 0.05]
            score = sum(0.5 + it["said"] for it in sel)
            if best is None or score > best[0]:
                best = (score, y0, sel)
        per_min = len(best[2]) / max(1e-6, duration / 60)
        spoken_share = np.mean([it["said"] for it in best[2]]) if best[2] else 0
        if len(best[2]) >= 4 and (spoken_share >= 0.4 or per_min >= 8):
            band = float(np.median([it["box"][1] + it["box"][3] / 2 for it in best[2]]))
    for it in items:
        if it["role"]:
            continue
        yc = it["box"][1] + it["box"][3] / 2
        if band is not None and abs(yc - band) < 0.06 and it["life"] <= 3.0 and len(tok(it["text"])) <= 8:
            it["role"] = "caption"
        elif CTA.search(it["text"]):
            it["role"] = "cta"
        elif it["box"][3] >= 0.03 and it["life"] >= 0.75:
            it["role"] = "title" if (it["a"] < 4.0 or it["box"][3] >= 0.045) else "label"
        else:
            it["role"] = "label"
    return band


# ─── style of a crop ─────────────────────────────────────────────────────────────────────────────

def text_style(frame: np.ndarray, box: list[float]) -> dict | None:
    """Fill, outline, shadow, box and highlight of the text in `box` (fractions) on a full-size frame."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    px, py = int(x * W), int(y * H)
    pw, ph = max(4, int(w * W)), max(4, int(h * H))
    m = int(1.5 * ph)   # wide enough that the crop's edge lies beyond a box behind the text
    x0, y0, x1, y1 = max(0, px - m), max(0, py - m), min(W, px + pw + m), min(H, py + ph + m)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    ch, cw = lab.shape[:2]
    border = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3), lab[:, :2].reshape(-1, 3), lab[:, -2:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # the text box, opened up a little so the letters no longer touch its edge
    e = max(2, int(0.15 * ph))
    ix0, iy0 = max(0, px - x0 - e), max(0, py - y0 - e)
    ix1, iy1 = min(cw, px - x0 + pw + e), min(ch, py - y0 + ph + e)
    inner_img = lab[iy0:iy1, ix0:ix1]
    inner = inner_img.reshape(-1, 3)
    if len(inner) < 30:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(inner.astype(np.float32), 3, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=3) / len(labels)
    lab_img = labels.reshape(inner_img.shape[:2])
    edge = np.concatenate([lab_img[0], lab_img[-1], lab_img[:, 0], lab_img[:, -1]])
    touch = [float((edge == k).mean()) for k in range(3)]
    # the fill: a colour that stands apart from the surroundings and doesn't reach the box's edge (the
    # background and a box behind the text do); of those, the biggest (a shadow is partly hidden behind it)
    ok = [k for k in range(3) if share[k] > 0.05 and touch[k] < 0.12 and np.linalg.norm(centres[k] - bg) > 20]
    if not ok:
        ok = [int(np.argmax([np.linalg.norm(c - bg) if s_ > 0.06 else -1 for c, s_ in zip(centres, share)]))]
    fill_k = max(ok, key=lambda k: share[k])
    if len(ok) >= 2:
        # an outline or a shadow hugs the letters, never the other way round: when one colour lies mostly in
        # the halo of another, the other is the fill
        r_h = max(2, int(0.12 * ph))
        masks = {k: (lab_img == k).astype(np.uint8) for k in ok}
        halo = {k: cv2.dilate(masks[k], np.ones((2 * r_h + 1,) * 2, np.uint8)) > 0 for k in ok}
        best_k, best_hug = None, 0.0
        for a_ in ok:
            for b_ in ok:
                if a_ != b_ and masks[b_].sum() > 10:
                    hug = float((halo[a_] & (masks[b_] > 0)).sum() / masks[b_].sum())
                    back = float((halo[b_] & (masks[a_] > 0)).sum() / max(1, masks[a_].sum()))
                    if hug > 0.7 and hug > back + 0.1 and hug > best_hug:
                        best_k, best_hug = a_, hug
        if best_k is not None:
            fill_k = best_k
    fill = centres[fill_k]
    mask = np.linalg.norm(lab - fill, axis=2) < 22
    region = np.zeros_like(mask)
    region[iy0:iy1, ix0:ix1] = True
    mask &= region          # only the letters in the text's own box, not a white wall beside it
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if mask.sum() < 20:
        return None
    to_rgb = lambda l: hexcolor(cv2.cvtColor(np.uint8([[np.clip(l, 0, 255)]]), cv2.COLOR_LAB2RGB)[0, 0])
    ys, xs = np.nonzero(mask)
    ink_h = int(ys.max() - ys.min() + 1)
    out = {"color": to_rgb(np.median(lab[mask], axis=0)), "ink_h_px": ink_h}
    # outline: a ring hugging the letters, one colour, unlike what lies beyond it
    r_ = max(1, int(round(0.07 * ink_h)))
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((2 * r_ + 1,) * 2, np.uint8)) > 0) & ~mask
    far = (cv2.dilate(mask.astype(np.uint8), np.ones((6 * r_ + 1,) * 2, np.uint8)) > 0) & ~ring & ~mask
    if ring.sum() > 20 and far.sum() > 20:
        rc, fc = np.median(lab[ring], axis=0), np.median(lab[far], axis=0)
        consistent = (np.linalg.norm(lab[ring] - rc, axis=1) < 20).mean()
        out["stroke"] = bool(np.linalg.norm(rc - fc) > 18 and consistent > 0.6 and np.linalg.norm(rc - fill) > 25)
        if out["stroke"]:
            out["stroke_color"] = to_rgb(rc)
            out["stroke_px"] = r_
    else:
        out["stroke"] = False
    # drop shadow: a darker copy of the letters at an offset
    best = None
    rng_ = max(2, int(0.15 * ink_h))
    base_l = float(np.median(lab[far][:, 0])) if far.sum() > 20 else float(bg[0])
    for dy in range(-rng_, rng_ + 1, max(1, rng_ // 4)):
        for dx in range(-rng_, rng_ + 1, max(1, rng_ // 4)):
            if dx == 0 and dy == 0:
                continue
            sh = np.roll(np.roll(mask, dy, axis=0), dx, axis=1) & ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if sh.sum() < 15:
                continue
            dark = base_l - float(lab[sh][:, 0].mean())
            if best is None or dark > best[0]:
                best = (dark, dx, dy)
    opposite = None
    if best:
        dx, dy = best[1], best[2]
        sh = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1) & ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        opposite = base_l - float(lab[sh][:, 0].mean()) if sh.sum() > 15 else 0.0
    # a shadow darkens one side more than the other; even darkening all round is a glow or soft outline
    out["shadow"] = bool(best and best[0] > 10 and (opposite is None or best[0] > opposite + 6))
    if out["shadow"]:
        out["shadow_offset_px"] = [best[1], best[2]]
    elif best and best[0] > 10 and not out.get("stroke"):
        out["glow"] = True
    # box: a plain area of one colour around the text, unlike what is beyond it
    pad = max(2, int(0.35 * ink_h))
    around = np.zeros_like(mask)
    around[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad] = True
    around &= ~cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if around.sum() > 30:
        ac = np.median(lab[around], axis=0)
        uniform = (np.linalg.norm(lab[around] - ac, axis=1) < 12).mean()
        out["box"] = bool(uniform > 0.8 and np.linalg.norm(ac - bg) > 12 and np.linalg.norm(ac - fill) > 25)
        if out["box"]:
            out["box_color"] = to_rgb(ac)
    # highlight: a second, saturated colour in the letters
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # away from anti-aliased edges
    g = lab[core] if core.sum() > 60 else lab[mask]
    if len(g) > 60:
        _, gl, gc = cv2.kmeans(g.astype(np.float32), 2, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        gs = np.bincount(gl.ravel(), minlength=2) / len(gl)
        if np.linalg.norm(gc[0] - gc[1]) > 35 and gs.min() > 0.15:
            sat = [np.hypot(c[1] - 128, c[2] - 128) for c in gc]
            k = int(np.argmax(sat))
            if sat[k] > 30:
                out["highlight_color"] = to_rgb(gc[k])
                out["color"] = to_rgb(gc[1 - k])
    out["mask"] = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return out


def case_of(texts: list[str]) -> str | None:
    letters = "".join(ch for t in texts for ch in t if ch.isalpha())
    if not letters:
        return None
    up = sum(ch.isupper() for ch in letters) / len(letters)
    if up > 0.9:
        return "upper"
    if up < 0.05:
        return "lower"
    starts = [t.strip()[0] for t in texts if t.strip() and t.strip()[0].isalpha()]
    return "sentence" if starts and sum(c.isupper() for c in starts) / len(starts) > 0.5 else "mixed"


# ─── entrance at the video's own rate ────────────────────────────────────────────────────────────

def entrance(ctx: Ctx, it: dict) -> dict | None:
    """When a caption really appears and how. Its own letters (from a frame while it is fully up) are looked
    for in each frame from 0.3 s before it was first read, at several scales: the first frame they show is when
    it appears; their scale, match strength and position over the next frames say pop, fade or slide."""
    x, y, w, h = it["box"]
    end = max(it["a"] + 0.05, it["b"] - 0.6 / FPS_TRACK)        # still this caption, not the next one
    a = max(0.0, it["a"] - 0.3)
    ts, fr = ctx.clip(a, min(ctx.duration, end), long_edge=720)
    if len(fr) < 4:
        return None
    H, W = fr.shape[1:3]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]
    tx0, ty0 = max(0, int(x * W)), max(0, int(y * H))
    tx1, ty1 = min(W, int((x + w) * W) + 1), min(H, int((y + h) * H) + 1)
    tmpl = gray[-1][ty0:ty1, tx0:tx1]
    if tmpl.size < 60 or tmpl.std() < 8:
        return None
    m = 0.6
    X0, Y0 = max(0, int((x - w * m) * W)), max(0, int((y - h * 2 * m) * H))
    X1, Y1 = min(W, int((x + w * (1 + m)) * W)), min(H, int((y + h * (1 + 2 * m)) * H))
    scales = np.arange(0.4, 1.31, 0.05)
    rows = []
    for t, g in zip(ts, gray):
        region = g[Y0:Y1, X0:X1]
        best = (-1.0, 1.0, 0.0)
        for sc in scales:
            th, tw = max(4, int(tmpl.shape[0] * sc)), max(4, int(tmpl.shape[1] * sc))
            if th >= region.shape[0] or tw >= region.shape[1]:
                continue
            r = cv2.matchTemplate(region, cv2.resize(tmpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (float(mx), float(sc), float(loc[1] + th / 2))
        rows.append((float(t), *best))
    score = np.array([r[1] for r in rows])
    seen = np.nonzero(score > 0.6)[0]
    if not len(seen):
        return None
    k0 = int(seen[0])
    out = {"t_in": r3(rows[k0][0])}
    tt = np.array([r[0] for r in rows[k0:]])
    sc_ = np.array([r[2] for r in rows[k0:]])
    sco = score[k0:]
    yy = np.array([r[3] for r in rows[k0:]])
    if len(tt) >= 3 and (sc_[0] < 0.9 or sc_.max() > 1.06):
        fit = ease.fit(tt, sc_, end_value=1.0, frame=1 / ctx.fps)
        out.update(kind="pop", scale_from=round(float(sc_[0]), 2), overshoot=round(float(max(0.0, sc_.max() - 1)), 2),
                   ease=fit["ease"] if fit else None, duration_s=fit["duration"] if fit else None)
    elif len(tt) >= 3 and abs(yy[0] - yy[-1]) > 0.25 * tmpl.shape[0]:
        fit = ease.fit(tt, yy, frame=1 / ctx.fps)
        out.update(kind="slide", ease=fit["ease"] if fit else None, duration_s=fit["duration"] if fit else None)
    elif len(tt) >= 3 and sco[0] < 0.8 and sco[min(2, len(sco) - 1)] > sco[0] + 0.1:
        fit = ease.fit(tt, sco, frame=1 / ctx.fps)
        out.update(kind="fade", ease=fit["ease"] if fit else None, duration_s=fit["duration"] if fit else None)
    else:
        out.update(kind="none", duration_s=0.0)
    return out


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def style_block(ctx: Ctx, its: list[dict], n: int = 6, match_font: bool = True) -> dict:
    """The common style of a set of items, from full-size crops of up to n of them."""
    pick = sorted(its, key=lambda it: -it["box"][3])[:max(n * 2, n)]
    pick = sorted(pick, key=lambda it: it["a"])[:: max(1, len(pick) // n)][:n]
    frames = ctx.frames_at([min(it["b"] - 0.1, it["a"] + 0.25) for it in pick], long_edge=None)
    styles = []
    for it, f in zip(pick, frames):
        st = text_style(f, it["box"])
        if st:
            styles.append((it, st))
    if not styles:
        return {}

    def vote(key, default=None):
        vals = [s.get(key) for _, s in styles if s.get(key) is not None]
        return max(set(vals), key=vals.count) if vals else default
    out = {"color": vote("color"), "stroke": vote("stroke", False), "shadow": vote("shadow", False),
           "box": vote("box", False), "glow": vote("glow", False)}
    if out["stroke"]:
        out["stroke_color"] = vote("stroke_color")
    if out["box"]:
        out["box_color"] = vote("box_color")
    hl = [s.get("highlight_color") for _, s in styles if s.get("highlight_color")]
    out["highlight_color"] = max(set(hl), key=hl.count) if len(hl) >= max(2, len(styles) // 4) else out["color"]
    sh = [s["shadow_offset_px"] for _, s in styles if s.get("shadow_offset_px")]
    if sh:
        out["shadow_offset_px"] = [int(np.median([v[0] for v in sh])), int(np.median([v[1] for v in sh]))]
    ink = float(np.median([s["ink_h_px"] for _, s in styles]))
    out["ink_h_px"] = round(ink)
    if match_font:
        # only clean samples: as many pieces as letters (give or take), sensible ink density and proportions,
        # so a crop with a hand over it or a misread word doesn't drag the match down
        def clean(it, s_):
            mk = s_.get("mask")
            n_l = sum(ch.isalnum() for ch in it["text"])
            if mk is None or n_l == 0:
                return False
            n_cc = cv2.connectedComponents(mk.astype(np.uint8), connectivity=8)[0] - 1
            dens = float(mk.mean())
            asp = mk.shape[1] / max(1, mk.shape[0])
            return 0.5 * n_l <= n_cc <= 2 * n_l + 2 and 0.12 <= dens <= 0.75 and 0.25 * n_l <= asp <= 1.4 * n_l
        good = [(it, s_) for it, s_ in styles if clean(it, s_)] or styles
        samples = [(s_["mask"], it["text"]) for it, s_ in good if s_.get("mask") is not None]
        fm = fonts.match(samples)
        out["font_candidates"] = fm["candidates"]
        out["fonts_offline"] = fm["offline"]
        out["font_quality"] = fm["quality"]
        out["font_style"] = fm["style_class"]
        bg_ = fm["best_google"]
        best = fm["best"]
        out["font_match"] = bg_["font"] if bg_ and best and bg_["score"] >= 0.85 * best["score"] else (best or {}).get("font")
        out["weight"] = (bg_ or best or {}).get("weight") if out["font_match"] == (bg_ or {}).get("font") else (best or {}).get("weight")
        out["font_best_overall"] = best
        # font size from the matched font's own metrics: ink height of these words at 100 px
        cand = next((c for c in fonts.candidates(online=not fm["offline"]) if c["font"] == out["font_match"]
                     and c["weight"] == out["weight"]), None)
        if cand:
            ratios = []
            for it, s in styles:
                f100 = fonts._font(cand["path"], cand.get("index", 0), 100, cand["weight"], bool(cand.get("variable")))
                bb = f100.getbbox(it["text"])
                if bb[3] - bb[1] > 0:
                    ratios.append(s["ink_h_px"] / (bb[3] - bb[1]) * 100)
            if ratios:
                out["font_px"] = round(float(np.median(ratios)))
        out["font_samples"] = len(samples)
        out["comparison"] = (good[0][0]["text"], good[0][1]["mask"], fm["candidates"])
    return out


def run(ctx: Ctx) -> dict:
    track = text_track(ctx)
    items = items_of(track)
    # how many lines share a moment with each item, to tell blocks of UI text from a lone line
    concurrent = {}
    for it in items:
        concurrent[id(it)] = max((sum(1 for l in fr["lines"] if abs((l["box"][1]) - it["box"][1]) < 0.25)
                                  for fr in track if it["a"] <= fr["t"] <= it["b"]), default=0)
    words = ctx.words()
    band = classify_roles(items, words, ctx.duration, concurrent)
    # text on the speaker (a hoodie's print) belongs to the scene, not to the edit's graphics
    check = ([it for it in items if it["role"] in ("title", "cta")] + [it for it in items if it["role"] == "label"])[:160]
    if check:
        masks = ctx.vision("person", ctx.frames_at([(it["a"] + it["b"]) / 2 for it in check], long_edge=640))
        for it, pm in zip(check, masks):
            mk = pm.get("mask")
            if isinstance(mk, np.ndarray):
                x, y, w, h = it["box"]
                Hm, Wm = mk.shape
                on_person = mk[int(y * Hm):int((y + h) * Hm) + 1, int(x * Wm):int((x + w) * Wm) + 1]
                # hands and the edges of clothes break the mask up, so most of the box is enough
                if on_person.size and float(on_person.mean()) >= 0.45:
                    it["role"] = "scene"
    caps = [it for it in items if it["role"] == "caption"]
    titles = [it for it in items if it["role"] in ("title", "cta")]
    roles: dict[str, float] = {}
    for it in items:
        roles[it["role"]] = roles.get(it["role"], 0) + 1
    d = ctx.duration
    long_side = max(ctx.W, ctx.H)
    summary, evidence = [], []
    captions: dict = {"present": bool(caps)}
    if caps:
        eprint(f"typography: {len(caps)} captions; style, font and entrance …")
        st = style_block(ctx, caps, n=10)
        comp = st.pop("comparison", None)
        wpc = [len(tok(it["text"])) for it in caps]
        ycs = [it["box"][1] + it["box"][3] / 2 for it in caps]
        xcs = [it["box"][0] + it["box"][2] / 2 for it in caps]
        # lines: captions read as several lines at the same moment, close above each other
        stacked = 0
        for it in caps:
            stacked += any(o is not it and o["role"] == "caption" and abs(o["a"] - it["a"]) < 0.13 and
                           0 < abs(o["box"][1] - it["box"][1]) < 2.5 * it["box"][3] for o in caps)
        ent = []
        for it in caps[:: max(1, len(caps) // 10)][:10]:
            e = entrance(ctx, it)
            if e:
                ent.append((it, e))
        kinds = [e["kind"] for _, e in ent]
        anim = max(set(kinds), key=kinds.count) if kinds else None
        ent_k = [e for _, e in ent if e["kind"] == anim]
        eases = [e.get("ease") for e in ent_k if e.get("ease")]
        sync = []
        for it, e in ent:
            ts = tok(it["text"])
            cands = [w for w in words if ts and tok(w["w"]) and tok(w["w"])[0] == ts[0] and abs(w["a"] - e["t_in"]) < 1.0]
            if cands:
                w = min(cands, key=lambda w: abs(w["a"] - e["t_in"]))
                sync.append(1000 * (e["t_in"] - w["a"]))
        holds = [it["life"] for it in caps]
        captions.update({
            "words_per_caption": int(round(float(np.median(wpc)))),
            "lines": 2 if stacked > len(caps) / 2 else 1,
            "y_pct": round(100 * float(np.median(ycs)), 1), "x_pct": round(100 * float(np.median(xcs)), 1),
            "size_pct": round(100 * st["font_px"] / long_side, 2) if st.get("font_px") else round(100 * st.get("ink_h_px", 0) / 0.72 / long_side, 2),
            "case": case_of([it["text"] for it in caps]),
            "font_match": st.get("font_match"), "weight": st.get("weight"),
            "color": st.get("color"), "highlight_color": st.get("highlight_color"),
            "stroke": st.get("stroke", False), "stroke_color": st.get("stroke_color"),
            "box": st.get("box", False), "box_color": st.get("box_color"),
            "shadow": bool(st.get("shadow") or st.get("glow")), "shadow_offset_px": st.get("shadow_offset_px"),
            "animation": {"pop": "pop", "fade": "fade", "slide": "slide", "none": "none"}.get(anim, "none") if anim else None,
            "entrance": anim, "ease": max(set(eases), key=eases.count) if eases else None,
            "ease_s": r3(np.median([e["duration_s"] for e in ent_k if e.get("duration_s") is not None])) if ent_k else None,
            "source": "measured (typography lens)",
            "sync_ms": {"median": round(float(np.median(sync))), "p10": round(float(np.percentile(sync, 10))),
                        "p90": round(float(np.percentile(sync, 90))), "n": len(sync)} if sync else None,
            "hold_s": r3(np.median(holds)), "per_min": round(len(caps) / (d / 60), 1),
            "first_s": caps[0]["a"], "band_y_pct": round(100 * band, 1) if band is not None else None,
            "font_candidates": st.get("font_candidates"), "font_best_overall": st.get("font_best_overall"),
            "fonts_offline": st.get("fonts_offline"), "font_px": st.get("font_px"),
            "font_quality": st.get("font_quality"), "font_style": st.get("font_style"),
        })
        c = captions
        summary.append(f"Captions: {len(caps)} ({c['per_min']}/min), {c['words_per_caption']} word(s) each, {c['lines']} line, "
                       f"at {c['y_pct']}% down, {c['case']} case, about {c['size_pct']}% of the long side.")
        summary.append(f"Caption font ({c.get('font_style') or '?'}; match {c.get('font_quality') or '?'}): {c['font_match']} {c['weight']}"
                       + (f" (best overall {c['font_best_overall']['font']} {c['font_best_overall']['weight']})"
                          if c.get("font_best_overall") and c["font_best_overall"]["font"] != c["font_match"] else "")
                       + f"; fill {c['color']}" + (f", highlight {c['highlight_color']}" if c["highlight_color"] != c["color"] else "")
                       + (f", outline {c['stroke_color']}" if c["stroke"] else "") + (f", box {c['box_color']}" if c["box"] else "")
                       + (", soft shadow" if c["shadow"] else "") + ".")
        if anim:
            summary.append(f"Caption entrance: {anim}" + (f", {c['ease']}" if c["ease"] else "")
                           + (f", {c['ease_s']} s" if c["ease_s"] else "") + f"; each stays {c['hold_s']} s"
                           + (f"; appears {c['sync_ms']['median']:+d} ms against its spoken word (p10 {c['sync_ms']['p10']:+d}, p90 {c['sync_ms']['p90']:+d})"
                              if c["sync_ms"] else "") + ".")
        if comp:
            text, mask, cands = comp
            if mask is not None and cands:
                evidence.append({"t": caps[0]["a"], "what": "caption font: on screen against the closest renders",
                                 "img": ctx.img("typography-font.png", fonts.comparison_image(mask, text, cands, fonts.candidates(online=not captions.get("fonts_offline"))))})
        mid = caps[len(caps) // 2]
        evidence.append({"t": mid["a"], "what": "a caption", "img": ctx.strip("typography-caption.jpg", [mid["a"] + 0.15], long_edge=540)})
    else:
        summary.append("No word-by-word captions found.")
    title_rows = []
    for it in titles[:8]:
        st = style_block(ctx, [it], n=1, match_font=True)
        st.pop("comparison", None)
        title_rows.append({"t": it["a"], "end": it["b"], "text": it["text"], "role": it["role"], "box": it["box"],
                           "size_pct": round(100 * st["font_px"] / long_side, 2) if st.get("font_px") else None,
                           "font_match": st.get("font_match"), "weight": st.get("weight"), "color": st.get("color"),
                           "highlight_color": st.get("highlight_color"), "case": case_of([it["text"]]),
                           "effects": [k for k in ("stroke", "shadow", "box", "glow") if st.get(k)],
                           "box_color": st.get("box_color")})
    if title_rows:
        summary.append("Titles and calls to action: " + "; ".join(
            f"{r['t']:.1f} s “{r['text'][:40]}” {r['font_match'] or '?'} {r['weight'] or ''} {r['color']}"
            + (f" on {r['box_color']}" if r.get("box_color") else "") for r in title_rows[:5]) + ".")
    hier = {}
    hs = {role: float(np.median([it["box"][3] for it in items if it["role"] == role])) for role in roles}
    if hs.get("caption"):
        hier = {f"{role}_vs_caption": round(v / hs["caption"], 2) for role, v in hs.items() if role != "caption"}
    summary.append("Text roles: " + ", ".join(f"{k} {int(v)}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])) + ".")
    return {"summary": summary, "captions": captions, "titles": title_rows, "roles": roles,
            "roles_pct": {k: pct(v, len(items)) for k, v in roles.items()}, "hierarchy": hier,
            "text_items": [{k: it[k] for k in ("a", "b", "text", "box", "role", "said", "life")} for it in items[:400]],
            "evidence": evidence}


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    from PIL import Image, ImageDraw
    cands = fonts.candidates(online=False)
    font = next(c for c in cands if c["font"] in ("Arial Black", "Futura", "Helvetica"))
    f = fonts._font(font["path"], font.get("index", 0), 60, font["weight"], bool(font.get("variable")))
    W, Hh = 540, 960
    img = Image.new("RGB", (W, Hh), (90, 110, 130))
    d = ImageDraw.Draw(img)
    # white word with a dark drop shadow 4 px down-right, in a yellow box elsewhere: a label
    d.text((154, 704), "Hello", font=f, fill=(20, 20, 20))
    d.text((150, 700), "Hello", font=f, fill=(255, 255, 255))
    d.rectangle([100, 200, 440, 290], fill=(255, 214, 10))
    d.text((130, 210), "TOP 50", font=f, fill=(15, 15, 15))
    arr = np.asarray(img)
    bb = f.getbbox("Hello")
    box = [150 / W, (700 + bb[1]) / Hh, (bb[2] - bb[0]) / W, (bb[3] - bb[1]) / Hh]
    st = text_style(arr, box)
    assert st and st["color"] in ("#FFFFFF", "#FEFEFE") and st["shadow"] and not st["box"], st
    bb2 = f.getbbox("TOP 50")
    st2 = text_style(arr, [130 / W, (210 + bb2[1]) / Hh, (bb2[2] - bb2[0]) / W, (bb2[3] - bb2[1]) / Hh])
    assert st2 and st2["box"] and st2["box_color"].startswith("#FF") and st2["color"] in ("#0F0F0F", "#101010", "#0E0E0E"), st2
    assert case_of(["TOP 50", "NEW"]) == "upper" and case_of(["hello", "moment"]) == "lower" and case_of(["Hello", "Moment"]) == "sentence"
    track = [{"t": t, "lines": [{"text": w, "box": [0.3, 0.72, 0.4, 0.04]}]} for t, w in
             zip(np.arange(0, 3, 0.25), ["Hallo"] * 2 + ["und"] * 2 + ["willkommen"] * 3 + ["zum"] * 2 + ["Video"] * 3)]
    its = items_of(track)
    assert [it["key"] for it in its] == ["hallo", "und", "willkommen", "zum", "video"], [it["key"] for it in its]
    words = [{"w": w, "a": a, "b": a + 0.3} for w, a in (("Hallo", 0.0), ("und", 0.5), ("willkommen", 1.0), ("zum", 1.75), ("Video", 2.25))]
    band = classify_roles(its, words, 3.0, {})
    assert band and abs(band - 0.74) < 0.03 and all(it["role"] == "caption" for it in its), [(it["role"], it["said"]) for it in its]
    print("typography demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
