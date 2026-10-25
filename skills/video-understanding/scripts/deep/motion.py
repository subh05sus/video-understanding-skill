"""motion: the elements laid over or into the picture (cards, screenshots, stickers, titles, picture-in-picture,
memes) and exactly how they move: how each one comes in and goes out, with its ease and duration, how it moves
while it is up, what staggers, and GSAP that rebuilds it, checked by rendering it and comparing with the video.

How it measures (grey frames at the video's own rate, 360 px on the long side):
  - Events: bursts of change outside the cuts (edit lens), the speaker (Apple Vision's person mask) and the
    caption band (typography lens). Comparing a steady frame before a burst with one after it shows what
    persists: a region that changed for good is an element arriving, leaving or being swapped. A gesture or a
    flicker leaves nothing behind and is dropped.
  - Entrance and exit: the settled element is looked for in every frame of the burst at several scales
    (normalised cross-correlation), giving its position and scale per frame; a blend against the frame before
    gives its opacity; the share of its pixels already in place gives a wipe's reveal. Each channel that moves
    is fitted with ease.fit (start, duration, GSAP ease, overshoot, cubic-bezier).
  - Check: the element is composited back over the frame before it arrived, moving by the fitted curves, and
    compared with the real frames. "match" is how much of the error a plain cut would leave the fit removes.
  - While up: its position and scale at 6 fps (drift, float, slow push) and whether its inside keeps changing
    (a video playing in it).
  - What it is (a guess, labelled so): a face inside is a video or picture-in-picture, many text lines a
    screenshot or UI, a ragged silhouette a sticker, one or two big lines a text card.

    python3 motion.py demo
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ease  # noqa: E402
from common import Ctx, eprint, pct, r3  # noqa: E402

VERSION = 1
EDGE = 360
DIFF = 22             # grey levels: a pixel that changed
MIN_AREA = 0.012      # an element covers at least 1.2% of the frame
MAX_BURST = 1.6       # seconds; longer change is ongoing motion, not an entrance
# a template shrunk under 0.4x matches ordinary texture almost anywhere, so smaller is not trusted
SCALES = np.r_[np.arange(0.4, 0.95, 0.05), np.arange(0.95, 1.06, 0.01), np.arange(1.08, 1.41, 0.04)]


# ─── frames and what to ignore ───────────────────────────────────────────────────────────────────

def load(ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
    ts, fr = [], []
    for s in ctx.sections:
        for t, g in ctx.frames(s["a"], s["b"], long_edge=EDGE, gray=True):
            ts.append(t)
            fr.append(cv2.GaussianBlur(g, (3, 3), 0))
    return np.array(ts), np.stack(fr) if fr else np.zeros((0, 1, 1), np.uint8)


def person_masks(ctx: Ctx, h: int, w: int) -> tuple[np.ndarray, list[np.ndarray]]:
    times = [float(t) for s in ctx.sections for t in np.arange(s["a"] + 0.1, s["b"], 0.5)]
    if not times:
        return np.zeros(0), []
    res = ctx.vision("person", ctx.frames_at(times, long_edge=EDGE))
    k = np.ones((max(3, w // 25),) * 2, np.uint8)
    masks = []
    for r in res:
        m = r.get("mask")
        m = (cv2.resize(m, (w, h)) > 0.35).astype(np.uint8) if isinstance(m, np.ndarray) else np.zeros((h, w), np.uint8)
        masks.append(cv2.dilate(m, k) > 0)
    return np.array(times), masks


def mask_at(times: np.ndarray, masks: list[np.ndarray], t: float, shape) -> np.ndarray:
    if not len(times):
        return np.zeros(shape, bool)
    return masks[int(np.argmin(np.abs(times - t)))]


def cut_zones(ctx: Ctx) -> list[tuple[float, float]]:
    edit = ctx.results.get("edit") or ctx.load("edit") or {}
    zones = []
    for c in edit.get("cuts", []):
        if isinstance(c, dict):
            zones.append((c["t"] - 0.08, c["t"] + (c.get("dur_s") or 0) + 0.08))
    if not zones:
        zones = [(c - 0.08, c + 0.08) for c in ctx.cuts()]
    return zones


def caption_band(ctx: Ctx, h: int) -> tuple[int, int] | None:
    typo = ctx.results.get("typography") or ctx.load("typography") or {}
    caps = typo.get("captions") or {}
    if not caps.get("present") or caps.get("y_pct") is None:
        return None
    y = caps["y_pct"] / 100
    half = max(0.035, (caps.get("size_pct") or 4) / 100 * (max(ctx.W, ctx.H) / ctx.H) * 0.9)
    return int((y - half) * h), int((y + half) * h)


# ─── events ──────────────────────────────────────────────────────────────────────────────────────

def bursts(ctx: Ctx, ts, fr, pt, pm, zones, band) -> list[tuple[int, int]]:
    n, h, w = fr.shape
    act = np.zeros(n)
    for i in range(1, n):
        if any(a <= ts[i] <= b for a, b in zones):
            continue
        d = cv2.absdiff(fr[i], fr[i - 1]) > DIFF
        d &= ~mask_at(pt, pm, ts[i], (h, w))
        if band:
            d[band[0]:band[1]] = False
        act[i] = d.mean()
    base = float(np.median(act))      # the usual level of change, still frames included
    thr = max(0.004, 3 * base)
    runs, i = [], 1
    while i < n:
        if act[i] > thr:
            j = i
            while j + 1 < n and (act[j + 1] > thr or (j + 2 < n and act[j + 2] > thr)):
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    # a step where most of the frame changes is a cut the edit lens may not list: not an element (the edit
    # lens's own cuts are already left out; a big card plus a camera move can change a third of the frame)
    return [(a, b) for a, b in runs if (ts[b] - ts[a]) <= MAX_BURST and act[a:b + 1].max() < 0.6]


def align(before: np.ndarray, after: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """`before` moved onto `after` when the camera moved (a pan, a push, a shake): a similarity transform from
    ORB features outside `ignore`, with RANSAC so a new element doesn't pull it. Unchanged if there is no
    clear camera move."""
    orb = cv2.ORB_create(800)
    m = (~ignore).astype(np.uint8) * 255
    k1, d1 = orb.detectAndCompute(before, m)
    k2, d2 = orb.detectAndCompute(after, m)
    if d1 is None or d2 is None or len(k1) < 25 or len(k2) < 25:
        return before
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 20:
        return before
    src = np.float32([k1[g.queryIdx].pt for g in good])
    dst = np.float32([k2[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < 20:
        return before
    moved = abs(math.hypot(M[0, 0], M[1, 0]) - 1) > 0.01 or math.hypot(M[0, 2], M[1, 2]) > 1.5
    return cv2.warpAffine(before, M, (before.shape[1], before.shape[0]), borderMode=cv2.BORDER_REPLICATE) if moved else before


def regions(ctx, ts, fr, a: int, b: int, pt, pm, band) -> list[dict]:
    """What a burst left behind: regions that differ between a steady frame before and one after, once the
    camera's own move is taken out. Pieces of one element are merged."""
    n, h, w = fr.shape
    i0, i1 = max(0, a - 2), min(n - 1, b + 2)
    ignore = mask_at(pt, pm, ts[i0], (h, w)) | mask_at(pt, pm, ts[i1], (h, w))
    if band:
        ignore = ignore.copy()
        ignore[band[0]:band[1]] = True
    before = align(fr[i0], fr[i1], ignore).astype(np.int16)
    after = fr[i1].astype(np.int16)
    p = (np.abs(after - before) > DIFF) & ~ignore
    p = cv2.morphologyEx(p.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_l, lab, stats, _ = cv2.connectedComponentsWithStats(p)
    boxes = []
    for k in range(1, n_l):
        x, y, bw, bh, area = stats[k]
        if area >= 0.25 * MIN_AREA * h * w:
            boxes.append([int(x), int(y), int(x + bw), int(y + bh), int(area)])
    # merge pieces that overlap or nearly touch (a card split by text, a hand or the caption band)
    gap = int(0.03 * max(h, w))
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                A, B = boxes[i], boxes[j]
                if A[0] - gap < B[2] and B[0] - gap < A[2] and A[1] - gap < B[3] and B[1] - gap < A[3]:
                    boxes[i] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]), A[4] + B[4]]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for x0, y0, x1, y1, area in boxes:
        bw, bh = x1 - x0, y1 - y0
        if area < MIN_AREA * h * w or bw < 12 or bh < 12 or bw * bh > 0.85 * h * w:
            continue
        out.append({"box": (x0, y0, bw, bh), "fill": round(area / (bw * bh), 2), "i0": i0, "i1": i1, "a": a, "b": b})
    return out


def card_at(ctx: Ctx, t: float, box, shape) -> tuple | None:
    """The card with text in it (design lens) around `box` at time t, or None."""
    from design import rectangles
    h, w = shape
    x, y, bw, bh = box
    f = ctx.frame(t, long_edge=EDGE)
    pad_x, pad_y = int(0.25 * bw), int(0.25 * bh)
    X0, Y0, X1, Y1 = max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    crop = f[Y0:Y1, X0:X1]
    if crop.shape[0] < 30 or crop.shape[1] < 30:
        return None
    lines = ctx.vision("text", [crop])[0].get("lines", []) or []
    rects = [r_ for r_ in rectangles(crop, min_area=0.3, lines=lines) if r_["kind"] == "text"]   # a plain wall is "solid" too
    if not rects:
        return None
    r_ = rects[0]
    return (X0 + r_["x"], Y0 + r_["y"], r_["w"], r_["h"])


def snap(ctx: Ctx, t: float, box, shape) -> tuple:
    """Snap a region to the card it belongs to: the biggest card-like rectangle around it (design lens)."""
    from design import rectangles
    h, w = shape
    x, y, bw, bh = box
    f = ctx.frame(t, long_edge=EDGE)
    pad_x, pad_y = int(0.25 * bw), int(0.25 * bh)
    X0, Y0, X1, Y1 = max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    crop = f[Y0:Y1, X0:X1]
    if crop.shape[0] < 30 or crop.shape[1] < 30:
        return box
    lines = ctx.vision("text", [crop])[0].get("lines", []) or []
    rects = rectangles(crop, min_area=0.3, lines=lines)
    if not rects:
        return box
    r = rects[0]
    nb = (X0 + r["x"], Y0 + r["y"], r["w"], r["h"])
    # keep the snap only when it really is the same thing: it overlaps most of the region
    ix = max(0, min(nb[0] + nb[2], x + bw) - max(nb[0], x))
    iy = max(0, min(nb[1] + nb[3], y + bh) - max(nb[1], y))
    return nb if ix * iy >= 0.5 * bw * bh else box


def structure(g: np.ndarray) -> float:
    """How much there is inside a region: edge density (an empty wall or canvas is low)."""
    e = cv2.Canny(g, 40, 120)
    return float(e.mean() / 255)


def direction(fr, reg) -> str:
    """in, out or swap: which side of the burst has the element."""
    x, y, bw, bh = reg["box"]
    b, a = fr[reg["i0"], y:y + bh, x:x + bw], fr[reg["i1"], y:y + bh, x:x + bw]
    sb, sa = structure(b), structure(a)
    if sa > 1.6 * sb + 0.005:
        return "in"
    if sb > 1.6 * sa + 0.005:
        return "out"
    return "swap"


# ─── tracking ────────────────────────────────────────────────────────────────────────────────────

def locate(g: np.ndarray, tmpl: np.ndarray, near: tuple[int, int, int, int], scales=SCALES) -> tuple[float, float, float, float]:
    """(score, scale, cx, cy) of the template's best match in g, searched around `near` (x, y, w, h)."""
    x, y, bw, bh = near
    H, W = g.shape
    X0, Y0 = max(0, x - bw), max(0, y - bh)
    X1, Y1 = min(W, x + 2 * bw), min(H, y + 2 * bh)
    region = g[Y0:Y1, X0:X1]
    best = (-1.0, 1.0, x + bw / 2, y + bh / 2)

    def at(sc):
        th, tw = int(round(tmpl.shape[0] * sc)), int(round(tmpl.shape[1] * sc))
        if th < 6 or tw < 6 or th > region.shape[0] or tw > region.shape[1]:
            return None
        t_ = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        if t_.std() < 3:
            return None
        r = cv2.matchTemplate(region, t_, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        return (float(mx), float(sc), X0 + loc[0] + tw / 2, Y0 + loc[1] + th / 2)
    scales = np.asarray(scales, float)
    if len(scales) > 12:
        # coarse to fine: every third scale first, then every scale near the best of those
        coarse = scales[::3]
        for sc in coarse:
            m = at(sc)
            if m and m[0] > best[0]:
                best = m
        step = float(np.median(np.diff(coarse))) if len(coarse) > 1 else 0.1
        fine = scales[np.abs(scales - best[1]) <= step]
    else:
        fine = scales
    for sc in fine:
        m = at(sc)
        if m and m[0] > best[0]:
            best = m
    return best


def alpha_at(frame: np.ndarray, bg: np.ndarray, tmpl: np.ndarray) -> float:
    """Opacity of the element at its final place: frame ≈ (1-a)·background + a·element."""
    a, b, m = bg.astype(np.float32).ravel(), tmpl.astype(np.float32).ravel(), frame.astype(np.float32).ravel()
    diff = b - a
    den = float(diff @ diff)
    return float(np.clip(((m - a) @ diff) / den, 0, 1)) if den > 1e-6 else 1.0


def track(ts, fr, idx: list[int], tmpl, box, bg) -> list[dict]:
    x, y, bw, bh = box
    rows = []
    for i in idx:
        sc, s, cx, cy = locate(fr[i], tmpl, box)
        crop = fr[i, y:y + bh, x:x + bw]
        reveal = float((np.abs(crop.astype(np.int16) - tmpl.astype(np.int16)) < DIFF).mean())
        rows.append({"i": i, "t": float(ts[i]), "score": sc, "scale": s, "cx": cx, "cy": cy,
                     "alpha": alpha_at(crop, bg, tmpl), "reveal": reveal})
    return rows


def move_window(ts, fr, box, i_lo: int, i_hi: int, tmpl, bg, exit_: bool, fps: float) -> tuple[int, int] | None:
    """The frames of the move itself. In: from the first frame the element really shows to the frame its
    move has run out. Out: from the frame it starts to go to the last frame it shows.
    "Really shows": the match must beat what the same template scores on a frame where the element is not
    there yet (small templates correlate with any texture). "Run out": the frame-to-frame change in scale and
    place has dropped to the slow drift that follows (cards often keep a gentle push after they pop)."""
    n, H, W = fr.shape
    lo, hi = max(0, i_lo), min(n - 1, i_hi)
    absent = lo - 2 if not exit_ else hi + 2
    absent = min(n - 1, max(0, absent))
    base = locate(fr[absent], tmpl, box)[0]
    vis_thr = max(0.6, base + 0.12)
    span = int(1.6 * fps)
    idx = list(range(lo, min(n, hi + span))) if not exit_ else list(range(max(0, lo - span), hi + 1))
    rows = track(ts, fr, idx, tmpl, box, bg)
    seen = [r for r in rows if r["score"] > vis_thr]
    if not seen:
        return None
    order = seen if not exit_ else seen[::-1]
    k0 = order[0]["i"]
    after = [r for r in (rows if not exit_ else rows[::-1]) if (r["i"] >= k0 if not exit_ else r["i"] <= k0) and r["score"] > vis_thr]
    if len(after) < 4:
        return (k0, after[-1]["i"]) if not exit_ else (after[-1]["i"], k0)
    sc = np.array([r["scale"] for r in after])
    pos = np.array([[r["cx"], r["cy"]] for r in after])
    v = np.abs(np.diff(np.log(sc))) + np.linalg.norm(np.diff(pos, axis=0), axis=1) / max(W, H)
    tail = np.median(v[len(v) // 2:]) if len(v) >= 6 else float(v.min())
    calm = v <= max(0.004, 2.0 * tail)
    ks = next((after[j + 1]["i"] for j in range(len(v) - 2) if calm[j] and calm[j + 1] and calm[j + 2]), after[-1]["i"])
    return (k0, ks) if not exit_ else (ks, k0)


def fit_motion(rows: list[dict], box, W: int, H: int, fps: float, exit_: bool = False) -> dict:
    """Which channels move and their fitted curves; the kind of entrance (or exit)."""
    x, y, bw, bh = box
    fx, fy = x + bw / 2, y + bh / 2
    vis = [r for r in rows if r["score"] > 0.45 or r["alpha"] > 0.2 or r["reveal"] > 0.2]
    if not vis:
        return {"kind": "cut", "duration_s": 0.0, "note": "not visible in between"}
    t = np.array([r["t"] for r in rows])
    out: dict = {"channels": {}}
    # the element's own motion: only frames where it was found with confidence
    good = [r for r in rows if r["score"] > 0.55]
    tg = np.array([r["t"] for r in good]) if good else np.zeros(0)
    if len(good) >= 3:
        sc = np.array([r["scale"] for r in good])
        dx = np.array([(r["cx"] - fx) / W for r in good])
        dy = np.array([(r["cy"] - fy) / H for r in good])
        if np.ptp(sc) > 0.08:
            f = ease.fit(tg, sc, end_value=None if exit_ else 1.0, start_value=1.0 if exit_ else None, frame=1 / fps)
            if f:
                out["channels"]["scale"] = f
        for name, v in (("x", dx), ("y", dy)):
            if np.ptp(v) > 0.03:
                f = ease.fit(tg, v, end_value=None if exit_ else 0.0, start_value=0.0 if exit_ else None, frame=1 / fps)
                if f:
                    out["channels"][name] = f
    al = np.array([r["alpha"] for r in rows])
    if np.ptp(al) > 0.35 and not out["channels"]:
        f = ease.fit(t, al, end_value=0.0 if exit_ else 1.0, start_value=1.0 if exit_ else 0.0, frame=1 / fps)
        if f:
            out["channels"]["opacity"] = f
    rv = np.array([r["reveal"] for r in rows])
    if np.ptp(rv) > 0.35 and not out["channels"]:
        f = ease.fit(t, rv, end_value=0.0 if exit_ else 1.0, frame=1 / fps)
        if f:
            out["channels"]["reveal"] = f
    ch = out["channels"]
    if not ch:
        out.update(kind="cut", duration_s=0.0)
        return out
    main = max(ch.values(), key=lambda f: f["duration"])
    if main["duration"] < 1.5 / fps:
        # over within a frame: it simply appears (or goes), however the channels fitted
        out.update(kind="cut", duration_s=0.0, channels={})
        return out
    kinds = []
    if "x" in ch or "y" in ch:
        kinds.append("slide")
    if "scale" in ch:
        kinds.append("scale")
    if "opacity" in ch:
        kinds.append("fade")
    if "reveal" in ch:
        kinds.append("wipe")
    out["kind"] = "+".join(kinds)
    out["duration_s"] = main["duration"]
    out["start"] = min(f["start"] for f in ch.values())
    out["ease"] = main["ease"]
    out["bezier"] = main["bezier"]
    out["css"] = main["css"]
    out["rmse"] = main["rmse"]
    if "scale" in ch:
        s = ch["scale"]
        out["scale_from"] = round(s["from"], 2) if not exit_ else 1.0
        out["scale_to"] = 1.0 if not exit_ else round(s["to"], 2)
        out["overshoot"] = s["overshoot"]
    for name in ("x", "y"):
        if name in ch:
            off = ch[name]["from"] if not exit_ else ch[name]["to"]
            side = ("left" if off < 0 else "right") if name == "x" else ("top" if off < 0 else "bottom")
            out["from" if not exit_ else "to"] = side
            out["distance_pct"] = round(abs(off) * 100, 1)
    out["fade"] = "opacity" in ch
    return out


def render_check(ts, fr, rows: list[dict], fit: dict, tmpl, box, bg_frame: np.ndarray, W: int, H: int) -> dict | None:
    """Composite the element moving by the fitted curves over the frame before it came, compare with the real
    frames, and score how much of a plain cut's error the fit removes (1 = all of it). Only the element's own
    footprint counts (where the fit, the cut or the tracker puts it), and only the frames of the move, so a
    speaker talking or a camera drifting behind it doesn't swamp the comparison."""
    ch = fit.get("channels") or {}
    if not ch or not rows:
        return None
    x, y, bw, bh = box
    fx, fy = x + bw / 2, y + bh / 2

    def value(name, t, rest):
        f = ch.get(name)
        if not f:
            return rest
        p = np.clip((t - f["start"]) / max(1e-6, f["duration"]), 0, 1)
        e = ease.evaluate(f["ease"], np.array([p]))[0]
        return f["from"] + (f["to"] - f["from"]) * e

    def rect(sc_, cx_, cy_):
        th, tw = int(round(bh * sc_)), int(round(bw * sc_))
        xa, ya = int(round(cx_ - tw / 2)), int(round(cy_ - th / 2))
        return xa, ya, tw, th

    def paint(canvas, foot, sc_, cx_, cy_, al):
        xa, ya, tw, th = rect(sc_, cx_, cy_)
        if th < 2 or tw < 2:
            return
        xs0, ys0, xs1, ys1 = max(0, xa), max(0, ya), min(W, xa + tw), min(H, ya + th)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        foot[ys0:ys1, xs0:xs1] = True
        if al > 0:
            t_ = cv2.resize(tmpl, (tw, th)).astype(np.float32)
            canvas[ys0:ys1, xs0:xs1] = (1 - al) * canvas[ys0:ys1, xs0:xs1] + al * t_[ys0 - ya:ys1 - ya, xs0 - xa:xs1 - xa]

    t_first = min(f["start"] for f in ch.values())
    t_last = max(f["start"] + f["duration"] for f in ch.values())
    errs_fit, errs_cut, used = [], [], []
    for r in rows:
        if not (t_first - 2.5 / 30 <= r["t"] <= t_last + 2.5 / 30):
            continue
        real = fr[r["i"]].astype(np.float32)
        canvas_fit = bg_frame.astype(np.float32).copy()
        canvas_cut = bg_frame.astype(np.float32).copy()
        foot = np.zeros((H, W), bool)
        on = r["t"] >= t_first
        a = value("opacity", r["t"], 1.0) if on else 0.0
        paint(canvas_fit, foot, value("scale", r["t"], 1.0), fx + value("x", r["t"], 0.0) * W, fy + value("y", r["t"], 0.0) * H, a)
        paint(canvas_cut, foot, 1.0, fx, fy, 1.0 if on else 0.0)
        if r["score"] > 0.6:
            xa, ya, tw, th = rect(r["scale"], r["cx"], r["cy"])
            foot[max(0, ya):max(0, ya + th), max(0, xa):max(0, xa + tw)] = True
        if foot.sum() < 20:
            continue
        errs_fit.append(float(np.abs(canvas_fit[foot] - real[foot]).mean()))
        errs_cut.append(float(np.abs(canvas_cut[foot] - real[foot]).mean()))
        used.append(r["t"])
    if not errs_fit:
        return None
    ef, ec = float(np.mean(errs_fit)), float(np.mean(errs_cut))
    match = 1.0 - ef / ec if ec > 1e-3 else None
    worst = int(np.argmax(np.array(errs_fit) - np.array(errs_cut)))
    return {"match": None if match is None else round(float(np.clip(match, 0, 1)), 2), "err_fit": round(ef, 1),
            "err_cut": round(ec, 1), "worst_t": r3(used[worst]), "frames": len(used)}


# ─── one element ─────────────────────────────────────────────────────────────────────────────────

def follow_while_up(ctx: Ctx, tmpl, box, a: float, b: float) -> dict:
    """Position, scale and inner change at 6 fps while the element is up."""
    if b - a < 0.6:
        return {"secondary": "still"}
    ts, fr = ctx.clip(a, b, fps=6, long_edge=EDGE, gray=True)
    if len(fr) < 3:
        return {"secondary": "still"}
    H, W = fr.shape[1:]
    x, y, bw, bh = box
    rows = []
    for g in fr:
        g = cv2.GaussianBlur(g, (3, 3), 0)
        rows.append(locate(g, tmpl, box, scales=np.arange(0.85, 1.16, 0.01)))
    good = [r for r in rows if r[0] > 0.5]
    inner = [float(cv2.absdiff(g[y:y + bh, x:x + bw], fr[0][y:y + bh, x:x + bw]).mean()) for g in fr]
    if len(good) < 3:
        return {"secondary": "video" if np.median(inner) > 12 else "unknown", "lost": True}
    sc = np.array([r[1] for r in good])
    cx = np.array([r[2] for r in good]) / W
    cy = np.array([r[3] for r in good]) / H
    dur = max(1e-6, ts[-1] - ts[0])
    drift = float(math.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) / dur * 100)       # % of the frame per second
    push = float((sc[-1] / sc[0] - 1) / dur * 100)                              # % scale per second
    wob = float(np.std(np.diff(cy)) * 100) if len(cy) > 3 else 0.0
    kind = "push" if abs(push) > 2 else "drift" if drift > 1 else "float" if wob > 0.5 else \
        "video" if np.median(inner) > 12 and np.median([r[0] for r in good]) < 0.85 else "still"
    return {"secondary": kind, "drift_pct_s": round(drift, 2), "push_pct_s": round(push, 2), "wobble": round(wob, 2)}


def looks(ctx: Ctx, items: list[tuple[float, tuple, float]], shape) -> list[dict]:
    """What each element is (a guess) and how it is styled, from full-size crops; Vision runs once for all
    of them. items: (time, box in work pixels, how much of its box the change filled)."""
    from design import _corner_radius, card_style
    h, w = shape
    if not items:
        return []
    frames = ctx.frames_at([t for t, _, _ in items], long_edge=960)
    crops, places = [], []
    for f, (_, box, _) in zip(frames, items):
        H, W = f.shape[:2]
        x, y, bw, bh = box
        X0, Y0, X1, Y1 = int(x / w * W), int(y / h * H), int((x + bw) / w * W), int((y + bh) / h * H)
        crop = f[Y0:Y1, X0:X1]
        crops.append(crop if crop.size else np.zeros((8, 8, 3), np.uint8))
        places.append((f, X0, Y0, X1, Y1, W))
    faces_all = ctx.vision("faces", crops)
    text_all = ctx.vision("text", crops)
    out = []
    for (t, box, fill), (f, X0, Y0, X1, Y1, W), fc, tx in zip(items, places, faces_all, text_all):
        faces = fc.get("faces", []) or []
        lines = tx.get("lines", []) or []
        big = [l for l in lines if l.get("box") and l["box"][3] > 0.12]
        if faces and max(fb["box"][2] * fb["box"][3] for fb in faces) > 0.02:
            kind = "video / picture-in-picture"
        elif len(lines) >= 3:
            kind = "screenshot / UI"
        elif fill < 0.7 and not lines:
            kind = "sticker / cut-out"
        elif lines and (big or len(lines) <= 2):
            kind = "text card"
        else:
            kind = "photo / graphic"
        look = {}
        try:
            st = card_style(f, {"x": X0, "y": Y0, "w": X1 - X0, "h": Y1 - Y0})
            rad = _corner_radius(f, X0, Y0, X1 - X0, Y1 - Y0)
            look = {"radius_px": None if rad is None else round(rad * ctx.W / W), "shadow": st.get("shadow"),
                    "border": st.get("border"), "fill": st.get("fill")}
        except Exception:
            pass
        out.append({"kind_guess": kind, "text": " ".join(l["text"] for l in lines)[:160], "faces": len(faces), "look": look})
    return out


def gsap(el: dict, W: int, H: int) -> str:
    """Ready-to-paste GSAP for one element, in a 1080-wide frame."""
    k = 1080 / max(1, W)
    sel = f'"#el{el["id"]}"'
    parts = []
    e = el.get("entrance") or {}
    ch = e.get("channels") or {}
    props = []
    if "x" in ch:
        props.append(f"x: {round(ch['x']['from'] * W * k)}")
    if "y" in ch:
        props.append(f"y: {round(ch['y']['from'] * H * k)}")
    if "scale" in ch:
        props.append(f"scale: {ch['scale']['from']:.2f}")
    if "opacity" in ch or e.get("fade"):
        props.append("opacity: 0")
    if "reveal" in ch and e.get("from"):
        side = {"left": "inset(0 100% 0 0)", "right": "inset(0 0 0 100%)", "top": "inset(0 0 100% 0)", "bottom": "inset(100% 0 0 0)"}
        props.append(f'clipPath: "{side.get(e["from"], "inset(0 100% 0 0)")}"')
    if props:
        parts.append(f'gsap.from({sel}, {{{", ".join(props)}, duration: {e["duration_s"]:.2f}, ease: "{e["ease"]}"}});')
    else:
        parts.append(f"gsap.set({sel}, {{autoAlpha: 1}});  // appears on a cut")
    x = el.get("exit") or {}
    xc = x.get("channels") or {}
    props = []
    if "x" in xc:
        props.append(f"x: {round(xc['x']['to'] * W * k)}")
    if "y" in xc:
        props.append(f"y: {round(xc['y']['to'] * H * k)}")
    if "scale" in xc:
        props.append(f"scale: {xc['scale']['to']:.2f}")
    if "opacity" in xc:
        props.append("opacity: 0")
    if props:
        parts.append(f'gsap.to({sel}, {{{", ".join(props)}, duration: {x["duration_s"]:.2f}, ease: "{x["ease"]}", delay: {max(0.0, el["t_out"] - el["t_in"] - x["duration_s"]):.2f}}});')
    return "\n".join(parts)


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def run(ctx: Ctx) -> dict:
    eprint("motion: reading frames …")
    ts, fr = load(ctx)
    if len(fr) < 10:
        return {"summary": ["Too short to follow motion."], "elements": [], "evidence": []}
    n, h, w = fr.shape
    pt, pm = person_masks(ctx, h, w)
    zones = cut_zones(ctx)
    band = caption_band(ctx, h)
    eprint("motion: finding elements coming and going …")
    regs = []
    for a, b in bursts(ctx, ts, fr, pt, pm, zones, band):
        lo_t, hi_t = float(ts[max(0, a - 2)]), float(ts[min(n - 1, b + 2)])
        crossing = [z1 for z0, z1 in zones if lo_t - 0.05 <= z0 <= hi_t + 0.05 or lo_t - 0.05 <= z1 <= hi_t + 0.05]
        if crossing:
            # the frames before and after belong to different shots: compare only after the cut, which still
            # catches an element that pops in with the cut and keeps animating
            a = int(np.searchsorted(ts, max(crossing) + 0.04)) + 2
            if b - a < 3:
                continue
        for r in regions(ctx, ts, fr, a, b, pt, pm, band):
            x, y, bw, bh = r["box"]
            edges = (x <= 1) + (y <= 1) + (x + bw >= w - 1) + (y + bh >= h - 1)
            if edges >= 2 and r["fill"] < 0.85:
                continue    # a ragged patch in a corner: background revealed by reframing, not a placed element
            r["dir"] = direction(fr, r)
            regs.append(r)
    # elements: an arrival, then the departure that clears the same place (or the next cut, or the end)
    cut_t = sorted(z[0] for z in zones)
    elements = []
    used = set()
    ins = [r for r in regs if r["dir"] in ("in", "swap")]
    outs = [r for r in regs if r["dir"] in ("out", "swap")]
    for k, r in enumerate(ins):
        x, y, bw, bh = r["box"]
        t_in = float(ts[r["a"]])
        later = [o for o in outs if id(o) not in used and o["a"] > r["b"] and _iou(o["box"], r["box"]) > 0.3]
        nxt_cut = next((c for c in cut_t if c > float(ts[r["b"]]) + 0.1), ctx.duration)
        o = later[0] if later and float(ts[later[0]["a"]]) < nxt_cut else None
        if o:
            used.add(id(o))
        elements.append({"in": r, "out": o, "t_in": t_in, "t_out": float(ts[o["b"]]) if o else nxt_cut})
    elements = elements[:90]
    eprint(f"motion: {len(elements)} elements; tracking their moves …")
    rows_out = []
    pending: list[tuple] = []
    fps = ctx.fps

    def one(k: int, el: dict):
        """Track one element's entrance and exit; None when it turns out not to be a new element."""
        r = el["in"]
        r["box"] = snap(ctx, float(ts[r["i1"]]), r["box"], (h, w))
        x, y, bw, bh = r["box"]
        settle = r["i1"]
        tmpl = fr[settle, y:y + bh, x:x + bw]
        bg = fr[r["i0"], y:y + bh, x:x + bw]
        # already there before the burst (its content scrolled or changed, it didn't arrive): a card-like
        # rectangle already sat in the same place. A wall or a room behind a new card has none.
        prior = card_at(ctx, float(ts[max(0, r["a"] - 3)]), r["box"], (h, w))
        if prior is not None and _iou(prior, r["box"]) >= 0.6:
            return None
        win = move_window(ts, fr, r["box"], r["a"] - 2, r["b"] + 2, tmpl, bg, False, fps)
        if win:
            # a real element holds together once it has arrived: its settled look is found again, in the
            # same place, for the next half second. Hands, hair and shifting background don't.
            k_s = win[1]
            chk_idx = list(range(k_s, min(n, k_s + int(0.5 * fps)), 2))
            t_s = fr[k_s, y:y + bh, x:x + bw]
            hold = [locate(fr[i], t_s, r["box"], scales=np.arange(0.9, 1.11, 0.02)) for i in chk_idx]
            if hold and sum(1 for hsc, hs, hx, hy in hold if hsc > 0.75 and math.hypot(hx - (x + bw / 2), hy - (y + bh / 2)) < 0.06 * max(w, h)) < 0.6 * len(hold):
                return None
        if win:
            k0, ks = win
            settle = ks
            tmpl = fr[ks, y:y + bh, x:x + bw]          # the element as it settles, not after its slow drift
            i_bg = max(0, k0 - 2)
            bg = fr[i_bg, y:y + bh, x:x + bw]
            idx = list(range(i_bg, min(n, ks + 4)))
        else:
            i_bg = r["i0"]
            idx = list(range(max(0, r["a"] - 2), min(n, r["b"] + 3)))
        rows = track(ts, fr, idx, tmpl, r["box"], bg)
        ent = fit_motion(rows, r["box"], w, h, fps)
        chk = render_check(ts, fr, rows, ent, tmpl, r["box"], fr[i_bg], w, h)
        if chk:
            ent["match"] = chk["match"]
            ent["check"] = chk
            if ent.get("kind") != "cut" and (chk["match"] is None or chk["match"] < 0.25):
                # the fitted move explains little of what happened: say so instead of passing it on
                ent["fitted_kind"] = ent["kind"]
                ent["kind"] = "unclear"
        ext = None
        if el["out"]:
            o = el["out"]
            tm2 = fr[o["i0"], y:y + bh, x:x + bw]
            bg2 = fr[o["i1"], y:y + bh, x:x + bw]
            win2 = move_window(ts, fr, r["box"], o["a"] - 2, o["b"] + 2, tm2, bg2, True, fps)
            if win2:
                ke, kl = win2
                tm2 = fr[ke, y:y + bh, x:x + bw]
                idx2 = list(range(max(0, ke - 2), min(n, kl + 3)))
            else:
                idx2 = list(range(max(0, o["a"] - 2), min(n, o["b"] + 3)))
            rows2 = track(ts, fr, idx2, tm2, r["box"], bg2)
            ext = fit_motion(rows2, r["box"], w, h, fps, exit_=True)
        up_a = float(ts[min(n - 1, r["b"] + 2)])
        up_b = float(el["t_out"]) - 0.1
        sec = follow_while_up(ctx, tmpl, r["box"], up_a, up_b) if up_b > up_a else {"secondary": "still"}
        item = (float(ts[settle]) + 0.05, r["box"], r["fill"])
        faces = [f for f in ctx.faces_track() if el["t_in"] <= f["t"] <= el["t_out"] and f["faces"]]
        bx = (x / w, y / h, bw / w, bh / h)
        covers = sum(1 for f in faces if _iou(f["faces"][0]["box"], bx, frac=True) > 0.05)
        e = {"id": 0, "t_in": r3(el["t_in"]), "t_out": r3(el["t_out"]),
             "box": [round(v, 3) for v in bx], "area_pct": round(100 * bx[2] * bx[3], 1),
             "kind_guess": None, "text": None, "look": None,
             "covers_face": bool(faces) and covers > len(faces) / 2,
             "entrance": {k2: v for k2, v in ent.items() if k2 != "channels"} | {"channels": ent.get("channels", {})},
             "exit": ext, **sec}
        e["_strip"] = (float(ts[max(0, (win[0] if win else r["a"]) - 2)]), float(ts[min(n - 1, (win[1] if win else r["b"]) + 2)]), tmpl)
        return e, item

    # elements are independent: track them on threads (OpenCV's matching and the ffmpeg/Vision helpers run
    # outside Python's lock, so this uses the cores)
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        done = list(ex.map(lambda kv: one(*kv), enumerate(elements)))
    for res in done:
        if res:
            rows_out.append(res[0])
            pending.append(res[1])
    for e, lk in zip(rows_out, looks(ctx, pending, (h, w))):
        e.update(kind_guess=lk["kind_guess"], text=lk["text"], look=lk["look"])
    # one arrival found twice (two pieces of one card, or a swap seen from both sides): keep the better fit
    kept: list[dict] = []
    for e in sorted(rows_out, key=lambda e: e["t_in"]):
        dup = next((k for k in kept if abs(k["t_in"] - e["t_in"]) <= 0.35 and
                    (_iou(k["box"], e["box"]) > 0.3 or _inside(k["box"], e["box"]) > 0.7 or _inside(e["box"], k["box"]) > 0.7)), None)
        if dup is None:
            kept.append(e)
            continue
        better = lambda z: ((z["entrance"] or {}).get("match") or 0, z["area_pct"])
        if better(e) > better(dup):
            kept[kept.index(dup)] = e
    for i, e in enumerate(kept, 1):
        e["id"] = i
        e["gsap"] = gsap(e, w, h)
        t0, t1, tmpl = e.pop("_strip")
        if i <= 8:
            mid = (t0 + t1) / 2
            e["strip"] = ctx.strip(f"motion-el{i}.jpg", [t0, (t0 + mid) / 2, mid, (mid + t1) / 2, t1], long_edge=300)
            e["template"] = ctx.img(f"motion-el{i}-tmpl.png", cv2.cvtColor(tmpl, cv2.COLOR_GRAY2RGB))
    return summarize(ctx, kept)


def _inside(a, b) -> float:
    """Share of box a that lies inside box b."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, a[2] * a[3])


def _iou(a, b, frac: bool = False) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def aggregate(moves: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for m in moves:
        if not m:
            continue
        key = (m.get("kind"), m.get("ease"))
        groups.setdefault(key, []).append(m)
    total = sum(len(v) for v in groups.values()) or 1
    out = []
    for (kind, ease_), ms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append({"kind": kind, "ease": ease_, "n": len(ms), "share_pct": pct(len(ms), total),
                    "duration_s": r3(np.median([m.get("duration_s") or 0 for m in ms])),
                    "overshoot": r3(np.median([m.get("overshoot") or 0 for m in ms])),
                    "fade": any(m.get("fade") for m in ms),
                    "from": max(set(f := [m.get("from") for m in ms if m.get("from")]), key=f.count) if any(m.get("from") for m in ms) else None,
                    "to": max(set(f2 := [m.get("to") for m in ms if m.get("to")]), key=f2.count) if any(m.get("to") for m in ms) else None,
                    "distance_pct": r3(np.median([m["distance_pct"] for m in ms if m.get("distance_pct") is not None])) if any(m.get("distance_pct") is not None for m in ms) else None,
                    "scale_from": r3(np.median([m["scale_from"] for m in ms if m.get("scale_from") is not None])) if any(m.get("scale_from") is not None for m in ms) else None,
                    "match": r3(np.median([m["match"] for m in ms if m.get("match") is not None])) if any(m.get("match") is not None for m in ms) else None})
    return out


def summarize(ctx: Ctx, els: list[dict]) -> dict:
    d = ctx.duration
    minutes = d / 60 if d else 1
    entrances = aggregate([e["entrance"] for e in els if (e["entrance"] or {}).get("kind") != "unclear"])
    exits = aggregate([e["exit"] for e in els if e.get("exit") and e["exit"].get("kind") != "unclear"])
    unclear = sum(1 for e in els if (e["entrance"] or {}).get("kind") == "unclear")
    stagger = []
    group = []
    for e in sorted(els, key=lambda e: e["t_in"]):
        if group and e["t_in"] - group[-1]["t_in"] <= 0.6:
            group.append(e)
        else:
            if len(group) >= 2:
                gaps = np.diff([g["t_in"] for g in group])
                stagger.append({"t": group[0]["t_in"], "n": len(group), "interval_s": r3(np.median(gaps)),
                                "order": "top to bottom" if group[-1]["box"][1] > group[0]["box"][1] else "bottom to top"})
            group = [e]
    if len(group) >= 2:
        gaps = np.diff([g["t_in"] for g in group])
        stagger.append({"t": group[0]["t_in"], "n": len(group), "interval_s": r3(np.median(gaps)),
                        "order": "top to bottom" if group[-1]["box"][1] > group[0]["box"][1] else "bottom to top"})
    screen: dict[str, float] = {}
    for e in els:
        screen[e["kind_guess"]] = screen.get(e["kind_guess"], 0) + (e["t_out"] - e["t_in"])
    tot = sum(screen.values()) or 1
    zones = {"top": 0.0, "middle": 0.0, "bottom": 0.0, "full": 0.0}
    for e in els:
        x, y, bw, bh = e["box"]
        key = "full" if bw * bh > 0.6 else "top" if y + bh / 2 < 0.38 else "bottom" if y + bh / 2 > 0.62 else "middle"
        zones[key] += e["t_out"] - e["t_in"]
    sec: dict[str, int] = {}
    for e in els:
        sec[e.get("secondary", "still")] = sec.get(e.get("secondary", "still"), 0) + 1
    layout = {"zones_pct": {k: pct(v, tot) for k, v in zones.items()},
              "median_box": [round(100 * float(np.median([e["box"][i] for e in els])), 1) for i in range(4)] if els else None,
              "area_pct": round(float(np.median([e["area_pct"] for e in els])), 1) if els else None,
              "covers_face_pct": pct(sum(1 for e in els if e["covers_face"]), len(els))}
    lang = None
    moving = [en for en in entrances if en["kind"] != "cut"]
    if entrances:
        top = moving[0] if moving else entrances[0]
        if top["kind"] == "cut":
            lang = "Elements mostly appear on a cut, with no move in."
        else:
            lang = (f"Elements mostly come in with a {top['kind']} ({top['ease']}, {top['duration_s']:.2f} s"
                    + (f", {top['overshoot'] * 100:.0f}% overshoot" if top.get("overshoot") else "") + ")")
            cut_share = next((en["share_pct"] for en in entrances if en["kind"] == "cut"), 0)
            lang += f"; {cut_share:.0f}% just appear on a cut." if cut_share else "."
        if exits:
            lang += f" Exits: mostly {exits[0]['kind']}" + (f" ({exits[0]['ease']})" if exits[0]["kind"] != "cut" else "") + "."
    summary = [f"{len(els)} elements come and go ({len(els) / minutes:.1f}/min): "
               + ", ".join(f"{k} {v / tot * 100:.0f}%" for k, v in sorted(screen.items(), key=lambda kv: -kv[1])) + " of their screen time."]
    if lang:
        summary.append(lang)
    if unclear:
        summary.append(f"{unclear} of {len(els)} entrances could not be fitted reliably (re-rendered match under 0.25) and are left out of these figures.")
    for e in [e for e in els if (e["entrance"] or {}).get("kind") not in (None, "cut", "unclear")][:5]:
        en = e["entrance"]
        summary.append(f"{e['t_in']:.1f} s {e['kind_guess']} comes in: {en['kind']}, {en.get('ease')}, {en.get('duration_s')} s"
                       + (f", from {en['scale_from']} scale" if en.get("scale_from") else "")
                       + (f" from the {en['from']} {en.get('distance_pct')}%" if en.get("from") else "")
                       + (f", {en['overshoot'] * 100:.0f}% overshoot" if en.get("overshoot") else "")
                       + (f" (re-rendered match {en['match']})" if en.get("match") is not None else "") + ".")
    if stagger:
        summary.append("Staggered: " + "; ".join(f"{s['n']} elements every {s['interval_s']} s at {s['t']:.1f} s" for s in stagger[:4]) + ".")
    if sec:
        summary.append("While up: " + ", ".join(f"{k} {v}" for k, v in sorted(sec.items(), key=lambda kv: -kv[1])) + ".")
    evidence = [{"t": e["t_in"], "what": f"element {e['id']}: {e['kind_guess']}, {(e['entrance'] or {}).get('kind')}", "img": e["strip"]}
                for e in els if e.get("strip")]
    return {"summary": summary, "elements": els, "entrances": entrances, "exits": exits, "stagger": stagger,
            "secondary_motion": {k: pct(v, len(els)) for k, v in sec.items()}, "per_min": round(len(els) / minutes, 1),
            "kinds": {k: pct(v, tot) for k, v in screen.items()}, "layout": layout, "language": lang, "evidence": evidence}


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    import os
    import subprocess
    import tempfile
    import vu
    # a grey canvas with texture; a card pops in at 1.0 s (scale 0.6 -> 1, back.out) and another slides up
    # from below at 2.5 s (power3.out, 0.4 s); 30 fps, 4 s
    W, H, fps = 360, 640, 30
    rng = np.random.default_rng(0)
    bgimg = np.full((H, W, 3), 70, np.uint8)
    bgimg += rng.integers(0, 12, (H, W, 1), dtype=np.uint8)
    card = np.full((140, 220, 3), 245, np.uint8)
    for k in range(5):
        cv2.rectangle(card, (16, 18 + k * 24), (200 - k * 20, 28 + k * 24), (40, 40, 40), -1)
    card2 = np.full((90, 260, 3), 30, np.uint8)
    cv2.putText(card2, "LABEL 2", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 220, 40), 3)
    with tempfile.TemporaryDirectory() as td:
        os.environ["VU_CACHE"] = str(Path(td) / "cache")
        vu.CACHE = Path(os.environ["VU_CACHE"])
        src = Path(td) / "m.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
                              "-i", "-", "-pix_fmt", "yuv420p", "-crf", "12", str(src)], stdin=subprocess.PIPE)
        for i in range(int(4 * fps)):
            t = i / fps
            f = bgimg.copy()
            p1 = np.clip((t - 1.0) / 0.45, 0, 1)
            if t >= 1.0:
                s = 0.6 + 0.4 * float(ease.evaluate("back.out(1.7)", np.array([p1]))[0])
                c = cv2.resize(card, (int(220 * s), int(140 * s)))
                cy, cx = 200, 180
                y0, x0 = cy - c.shape[0] // 2, cx - c.shape[1] // 2
                f[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
            if t >= 2.5:
                p2 = float(ease.evaluate("power3.out", np.array([np.clip((t - 2.5) / 0.4, 0, 1)]))[0])
                yy = int(470 + (1 - p2) * 170)
                y1 = min(H, yy + 90)
                if yy < H:
                    f[yy:y1, 50:310] = card2[:y1 - yy]
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
        ctx = Ctx(vu.Video(src))
        ctx.results["edit"] = {"cuts": []}
        ctx.results["typography"] = {"captions": {"present": False}}
        res = run(ctx)
        ens = [e["entrance"] for e in res["elements"]]
        print("  ", [(e["t_in"], (e["entrance"] or {}).get("kind"), (e["entrance"] or {}).get("ease"), (e["entrance"] or {}).get("duration_s"), (e["entrance"] or {}).get("match")) for e in res["elements"]])
        pop = next((e for e in res["elements"] if abs(e["t_in"] - 1.0) < 0.15), None)
        slide = next((e for e in res["elements"] if abs(e["t_in"] - 2.5) < 0.15), None)
        assert pop and "scale" in pop["entrance"]["kind"], res["elements"]
        assert abs(pop["entrance"]["duration_s"] - 0.45) < 0.15 and pop["entrance"]["scale_from"] < 0.75, pop["entrance"]
        assert pop["entrance"]["ease"].split("(")[0] in ("back.out", "elastic.out") or pop["entrance"]["overshoot"] > 0.03, pop["entrance"]
        assert slide and "slide" in slide["entrance"]["kind"] and slide["entrance"]["from"] == "bottom", slide and slide["entrance"]
        assert abs(slide["entrance"]["duration_s"] - 0.4) < 0.15, slide["entrance"]
        assert pop["entrance"].get("match") is not None and pop["entrance"]["match"] > 0.6, pop["entrance"]
        assert "gsap.from" in pop["gsap"] and "scale" in pop["gsap"]
    print("motion demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
