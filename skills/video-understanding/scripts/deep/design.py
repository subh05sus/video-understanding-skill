"""design: the look of the video. Its colours (canvas, accents, footage), the grade, what kind of background each
shot sits on, the layout (full footage, split screen, picture-in-picture, cards, graphics canvas), how the
speaker is framed, what crowds the platform's UI zones, and how the cards are styled.

How it measures, on one frame from the middle of each shot (from the edit lens, else from vu's cuts):
  - Colours: k-means in Lab per frame, the speaker masked out with Apple Vision's person mask, merged across
    shots by screen time. Canvas = large flat areas; accents = saturated colours on a small share of the frame
    that come back across shots; footage = the speaker and textured, photographed areas.
  - Grade: lightness (L*) mean, contrast (5th-95th percentile), black and white points, chroma, warmth (b*),
    tint (a*), the tint of shadows and highlights, vignette (corners against the centre) and sharpness.
  - Background: flat colour, gradient (a plane fitted in Lab explains it), grid or lines (regular peaks in the
    row and column edge profiles, with their spacing), blurred footage, or a real room.
  - Layout: the edit lens's split line and fills, plus rectangles found as contours (a card or a
    picture-in-picture: what is inside them decides which).
  - Cards: corner radius from the area a rounded corner leaves out of its box, r = sqrt(missing / (1 - pi/4));
    shadow from a darker band just outside the bottom and right edges; border from a ring of other colour inside.
  - Framing: the face box gives shot size, position, headroom and how close the eyes sit to the upper third.
  - Safe zones (9:16): faces and text in the right rail, the bottom caption band and the top bar.

    python3 design.py demo
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Ctx, eprint, hexcolor, pct, r3  # noqa: E402

VERSION = 1
EDGE = 480
# 9:16 platform UI (TikTok/Reels/Shorts), fractions of the frame: x, y, w, h
ZONES = {"right_rail": (0.84, 0.35, 0.16, 0.5), "bottom_band": (0.0, 0.8, 1.0, 0.2), "top_bar": (0.0, 0.0, 1.0, 0.08)}


# ─── colour ──────────────────────────────────────────────────────────────────────────────────────

def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float Lab with L in 0-100 and a, b around 0."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def lab_to_hex(lab) -> str:
    l, a, b = lab
    px = np.array([[[l * 255 / 100, a + 128, b + 128]]], np.float32).clip(0, 255).astype(np.uint8)
    return hexcolor(cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0])


def kmeans(lab_pixels: np.ndarray, k: int = 6) -> list[tuple[np.ndarray, float]]:
    """(centre, share) of k colour clusters, biggest first."""
    if len(lab_pixels) < k * 4:
        return []
    data = lab_pixels.reshape(-1, 3).astype(np.float32)
    if len(data) > 20000:
        data = data[np.random.default_rng(0).choice(len(data), 20000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    return sorted(((centres[i], float(shares[i])) for i in range(k)), key=lambda x: -x[1])


def merge_colours(items: list[tuple[np.ndarray, float]], de: float = 9.0) -> list[dict]:
    """Merge colours closer than `de` (a rough delta E) and weight by share; biggest first."""
    out: list[list] = []
    for c, w in sorted(items, key=lambda x: -x[1]):
        for o in out:
            if np.linalg.norm(o[0] - c) < de:
                o[0] = (o[0] * o[1] + c * w) / (o[1] + w)
                o[1] += w
                o[2] += 1
                break
        else:
            out.append([np.array(c, float), w, 1])
    total = sum(o[1] for o in out) or 1
    return [{"hex": lab_to_hex(o[0]), "pct": round(100 * o[1] / total, 1), "shots": o[2],
             "lab": [round(float(v), 1) for v in o[0]]} for o in sorted(out, key=lambda o: -o[1])]


def flat_mask(rgb: np.ndarray, thr: float = 6.0) -> np.ndarray:
    """Pixels in smooth areas (a canvas, a card's fill), as opposed to texture and edges."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = cv2.erode((mag < thr).astype(np.uint8), np.ones((5, 5), np.uint8))
    return m > 0


# ─── background and pattern ──────────────────────────────────────────────────────────────────────

def periodic_lines(gray: np.ndarray, mask: np.ndarray) -> dict | None:
    """Grid paper or ruled lines: regular peaks in the column and row profiles of edge strength."""
    g = gray.astype(np.float32)
    # only edges with real contrast: a smooth gradient's 8-bit steps repeat regularly too, but faintly
    ex = (np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) > 12) * mask
    ey = (np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)) > 12) * mask
    found = {}
    for name, prof in (("vertical", ex.sum(axis=0)), ("horizontal", ey.sum(axis=1))):
        p = prof - prof.mean()
        if p.std() < 1e-6 or len(p) < 40:
            continue
        ac = np.correlate(p, p, mode="full")[len(p) - 1:]
        ac /= ac[0]
        lo, hi = 6, len(p) // 3
        k = lo + int(np.argmax(ac[lo:hi]))
        # regular: the second multiple agrees
        if not (ac[k] > 0.3 and (2 * k >= len(ac) or ac[2 * k] > 0.15)):
            continue
        # real ruled paper repeats across the frame: at every spacing step a line runs (mostly) unbroken.
        # A room's window frames are a few strong edges, not four or more evenly spaced lines.
        e = ex if name == "vertical" else ey.T
        m_ = mask if name == "vertical" else mask.T
        phase = int(np.argmax([prof[ph::k].mean() for ph in range(k)]))
        cover = []
        for c in range(phase, e.shape[1], k):
            if m_[:, c].sum() > 20:
                col = np.maximum.reduce([e[:, max(0, c - 1)], e[:, c], e[:, min(e.shape[1] - 1, c + 1)]])
                cover.append(float(col[m_[:, c]].mean()))
        if len(cover) >= 4 and sum(1 for v in cover if v > 0.5) >= 0.6 * len(cover):
            found[name] = (k, float(ac[k]))
    if not found:
        return None
    spacing = float(np.median([v[0] for v in found.values()]))
    return {"pattern": "grid" if len(found) == 2 else "lines", "spacing_px_work": spacing,
            "strength": round(float(np.mean([v[1] for v in found.values()])), 2)}


def background_of(rgb: np.ndarray, exclude: np.ndarray | None, ctx_scale: float, footage: bool = False) -> dict:
    """What a shot sits on. `exclude` (0..1) marks the speaker, cards and text, which are not background.
    Footage (a speaker filmed in a room) is a room, blurred footage or a backdrop; a made canvas (a split
    screen's other half, a graphics frame) is flat colour, a gradient, grid or ruled paper, or black, judged
    on its smooth pixels, so a phone screenshot or a card on it doesn't hide it."""
    h, w = rgb.shape[:2]
    keep = np.ones((h, w), bool) if exclude is None else exclude < 0.3
    if keep.mean() < 0.1:
        return {"kind": "speaker fills the frame"}
    rim = np.zeros((h, w), bool)
    bx, by = max(2, int(0.18 * w)), max(2, int(0.18 * h))
    rim[:, :bx] = rim[:, -bx:] = rim[:by, :] = rim[-by:, :] = True
    if (keep & rim).mean() > 0.05:
        keep = keep & rim     # a canvas shows at the margins; the middle holds the content
    lab = to_lab(rgb)
    px = lab[keep]
    flat = flat_mask(rgb)
    fl = float(flat[keep].mean())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F)[keep].var())
    if float(np.percentile(px[:, 0], 90)) < 8:
        return {"kind": "black", "flat": round(fl, 2)}
    pat = periodic_lines(gray, keep)
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    smooth = float((np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) < 12)[keep].mean())
    if pat and smooth > 0.6:   # regular lines on an otherwise smooth ground: ruled or grid paper
        pat["spacing_px"] = round(pat.pop("spacing_px_work") * ctx_scale)
        return {"kind": f"{pat['pattern']} paper", **pat, "colour": lab_to_hex(np.median(px, axis=0)), "flat": round(fl, 2)}
    if footage:
        if fl > 0.8 and float(np.percentile(px[:, 0], 95) - np.percentile(px[:, 0], 5)) < 12:
            return {"kind": "backdrop", "colour": lab_to_hex(np.median(px, axis=0)), "flat": round(fl, 2)}
        if sharp < 40:
            return {"kind": "blurred footage", "sharpness": round(sharp), "flat": round(fl, 2)}
        return {"kind": "real room / footage", "sharpness": round(sharp), "flat": round(fl, 2)}
    sel_m = keep & flat
    if sel_m.mean() < 0.12 * keep.mean() + 1e-9 or sel_m.sum() < 200:
        return {"kind": "busy (footage or texture)", "sharpness": round(sharp), "flat": round(fl, 2)}
    fpx = lab[sel_m]
    spread = float(np.percentile(fpx[:, 0], 95) - np.percentile(fpx[:, 0], 5)) + float(np.linalg.norm(
        np.percentile(fpx[:, 1:], 95, axis=0) - np.percentile(fpx[:, 1:], 5, axis=0)))
    if spread < 10:
        return {"kind": "flat colour", "colour": lab_to_hex(np.median(fpx, axis=0)), "flat": round(fl, 2)}
    # a gradient: a plane in x, y explains the colour of the smooth pixels
    ys, xs = np.nonzero(sel_m)
    pick = np.random.default_rng(0).choice(len(xs), min(6000, len(xs)), replace=False)
    A = np.c_[xs[pick] / w, ys[pick] / h, np.ones(len(pick))]
    Y = lab[ys[pick], xs[pick]]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    r2 = 1 - ((Y - A @ coef) ** 2).sum() / max(1e-6, ((Y - Y.mean(axis=0)) ** 2).sum())
    if r2 > 0.6:
        gx, gy = np.linalg.norm(coef[0]), np.linalg.norm(coef[1])
        direction = "left to right" if gx >= gy else "top to bottom"
        if min(gx, gy) > 0.5 * max(gx, gy):
            direction = "diagonal"
        # the colour at each end of the gradient, read along the frame's centre line (or corner to corner)
        if direction == "top to bottom":
            start, end = coef[2] + 0.5 * coef[0], coef[2] + 0.5 * coef[0] + coef[1]
        elif direction == "left to right":
            start, end = coef[2] + 0.5 * coef[1], coef[2] + 0.5 * coef[1] + coef[0]
        else:
            start, end = coef[2], coef[2] + coef[0] + coef[1]
        return {"kind": "gradient", "direction": direction, "stops": [lab_to_hex(start), lab_to_hex(end)],
                "fit": round(float(r2), 2), "flat": round(fl, 2)}
    return {"kind": "soft colour field", "colour": lab_to_hex(np.median(fpx, axis=0)), "flat": round(fl, 2)}


# ─── cards ───────────────────────────────────────────────────────────────────────────────────────

def _corner_radius(rgb: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
    """Corner radius of a card from its own fill: the pixels of the card's colour (holes such as text filled in)
    leave r^2 (1 - pi/4) out of each corner of their box."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)
    lab = to_lab(rgb[y0:y1, x0:x1])
    ch, cw = lab.shape[:2]
    body = np.median(lab[ch // 3:2 * ch // 3, cw // 3:2 * cw // 3].reshape(-1, 3), axis=0)
    # how far the card stands from what is around it decides how strict "the card's colour" is
    ox0, oy0, ox1, oy1 = max(0, x - 12), max(0, y - 12), min(w, x + bw + 12), min(h, y + bh + 12)
    ring = to_lab(rgb[oy0:oy1, ox0:ox1])
    ring_px = np.concatenate([ring[:6].reshape(-1, 3), ring[-6:].reshape(-1, 3), ring[:, :6].reshape(-1, 3), ring[:, -6:].reshape(-1, 3)])
    sep = float(np.linalg.norm(np.median(ring_px, axis=0) - body)) if len(ring_px) else 20.0
    m = (np.linalg.norm(lab - body, axis=2) < max(2.0, min(10.0, 0.5 * sep))).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 1, -1)
    fx, fy, fw, fh = cv2.boundingRect(c)
    if fw * fh < 0.7 * bw * bh:
        return None     # the card's colour doesn't fill its box: not a solid card, no radius to read
    q = max(3, min(fw, fh) // 4)
    corners = [q * q - float(filled[yy:yy + q, xx:xx + q].sum()) for yy, xx in
               ((fy, fx), (fy, fx + fw - q), (fy + fh - q, fx), (fy + fh - q, fx + fw - q))]
    if max(corners) > 3 * max(1.0, min(corners)) + 4:
        return None     # the corners disagree: something overlaps the card
    missing = float(np.mean(corners))
    return math.sqrt(missing / (1 - math.pi / 4)) if missing > 0.5 else 0.0


def rectangles(rgb: np.ndarray, min_area: float = 0.04, lines: list[dict] | None = None,
               faces: list[dict] | None = None) -> list[dict]:
    """Cards: contours whose shape is nearly their box, standing out from what is around them, and holding
    text (a screenshot, a text card), a solid fill, or a face (a picture-in-picture). A door or a window in a
    room is a rectangle too, but none of these."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh / (w * h)
        if area < min_area or area > 0.92 or bw < 30 or bh < 30:
            continue
        hull = cv2.convexHull(c)
        fill = cv2.contourArea(hull) / (bw * bh)
        if fill < 0.88:
            continue
        if any(abs(x - o["x"]) < 6 and abs(y - o["y"]) < 6 and abs(bw - o["w"]) < 12 for o in out):
            continue
        fx0, fy0, fx1, fy1 = x / w, y / h, (x + bw) / w, (y + bh) / h
        n_text = sum(1 for l in lines or [] if l.get("box") and fx0 <= l["box"][0] + l["box"][2] / 2 <= fx1
                     and fy0 <= l["box"][1] + l["box"][3] / 2 <= fy1)
        has_face = any(fx0 <= f["box"][0] + f["box"][2] / 2 <= fx1 and fy0 <= f["box"][1] + f["box"][3] / 2 <= fy1
                       and f["box"][2] < 0.8 * (fx1 - fx0) for f in faces or [])
        inner = to_lab(rgb[y + bh // 6:y + 5 * bh // 6, x + bw // 6:x + 5 * bw // 6])
        solid = float(inner[..., 0].std()) < 6
        kind = "footage" if has_face else "text" if n_text >= 2 else "solid" if solid else None
        if kind is None:
            continue
        radius = _corner_radius(rgb, x, y, bw, bh) if kind != "footage" else None
        out.append({"x": x, "y": y, "w": bw, "h": bh, "area": round(area, 3), "kind": kind,
                    "radius_work": None if radius is None else round(radius, 1)})
    return sorted(out, key=lambda r: -r["area"])[:6]


def card_style(rgb: np.ndarray, r: dict) -> dict:
    """Shadow (a darker band outside bottom and right) and border (a ring of other colour inside the edge)."""
    h, w = rgb.shape[:2]
    lab = to_lab(rgb)
    x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
    band = max(3, int(0.03 * min(bw, bh)))

    def mean_l(x0, y0, x1, y1):
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        return float(lab[y0:y1, x0:x1, 0].mean()) if x1 > x0 and y1 > y0 else None
    far = max(4 * band, int(0.12 * min(bw, bh)))   # clear of a soft shadow's reach
    near_b, far_b = mean_l(x, y + bh, x + bw, y + bh + band), mean_l(x, y + bh + far, x + bw, y + bh + far + band)
    near_t, far_t = mean_l(x, y - band, x + bw, y), mean_l(x, y - far - band, x + bw, y - far)
    shadow = None
    if near_b is not None and far_b is not None and far_b - near_b > 4:
        shadow = {"below_darker_by": round(far_b - near_b, 1),
                  "offset": "down" if near_t is None or far_t is None or (far_t - near_t) < 2 else "all around"}
    inner = lab[y + 2:y + 4, x + band:x + bw - band].reshape(-1, 3)
    body = lab[y + bh // 3:y + 2 * bh // 3, x + bw // 3:x + 2 * bw // 3].reshape(-1, 3)
    border = None
    if len(inner) and len(body) and np.linalg.norm(np.median(inner, axis=0) - np.median(body, axis=0)) > 18:
        border = lab_to_hex(np.median(inner, axis=0))
    fill = lab_to_hex(np.median(body, axis=0)) if len(body) else None
    return {"shadow": shadow, "border": border, "fill": fill}


# ─── run ─────────────────────────────────────────────────────────────────────────────────────────

def shot_list(ctx: Ctx) -> list[dict]:
    edit = ctx.results.get("edit") or ctx.load("edit") or {}
    shots = [s for s in edit.get("shots", []) if isinstance(s, dict)]
    if not shots:
        bounds = [0.0] + ctx.cuts() + [ctx.duration]
        shots = [{"a": a, "b": b, "len": b - a} for a, b in zip(bounds, bounds[1:]) if b - a > 0.05]
    shots = [s for s in shots if ctx.in_sections((s["a"] + s["b"]) / 2)]
    if len(shots) > 60:   # keep the longest, they are most of the screen time
        shots = sorted(sorted(shots, key=lambda s: -s["len"])[:60], key=lambda s: s["a"])
    return shots


def layout_of(shot: dict, rects: list[dict], faces: list[dict], ocr: list[dict], w: int, h: int) -> str:
    fill = shot.get("fill", "")
    if shot.get("split_at"):
        return "split screen"
    face_boxes = [f["box"] for f in faces]
    if any(r["kind"] == "footage" and r["area"] < 0.6 for r in rects):
        return "picture-in-picture"
    if face_boxes and any(r["kind"] in ("text", "solid") for r in rects):
        return "card over footage"
    if fill in ("graphic", "screen") or (not face_boxes and rects):
        return "graphics canvas"
    if face_boxes:
        return "full footage"
    return "b-roll / full graphic"


def run(ctx: Ctx) -> dict:
    shots = shot_list(ctx)
    if not shots:
        return {"summary": ["No shots to look at."], "evidence": []}
    eprint(f"design: one frame from each of {len(shots)} shots …")
    mids = [(s["a"] + s["b"]) / 2 for s in shots]
    frames = ctx.frames_at(mids, long_edge=EDGE)
    persons = ctx.vision("person", frames)
    texts = ctx.vision("text", frames)
    faces_all = ctx.faces_track()
    h, w = frames[0].shape[:2]
    scale = ctx.H / h   # work pixels -> source pixels

    canvas_items, accent_items, footage_items, all_items = [], [], [], []
    per_shot, grades, cards = [], [], []
    layout_time: dict[str, float] = {}
    face_rows = []
    zone_hits = {k: 0 for k in ZONES}
    zone_text = {k: 0 for k in ZONES}
    for sh, f, pr, tx in zip(shots, frames, persons, texts):
        weight = sh["len"]
        pm = pr.get("mask") if isinstance(pr.get("mask"), np.ndarray) else None
        lab = to_lab(f)
        keep = np.ones((h, w), bool) if pm is None else pm < 0.3
        flat = flat_mask(f)
        cl = kmeans(lab[::3, ::3].reshape(-1, 3), 6)
        all_items += [(c, s_ * weight) for c, s_ in cl]
        canv = flat & keep
        if canv.mean() > 0.08:
            canvas_items += [(c, s_ * weight * canv.mean()) for c, s_ in kmeans(lab[canv][::4], 3)]
        foot = ~flat | (~keep)
        if foot.mean() > 0.08:
            footage_items += [(c, s_ * weight * foot.mean()) for c, s_ in kmeans(lab[foot][::4], 4)]
        # accents: saturated colours on a small share of the frame
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        sat = (chroma > 35) & keep
        if 0.002 < sat.mean() < 0.12:
            accent_items += [(c, s_ * weight) for c, s_ in kmeans(lab[sat][::2], 2)]
        L = lab[..., 0]
        c_ = lab[h // 4:3 * h // 4, w // 4:3 * w // 4, 0].mean()
        corners = np.concatenate([L[:h // 8, :w // 8].ravel(), L[:h // 8, -w // 8:].ravel(), L[-h // 8:, :w // 8].ravel(), L[-h // 8:, -w // 8:].ravel()])
        region = ~flat | ~keep   # the photographed part
        g_px = lab[region] if region.mean() > 0.1 else lab.reshape(-1, 3)
        grades.append({"w": weight, "L": float(g_px[:, 0].mean()), "contrast": float(np.percentile(g_px[:, 0], 95) - np.percentile(g_px[:, 0], 5)),
                       "black": float(np.percentile(g_px[:, 0], 1)), "white": float(np.percentile(g_px[:, 0], 99)),
                       "chroma": float(np.hypot(g_px[:, 1], g_px[:, 2]).mean()), "warmth": float(g_px[:, 2].mean()),
                       "tint": float(g_px[:, 1].mean()),
                       "shadow_ab": np.median(g_px[g_px[:, 0] < 25][:, 1:], axis=0).tolist() if (g_px[:, 0] < 25).sum() > 50 else None,
                       "high_ab": np.median(g_px[g_px[:, 0] > 75][:, 1:], axis=0).tolist() if (g_px[:, 0] > 75).sum() > 50 else None,
                       "vignette": float(corners.mean() - c_),
                       "sharp": float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())})
        ftrack = min(faces_all, key=lambda x: abs(x["t"] - (sh["a"] + sh["b"]) / 2)) if faces_all else {"faces": []}
        fcs = ftrack["faces"] if abs(ftrack.get("t", 1e9) - (sh["a"] + sh["b"]) / 2) < 1.0 else []
        lines = tx.get("lines", []) or []
        rects = rectangles(f, lines=lines, faces=fcs)
        # the background is what is around the content: leave out the speaker, the cards and the text
        content = np.zeros((h, w), np.float32) if pm is None else pm.copy()
        for r in rects:
            content[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]] = 1.0
        for l in lines:
            if l.get("box"):
                bx, by, bw_, bh_ = l["box"]
                content[max(0, int((by - bh_ * 0.5) * h)):int((by + bh_ * 1.5) * h), max(0, int((bx - 0.02) * w)):int((bx + bw_ + 0.02) * w)] = 1.0
        split = sh.get("split_at")
        if split and fcs:
            # a split screen: the designed half is the one without the speaker
            cut_y = int(split * h)
            face_below = (fcs[0]["box"][1] + fcs[0]["box"][3] / 2) > split
            sub = f[:cut_y] if face_below else f[cut_y:]
            sub_c = content[:cut_y] if face_below else content[cut_y:]
            bg = background_of(sub, sub_c, scale) if min(sub.shape[:2]) > 40 else background_of(f, content, scale, footage=True)
            bg["region"] = "top half" if face_below else "bottom half"
        else:
            filmed = bool(fcs) or (pm is not None and float(pm.mean()) > 0.05) or sh.get("fill") in ("face", "broll")
            bg = background_of(f, content, scale, footage=filmed)
        lay = layout_of(sh, rects, fcs, lines, w, h)
        layout_time[lay] = layout_time.get(lay, 0) + weight
        for r in rects[:3]:
            st = card_style(f, r)
            if r["area"] < 0.85:
                cards.append({"t": r3((sh["a"] + sh["b"]) / 2), "kind": r["kind"],
                              "box": [round(r["x"] / w, 3), round(r["y"] / h, 3), round(r["w"] / w, 3), round(r["h"] / h, 3)],
                              "radius_px": None if r["radius_work"] is None else round(r["radius_work"] * scale), **st})
        if fcs:
            fb = fcs[0]["box"]
            face_rows.append({"w": weight, "h": fb[3], "cx": fb[0] + fb[2] / 2, "cy": fb[1] + fb[3] / 2, "top": fb[1],
                              "eyes": fb[1] + 0.4 * fb[3]})
        for name, (zx, zy, zw, zh) in ZONES.items():
            def inz(b):
                return b and b[0] < zx + zw and b[0] + b[2] > zx and b[1] < zy + zh and b[1] + b[3] > zy
            if any(inz(fc["box"]) for fc in fcs):
                zone_hits[name] += 1
            if any(inz(l.get("box")) and l["box"][3] > 0.012 for l in lines):
                zone_text[name] += 1
        per_shot.append({"a": sh["a"], "b": sh["b"], "layout": lay, "background": bg.get("kind"), "bg": bg,
                         "palette": [{"hex": lab_to_hex(c), "pct": round(100 * s_, 1)} for c, s_ in cl[:5]],
                         "luma": round(grades[-1]["L"], 1), "chroma": round(grades[-1]["chroma"], 1), "cards": len(rects)})
    total_w = sum(g["w"] for g in grades) or 1

    def wavg(key):
        return round(sum(g[key] * g["w"] for g in grades) / total_w, 1)
    sh_ab = [g["shadow_ab"] for g in grades if g["shadow_ab"]]
    hi_ab = [g["high_ab"] for g in grades if g["high_ab"]]
    grade = {"lightness": wavg("L"), "contrast": wavg("contrast"), "black_point": wavg("black"), "white_point": wavg("white"),
             "chroma": wavg("chroma"), "warmth_b": wavg("warmth"), "tint_a": wavg("tint"), "vignette": wavg("vignette"),
             "sharpness": round(float(np.median([g["sharp"] for g in grades]))),
             "shadows_ab": [round(float(v), 1) for v in np.median(sh_ab, axis=0)] if sh_ab else None,
             "highlights_ab": [round(float(v), 1) for v in np.median(hi_ab, axis=0)] if hi_ab else None}
    grade["reads"] = ", ".join(x for x in (
        "warm" if grade["warmth_b"] > 8 else "cool" if grade["warmth_b"] < -3 else "neutral",
        "punchy" if grade["contrast"] > 70 else "flat/soft" if grade["contrast"] < 45 else "normal contrast",
        "saturated" if grade["chroma"] > 30 else "muted" if grade["chroma"] < 15 else "natural saturation",
        "bright" if grade["lightness"] > 65 else "dark" if grade["lightness"] < 35 else "",
        "vignetted" if grade["vignette"] < -8 else "") if x)
    palette = {"overall": merge_colours(all_items)[:8], "canvas": merge_colours(canvas_items)[:4],
               "accents": [c for c in merge_colours(accent_items, de=14) if c["shots"] >= min(2, len(shots))][:5],
               "footage": merge_colours(footage_items)[:5]}
    # backgrounds and layouts by screen time
    bgs: dict[str, float] = {}
    for p, sh in zip(per_shot, shots):
        bgs[p["background"]] = bgs.get(p["background"], 0) + sh["len"]
    background = {"mix_pct": {k: pct(v, total_w) for k, v in sorted(bgs.items(), key=lambda kv: -kv[1])}}
    pats = [p["bg"] for p in per_shot if "paper" in str(p["background"])]
    if pats:
        background |= {"pattern": pats[0]["pattern"], "pattern_size": int(np.median([p["spacing_px"] for p in pats])),
                       "pattern_colour": pats[0].get("colour")}
    grads = [p["bg"] for p in per_shot if p["background"] == "gradient"]
    if grads:
        background["gradient"] = {"direction": grads[0]["direction"], "stops": grads[0]["stops"]}
    layout = {"mix_pct": {k: pct(v, total_w) for k, v in sorted(layout_time.items(), key=lambda kv: -kv[1])},
              "timeline": [{"a": p["a"], "b": p["b"], "layout": p["layout"]} for p in per_shot]}
    comp = {}
    if face_rows:
        fw = sum(r["w"] for r in face_rows)

        def fav(k):
            return sum(r[k] * r["w"] for r in face_rows) / fw
        sizes = {"extreme close-up": 0.0, "close-up": 0.0, "medium close-up": 0.0, "medium": 0.0, "wide": 0.0}
        for r in face_rows:
            k = "extreme close-up" if r["h"] > 0.45 else "close-up" if r["h"] > 0.25 else "medium close-up" if r["h"] > 0.12 else "medium" if r["h"] > 0.06 else "wide"
            sizes[k] += r["w"]
        comp = {"face_in_shots_pct": pct(fw, total_w), "shot_size_pct": {k: pct(v, fw) for k, v in sizes.items() if v},
                "face_centre_pct": [round(100 * fav("cx"), 1), round(100 * fav("cy"), 1)],
                "face_height_pct": round(100 * fav("h"), 1), "headroom_pct": round(100 * fav("top"), 1),
                "eyes_from_upper_third_pct": round(100 * (fav("eyes") - 1 / 3), 1),
                "off_centre_pct": round(100 * abs(fav("cx") - 0.5), 1)}
    safe = None
    if ctx.portrait:
        n = len(shots)
        safe = {"zones": {k: list(v) for k, v in ZONES.items()},
                "faces_in_zone_pct": {k: pct(v, n) for k, v in zone_hits.items()},
                "text_in_zone_pct": {k: pct(v, n) for k, v in zone_text.items()}}
    radii = [c["radius_px"] for c in cards if c["radius_px"] is not None]
    cards_sum = {"n": len(cards), "radius_px": int(np.median(radii)) if radii else None, "radius_read_on": len(radii),
                 "shadow": ("soft" if sum(1 for c in cards if c["shadow"]) > len(cards) / 2 else "none") if cards else None,
                 "border": next((c["border"] for c in cards if c["border"]), None), "examples": cards[:6]}
    summary = [
        "Palette (screen time): " + ", ".join(f"{c['hex']} {c['pct']:.0f}%" for c in palette["overall"][:6]) + ".",
    ]
    if palette["canvas"]:
        summary.append("Canvas: " + ", ".join(f"{c['hex']} {c['pct']:.0f}%" for c in palette["canvas"][:3])
                       + ("; accents: " + ", ".join(c["hex"] for c in palette["accents"]) if palette["accents"] else "; no recurring accent") + ".")
    summary.append(f"Grade: {grade['reads']} (L* {grade['lightness']}, contrast {grade['contrast']}, chroma {grade['chroma']}, "
                   f"warmth b* {grade['warmth_b']}, vignette {grade['vignette']}).")
    summary.append("Backgrounds: " + ", ".join(f"{k} {v:.0f}%" for k, v in background["mix_pct"].items())
                   + (f"; {background['pattern']} pattern every {background['pattern_size']} px" if background.get("pattern") else "")
                   + (f"; gradient {background['gradient']['stops'][0]}→{background['gradient']['stops'][1]} {background['gradient']['direction']}" if background.get("gradient") else "") + ".")
    summary.append("Layouts: " + ", ".join(f"{k} {v:.0f}%" for k, v in layout["mix_pct"].items()) + ".")
    if comp:
        summary.append(f"Framing: face in {comp['face_in_shots_pct']:.0f}% of shot time, mostly "
                       f"{max(comp['shot_size_pct'], key=comp['shot_size_pct'].get)}; face centre at {comp['face_centre_pct'][0]}% across, "
                       f"{comp['face_centre_pct'][1]}% down; eyes {comp['eyes_from_upper_third_pct']:+.0f}% from the upper third.")
    if cards:
        summary.append(f"Cards: {len(cards)} seen" + (f", corner radius about {cards_sum['radius_px']} px (source, read on {len(radii)})" if radii else ", corner radius not readable") + ", "
                       f"shadow {cards_sum['shadow']}" + (f", border {cards_sum['border']}" if cards_sum["border"] else "") + ".")
    if safe:
        crowd = [f"{k.replace('_', ' ')} (text in {v:.0f}% of shots)" for k, v in safe["text_in_zone_pct"].items() if v >= 20]
        summary.append("Safe zones: " + ("text sits in the " + ", ".join(crowd) if crowd else "text keeps clear of the platform UI") + ".")
    evidence = [{"t": 0.0, "what": "palette: overall, canvas, accents", "img": swatches(ctx, palette)}]
    seen_l = set()
    for p, f in zip(per_shot, frames):
        if p["layout"] not in seen_l:
            seen_l.add(p["layout"])
            evidence.append({"t": r3((p["a"] + p["b"]) / 2), "what": f"layout: {p['layout']}; background: {p['background']}",
                             "img": ctx.img(f"design-layout-{len(seen_l)}.jpg", cv2.resize(f, (w // 2, h // 2)))})
    return {"summary": summary, "palette": palette, "grade": grade, "background": background, "layout": layout,
            "composition": comp, "safe_zone": safe, "cards": cards_sum, "per_shot": per_shot, "evidence": evidence}


def swatches(ctx: Ctx, palette: dict) -> str:
    from PIL import Image, ImageDraw
    rows = [("overall", palette["overall"][:8]), ("canvas", palette["canvas"][:4]), ("accents", palette["accents"][:5]),
            ("footage", palette["footage"][:5])]
    img = Image.new("RGB", (900, 60 * len(rows)), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for i, (name, cols) in enumerate(rows):
        d.text((8, i * 60 + 22), name, fill=(60, 60, 60))
        for j, c in enumerate(cols):
            x = 90 + j * 100
            d.rectangle([x, i * 60 + 6, x + 92, i * 60 + 40], fill=c["hex"])
            d.text((x, i * 60 + 44), f"{c['hex']} {c['pct']:.0f}%", fill=(60, 60, 60))
    return ctx.img("design-palette.png", np.asarray(img))


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    rng = np.random.default_rng(0)
    w, h = 270, 480
    # grid paper with a rounded white card that casts a shadow, and a yellow label
    paper = np.full((h, w, 3), (246, 243, 234), np.uint8)
    for x in range(0, w, 18):
        paper[:, x] = (225, 222, 214)
    for y in range(0, h, 18):
        paper[y, :] = (225, 222, 214)
    bg = background_of(paper, None, 4.0)
    assert bg["kind"] == "grid paper" and abs(bg["spacing_px"] - 72) <= 8, bg
    img = paper.copy()
    shade = np.zeros((h, w), np.float32)
    cv2.rectangle(shade, (60, 118), (210, 318), 1.0, -1)                  # a soft shadow, 8 px down
    shade = cv2.GaussianBlur(shade, (31, 31), 0)
    img = (img * (1 - 0.35 * shade[..., None])).astype(np.uint8)
    card = np.zeros((h, w), np.uint8)
    cv2.rectangle(card, (60, 110), (210, 310), 255, -1)
    r = 20
    for cx, cy in ((60, 110), (210, 110), (60, 310), (210, 310)):
        card[cy - r if cy > 200 else cy:cy + 1 if cy > 200 else cy + r, cx - r if cx > 100 else cx:cx + 1 if cx > 100 else cx + r] = 0
    for cx, cy in ((60 + r, 110 + r), (210 - r, 110 + r), (60 + r, 310 - r), (210 - r, 310 - r)):
        cv2.circle(card, (cx, cy), r, 255, -1)
    img[card > 0] = (255, 255, 255)
    rects = rectangles(img)
    assert rects and abs(rects[0]["radius_work"] - r) < 6, rects
    st = card_style(img, rects[0])
    assert st["shadow"] is not None and st["fill"] == "#FFFFFF", st
    grad = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        grad[y] = (np.array([200, 180, 240]) * (1 - y / h) + np.array([250, 200, 190]) * (y / h)).astype(np.uint8)
    g = background_of(grad + rng.integers(0, 2, grad.shape, dtype=np.uint8), None, 1.0)
    assert g["kind"] == "gradient" and g["direction"] == "top to bottom", g
    assert background_of(np.full((h, w, 3), 30, np.uint8), None, 1)["kind"] == "flat colour"
    cols = merge_colours([(np.array([50., 0, 0]), 0.5), (np.array([52., 1, 0]), 0.2), (np.array([90., 0, 60]), 0.3)])
    assert len(cols) == 2 and cols[0]["pct"] == 70.0, cols
    print("design demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
