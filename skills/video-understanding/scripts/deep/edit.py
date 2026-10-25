"""edit: how the video is cut. Every cut at the video's own frame rate and what kind it is, the shots and what
fills them, the camera (punch-ins, pushes, pans, shake), and the rhythm: how often something new happens.

How it measures:
  - Every frame is shrunk to 160 px and compared with the one before (grey difference and a colour histogram
    distance). A strong one-frame change is an instant cut; a run of strong changes between two steady pictures
    is a transition; a lone spike well above its neighbours is a quieter cut (a jump cut keeps the room, so the
    histogram barely moves but the picture jumps).
  - A transition is tested as a blend of the frames either side (dissolve, or a fade when one side is black or
    white), as a region of the new picture that grows (wipe, or a mask when it grows in scattered blocks, like a
    pixel dissolve), or as blur (whip). A change that snaps back within a few frames is a flash or flicker, not a
    cut. A change in only part of the frame (the top half of a split screen swaps) is an insert, not a cut.
  - An instant cut is matched with ORB features: the same picture a little shifted is a jump cut, the same
    picture scaled is a zoom punch (with its scale), a very different scale is a reframe of the layout, and no
    match is a hard cut.
  - The camera inside each shot: sparse optical flow on 320 px grey frames, a similarity transform per step,
    accumulated into scale (a push), position (a pan) and jitter (handheld shake). A push is fitted with ease.fit.

    python3 edit.py demo
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ease  # noqa: E402
from common import Ctx, eprint, hex_to_rgb, hexcolor, near, pct, r3  # noqa: E402

VERSION = 1
WORK = 160            # long edge for cut detection
THUMB = (36, 64)      # (w, h) grey thumbnails kept for every frame (portrait); swapped for landscape
STRONG_H = 0.30       # histogram distance of a clear change
STRONG_D = 0.15       # grey difference of a clear change
SPIKE = 2.5           # a quiet cut stands this far above the frames around it
MAX_TRANSITION = 0.8  # seconds; a longer run of change is movement inside a shot
CAM_FPS = 15
CAM_EDGE = 320
FULL_CUTS = {"hard", "jump", "match", "zoom-punch", "layout", "zoom", "dissolve", "fade", "wipe", "mask", "whip", "transition"}


# ─── per-frame signals ───────────────────────────────────────────────────────────────────────────

def _hist(f: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(1.0, h.sum())


MATCH_EDGE = 360      # long edge of the grey frames kept for feature matching


def ffmpeg_small(shape) -> tuple[int, int]:
    """(w, h) of the 160 px working frame for a frame of this shape."""
    h, w = shape[:2]
    return (WORK, max(2, round(h * WORK / w))) if w >= h else (max(2, round(w * WORK / h)), WORK)


def gray_at(ctx: Ctx, t: float) -> np.ndarray:
    """The grey frame showing at t, from the copy kept while reading (or one seek for a long video)."""
    g = getattr(ctx, "_g360", None)
    if g:
        return g[min(len(g) - 1, max(0, int(t * ctx.fps + 1e-6)))]
    return cv2.cvtColor(ctx.frame(t, long_edge=MATCH_EDGE), cv2.COLOR_RGB2GRAY)


def signals(ctx: Ctx) -> dict:
    """Walk the whole video once at its own fps: times, grey difference d[i] and histogram distance dh[i] between
    frame i-1 and i (d[0] = 0), each frame's mean luma and contrast, and a small grey thumbnail of every frame."""
    tw, th = THUMB if ctx.portrait else THUMB[::-1]
    ts, d, dh, luma, std, thumbs, hists = [], [], [], [], [], [], []
    prev_g = prev_h = None
    # keep a 360 px grey copy of every frame of a short video: the cut checks look at many single frames, and
    # seeking into a long-GOP 1080p file for each one costs seconds apiece
    keep = ctx.duration * ctx.fps <= 6000
    ctx._g360 = [] if keep else None
    for t, big in ctx.frames(0, ctx.duration, long_edge=MATCH_EDGE):
        if keep:
            ctx._g360.append(cv2.cvtColor(big, cv2.COLOR_RGB2GRAY))
        f = cv2.resize(big, ffmpeg_small(big.shape), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h = _hist(f)
        ts.append(t)
        d.append(0.0 if prev_g is None else float(np.abs(g - prev_g).mean() / 255))
        dh.append(0.0 if prev_h is None else float(0.5 * np.abs(h - prev_h).sum()))
        luma.append(float(g.mean()))
        std.append(float(g.std()))
        thumbs.append(cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA))
        hists.append(h)
        prev_g, prev_h = g, h
    return {"t": np.array(ts), "d": np.array(d), "dh": np.array(dh), "luma": np.array(luma), "std": np.array(std),
            "thumb": np.stack(thumbs) if thumbs else np.zeros((0, th, tw), np.float32),
            "hist": np.stack(hists) if hists else np.zeros((0, 192))}


def _dist(s: dict, i: int, j: int) -> float:
    """How different frames i and j are (0 same .. 1): histogram distance and thumbnail difference."""
    i, j = max(0, min(len(s["t"]) - 1, i)), max(0, min(len(s["t"]) - 1, j))
    hd = 0.5 * np.abs(s["hist"][i] - s["hist"][j]).sum()
    td = np.abs(s["thumb"][i] - s["thumb"][j]).mean() / 255
    return float(max(hd, min(1.0, td * 4)))


def _uniform(s: dict, i: int) -> str | None:
    if s["std"][i] < 14:
        if s["luma"][i] > 220:
            return "white"
        if s["luma"][i] < 30:
            return "black"
    return None


# ─── finding cuts and transitions ────────────────────────────────────────────────────────────────

def activity(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """How much each frame step changes the picture (the larger of the grey difference and part of the colour
    histogram distance, so a yellow-to-white change counts as much as a dark-to-light one), and the shot's own
    level of change around it (the median over a second either side)."""
    c = np.maximum(s["d"], 0.6 * s["dh"])
    n = len(c)
    base = np.array([np.median(c[max(0, i - 15):i + 16]) for i in range(n)])
    return c, base


def candidates(s: dict, fps: float) -> tuple[list[tuple[int, int]], list[int]]:
    """Runs [i0, i1] of frame steps changing well above the shot's own level, and lone spikes (a quiet jump cut
    in a talking head: a step standing well above its neighbours)."""
    d, dh = s["d"], s["dh"]
    c, base = activity(s)
    n = len(c)
    thr = np.maximum(0.05, 2.5 * base)
    runs: list[list[int]] = []
    for i in range(1, n):
        if c[i] <= thr[i]:
            continue
        # one calmer frame inside a transition does not end it, but a steady frame does
        if runs and (i - runs[-1][1] == 1 or (i - runs[-1][1] == 2 and c[i - 1] >= 0.6 * thr[i - 1])):
            runs[-1][1] = i
        else:
            runs.append([i, i])
    # a transition that pauses for a frame or two (a page that drops, stops, then settles) is still one
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[1] > r[0] and merged[-1][1] > merged[-1][0] and r[0] - merged[-1][1] <= 3:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = merged
    out = []
    max_len = int(MAX_TRANSITION * fps)
    for a, b in runs:
        if b - a + 1 <= max_len:
            out.append((a, b))
            continue
        # a long run is movement; only frames that tower over the run are cuts inside it
        med = float(np.median(dh[a:b + 1]))
        out.extend((k, k) for k in range(a, b + 1) if dh[k] > 0.5 and dh[k] > 3 * med)
    taken = {k for a, b in runs for k in range(a, b + 1)}
    spikes = []
    for i in range(2, n - 2):
        if i in taken or d[i] < 0.03:
            continue
        around = max(d[i - 2], d[i - 1], d[i + 1], d[i + 2])
        if d[i] > SPIKE * max(around, 0.006) and dh[i] > 0.04:
            spikes.append(i)
    return out, spikes


def _blend_fit(pre: np.ndarray, mid: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    """Best alpha with mid ≈ (1-alpha)·pre + alpha·post, and the leftover as a share of the pre/post gap."""
    a, b, m = pre.ravel(), post.ravel(), mid.ravel()
    diff = b - a
    den = float(diff @ diff)
    if den < 1e-6:
        return 0.0, 1.0
    alpha = float(np.clip(((m - a) @ diff) / den, 0, 1))
    res = np.abs(m - (a + alpha * diff)).mean()
    return alpha, float(res / max(1e-6, np.abs(diff).mean()))


def _side(mask: np.ndarray) -> str | None:
    """Which edge of the frame a region touches most: where something enters from."""
    h, w = mask.shape
    if not mask.any():
        return None
    edges = {"left": mask[:, 0].mean(), "right": mask[:, -1].mean(), "top": mask[0, :].mean(), "bottom": mask[-1, :].mean()}
    side = max(edges, key=edges.get)
    return side if edges[side] > 0.2 else None


def _pieces(mask: np.ndarray) -> bool:
    """Is this mask scattered blocks: 8+ pieces, none holding half of it, together covering 15%+ of the frame?"""
    if mask.mean() < 0.15:
        return False
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= 2]
    return len(areas) >= 8 and areas.max() < 0.5 * areas.sum()


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return (xs.mean() / mask.shape[1], ys.mean() / mask.shape[0]) if len(xs) else (0.5, 0.5)


def _sweep_from(c0: tuple[float, float], c1: tuple[float, float]) -> str | None:
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    if max(abs(dx), abs(dy)) < 0.08:
        return None
    h = "left" if dx > 0 else "right"
    v = "top" if dy > 0 else "bottom"
    if min(abs(dx), abs(dy)) > 0.5 * max(abs(dx), abs(dy)):
        return f"{v}-{h}"   # a diagonal sweep
    return h if abs(dx) >= abs(dy) else v


def _shape(mask: np.ndarray) -> str | None:
    """The edge of a growing reveal: "circle" (an iris) or "line" (a straight wipe), from its boundary points
    away from the frame's border."""
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n_lab < 2:
        return None
    big = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = cs[0][:, 0, :].astype(np.float64)
    h, w = mask.shape
    pts = pts[(pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)]
    if len(pts) < 8:
        return None
    # straight: the points lie on a line
    c = pts - pts.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    if sv[1] < 0.08 * sv[0]:
        return "line"
    # round: least-squares circle x^2 + y^2 + Dx + Ey + F = 0
    A = np.c_[pts, np.ones(len(pts))]
    rhs = -(pts ** 2).sum(axis=1)
    D, E, F = np.linalg.lstsq(A, rhs, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = math.sqrt(max(1e-6, cx * cx + cy * cy - F))
    err = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).mean()
    # a real arc: a radius on the frame's scale and the points spread over more than a quarter turn
    ang = np.sort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    spread = 2 * math.pi - max(np.max(np.diff(ang)), 2 * math.pi - (ang[-1] - ang[0]))
    if err < 0.06 * r and 3 < r < 1.2 * max(h, w) and spread > math.pi / 2:
        return "circle"
    return None


def _zoom_transition(ctx: Ctx, t_pre: float, t_post: float, n: int) -> dict | None:
    """The picture keeps its content but changes scale steadily through the transition (footage shrinking into
    a card, a card growing to fill the frame): a zoom transition. The scale is chained frame to frame (neighbours
    always match well, while the two ends may share only a small face), and the curve is fitted to an ease."""
    fps = ctx.fps
    times = [t_pre + k / fps for k in range(n)]
    frames = [gray_at(ctx, t + 0.2 / fps) for t in times]
    logs = []
    for f0, f1 in zip(frames, frames[1:]):
        m = _orb_match(f0, f1)
        if not m or m.get("inliers", 0) < 20 or m.get("ratio", 0) < 0.4:
            return None
        logs.append(math.log(m["scale"]))
    total = math.exp(sum(logs))
    moving = [x for x in logs if abs(x) > 0.005]
    if 0.9 <= total <= 1.1 or not moving or sum(1 for x in moving if (x > 0) == (total > 1)) < 0.8 * len(moving):
        return None
    scales = np.exp(np.r_[0.0, np.cumsum(logs)])
    fit = ease.fit(np.array(times), scales, frame=1 / fps) if len(times) >= 4 else None
    return {"kind": "zoom", "direction": "out" if total < 1 else "in", "scale_to": round(total, 3),
            "ease": fit["ease"] if fit else None, "ease_rmse": fit["rmse"] if fit else None,
            "overshoot": fit["overshoot"] if fit else None, "conf": 0.8}


def _best_shift(F: np.ndarray, X: np.ndarray, axis: int) -> tuple[int, float, float]:
    """The shift along `axis` (0 = down, 1 = right, in thumbnail pixels) that best lines picture X up with part
    of frame F: (shift, share of the frame it explains, how much more that is than with no shift). A flat
    picture lines up anywhere, so only a shift that explains clearly more than staying put counts."""
    n = X.shape[axis]
    shares = {}
    for sh in range(-n + n // 6, n - n // 6):
        if axis == 0:
            a, b = (F[sh:], X[:n - sh]) if sh >= 0 else (F[:n + sh], X[-sh:])
        else:
            a, b = (F[:, sh:], X[:, :n - sh]) if sh >= 0 else (F[:, :n + sh], X[:, -sh:])
        shares[sh] = float((np.abs(a - b).max(axis=2) < 20).sum() / F[..., 0].size)
    best = max(shares, key=lambda k: (shares[k], -abs(k)))
    return best, shares[best], shares[best] - shares.get(0, 0.0)


def _slide(P: np.ndarray, Q: np.ndarray, mids: list[np.ndarray]) -> dict | None:
    """A slide or push: the old picture travels out and/or the new one travels in, along one axis."""
    for axis, (neg, pos) in ((0, ("top", "bottom")), (1, ("left", "right"))):
        old = [_best_shift(F, P, axis) for F in mids]
        new = [_best_shift(F, Q, axis) for F in mids]
        n = P.shape[axis]
        old_moves = [o for o in old if o[1] > 0.2 and abs(o[0]) >= 2 and o[2] > 0.1]
        new_moves = [o for o in new if o[1] > 0.2 and abs(o[0]) >= 2 and o[2] > 0.1]
        o_ok = len(old_moves) >= max(1, len(mids) // 2) and all(abs(b[0]) >= abs(a[0]) for a, b in zip(old_moves, old_moves[1:]))
        n_ok = len(new_moves) >= max(1, len(mids) // 2) and all(abs(b[0]) <= abs(a[0]) for a, b in zip(new_moves, new_moves[1:]))
        if not (o_ok or n_ok):
            continue
        out = {"kind": "slide", "axis": "vertical" if axis == 0 else "horizontal"}
        if n_ok:
            # the new picture starts offset towards the side it enters from
            out["from"] = pos if new_moves[0][0] > 0 else neg
        if o_ok:
            out["old_exits"] = pos if old_moves[-1][0] > 0 else neg
        if o_ok and n_ok:
            out["kind"] = "push"     # both travel together
        out["travel_pct"] = round(100 * max([abs(m[0]) for m in old_moves + new_moves]) / n)
        return out
    return None


def classify_transition(s: dict, a: int, b: int, ctx: Ctx) -> dict:
    """Frames a-1 (the picture before) .. b (the picture after), the ones between changing:
    dissolve, fade (through a colour), mask (scattered blocks, maybe over a colour), wipe (the new picture
    spreads from one side; it may slide in), whip (blur), or transition (none of these).
    Judged in colour: in grey a yellow screen passes for "halfway to white"."""
    pre, post = a - 1, b
    fps = ctx.fps
    out = {"t": r3(s["t"][a]), "dur_s": r3((b - a + 1) / fps), "frames": b - a + 1}
    if b - a < 1:
        return out | {"kind": "transition"}
    _, clip = ctx.clip(float(s["t"][pre]), float(s["t"][post]) + 0.5 / fps, long_edge=WORK)
    if len(clip) < 3:
        return out | {"kind": "transition"}
    zoom = _zoom_transition(ctx, float(s["t"][pre]), float(s["t"][post]), len(clip))
    if zoom:
        return out | zoom
    tw, th = THUMB if ctx.portrait else THUMB[::-1]
    fr = [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) for f in clip]
    P, Q, mids = fr[0], fr[-1], fr[1:-1]
    gap = np.abs(P - Q).max(axis=2)
    tol = np.maximum(18, 0.3 * gap)
    changed = gap > 25
    fits, shares, others, coherent, flat_other, other_rgb = [], [], [], [], [], []
    news: list[np.ndarray] = []
    others_m: list[np.ndarray] = []
    first_new = None
    for F in mids:
        fits.append(_blend_fit(P, F, Q))
        near_pre = np.abs(F - P).max(axis=2) < tol
        near_post = np.abs(F - Q).max(axis=2) < tol
        new = near_post & ~near_pre & changed
        other = ~near_pre & ~near_post             # neither picture: an in-between colour or effect
        shares.append(new.sum() / max(1, changed.sum()))
        others.append(float(other.mean()))
        flat_other.append(float(F[other].std(axis=0).mean()) if other.sum() > 20 else 99.0)
        other_rgb.append(np.median(F[other], axis=0) if other.sum() > 20 else None)
        # a block mask breaks the new picture (or the colour it passes through) into many like-sized pieces
        # covering much of the frame; a slide or wipe has one region that dominates, with a thin fringe
        coherent.append(any(_pieces(m) for m in (new, other)))
        if first_new is None and shares[-1] > 0.15:
            first_new = new
        news.append(new)
        others_m.append(other)
    alphas = [f[0] for f in fits]
    resid = float(np.median([f[1] for f in fits]))
    rising = all(alphas[k + 1] >= alphas[k] - 0.15 for k in range(len(alphas) - 1))
    k_through = int(np.argmax(others))
    through = None
    if others[k_through] > 0.4 and flat_other[k_through] < 20 and other_rgb[k_through] is not None:
        through = hexcolor(other_rgb[k_through])
    # a blend first: in a block mask each pixel is wholly old or wholly new, which no blend can fit
    if resid < 0.22 and rising and not through:
        ends = _uniform(s, pre) or _uniform(s, post)
        return out | {"kind": "fade" if ends else "dissolve", "through": ends, "conf": round(1 - resid, 2)}
    if through:
        C = np.ones_like(P) * np.array(hex_to_rgb(through), np.float32)
        r_th = float(np.median([min(_blend_fit(P, F, C)[1], _blend_fit(C, F, Q)[1]) for F in mids]))
        if r_th < 0.25:
            return out | {"kind": "fade", "through": through, "conf": round(1 - r_th, 2)}
    slide = _slide(P, Q, mids)
    if slide:
        return out | slide | {"conf": 0.7}
    scattered = sum(coherent) >= 2
    if scattered:
        return out | {"kind": "mask", "pattern": "blocks", "through": through, "conf": 0.7}
    # the new picture takes over: it ends up showing in most of what changes and mostly rises on the way
    # (a slide may pause or settle, so not strictly)
    rises = sum(1 for k in range(len(shares) - 1) if shares[k + 1] >= shares[k] - 0.02)
    grows = max(shares) > 0.5 and shares[-1] > 0.3 and rises >= 0.7 * (len(shares) - 1)
    if grows:
        if through:
            # a band of colour sweeps across: it enters on the side it moves away from
            seen = [_centroid(m) for m in others_m if m.mean() > 0.1]
            frm = _sweep_from(seen[0], seen[-1]) if len(seen) >= 2 else None
        else:
            # the new picture's centre travels away from the side it comes in from (edges can hide it when
            # both pictures share a colour there)
            seen = [_centroid(m) for m, sh in zip(news, shares) if sh > 0.15]
            frm = _sweep_from(seen[0], _centroid(news[-1])) if seen else None
            if frm is None and first_new is not None:
                frm = _side(first_new)
        mid = next((m for m, sh in zip(news, shares) if sh > 0.3), None)
        return out | {"kind": "wipe", "from": frm, "shape": _shape(mid) if mid is not None else None,
                      "through": through, "conf": 0.6}
    if through:
        return out | {"kind": "fade", "through": through, "conf": 0.5}
    sharp = [cv2.Laplacian(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in fr]
    if min(sharp[1:-1]) < 0.35 * min(sharp[0], sharp[-1]):
        return out | {"kind": "whip", "conf": 0.5}
    return out | {"kind": "transition", "conf": 0.4}


def _orb_match(f1: np.ndarray, f2: np.ndarray) -> dict | None:
    """Similarity transform from frame 1 to frame 2 (scale, shift as a fraction of the width) and how well it fits."""
    g1 = f1 if f1.ndim == 2 else cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = f2 if f2.ndim == 2 else cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(1500)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return {"inliers": len(good), "ratio": 0.0}
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return {"inliers": 0, "ratio": 0.0}
    n_in = int(inl.sum())
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    w = g1.shape[1]
    return {"inliers": n_in, "ratio": n_in / len(good), "scale": scale,
            "dx": float(M[0, 2] / w), "dy": float(M[1, 2] / w), "rot": float(math.degrees(math.atan2(M[1, 0], M[0, 0])))}


def _partial(s: dict, i: int) -> dict | None:
    """Is only part of the frame changing at step i (an insert)? A band (top/bottom/left/right) or a box."""
    P, Q = s["thumb"][i - 1], s["thumb"][i]
    h, w = P.shape
    bh, bw = h // 12, w // 6
    grid = np.zeros((12, 6))
    for r in range(12):
        for c in range(6):
            grid[r, c] = np.abs(P[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] - Q[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]).mean() / 255
    changed = grid > 0.08
    frac = changed.mean()
    if frac < 0.12 or frac > 0.8:
        return None
    rows = changed.mean(axis=1)
    for r in range(2, 11):
        top, bot = rows[:r].mean(), rows[r:].mean()
        if top > 0.8 and bot < 0.15:
            return {"region": "top", "split_pct": round(100 * r / 12), "share": round(float(frac), 2)}
        if bot > 0.8 and top < 0.15:
            return {"region": "bottom", "split_pct": round(100 * r / 12), "share": round(float(frac), 2)}
    ys, xs = np.nonzero(changed)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0.7 * 72 and changed[box[1]:box[3], box[0]:box[2]].mean() > 0.8 and grid[changed].mean() > 0.12:
        return {"region": "box", "box": [round(box[0] / 6, 2), round(box[1] / 12, 2), round((box[2] - box[0]) / 6, 2),
                                         round((box[3] - box[1]) / 12, 2)], "share": round(float(frac), 2)}
    return None


def classify_instant(s: dict, i: int, ctx: Ctx, pre: int | None = None) -> dict:
    """A cut at frame i (the first new frame), compared with frame `pre` (default i-1)."""
    t = float(s["t"][i])
    pre = i - 1 if pre is None else pre
    out = {"t": r3(t)}
    n = len(s["t"])
    # snaps back within a few frames: a flash or a flicker, not a cut
    back = min((_dist(s, i - 1, j) for j in range(i + 1, min(n, i + 6))), default=1.0)
    if back < 0.12:
        u = _uniform(s, i)
        return out | {"kind": "flash" if u else "flicker", "through": u, "cut": False}
    if _uniform(s, i) and i + 1 < n:
        return out | {"kind": "flash", "through": _uniform(s, i), "cut": True, "conf": 0.6}
    part = _partial(s, i)
    if part:
        return out | {"kind": "insert", **part, "cut": False, "conf": 0.7}
    f1 = gray_at(ctx, float(s["t"][pre]) + 0.2 / ctx.fps)
    f2 = gray_at(ctx, t + 0.2 / ctx.fps)
    m = _orb_match(f1, f2)
    if m and m.get("inliers", 0) >= 25 and m.get("ratio", 0) >= 0.35:
        sc, shift = m["scale"], math.hypot(m["dx"], m["dy"])
        conf = round(min(1.0, m["ratio"]), 2)
        if 0.96 <= sc <= 1.04:
            if s["dh"][i] > STRONG_H:
                # the subject holds its place while the colours of everything change: a match cut
                return out | {"kind": "match", "conf": conf, "cut": True}
            change = np.abs(s["thumb"][pre] - s["thumb"][i]) > 25
            ys, xs = np.nonzero(change)
            if shift < 0.004 and len(xs):
                h, w = change.shape
                box = [xs.min() / w, ys.min() / h, (xs.max() + 1 - xs.min()) / w, (ys.max() + 1 - ys.min()) / h]
                if box[2] * box[3] < 0.45:
                    # nothing moved but one region: a cut inside a card or a split, not across the frame
                    return out | {"kind": "insert", "region": "box", "box": [round(x, 2) for x in box], "cut": False, "conf": conf}
            return out | {"kind": "jump", "shift": round(shift, 3), "conf": conf, "cut": True}
        if 0.7 <= sc <= 1.45:
            return out | {"kind": "zoom-punch", "scale": round(sc, 3), "conf": conf, "cut": True}
        return out | {"kind": "layout", "scale": round(sc, 3), "conf": conf, "cut": True}
    return out | {"kind": "hard", "conf": 0.8 if s["dh"][i] > STRONG_H else 0.6, "cut": True}


def gradual(ctx: Ctx, s: dict, taken: set[int]) -> list[dict]:
    """Slow transitions no single frame gives away: a window whose two ends differ a lot, with the change spread
    over its frames and a steady picture on both sides. Kept only when it really is a blend, so a slow pan, a
    gesture or a graphic gliding across is never called a cut."""
    n = len(s["t"])
    found = []
    for L in sorted({max(3, int(ctx.fps * x)) for x in (0.2, 0.4, 0.7, 1.0)}):
        i = 4
        while i + L + 3 < n:
            if any(k in taken for k in range(i - 1, i + L + 2)):
                i += 1
                continue
            total = _dist(s, i - 1, i + L)
            if total > 0.35:
                steps = s["d"][i:i + L + 1]
                # spread out: no single frame step carries most of the change (that would be an instant cut).
                # Judged on grey pixels: a blend moves them smoothly while the colour histogram jumps between bins.
                pix_total = float(np.abs(s["thumb"][i - 1] - s["thumb"][i + L]).mean() / 255)
                if steps.max() < 0.5 * pix_total and _dist(s, i - 4, i - 1) < 0.1 and _dist(s, i + L, i + L + 3) < 0.1:
                    # trim quiet frames at either end so the transition starts and ends where the change does
                    act = np.nonzero(steps > 0.25 * np.median(steps[steps > 0]) if (steps > 0).any() else steps > 0)[0]
                    a, b = i + int(act[0]), i + int(act[-1])
                    ev = classify_transition(s, a, b, ctx)
                    # only true blends: wipes and masks are quick and the runs find them; a slow "mask" is
                    # hands waving or a page scrolling
                    if ev["kind"] in ("dissolve", "fade"):
                        found.append(ev)
                        taken.update(range(a - 1, b + 2))
                        i = b + 1
                        continue
            i += 1
    return found


def find_cuts(ctx: Ctx, s: dict) -> tuple[list[dict], list[dict]]:
    """(cuts, other events): cuts split shots; flashes, flickers and inserts are events inside shots."""
    cuts, other = [], []
    runs, spikes = candidates(s, ctx.fps)
    n = len(s["t"])
    taken: set[int] = set()
    for a, b in runs:
        if a < 1:
            continue
        taken.update(range(a, b + 1))
        if b == a:
            ev = classify_instant(s, a, ctx)
        else:
            gap = _dist(s, a - 1, min(n - 1, b))
            if gap < 0.12:
                # a burst of change that ends where it began: a flash, a glitch or a gesture, not a cut
                if b - a < 6:
                    other.append({"t": r3(s["t"][a]), "kind": "flicker", "dur_s": r3((b - a + 1) / ctx.fps)})
                continue
            if gap < 0.3:
                continue   # the picture moved but stayed the same shot (a graphic animating, a pan)
            ev = classify_transition(s, a, b, ctx)
            ev["cut"] = True
            if ev["kind"] == "dissolve" and ev["frames"] <= 2:
                # one or two blended frames: a soft cut (often re-encoding), judged as the cut it is
                ev = {**classify_instant(s, b, ctx, pre=a - 1), "t": r3(s["t"][a]), "blend_frames": ev["frames"]}
        (cuts if ev.pop("cut", ev["kind"] in FULL_CUTS) else other).append(ev)
    cuts.extend(gradual(ctx, s, taken))   # slow dissolves in the quiet stretches; adds their frames to `taken`
    for i in spikes:
        if i in taken or i < 1:
            continue
        ev = classify_instant(s, i, ctx)
        (cuts if ev.pop("cut", ev["kind"] in FULL_CUTS) else other).append(ev)
        taken.add(i)
    # quiet cuts in a busy shot: a mild punch-in or jump while hands move barely stands out, so a smaller
    # spike counts too, but only when matching features confirm the picture itself jumped
    have = [c["t"] for c in cuts] + [o["t"] for o in other]
    d = s["d"]
    quiet = []
    for i in range(2, len(d) - 2):
        if i in taken or d[i] < 0.02 or any(abs(s["t"][i] - t) < 0.2 for t in have):
            continue
        around = max(d[i - 2], d[i - 1], d[i + 1], d[i + 2])
        if d[i] >= 1.8 * max(around, 0.005):
            quiet.append((d[i] / max(around, 0.005), i))
    # checking costs two seeks and a feature match each: only the 30 that stand out most
    for _, i in sorted(quiet, reverse=True)[:30]:
        if any(abs(s["t"][i] - t) < 0.2 for t in have):
            continue
        f1 = gray_at(ctx, float(s["t"][i - 1]) + 0.2 / ctx.fps)
        f2 = gray_at(ctx, float(s["t"][i]) + 0.2 / ctx.fps)
        m = _orb_match(f1, f2)
        if not m or m.get("inliers", 0) < 40 or m.get("ratio", 0) < 0.5:
            continue
        shift = math.hypot(m["dx"], m["dy"])
        if 1.04 <= m["scale"] <= 1.45 or 0.7 <= m["scale"] <= 0.96:
            cuts.append({"t": r3(s["t"][i]), "kind": "zoom-punch", "scale": round(m["scale"], 3), "conf": round(m["ratio"], 2), "quiet": True})
        elif shift > 0.02:
            cuts.append({"t": r3(s["t"][i]), "kind": "jump", "shift": round(shift, 3), "conf": round(m["ratio"], 2), "quiet": True})
        else:
            continue
        have.append(float(s["t"][i]))
    return sorted(cuts, key=lambda c: c["t"]), sorted(other, key=lambda c: c["t"])


# ─── shots, what fills them, the camera ──────────────────────────────────────────────────────────

def shots_of(ctx: Ctx, cuts: list[dict]) -> list[dict]:
    bounds = [0.0] + [c["t"] for c in cuts] + [ctx.duration]
    shots = []
    for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
        if b - a > 1e-3:
            shots.append({"i": k, "a": r3(a), "b": r3(b), "len": r3(b - a)})
    return shots


def _flatness(f: np.ndarray) -> float:
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float((np.hypot(gx, gy) < 6).mean())


def split_line(f: np.ndarray) -> float | None:
    """A split screen's divider: a row between 25% and 75% of the height where the picture changes across most
    of the width, standing well above the rows around it (a line, not texture). Its height as a fraction, or None."""
    f = cv2.resize(f, (270, max(2, int(270 * f.shape[0] / f.shape[1]))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    h = g.shape[0]
    strong = (np.abs(g[2:] - g[:-2]) > 10).mean(axis=1)   # across two rows: a soft edge still counts
    lo, hi = int(0.25 * h), int(0.75 * h)
    r = lo + int(np.argmax(strong[lo:hi]))
    if strong[r] >= 0.65 and strong[r] - float(np.median(strong[lo:hi])) >= 0.35:
        return round((r + 1) / h, 3)
    return None


def fill_shots(ctx: Ctx, shots: list[dict]) -> None:
    """What fills each shot (a code guess): face, screen, face+screen, graphic, broll."""
    faces = ctx.faces_track()
    mids = [(s["a"] + s["b"]) / 2 for s in shots]
    frames = ctx.frames_at(mids, long_edge=640)
    texts = ctx.vision("text", frames)
    for s, f, tx in zip(shots, frames, texts):
        fs = [x for x in faces if s["a"] <= x["t"] <= s["b"]] or \
             ([min(faces, key=lambda x: abs(x["t"] - (s["a"] + s["b"]) / 2))] if faces else [])
        face_area = np.median([max([fc["box"][2] * fc["box"][3] for fc in x["faces"]] or [0]) for x in fs]) if fs else 0
        lines = tx.get("lines", []) or []
        small = [l for l in lines if l.get("box") and l["box"][3] < 0.03]
        text_area = sum(l["box"][2] * l["box"][3] for l in lines if l.get("box"))
        flat = _flatness(f)
        face = face_area > 0.012
        screen = len(small) >= 6 or text_area > 0.08
        split = split_line(f)
        if split and face and fs:
            # a split screen: what fills the half without the face
            fy = np.median([x["faces"][0]["box"][1] + x["faces"][0]["box"][3] / 2 for x in fs if x["faces"]] or [0.5])
            other = [l for l in lines if l.get("box") and ((l["box"][1] < split) == (fy > split))]
            kind = "face+screen" if len(other) >= 3 or screen else "face+graphic"
        else:
            kind = "face+screen" if face and screen else "face" if face else "screen" if screen else \
                "graphic" if flat > 0.6 else "broll"
        s.update(fill=kind, face_area_pct=round(100 * float(face_area), 1), text_lines=len(lines), flat=round(flat, 2),
                 split_at=split)


def camera(ctx: Ctx, shot: dict, person: np.ndarray | None = None, digital: bool = False) -> dict:
    """The camera inside one shot (between its transitions): cumulative scale (a push), shift (a pan) and jitter
    (handheld shake), from frame-to-frame similarity transforms of the background. The speaker is masked out
    (`person`: Apple Vision's person mask, 0..1) and a step only counts when most tracked points agree, so a
    gesture or an animating graphic is not taken for the camera. A push or pan has to be steady in one direction
    and confirmed by matching the shot's first and last frames; otherwise the camera is "static" or "unknown"."""
    a, b = shot.get("a_clean", shot["a"]) + 0.5 / ctx.fps, shot["b"] - 0.5 / ctx.fps
    if b - a < 0.6:
        return {"kind": "static", "short": True}
    ts, gs = [], []
    for t, g in ctx.frames(a, b, fps=CAM_FPS, long_edge=CAM_EDGE, gray=True):
        ts.append(t)
        gs.append(g)
    if len(gs) < 6:
        return {"kind": "static", "short": True}
    h, w = gs[0].shape
    mask = np.full((h, w), 255, np.uint8)
    if person is not None and person.size:
        pm = cv2.resize(person, (w, h)) > 0.4
        pm = cv2.dilate(pm.astype(np.uint8), np.ones((max(3, w // 20),) * 2, np.uint8)) > 0
        mask[pm] = 0
    if (mask > 0).mean() < 0.15:
        return {"kind": "unknown", "note": "the speaker fills the frame; no background to follow"}
    steps: list[tuple[float, float, float] | None] = []
    for g0, g1 in zip(gs, gs[1:]):
        p0 = cv2.goodFeaturesToTrack(g0, maxCorners=300, qualityLevel=0.01, minDistance=6, mask=mask)
        step = None
        if p0 is not None and len(p0) >= 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 20:
                M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                if M is not None and inl is not None and inl.sum() >= 20 and inl.mean() >= 0.6:
                    step = (float(math.hypot(M[0, 0], M[1, 0])), float(M[0, 2] / w), float(M[1, 2] / w))
        steps.append(step)
    known = [st for st in steps if st is not None]
    out: dict = {"known_pct": pct(len(known), len(steps))}
    if len(known) < 0.7 * len(steps):
        return out | {"kind": "unknown", "note": "too few steps where the background agreed"}
    sc = np.array([st[0] for st in known])
    dx = np.array([st[1] for st in known])
    dy = np.array([st[2] for st in known])
    logS = np.cumsum(np.log(sc))
    X, Y = np.cumsum(dx), np.cumsum(dy)
    # handheld: frame-to-frame wobble left after taking out the smooth movement (% of the width per step)
    def wobble(v):
        return float(np.std(v - np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")))
    shake = max(wobble(dx), wobble(dy)) * 100
    out["shake"] = round(shake, 3)
    kinds = []
    total_s = float(np.exp(logS[-1]))
    steady_s = (np.sign(np.log(sc)) == np.sign(logS[-1]))[np.abs(np.log(sc)) > 1e-4].mean() if (np.abs(np.log(sc)) > 1e-4).any() else 0
    if abs(total_s - 1) > 0.05 and steady_s > 0.75:
        f0 = gray_at(ctx, ts[0])
        f1 = gray_at(ctx, ts[-1])
        check = _orb_match(f0, f1)
        if check and check.get("inliers", 0) >= 25 and abs(check["scale"] - total_s) < 0.04:
            fit = ease.fit(np.array([ts[0]] + [ts[i + 1] for i, st in enumerate(steps) if st is not None]),
                           np.exp(np.r_[0.0, logS]), frame=1 / CAM_FPS)
            out["push"] = {"direction": "in" if total_s > 1 else "out", "scale_from": 1.0, "scale_to": round(total_s, 3),
                           "start": fit["start"] if fit else r3(ts[0]), "duration_s": fit["duration"] if fit else r3(ts[-1] - ts[0]),
                           "ease": fit["ease"] if fit else None, "rmse": fit["rmse"] if fit else None}
            kinds.append("push")
    move = max(abs(X[-1]), abs(Y[-1]))
    main = dx if abs(X[-1]) >= abs(Y[-1]) else dy
    steady_m = (np.sign(main) == np.sign(main.sum()))[np.abs(main) > 5e-4].mean() if (np.abs(main) > 5e-4).any() else 0
    if move > 0.06 and steady_m > 0.75:
        # the picture moves one way when the camera moves the other
        out["pan"] = {"direction": ("left" if X[-1] > 0 else "right") if abs(X[-1]) >= abs(Y[-1]) else ("up" if Y[-1] > 0 else "down"),
                      "amount_pct": round(100 * move, 1)}
        kinds.append("pan")
    # graphics wobble only when an element animates, so a made canvas needs a clear, deliberate shake
    if shake > (1.0 if digital else 0.35):
        kinds.append("shake" if digital else "handheld")
    out["kind"] = "+".join(kinds) or "static"
    return out


# ─── putting it together ─────────────────────────────────────────────────────────────────────────

def sentence_starts(words: list[dict]) -> list[float]:
    out = []
    prev = None
    for w in words:
        if prev is None or w["a"] - prev["b"] >= 0.25 or str(prev["w"]).rstrip().endswith((".", "?", "!", "।")):
            out.append(w["a"])
        prev = w
    return out


def zoom_on(zt: list[float], n_punch: int, n_cuts: int, words: list[dict], stressed: list[float]) -> str | None:
    """What the zooms are timed to: sentence starts, stressed words, every cut, or something else."""
    if not zt:
        return None
    starts = sentence_starts(words)
    votes = {"sentence_start": sum(near(t, starts, 0.25) for t in zt),
             "emphasis": sum(near(t, stressed, 0.3) for t in zt) if stressed else 0,
             "every_cut": n_punch if n_punch and n_punch >= 0.6 * n_cuts else 0}
    best = max(votes, key=votes.get)
    return best if votes[best] >= max(2, 0.4 * len(zt)) else "other"


def rolling(times: list[float], duration: float, window: float = 10.0, step: float = 2.0) -> list[list[float]]:
    out = []
    t = 0.0
    while t <= max(0.0, duration - window) + 1e-6:
        n = sum(1 for x in times if t <= x < t + window)
        out.append([round(t + window / 2, 1), round(n * 10.0 / window, 2)])
        t += step
    if not out and duration > 0:
        out.append([round(duration / 2, 1), round(len(times) * 10.0 / duration, 2)])
    return out


def run(ctx: Ctx) -> dict:
    eprint("edit: reading every frame …")
    s = signals(ctx)
    cuts, other = find_cuts(ctx, s)
    shots = shots_of(ctx, cuts)
    eprint(f"edit: {len(cuts)} cuts, {len(shots)} shots; what fills them …")
    fill_shots(ctx, shots)
    eprint("edit: following the camera …")
    ends = {c["t"]: c["t"] + (c.get("dur_s") or 0) for c in cuts}
    for sh in shots:
        sh["a_clean"] = r3(ends.get(sh["a"], sh["a"]))   # after the transition that opens the shot
    follow = [sh for sh in shots if ctx.in_sections((sh["a"] + sh["b"]) / 2) and sh["b"] - sh["a_clean"] >= 0.6]
    masks = ctx.vision("person", ctx.frames_at([(sh["a_clean"] + sh["b"]) / 2 for sh in follow], long_edge=CAM_EDGE)) if follow else []
    for sh, m in zip(follow, masks):
        sh["camera"] = camera(ctx, sh, m.get("mask") if isinstance(m.get("mask"), np.ndarray) else None,
                              digital=sh.get("fill") in ("graphic", "screen"))
    d = ctx.duration
    lens = [sh["len"] for sh in shots]
    cut_t = [c["t"] for c in cuts]
    kinds: dict[str, int] = {}
    for c in cuts:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    cut_kinds = {k: pct(v, len(cuts)) for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])}
    # bursts: four or more cuts with under 0.4 s between them (a montage)
    bursts, cur = [], [cut_t[0]] if cut_t else []
    for x, y in zip(cut_t, cut_t[1:]):
        if y - x < 0.4:
            cur.append(y)
        else:
            if len(cur) >= 4:
                bursts.append({"a": cur[0], "b": cur[-1], "cuts": len(cur)})
            cur = [y]
    if len(cur) >= 4:
        bursts.append({"a": cur[0], "b": cur[-1], "cuts": len(cur)})
    hook_n = sum(1 for x in cut_t if x < min(10.0, d))
    pace = {
        "cuts": len(cuts), "cuts_per_10s": round(len(cuts) * 10 / d, 2) if d else 0,
        "cuts_per_min": round(len(cuts) * 60 / d, 1) if d else 0,
        "median_shot_s": r3(np.median(lens)) if lens else None, "mean_shot_s": r3(np.mean(lens)) if lens else None,
        "longest_hold_s": r3(max(lens)) if lens else None,
        "longest_hold_at": shots[int(np.argmax(lens))]["a"] if lens else None,
        "hist": {k: sum(1 for x in lens if lo <= x < hi) for k, lo, hi in
                 (("<0.5s", 0, .5), ("0.5-1s", .5, 1), ("1-2s", 1, 2), ("2-4s", 2, 4), ("4-8s", 4, 8), ("8s+", 8, 1e9))},
        "curve": rolling(cut_t, d),
        "hook_cuts_per_10s": round(hook_n * 10 / min(10.0, d), 2) if d else 0,
        "bursts": bursts,
    }
    # zoom: punch-ins on cuts and pushes inside shots
    punches = [c for c in cuts if c["kind"] == "zoom-punch"]
    pushes = [{"t": sh["camera"]["push"]["start"], **sh["camera"]["push"]} for sh in shots
              if isinstance(sh.get("camera"), dict) and sh["camera"].get("push")]
    zt = [p["t"] for p in punches] + [p["t"] for p in pushes]
    stressed = [e["t"] for e in (ctx.results.get("delivery") or {}).get("emphasis", []) if isinstance(e, dict)]
    on = zoom_on(zt, len(punches), len(cuts), ctx.words(), stressed)
    minutes = d / 60 if d else 1
    zoom = {
        "per_min": round(len(zt) / minutes, 1),
        "kind": "both" if punches and pushes else "punch" if punches else "push" if pushes else "none",
        "scale": round(float(np.median([p["scale"] for p in punches])), 3) if punches else
                 (round(float(np.median([p["scale_to"] for p in pushes])), 3) if pushes else None),
        "duration_s": 0.0 if punches and not pushes else (r3(np.median([p["duration_s"] for p in pushes])) if pushes else None),
        "ease": "none" if punches and not pushes else (pushes[0].get("ease") if pushes else None),
        "on": on,
        "punches": [{"t": p["t"], "scale": p["scale"]} for p in punches],
        "pushes": pushes,
    }
    cams = [sh["camera"] for sh in shots if isinstance(sh.get("camera"), dict) and not sh["camera"].get("short")]
    cam_time = sum(sh["len"] for sh in shots if isinstance(sh.get("camera"), dict) and not sh["camera"].get("short")) or 1
    camera_sum = {
        "pan_per_min": round(sum(1 for c in cams if "pan" in c.get("kind", "")) / minutes, 1),
        "push_per_min": round(len(pushes) / minutes, 1),
        "shake_pct": pct(sum(sh["len"] for sh in shots if any(k in str((sh.get("camera") or {}).get("kind")) for k in ("handheld", "shake"))), cam_time),
        "static_pct": pct(sum(sh["len"] for sh in shots if (sh.get("camera") or {}).get("kind") == "static"
                               and not sh["camera"].get("short")), cam_time),
    }
    # rhythm: every visual change the edit makes (graphics are added by cross.py)
    events = [{"t": c["t"], "kind": "cut:" + c["kind"]} for c in cuts]
    events += [{"t": e["t"], "kind": e["kind"]} for e in other if e["kind"] in ("insert", "flash")]
    events += [{"t": p["t"], "kind": "zoom-push"} for p in pushes]
    events.sort(key=lambda e: e["t"])
    et = [0.0] + [e["t"] for e in events] + [d]
    gaps = np.diff(et) if len(et) > 1 else np.array([d])
    rhythm = {"change_every_s": r3(np.median(gaps)), "p90_s": r3(np.percentile(gaps, 90)),
              "longest_static_s": r3(gaps.max()), "longest_static_at": r3(et[int(np.argmax(gaps))]),
              "events": events, "curve": rolling([e["t"] for e in events], d)}
    # fills as share of time
    fills: dict[str, float] = {}
    for sh in shots:
        fills[sh.get("fill", "?")] = fills.get(sh.get("fill", "?"), 0) + sh["len"]
    fill_pct = {k: pct(v, d) for k, v in sorted(fills.items(), key=lambda kv: -kv[1])}

    evidence = []
    shown: dict[str, int] = {}
    for c in cuts + [e for e in other if e["kind"] in ("insert", "flash")]:
        k = c["kind"]
        if shown.get(k, 0) >= 2:
            continue
        shown[k] = shown.get(k, 0) + 1
        dur = c.get("dur_s") or 0
        times = [max(0.0, c["t"] - 1.5 / ctx.fps)] + ([c["t"] + dur * f for f in (0.33, 0.66)] if dur else []) + [c["t"] + dur + 0.5 / ctx.fps]
        img = ctx.strip(f"edit-{k}-{shown[k]}.jpg", times, long_edge=240)
        evidence.append({"t": c["t"], "what": f"{k} cut" if c in cuts else k, "img": img})
    evidence.insert(0, {"t": 0.0, "what": "shot timeline (colour = what fills the shot, ticks = cuts)",
                        "img": shot_strip(ctx, shots, cuts)})

    summary = [
        f"{len(cuts)} cuts in {d:.1f} s: {pace['cuts_per_10s']} per 10 s, median shot {pace['median_shot_s']} s, "
        f"longest hold {pace['longest_hold_s']} s at {pace['longest_hold_at']} s.",
        "Cut kinds: " + ", ".join(f"{k} {v:.0f}%" for k, v in cut_kinds.items()) + "." if cut_kinds else "No cuts.",
        "What fills the screen (guess): " + ", ".join(f"{k} {v:.0f}%" for k, v in fill_pct.items()) + ".",
        f"Something new from the edit every {rhythm['change_every_s']} s (median; p90 {rhythm['p90_s']} s); "
        f"longest stretch without one {rhythm['longest_static_s']} s from {rhythm['longest_static_at']} s.",
    ]
    if bursts:
        summary.append("Montage bursts: " + "; ".join(f"{b['cuts']} cuts in {b['b'] - b['a']:.1f} s at {b['a']:.1f} s" for b in bursts) + ".")
    if pace["hook_cuts_per_10s"] and pace["cuts_per_10s"]:
        summary.append(f"Hook pace: {pace['hook_cuts_per_10s']} cuts per 10 s in the first 10 s against {pace['cuts_per_10s']} overall.")
    if zoom["kind"] != "none":
        summary.append(f"Zoom: {zoom['kind']}, {zoom['per_min']}/min, scale {zoom['scale']}"
                       + (f", on {zoom['on']}" if zoom["on"] else "") + ".")
    inserts = [e for e in other if e["kind"] == "insert"]
    if inserts:
        summary.append(f"{len(inserts)} inserts swap part of the frame without a full cut (e.g. {inserts[0]['region']} at {inserts[0]['t']} s).")
    summary.append(f"Camera: shake in {camera_sum['shake_pct']}% of followed time, static {camera_sum['static_pct']}%, "
                   f"{camera_sum['pan_per_min']} pans/min, {camera_sum['push_per_min']} pushes/min.")
    return {"summary": summary, "cuts": cuts, "events_other": other, "cut_kinds": cut_kinds, "shots": shots,
            "fill_pct": fill_pct, "pace": pace, "zoom": zoom, "camera": camera_sum, "rhythm": rhythm, "evidence": evidence}


FILL_RGB = {"face": (10, 132, 255), "face+screen": (94, 92, 230), "face+graphic": (100, 210, 255), "screen": (48, 209, 88), "graphic": (255, 159, 10),
            "broll": (191, 90, 242), "?": (142, 142, 147)}


def shot_strip(ctx: Ctx, shots: list[dict], cuts: list[dict]) -> str:
    W, H = 1000, 46
    img = np.full((H + 16, W, 3), 250, np.uint8)
    d = max(ctx.duration, 1e-6)
    for sh in shots:
        x0, x1 = int(sh["a"] / d * W), max(int(sh["a"] / d * W) + 1, int(sh["b"] / d * W))
        img[4:H - 4, x0:x1] = FILL_RGB.get(sh.get("fill", "?"), FILL_RGB["?"])
    for c in cuts:
        x = min(W - 1, int(c["t"] / d * W))
        img[0:H, x:x + 1] = 20
    return ctx.img("edit-shots.png", img)


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    import os
    import subprocess
    import tempfile
    import vu
    with tempfile.TemporaryDirectory() as td:
        os.environ["VU_CACHE"] = str(Path(td) / "cache")
        vu.CACHE = Path(os.environ["VU_CACHE"])
        src = Path(td) / "e.mp4"
        # 0-1.5 s a moving test card; 1.5 s the same card punched in 1.25x; 3.0 s a hard cut to a fractal;
        # 4.5 s a 0.4 s dissolve into colour bars
        g = ("testsrc2=s=360x640:r=30:d=3,split[a][b];[a]trim=0:1.5,setpts=PTS-STARTPTS[a1];"
             "[b]trim=1.5:3,setpts=PTS-STARTPTS,crop=iw/1.25:ih/1.25,scale=360:640,setsar=1[b1];"
             "mandelbrot=s=360x640:r=30,trim=0:1.9,setpts=PTS-STARTPTS,format=yuv420p,setsar=1[c];"
             "smptebars=s=360x640:r=30:d=1.5,format=yuv420p,setsar=1[e];"
             "[c][e]xfade=transition=fade:duration=0.4:offset=1.5[ce];"
             "[a1][b1][ce]concat=n=3:v=1:a=0,format=yuv420p[v]")
        subprocess.run(["ffmpeg", "-v", "error", "-filter_complex", g, "-map", "[v]", str(src)], check=True)
        ctx = Ctx(vu.Video(src))
        s = signals(ctx)
        cuts, other = find_cuts(ctx, s)
        got = [(c["kind"], c["t"]) for c in cuts]
        assert len(cuts) == 3, got
        assert cuts[0]["kind"] == "zoom-punch" and abs(cuts[0]["t"] - 1.5) < 0.05 and abs(cuts[0]["scale"] - 1.25) < 0.05, cuts[0]
        assert cuts[1]["kind"] == "hard" and abs(cuts[1]["t"] - 3.0) < 0.05, cuts[1]
        assert cuts[2]["kind"] == "dissolve" and abs(cuts[2]["t"] - 4.5) < 0.1 and 0.25 <= cuts[2]["dur_s"] <= 0.5, cuts[2]
        assert sentence_starts([{"w": "a.", "a": 0, "b": .2}, {"w": "b", "a": .25, "b": .4}, {"w": "c", "a": .5, "b": .6},
                                {"w": "d", "a": 1.0, "b": 1.1}]) == [0, .25, 1.0]
        print("  cuts:", got)
    print("edit demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
