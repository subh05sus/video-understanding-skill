"""Easing: GSAP's named eases, a free cubic-bezier, and a fitter that finds which one a measured movement
used, with its start, duration and overshoot.

A measured channel (x, y, scale, opacity, a wipe's visible fraction ...) goes in as times and values. The
fitter normalises it to progress 0 -> 1 (so 1.08 means 8% past the rest), finds when the movement starts
and how long it runs to sub-frame precision, and scores every GSAP ease against it. The answer names the
closest GSAP ease (what to write in `gsap.to`), a CSS/GSAP CustomEase cubic-bezier fitted freely, and the
error of each in progress units, so "power3.out, 0.42 s, rmse 0.021" can be trusted or not.

    python3 ease.py demo
"""
from __future__ import annotations

import math

import numpy as np

# ─── GSAP eases ──────────────────────────────────────────────────────────────────────────────────

def _power(k):
    def e_in(p):
        return p ** k

    def e_out(p):
        return 1 - (1 - p) ** k

    def e_inout(p):
        return np.where(p < 0.5, (2 * p) ** k / 2, 1 - (2 * (1 - p)) ** k / 2)
    return e_in, e_out, e_inout


def _sine():
    return (lambda p: 1 - np.cos(p * math.pi / 2),
            lambda p: np.sin(p * math.pi / 2),
            lambda p: -(np.cos(math.pi * p) - 1) / 2)


def _expo():
    def e_in(p):
        return np.where(p <= 0, 0.0, 2 ** (10 * (p - 1)))

    def e_out(p):
        return np.where(p >= 1, 1.0, 1 - 2 ** (-10 * p))

    def e_inout(p):
        return np.where(p < 0.5, 2 ** (20 * p - 10) / 2, (2 - 2 ** (-20 * p + 10)) / 2).clip(0, 1)
    return e_in, e_out, e_inout


def _circ():
    return (lambda p: 1 - np.sqrt(np.clip(1 - p * p, 0, 1)),
            lambda p: np.sqrt(np.clip(1 - (p - 1) ** 2, 0, 1)),
            lambda p: np.where(p < 0.5, (1 - np.sqrt(np.clip(1 - (2 * p) ** 2, 0, 1))) / 2,
                               (np.sqrt(np.clip(1 - (-2 * p + 2) ** 2, 0, 1)) + 1) / 2))


def back_out(s=1.70158):
    return lambda p: 1 + (s + 1) * (p - 1) ** 3 + s * (p - 1) ** 2


def back_in(s=1.70158):
    return lambda p: (s + 1) * p ** 3 - s * p ** 2


def elastic_out(amp=1.0, period=0.3):
    amp = max(amp, 1.0)
    s = period / (2 * math.pi) * math.asin(1 / amp)
    return lambda p: np.where(p >= 1, 1.0, amp * 2 ** (-10 * p) * np.sin((p - s) * (2 * math.pi) / period) + 1)


def bounce_out(p):
    n1, d1 = 7.5625, 2.75
    p = np.asarray(p, dtype=float)
    return np.where(p < 1 / d1, n1 * p * p,
           np.where(p < 2 / d1, n1 * (p - 1.5 / d1) ** 2 + 0.75,
           np.where(p < 2.5 / d1, n1 * (p - 2.25 / d1) ** 2 + 0.9375, n1 * (p - 2.625 / d1) ** 2 + 0.984375)))


EASES: dict[str, callable] = {"none": lambda p: p}
for _k, _name in ((2, "power1"), (3, "power2"), (4, "power3"), (5, "power4")):
    _i, _o, _io = _power(_k)
    EASES[f"{_name}.in"], EASES[f"{_name}.out"], EASES[f"{_name}.inOut"] = _i, _o, _io
for _name, _fam in (("sine", _sine()), ("expo", _expo()), ("circ", _circ())):
    EASES[f"{_name}.in"], EASES[f"{_name}.out"], EASES[f"{_name}.inOut"] = _fam
for _s in (0.8, 1.2, 1.7, 2.2, 2.8, 3.5):
    EASES[f"back.out({_s})"] = back_out(_s)
    EASES[f"back.in({_s})"] = back_in(_s)
for _a, _p in ((1, 0.3), (1, 0.45), (1.2, 0.4)):
    EASES[f"elastic.out({_a},{_p})"] = elastic_out(_a, _p)
EASES["bounce.out"] = bounce_out

# what a person would call each family
FEEL = {
    "none": "linear, mechanical",
    "power1": "gentle", "power2": "smooth", "power3": "snappy", "power4": "very snappy",
    "expo": "whip-fast, then a long soft landing", "circ": "fast with an abrupt settle", "sine": "soft, barely eased",
    "back": "overshoots and springs back", "elastic": "wobbles into place", "bounce": "bounces like a dropped object",
}


def feel(name: str) -> str:
    fam = name.split(".")[0]
    base = FEEL.get(fam, name)
    if name.endswith(".in") or ".in(" in name:
        return base + " (accelerates; used for exits)"
    if ".inOut" in name:
        return base + " (eases both ends)"
    return base


def evaluate(name: str, p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return np.asarray(EASES[name](p), dtype=float)


# ─── cubic-bezier (CSS / GSAP CustomEase) ────────────────────────────────────────────────────────

def bezier(x1: float, y1: float, x2: float, y2: float, x):
    """y at x for the CSS cubic-bezier(x1, y1, x2, y2), solved by bisection (x is monotonic for x1, x2 in [0, 1])."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    lo, hi = np.zeros_like(x), np.ones_like(x)
    for _ in range(30):
        t = (lo + hi) / 2
        bx = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3
        lo = np.where(bx < x, t, lo)
        hi = np.where(bx < x, hi, t)
    t = (lo + hi) / 2
    return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3


def css(b) -> str:
    return "cubic-bezier({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(*b)


def custom_ease(b) -> str:
    return 'CustomEase.create("custom", "M0,0 C{:.3f},{:.3f} {:.3f},{:.3f} 1,1")'.format(*b)


def _nelder_mead(f, x0, step=0.15, iters=160):
    n = len(x0)
    pts = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        x[i] += step
        pts.append(x)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        r = c + (c - pts[-1])
        fr = f(r)
        if fr < vals[0]:
            e = c + 2 * (c - pts[-1])
            fe = f(e)
            pts[-1], vals[-1] = (e, fe) if fe < fr else (r, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = r, fr
        else:
            k = c + 0.5 * (pts[-1] - c)
            fk = f(k)
            if fk < vals[-1]:
                pts[-1], vals[-1] = k, fk
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_bezier(x, y):
    """The cubic-bezier closest to progress y over normalised time x; y may overshoot (y1/y2 outside 0-1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def cost(b):
        x1, y1, x2, y2 = b
        pen = 0.0
        if not (0 <= x1 <= 1 and 0 <= x2 <= 1):
            pen += 10 * (max(0, -x1, x1 - 1) + max(0, -x2, x2 - 1))
            x1, x2 = min(max(x1, 0), 1), min(max(x2, 0), 1)
        return float(np.sqrt(np.mean((bezier(x1, y1, x2, y2, x) - y) ** 2))) + pen

    best = None
    for start in ((0.25, 0.1, 0.25, 1.0), (0.2, 0.8, 0.2, 1.0), (0.42, 0.0, 0.58, 1.0), (0.3, 1.4, 0.4, 1.0), (0.6, 0.0, 0.8, 0.4)):
        b, v = _nelder_mead(cost, start)
        if best is None or v < best[1]:
            best = (b, v)
    b = best[0]
    b = [min(max(b[0], 0), 1), b[1], min(max(b[2], 0), 1), b[3]]
    return [round(float(v), 3) for v in b], round(float(best[1]), 4)


# ─── fitting a measured movement ─────────────────────────────────────────────────────────────────

def progress(values, start_value=None, end_value=None):
    """Values -> progress, 0 at the start value, 1 at the settled value (medians of the first and last few)."""
    v = np.asarray(values, dtype=float)
    k = max(1, min(3, len(v) // 6))
    a = float(np.median(v[:k])) if start_value is None else start_value
    b = float(np.median(v[-k:])) if end_value is None else end_value
    if abs(b - a) < 1e-9:
        return np.zeros_like(v), a, b
    return (v - a) / (b - a), a, b


def _active_span(t, p, lo=0.03, hi=0.97):
    """First time progress leaves the start and the time it settles (stays within 3% of the end)."""
    moving = np.where(np.abs(p) > lo)[0]
    if not len(moving):
        return None
    i0 = max(0, moving[0] - 1)
    off = np.where(np.abs(p - 1) > 1 - hi)[0]
    i1 = min(len(p) - 1, (off[-1] + 1) if len(off) else len(p) - 1)
    return t[i0], t[i1]


def fit(times, values, *, start_value=None, end_value=None, names=None, frame=1 / 30) -> dict | None:
    """Fit a measured channel. Returns start, duration, overshoot, the best GSAP ease, the free bezier,
    and the runner-up eases, or None when the channel barely moves."""
    t = np.asarray(times, dtype=float)
    p, a, b = progress(values, start_value, end_value)
    if len(t) < 4 or not np.isfinite(p).all():
        return None
    span = _active_span(t, p)
    if span is None:
        return None
    s0, s1 = span
    overshoot = float(max(0.0, p.max() - 1.0))
    undershoot = float(max(0.0, -p.min()))
    names = names or list(EASES)
    results = []
    # search the start within +-1.5 frames and the duration around the measured span, in quarter frames
    starts = s0 + frame * np.arange(-1.5, 1.51, 0.25)
    d_meas = max(frame, s1 - s0)
    durs = d_meas * np.linspace(0.6, 2.6, 41)  # long-tail eases (expo, power4) settle visibly early
    for name in names:
        fn = EASES[name]
        best = None
        for st in starts:
            for d in durs:
                x = np.clip((t - st) / d, 0, 1)
                err = float(np.sqrt(np.mean((np.asarray(fn(x), float) - p) ** 2)))
                if best is None or err < best[0]:
                    best = (err, st, d)
        results.append({"ease": name, "rmse": round(best[0], 4), "start": round(float(best[1]), 3),
                        "duration": round(float(best[2]), 3)})
    results.sort(key=lambda r: r["rmse"])
    top = results[0]
    x = np.clip((t - top["start"]) / top["duration"], 0, 1)
    bez, bez_err = fit_bezier(x, p)
    return {
        "from": round(a, 4), "to": round(b, 4),
        "visible": round(float(s1 - s0), 3),  # from the first movement until it looks settled
        "start": top["start"], "duration": top["duration"], "end": round(top["start"] + top["duration"], 3),
        "ease": top["ease"], "rmse": top["rmse"], "feel": feel(top["ease"]),
        "overshoot": round(overshoot, 3), "undershoot": round(undershoot, 3),
        "bezier": bez, "bezier_rmse": bez_err, "css": css(bez),
        "alternatives": results[1:4],
        "samples": len(t),
    }


def curve_points(name_or_bezier, n=48):
    """(x, y) points of an ease for drawing it."""
    x = np.linspace(0, 1, n)
    if isinstance(name_or_bezier, str):
        y = evaluate(name_or_bezier, x)
    else:
        y = bezier(*name_or_bezier, x)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(x, y)]


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    fps = 30
    t = np.arange(0, 1.2, 1 / fps)
    for name, st, d in (("power3.out", 0.21, 0.4), ("back.out(1.7)", 0.1, 0.5), ("sine.inOut", 0.3, 0.6),
                        ("expo.out", 0.15, 0.7), ("none", 0.2, 0.3)):
        x = np.clip((t - st) / d, 0, 1)
        y = 100 + 300 * evaluate(name, x)  # a slide from x=100 to x=400 px
        r = fit(t, y)
        assert r is not None, name
        assert abs(r["start"] - st) <= 1 / fps, (name, r["start"])
        # a long-tail ease's last 3% is invisible, so its duration only has to reproduce the curve
        assert abs(r["duration"] - d) / d < 0.2 or r["rmse"] < 0.01, (name, r["duration"])
        fam_ok = r["ease"].split("(")[0] == name.split("(")[0] or r["rmse"] < 0.02
        assert fam_ok, (name, r["ease"], r["rmse"])
        assert r["bezier_rmse"] < 0.05, (name, r["bezier_rmse"])
        print(f"  {name:14s} -> {r['ease']:14s} start {r['start']:.3f} dur {r['duration']:.3f} "
              f"rmse {r['rmse']:.4f} overshoot {r['overshoot']:.2f} {r['css']}")
    assert fit(t, np.ones_like(t)) is None
    print("ease demo ok")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
