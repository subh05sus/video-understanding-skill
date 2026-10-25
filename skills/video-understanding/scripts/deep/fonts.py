"""fonts: which typeface a piece of on-screen text is set in, by rendering it.

The real text is cut out of the frame and turned into a glyph mask. The same words are rendered in each
candidate font and weight, scaled to the same height, and compared: how well the shapes overlap (IoU, with a
pixel of slack), how close the widths are (wide, regular, condensed), and how close the stroke thickness is
(light to black). Scores average over several samples of the same style.

Candidates: popular free Google Fonts used in short-form video (fetched once, without a key, from Google Fonts'
public CSS endpoint into ~/.cache/video-understanding/fonts) and the fonts installed on this machine (SF, Helvetica,
Arial, Avenir, Futura ...). Offline, only installed fonts are tried, and the result says so. A Google Font is
named first when one is close, because an editor can ship it.

    python3 fonts.py demo
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import eprint  # noqa: E402
import vu  # noqa: E402

FONT_DIR = vu.CACHE / "fonts"
H = 64                     # glyph masks are compared at this height
WEIGHTS = (400, 500, 600, 700, 800, 900)
GOOGLE = ["Montserrat", "Poppins", "Inter", "Roboto", "Open Sans", "Lato", "Raleway", "Nunito", "DM Sans",
          "Plus Jakarta Sans", "Manrope", "Outfit", "Urbanist", "Sora", "Figtree", "Work Sans", "Rubik", "Barlow",
          "League Spartan", "Kanit", "Archivo", "Archivo Black", "Anton", "Bebas Neue", "Oswald", "Bangers",
          "Permanent Marker", "Caveat", "Press Start 2P", "VT323", "Silkscreen", "Pixelify Sans", "Space Grotesk",
          "Space Mono", "Playfair Display", "Lora", "Instrument Serif", "Bricolage Grotesque", "Lexend", "Fredoka",
          "Luckiest Guy", "Titan One", "TikTok Sans", "Noto Sans", "Source Sans 3", "IBM Plex Sans", "JetBrains Mono",
          "Tiny5", "Micro 5", "Jersey 10", "Handjet", "DotGothic16", "Bytesized", "Workbench", "Sixtyfour", "Doto"]
SYSTEM = {  # name: (path, face index or None)
    "SF Pro": ("/System/Library/Fonts/SFNS.ttf", None),
    "Helvetica Neue": ("/System/Library/Fonts/HelveticaNeue.ttc", None),
    "Helvetica": ("/System/Library/Fonts/Helvetica.ttc", None),
    "Arial": ("/System/Library/Fonts/Supplemental/Arial.ttf", None),
    "Arial Black": ("/System/Library/Fonts/Supplemental/Arial Black.ttf", None),
    "Avenir Next": ("/System/Library/Fonts/Avenir Next.ttc", None),
    "Futura": ("/System/Library/Fonts/Supplemental/Futura.ttc", None),
    "Impact": ("/System/Library/Fonts/Supplemental/Impact.ttf", None),
    "Georgia": ("/System/Library/Fonts/Supplemental/Georgia.ttf", None),
}


# ─── getting fonts ───────────────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "video-understanding"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def google_font(family: str) -> dict[int, Path]:
    """{weight: ttf path} for a Google Font, downloaded once. Empty when offline or unknown."""
    d = FONT_DIR / re.sub(r"[^a-z0-9]+", "-", family.lower())
    idx = d / "index.json"
    if idx.exists():
        return {int(k): d / v for k, v in json.loads(idx.read_text()).items() if (d / v).exists()}
    q = family.replace(" ", "+")
    css = ""
    # a weight range for variable families; explicit lists for families with fewer weights or more axes
    tries = [f"https://fonts.googleapis.com/css2?family={q}:wght@100..900"]
    tries += [f"https://fonts.googleapis.com/css2?family={q}:wght@{';'.join(map(str, ws))}"
              for ws in ((400, 500, 600, 700, 800, 900), (400, 500, 600, 700, 800), (400, 500, 600, 700), (400, 700))]
    tries.append(f"https://fonts.googleapis.com/css2?family={q}")
    for url in tries:
        try:
            css = _get(url).decode("utf-8", "replace")
        except Exception:
            continue
        if "@font-face" in css:
            break
    blocks = re.findall(r"@font-face\s*{([^}]*)}", css)
    got = {}
    d.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        if "font-style: italic" in b:
            continue
        w = re.search(r"font-weight:\s*(\d+)", b)
        u = re.search(r"url\((https://[^)]+\.ttf)\)", b)
        if not (w and u) or int(w.group(1)) not in WEIGHTS:
            continue
        fn = f"{int(w.group(1))}.ttf"
        try:
            (d / fn).write_bytes(_get(u.group(1)))
            got[int(w.group(1))] = fn
        except Exception:
            continue
    if got:
        idx.write_text(json.dumps(got))
    return {k: d / v for k, v in got.items()}


def candidates(online: bool = True) -> list[dict]:
    """[{"font", "weight", "path", "index", "google"}] for every font and weight we can render."""
    out = []
    if online:
        for fam in GOOGLE:
            for w, p in sorted(google_font(fam).items()):
                out.append({"font": fam, "weight": w, "path": str(p), "index": 0, "google": True})
    for name, (path, _) in SYSTEM.items():
        if not Path(path).exists():
            continue
        if path.endswith(".ttc"):
            # each face of a collection: read its style name to get the weight
            for i in range(16):
                try:
                    f = ImageFont.truetype(path, 40, index=i)
                except Exception:
                    break
                style = (f.getname()[1] or "").lower()
                if "italic" in style or "oblique" in style or "condensed" in style:
                    continue
                w = 900 if "black" in style or "heavy" in style else 800 if "extra" in style and "bold" in style else \
                    700 if "bold" in style else 600 if "semi" in style or "demi" in style else 500 if "medium" in style else \
                    300 if "light" in style or "thin" in style else 400
                if w in WEIGHTS:
                    out.append({"font": name, "weight": w, "path": path, "index": i, "google": False})
        elif path.endswith("SFNS.ttf"):
            for w in WEIGHTS:
                out.append({"font": name, "weight": w, "path": path, "index": 0, "google": False, "variable": True})
        else:
            out.append({"font": name, "weight": 900 if "Black" in path or "Impact" in path else 400, "path": path,
                        "index": 0, "google": False})
    # one entry per font and weight
    seen, uniq = set(), []
    for c in out:
        k = (c["font"], c["weight"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


@lru_cache(maxsize=4096)
def _font(path: str, index: int, size: int, weight: int, variable: bool) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size, index=index)
    if variable:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


# ─── glyph masks ─────────────────────────────────────────────────────────────────────────────────

def tight(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def norm_height(mask: np.ndarray, h: int = H) -> np.ndarray:
    r = h / mask.shape[0]
    return cv2.resize(mask.astype(np.uint8) * 255, (max(4, int(round(mask.shape[1] * r))), h), interpolation=cv2.INTER_AREA) > 127


@lru_cache(maxsize=8192)
def _render(text: str, path: str, index: int, weight: int, variable: bool) -> np.ndarray | None:
    f = _font(path, index, 100, weight, variable)
    try:
        box = f.getbbox(text)
    except Exception:
        return None
    w, h = box[2] - box[0] + 20, box[3] - box[1] + 20
    if w <= 0 or h <= 0 or w > 6000:
        return None
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((10 - box[0], 10 - box[1]), text, font=f, fill=255)
    m = tight(np.asarray(img) > 127)
    return None if m is None else norm_height(m)


def render_mask(text: str, cand: dict) -> np.ndarray | None:
    return _render(text, cand["path"], cand.get("index", 0), cand["weight"], bool(cand.get("variable")))


def stroke(mask: np.ndarray) -> float:
    """Stroke thickness as a share of the height: twice the median of the distance-transform ridge."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)]
    return float(2 * np.median(ridge) / mask.shape[0]) if len(ridge) else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def edge_angles(mask: np.ndarray) -> np.ndarray:
    """Histogram (8 bins over 0-180°) of the glyphs' edge directions: pixel fonts are almost all 0° and 90°,
    geometric fonts have round sweeps, condensed fonts long verticals."""
    g = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    sel = mag > 0.2 * mag.max() if mag.max() > 0 else mag > 0
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 180) % 180
    h, _ = np.histogram(ang, bins=8, range=(0, 180), weights=mag[sel])
    return h / max(1e-9, h.sum())


def pixel_share(mask: np.ndarray) -> float:
    """How much of the edge runs straight across or straight down (1.0 for a pixel font)."""
    h = edge_angles(mask)
    return float(h[0] + h[4])      # the bins around 0°/180° and 90°


def compare(real: np.ndarray, cand: np.ndarray) -> float:
    """0..1 from five agreements at the same size: shape overlap with a pixel of slack, strict overlap, width
    (wide or condensed), stroke thickness (light or black) and the row profile (x-height, ascenders, where the
    ink sits) and column profile (letter widths and spacing)."""
    wr = min(real.shape[1], cand.shape[1]) / max(real.shape[1], cand.shape[1])
    c = cv2.resize(cand.astype(np.uint8), (real.shape[1], real.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    k = np.ones((3, 3), np.uint8)
    rd, cd = cv2.dilate(real.astype(np.uint8), k) > 0, cv2.dilate(c.astype(np.uint8), k) > 0
    slack = ((real & cd).sum() + (c & rd).sum()) / max(1, real.sum() + c.sum())
    strict = (real & c).sum() / max(1, (real | c).sum())
    sr, sc = stroke(real), stroke(c)
    st = np.exp(-abs(sr - sc) / max(1e-3, 0.25 * max(sr, sc)))
    rows = max(0.0, _corr(real.sum(axis=1).astype(float), c.sum(axis=1).astype(float)))
    cols = max(0.0, _corr(real.sum(axis=0).astype(float), c.sum(axis=0).astype(float)))
    edges = 1 - 0.5 * float(np.abs(edge_angles(real) - edge_angles(c)).sum())   # same kinds of curves and corners
    return float(slack * np.sqrt(strict) * wr ** 1.5 * st * (0.5 + 0.5 * rows) * (0.5 + 0.5 * cols) * (0.3 + 0.7 * edges))


def glyph_mask(crop: np.ndarray, fill_rgb: tuple[int, int, int] | None = None) -> np.ndarray | None:
    """The text's fill as a mask: the pixels of the fill colour (given, or the colour farthest from the border)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    if fill_rgb is None:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.median(border, axis=0)
        data = lab.reshape(-1, 3)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 3 if len(data) > 30 else 2
        _, labels, centres = cv2.kmeans(data, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(labels.ravel(), minlength=k) / len(labels)
        score = [np.linalg.norm(c - bg) * (s > 0.05) for c, s in zip(centres, share)]
        fill = centres[int(np.argmax(score))]
    else:
        fill = cv2.cvtColor(np.uint8([[fill_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    m = np.linalg.norm(lab - fill, axis=2) < 28
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return tight(m)


# ─── matching ────────────────────────────────────────────────────────────────────────────────────

def match(samples: list[tuple[np.ndarray, str]], online: bool = True, top: int = 5) -> dict:
    """samples: [(glyph mask, its text)]. Returns {"best", "best_google", "candidates": [...], "offline"}."""
    cands = candidates(online)
    offline = not any(c["google"] for c in cands)
    # pixel text is only compared with pixel fonts (video blur lowers its share a little, hence 0.7)
    px_real = [pixel_share(norm_height(m)) for m, _ in samples if m is not None]
    if px_real and float(np.median(px_real)) >= 0.7:
        probe = next((t for m, t in samples if m is not None and t.strip()), "Ag")
        pix = [c for c in cands if (r := render_mask(probe.strip(), c)) is not None and pixel_share(r) >= 0.7]
        if pix:
            cands = pix
    if offline and online:
        eprint("fonts: Google Fonts unreachable; matching against installed fonts only")
    scores: dict[tuple, list[float]] = {}
    for mask, text in samples:
        text = text.strip()
        if not text or mask is None:
            continue
        real = norm_height(mask)
        for c in cands:
            r = render_mask(text, c)
            if r is None:
                continue
            scores.setdefault((c["font"], c["weight"], c["google"]), []).append(compare(real, r))
    ranked = sorted(((np.mean(v), k) for k, v in scores.items() if len(v) == max(len(x) for x in scores.values())),
                    key=lambda x: -x[0])
    out = [{"font": k[0], "weight": k[1], "google": k[2], "score": round(float(s), 3)} for s, k in ranked[:top]]
    best_g = next(({"font": k[0], "weight": k[1], "score": round(float(s), 3)} for s, k in ranked if k[2]), None)
    px = [pixel_share(norm_height(m)) for m, _ in samples if m is not None]
    strokes = [stroke(norm_height(m)) for m, _ in samples if m is not None]
    style = None
    if px:
        style = "pixel" if np.median(px) >= 0.7 else "heavy display" if np.median(strokes) > 0.2 else \
            "bold sans" if np.median(strokes) > 0.13 else "regular"
    top_s = out[0]["score"] if out else 0.0
    quality = "good" if top_s >= 0.5 else "fair" if top_s >= 0.3 else "weak (closest free font, not the font itself)"
    return {"best": out[0] if out else None, "best_google": best_g, "candidates": out, "offline": offline,
            "samples": len(samples), "style_class": style, "quality": quality}


def comparison_image(mask: np.ndarray, text: str, cands: list[dict], all_c: list[dict]) -> np.ndarray:
    """The real glyphs over the top candidates' renders, for a person to judge."""
    rows = [norm_height(mask, 48)]
    labels = ["on screen"]
    for c in cands[:3]:
        src = next((x for x in all_c if x["font"] == c["font"] and x["weight"] == c["weight"]), None)
        r = render_mask(text, src) if src else None
        if r is not None:
            rows.append(norm_height(r, 48))
            labels.append(f"{c['font']} {c['weight']} ({c['score']:.2f})")
    W = max(r.shape[1] for r in rows) + 260
    img = Image.new("RGB", (W, 60 * len(rows)), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for i, (r, lab) in enumerate(zip(rows, labels)):
        img.paste(Image.fromarray(np.where(r, 20, 250).astype(np.uint8)).convert("RGB"), (250, i * 60 + 6))
        d.text((8, i * 60 + 22), lab, fill=(40, 40, 40))
    return np.asarray(img)


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    # render a word in an installed font, add blur and noise like a video frame, and find the font again
    cands = candidates(online=False)
    assert cands, "no installed fonts found"
    target = next((c for c in cands if c["font"] == "Futura" and c["weight"] == 700), None) or \
        next(c for c in cands if c["font"] == "Arial Black")
    rng = np.random.default_rng(0)
    samples = []
    for word in ("Hello", "Moment", "design"):
        f = _font(target["path"], target.get("index", 0), 70, target["weight"], bool(target.get("variable")))
        box = f.getbbox(word)
        img = Image.new("RGB", (box[2] + 40, box[3] + 40), (60, 70, 80))
        ImageDraw.Draw(img).text((20 - box[0] + box[0], 20), word, font=f, fill=(255, 255, 255))
        arr = cv2.GaussianBlur(np.asarray(img), (3, 3), 0).astype(np.int16) + rng.integers(-8, 8, (img.size[1], img.size[0], 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        samples.append((glyph_mask(arr), word))
    res = match(samples, online=False)
    names = [(c["font"], c["weight"]) for c in res["candidates"][:2]]
    assert (target["font"], target["weight"]) in names, (target["font"], target["weight"], res["candidates"])
    print(f"  {target['font']} {target['weight']} found: {res['candidates'][:3]}")
    print("fonts demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    elif sys.argv[1:] == ["fetch"]:
        for fam in GOOGLE:
            print(fam, sorted(google_font(fam)))
    else:
        print(__doc__)
