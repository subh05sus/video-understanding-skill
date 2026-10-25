"""style.json: the teardown's numbers in the shape an editor reads.

The keys follow the creator-teardown contract (motionmaxxing's edit skill reads them: pace, zoom, camera,
captions, graphics, sound, hook, design), measured from one video instead of a creator's set, so the editor can
cut a video "like this one". Everything else the lenses measured that the contract has no key for (palette,
grade, layout mix, delivery, beats, every element's GSAP) goes under "deep", for reelcut and for people.

    build(results, cross, meta) -> dict
    python3 style.py demo
"""
from __future__ import annotations

import sys
from fractions import Fraction

CAPTION_KEYS = ("present", "words_per_caption", "lines", "y_pct", "x_pct", "size_pct", "case", "font_match", "weight",
                "color", "highlight_color", "stroke", "stroke_color", "box", "box_color", "shadow", "animation",
                "entrance", "ease_s", "source")
CUT_KINDS = ("hard", "jump", "zoom-punch", "whip", "match", "mask", "dissolve")
SFX_KINDS = ("whoosh", "pop", "click", "riser", "impact", "ding", "other")


def aspect(w: int, h: int) -> str:
    known = {(9, 16), (16, 9), (1, 1), (4, 5), (4, 3), (3, 4)}
    f = Fraction(w, h).limit_denominator(20)
    for a, b in known:
        if abs(w / h - a / b) < 0.02:
            return f"{a}:{b}"
    return f"{f.numerator}:{f.denominator}"


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


def _cut_kinds(edit: dict) -> dict:
    """creator-teardown's cut kinds; this teardown's extra kinds (flash, layout, fade) fold into the nearest."""
    raw = edit.get("cut_kinds") or {}
    fold = {"fade": "dissolve", "flash": "hard", "wipe": "mask"}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k == "layout" or not _num(v):
            continue
        k2 = fold.get(k, k)
        if k2 in CUT_KINDS:
            out[k2] = round(out.get(k2, 0) + v, 1)
    total = sum(out.values())
    return {k: round(100 * v / total, 1) for k, v in out.items()} if total else {}


def _pattern(design: dict) -> tuple[str | None, int | None]:
    bg = design.get("background") or {}
    kind = str(bg.get("pattern") or bg.get("kind") or "").lower()
    size = _num(bg.get("pattern_size") or bg.get("spacing_px") or bg.get("grid_px"))
    for p in ("grid", "dots", "lines"):
        if p in kind:
            return p, int(size) if size else None
    return ("none", None) if bg else (None, None)


def _design_block(design: dict, motion: dict, typo: dict) -> dict:
    """creator-teardown's style.json "design": the graphics' ground, ink, accent, cards and type."""
    pal = design.get("palette") or {}
    canvas = [c for c in pal.get("canvas") or [] if isinstance(c, dict) and c.get("hex")]
    accents = [c for c in pal.get("accents") or [] if isinstance(c, dict) and c.get("hex")]
    ink = None
    titles = [t for t in typo.get("titles") or [] if isinstance(t, dict)]
    for t in titles:
        if isinstance(t.get("color"), str) and t["color"].startswith("#"):
            ink = t["color"]
            break
    cards = design.get("cards") or {}
    looks = [(e.get("look") or {}) for e in motion.get("elements") or [] if isinstance(e, dict)]
    radii = sorted(_num(l.get("radius_px")) for l in looks if _num(l.get("radius_px")) is not None)
    radius = _num(cards.get("radius_px")) or (radii[len(radii) // 2] if radii else None)
    shadows = [l.get("shadow") for l in looks if l.get("shadow") is not None]
    shadow = cards.get("shadow")
    if shadow is None and shadows:
        yes = sum(1 for s in shadows if s and s != "none")
        shadow = "soft" if yes > len(shadows) / 2 else "none"
    if isinstance(shadow, dict):
        shadow = "hard" if (shadow.get("blur_px") or 99) < 3 else "soft"
    if shadow is True:
        shadow = "soft"
    if shadow is False:
        shadow = "none"
    pattern, psize = _pattern(design)
    ent = (motion.get("entrances") or [None])[0] or {}
    font = (titles[0] if titles else {}) or {}
    return _clean({
        "ground": canvas[0]["hex"] if canvas else None,
        "ink": ink,
        "accent": accents[0]["hex"] if accents else None,
        "pattern": pattern, "pattern_size": psize,
        "radius": round(radius) if radius is not None else None,
        "shadow": shadow if shadow in ("none", "soft", "hard") else None,
        "font_family": font.get("font_match"),
        "font_weight": _num(font.get("weight")),
        "entrance": ent.get("kind"), "ease": ent.get("ease"), "dur": _num(ent.get("duration_s")),
        "source": "video-understanding teardown (measured)",
    })


def build(results: dict, cross: dict, meta: dict) -> dict:
    edit, sound, motion = results.get("edit") or {}, results.get("sound") or {}, results.get("motion") or {}
    typo, deliv, design = results.get("typography") or {}, results.get("delivery") or {}, results.get("design") or {}
    pace, epace = deliv.get("pace") or {}, edit.get("pace") or {}
    caps = typo.get("captions") or {}
    music = sound.get("music") or {}
    xs = cross.get("sfx") or {}
    hook = dict(cross.get("hook") or {})
    title_on = hook.pop("title_on_screen", None)
    name = meta.get("title") or meta.get("name") or "video"

    style = {
        "handle": name,
        "name": name,
        "videos": 1,
        "source": {"kind": "video-understanding teardown", "video": meta.get("path"), "url": meta.get("url"),
                   "duration_s": round(meta.get("duration") or 0, 2)},
        "aspect": aspect(meta.get("width") or 9, meta.get("height") or 16),
        "pace": _clean({
            "wpm": _num(pace.get("wpm")),
            "max_pause_s": _num((deliv.get("pauses") or {}).get("max_pause_s")),
            "cuts_per_10s": _num(epace.get("cuts_per_10s")),
            "median_shot_s": _num(epace.get("median_shot_s")),
            "cut_kinds": _cut_kinds(edit),
        }),
        "zoom": _clean(dict(edit.get("zoom") or {})),
        "camera": _clean(dict(edit.get("camera") or {})),
        "captions": _clean({k: caps.get(k) for k in CAPTION_KEYS}),
        "graphics": _clean({
            "per_min": _num(motion.get("per_min")),
            "kinds": motion.get("kinds"),
            "kinds_source": "code",
            "layout": motion.get("layout"),
            "entrances": motion.get("entrances"),
            "exits": motion.get("exits"),
            "secondary_motion": motion.get("secondary_motion"),
            "crop_palette": [c.get("hex") for c in (design.get("palette") or {}).get("accents") or [] if isinstance(c, dict)][:6] or None,
        }),
        "sound": _clean({
            "sfx_per_min": _num(sound.get("sfx_per_min")),
            "kinds": {k: v for k, v in (sound.get("kinds") or {}).items() if k in SFX_KINDS} or None,
            "on_cut_pct": xs.get("on_cut_pct"),
            "on_graphic_pct": xs.get("on_graphic_pct"),
            "in_voice_pct": xs.get("in_voice_pct"),
            "music": _clean({"present_pct": _num(music.get("present_pct")), "level_db": _num(music.get("level_db"))}),
        }),
        # one video is its own reference: it fills creator-teardown's "winner" medians
        "hook": _clean({"winner": _clean(hook), "control": None, "title_on_screen_pct": (100 if title_on else 0) if title_on is not None else None}),
        "design": _design_block(design, motion, typo),
        "deep": _clean({
            "palette": design.get("palette"),
            "grade": design.get("grade"),
            "layout": design.get("layout"),
            "background": design.get("background"),
            "composition": design.get("composition"),
            "safe_zone": design.get("safe_zone"),
            "titles": typo.get("titles"),
            "hierarchy": typo.get("hierarchy"),
            "caption_sync_ms": caps.get("sync_ms"),
            "font_candidates": caps.get("font_candidates"),
            "delivery": _clean({k: deliv.get(k) for k in ("language", "pace", "pauses", "fillers", "prosody", "hook", "body")}),
            "music": _clean({k: music.get(k) for k in ("bpm", "bpm_conf", "feel", "changes")}),
            "loudness": sound.get("loudness"),
            "rhythm": edit.get("rhythm"),
            "motion_language": motion.get("language"),
            "stagger": motion.get("stagger"),
            "animations": [_clean({"id": e.get("id"), "t_in": e.get("t_in"), "t_out": e.get("t_out"), "kind": e.get("kind_guess"),
                                   "box": e.get("box"), "entrance": e.get("entrance"), "exit": e.get("exit"), "gsap": e.get("gsap")})
                           for e in motion.get("elements") or [] if isinstance(e, dict)],
            "beat": cross.get("beat"),
            "emphasis": cross.get("emphasis"),
            "show_what_you_say": {k: v for k, v in (cross.get("show_what_you_say") or {}).items() if k != "items"} or None,
        }),
    }
    if music.get("present_pct") is not None and _num(music.get("present_pct")) is not None:
        style["music"] = music["present_pct"] >= 20
    style["warnings"] = [f"{k} lens failed: {(results.get(k) or {}).get('error')}" for k in results
                         if isinstance(results.get(k), dict) and results[k].get("error")]
    return _clean(style) | {"warnings": style["warnings"]}


def demo():
    res = {
        "edit": {"pace": {"cuts_per_10s": 3.2, "median_shot_s": 2.1}, "cut_kinds": {"hard": 50, "jump": 30, "layout": 10, "fade": 10},
                 "zoom": {"per_min": 6, "kind": "punch", "scale": 1.15}},
        "sound": {"sfx_per_min": 9, "kinds": {"whoosh": 60, "pop": 40}, "music": {"present_pct": 95, "level_db": -18, "bpm": 120}},
        "typography": {"captions": {"present": True, "font_match": "Montserrat", "weight": 800, "y_pct": 72, "extra": 1},
                       "titles": [{"t": 0.2, "font_match": "Anton", "weight": 400, "color": "#FFFFFF"}]},
        "delivery": {"pace": {"wpm": 190}, "pauses": {"max_pause_s": 0.3}},
        "design": {"palette": {"canvas": [{"hex": "#F6F3EA", "pct": 60}], "accents": [{"hex": "#FFD400", "pct": 3}]},
                   "background": {"kind": "grid paper", "pattern_size": 32}},
        "motion": {"per_min": 12, "entrances": [{"kind": "slide", "ease": "power3.out", "duration_s": 0.4}],
                   "elements": [{"id": 1, "look": {"radius_px": 24, "shadow": True}}]},
    }
    s = build(res, {"sfx": {"on_cut_pct": 70}, "hook": {"first_cut_s": 1.2, "title_on_screen": True}},
              {"name": "demo.mp4", "width": 1080, "height": 1920, "duration": 30})
    assert s["aspect"] == "9:16" and s["videos"] == 1
    assert s["pace"] == {"wpm": 190, "max_pause_s": 0.3, "cuts_per_10s": 3.2, "median_shot_s": 2.1,
                         "cut_kinds": {"hard": 55.6, "jump": 33.3, "dissolve": 11.1}}, s["pace"]
    assert "extra" not in s["captions"] and s["captions"]["font_match"] == "Montserrat"
    assert s["design"]["ground"] == "#F6F3EA" and s["design"]["pattern"] == "grid" and s["design"]["radius"] == 24
    assert s["design"]["shadow"] == "soft" and s["design"]["font_family"] == "Anton"
    assert s["hook"]["winner"] == {"first_cut_s": 1.2} and s["hook"]["title_on_screen_pct"] == 100
    assert s["sound"]["on_cut_pct"] == 70 and s["music"] is True and s["warnings"] == []
    print("style demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
