"""Findings that only show when the lenses are laid over each other: a sound effect on every cut, cuts on the
beat, a zoom on the stressed word, a graphic appearing the moment its word is said, how fast the hook gets to
its first cut, graphic, text and word. Every rate comes with what chance would give, so "62% of cuts on the
beat" can be read against "38% by luck at this tempo".

    run(results, duration) -> dict      results: lens name -> lens dict (any may be missing)
    python3 cross.py demo
"""
from __future__ import annotations

import re
import sys
import unicodedata

FRAME = 1 / 30
ON_CUT_S = 0.1        # a sound effect "on" a cut: within 3 frames
ON_BEAT_S = 0.07      # a cut "on" the beat: within 2 frames
ON_WORD_S = 0.3       # a visual event "on" a stressed word
SAID_SHOWN_S = 2.0    # a graphic "shows what is said" when its text is spoken within 2 s


def _times(items, key="t"):
    out = []
    for x in items or []:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, dict) and isinstance(x.get(key), (int, float)):
            out.append(float(x[key]))
    return sorted(out)


def _hit_rate(events: list[float], targets: list[float], tol: float) -> float | None:
    if not events or not targets:
        return None
    hits = sum(1 for e in events if any(abs(e - t) <= tol for t in targets))
    return round(100.0 * hits / len(events), 1)


def _chance(targets: list[float], tol: float, duration: float) -> float | None:
    """The share of random moments that would land within tol of a target."""
    if not targets or duration <= 0:
        return None
    covered, last = 0.0, -1e9
    for t in sorted(targets):
        a, b = max(t - tol, last), t + tol
        if b > a:
            covered += b - a
        last = max(last, b)
    return round(100.0 * min(1.0, covered / duration), 1)


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s or "").lower()
    word, out = [], []
    for ch in s:
        if unicodedata.category(ch)[0] in "LMN":
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return [w for w in out if len(w) >= 3]


def run(results: dict, duration: float) -> dict:
    edit, sound, motion = results.get("edit") or {}, results.get("sound") or {}, results.get("motion") or {}
    typo, deliv = results.get("typography") or {}, results.get("delivery") or {}
    cuts = _times(edit.get("cuts"))
    hard_cuts = _times([c for c in edit.get("cuts") or [] if isinstance(c, dict) and c.get("kind") != "layout"])
    beats = _times((sound.get("music") or {}).get("beats"))
    sfx = _times(sound.get("sfx"))
    elements = [e for e in motion.get("elements") or [] if isinstance(e, dict)]
    g_in = _times(elements, "t_in")
    g_out = _times(elements, "t_out")
    zooms = _times([e for e in (edit.get("rhythm") or {}).get("events", []) if isinstance(e, dict) and "zoom" in str(e.get("kind"))])
    emph = _times(deliv.get("emphasis"))
    words = results.get("_words") or []
    out: dict = {"summary": []}

    # sound effects against picture
    if sfx:
        on_cut = _hit_rate(sfx, cuts, ON_CUT_S)
        on_gfx = _hit_rate(sfx, g_in + g_out, ON_CUT_S)
        in_voice = None
        spans = [(w["a"], w["b"]) for w in words if isinstance(w, dict) and "a" in w]
        if spans:
            in_voice = round(100.0 * sum(1 for t in sfx if any(a <= t <= b for a, b in spans)) / len(sfx), 1)
        out["sfx"] = {"on_cut_pct": on_cut, "on_cut_chance_pct": _chance(cuts, ON_CUT_S, duration),
                      "on_graphic_pct": on_gfx, "on_graphic_chance_pct": _chance(g_in + g_out, ON_CUT_S, duration),
                      "in_voice_pct": in_voice,
                      "cuts_with_sfx_pct": _hit_rate(cuts, sfx, ON_CUT_S),
                      "graphics_with_sfx_pct": _hit_rate(g_in, sfx, ON_CUT_S)}
        s = out["sfx"]
        if s["on_cut_pct"] is not None or s["on_graphic_pct"] is not None:
            out["summary"].append(
                f"Sound effects: {s['on_cut_pct'] or 0:.0f}% land on a cut (chance {s['on_cut_chance_pct'] or 0:.0f}%), "
                f"{s['on_graphic_pct'] or 0:.0f}% on a graphic coming or going (chance {s['on_graphic_chance_pct'] or 0:.0f}%).")

    # the edit against the music
    if beats and hard_cuts:
        on = _hit_rate(hard_cuts, beats, ON_BEAT_S)
        ch = _chance(beats, ON_BEAT_S, duration)
        out["beat"] = {"cuts_on_beat_pct": on, "chance_pct": ch,
                       "graphics_on_beat_pct": _hit_rate(g_in, beats, ON_BEAT_S),
                       "cut_to_beat": bool(on is not None and ch is not None and on >= ch + 20)}
        verdict = "cut to the beat" if out["beat"]["cut_to_beat"] else "not cut to the beat"
        out["summary"].append(f"Edit vs music: {on:.0f}% of cuts on a beat against {ch:.0f}% by chance: {verdict}.")

    # what happens on the stressed words
    if emph:
        visual = sorted(set(cuts + g_in + zooms))
        out["emphasis"] = {"with_visual_pct": _hit_rate(emph, visual, ON_WORD_S),
                           "with_zoom_pct": _hit_rate(emph, zooms, ON_WORD_S),
                           "with_graphic_pct": _hit_rate(emph, g_in, ON_WORD_S),
                           "with_sfx_pct": _hit_rate(emph, sfx, ON_WORD_S),
                           "chance_pct": _chance(visual, ON_WORD_S, duration)}
        e = out["emphasis"]
        out["summary"].append(f"Stressed words: {e['with_visual_pct'] or 0:.0f}% get a cut, zoom or graphic within "
                              f"{ON_WORD_S:.1f} s (chance {e['chance_pct'] or 0:.0f}%).")

    # show what you say: graphic text spoken around its entrance
    said = []
    for el in elements:
        toks = set(_tokens(el.get("text") or ""))
        if not toks or el.get("t_in") is None:
            continue
        near = [w for w in words if abs(w.get("a", -99) - el["t_in"]) <= SAID_SHOWN_S]
        spoken = {t for w in near for t in _tokens(w.get("w", ""))}
        hit = toks & spoken
        said.append({"t": el["t_in"], "id": el.get("id"), "match": sorted(hit)[:5], "shown_with_word": bool(hit)})
    if said:
        n = sum(1 for s in said if s["shown_with_word"])
        out["show_what_you_say"] = {"pct": round(100.0 * n / len(said), 1), "n": len(said), "items": said}
        out["summary"].append(f"Show what you say: {n} of {len(said)} graphics with text appear within "
                              f"{SAID_SHOWN_S:.0f} s of a matching spoken word.")

    # the hook: how soon each thing arrives (creator-teardown's hook fields)
    def first(xs):
        return round(min(xs), 2) if xs else None
    caps = typo.get("captions") or {}
    titles = [t for t in typo.get("titles") or [] if isinstance(t, dict)]
    text_items = [t for t in typo.get("text_items") or [] if isinstance(t, dict)]
    faces = results.get("_faces") or []
    early = [f for f in faces if f.get("t", 99) <= 3.0]
    hook = {
        "first_cut_s": first([c for c in cuts if c > 0.05]),
        "first_graphic_s": first(g_in),
        "first_text_s": first(_times(text_items, "a") + _times(titles, "t") + _times(text_items, "t")),
        "first_caption_s": caps.get("first_s") if isinstance(caps.get("first_s"), (int, float)) else None,
        "first_word_s": first([w["a"] for w in words if isinstance(w, dict) and "a" in w]),
        "face_pct": round(100.0 * sum(1 for f in early if f.get("faces")) / len(early)) if early else None,
        "title_on_screen": bool([t for t in titles if (t.get("t") or 99) <= 3.0]),
    }
    out["hook"] = hook
    bits = [f"{k.replace('first_', '').replace('_s', '')} {v:.1f} s" for k, v in hook.items()
            if k.startswith("first_") and isinstance(v, (int, float))]
    if bits:
        out["summary"].append("Hook, first arrivals: " + ", ".join(bits) + ".")

    # one merged timeline for the report
    tl = []
    for c in edit.get("cuts") or []:
        if isinstance(c, dict):
            tl.append({"t": c.get("t"), "track": "cut", "label": c.get("kind", "cut")})
    for e in elements:
        if e.get("t_in") is not None:
            tl.append({"t": e["t_in"], "end": e.get("t_out"), "track": "graphic",
                       "label": ((e.get("entrance") or {}).get("kind") or "in") + (f" · {e.get('kind_guess')}" if e.get("kind_guess") else "")})
    for s in sound.get("sfx") or []:
        if isinstance(s, dict):
            tl.append({"t": s.get("t"), "track": "sfx", "label": s.get("kind", "sfx")})
    for e in deliv.get("emphasis") or []:
        if isinstance(e, dict):
            tl.append({"t": e.get("t"), "track": "stress", "label": e.get("w", "")})
    out["timeline"] = sorted([x for x in tl if isinstance(x.get("t"), (int, float))], key=lambda x: x["t"])
    return out


def demo():
    res = {
        "edit": {"cuts": [{"t": 1.0, "kind": "hard"}, {"t": 2.0, "kind": "jump"}, {"t": 3.0, "kind": "hard"}],
                 "rhythm": {"events": [{"t": 4.0, "kind": "zoom"}]}},
        "sound": {"music": {"beats": [0.5 * i for i in range(20)]}, "sfx": [{"t": 1.02, "kind": "whoosh"}, {"t": 5.5, "kind": "pop"}]},
        "motion": {"elements": [{"id": 1, "t_in": 5.5, "t_out": 7.0, "text": "Pricing table"}]},
        "delivery": {"emphasis": [{"t": 4.1, "w": "never"}]},
        "typography": {"captions": {"first_s": 0.4}, "titles": [{"t": 0.2, "text": "X"}]},
        "_words": [{"w": "the", "a": 0.4, "b": 0.5}, {"w": "pricing", "a": 5.0, "b": 5.4}],
        "_faces": [{"t": 0.5, "faces": [{"box": [0, 0, 1, 1]}]}, {"t": 2.5, "faces": []}],
    }
    out = run(res, 10.0)
    assert out["sfx"]["on_cut_pct"] == 50.0 and out["sfx"]["on_graphic_pct"] == 50.0, out["sfx"]
    assert out["beat"]["cuts_on_beat_pct"] == 100.0 and out["beat"]["cut_to_beat"], out["beat"]
    assert out["emphasis"]["with_zoom_pct"] == 100.0
    assert out["show_what_you_say"]["pct"] == 100.0
    assert out["hook"]["first_cut_s"] == 1.0 and out["hook"]["face_pct"] == 50 and out["hook"]["title_on_screen"]
    assert _tokens("Open-Source-Tools für दुकान") == ["open", "source", "tools", "für", "दुकान"]
    assert out["timeline"][0]["track"] == "cut"
    print("cross demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
