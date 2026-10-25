"""report: the teardown as a page people read (teardown.html) and as a digest Claude reads (teardown.md).

    html(ctx, cross, style, analysis_md) -> Path      the page: video, at a glance, timeline, every lens,
                                                    each animation with its ease curve and GSAP, evidence
    digest(ctx, cross, style) -> Path                 the numbers and findings in a few hundred lines, plus which
                                                    evidence images to look at for the qualitative pass

Images are embedded (base64) up to a size budget so the page is one file; past it they are linked.
    python3 report.py demo
"""
from __future__ import annotations

import base64
import html as H
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ease  # noqa: E402
from common import Ctx  # noqa: E402
import vu  # noqa: E402

BUDGET_MB = 14.0
TRACK_COL = {"cut": "var(--ink)", "graphic": "var(--acc)", "sfx": "var(--red)", "stress": "var(--blue)", "caption": "var(--muted)"}


def esc(x) -> str:
    return H.escape(str(x)) if x is not None else "—"


def ts_link(t, label=None) -> str:
    if t is None:
        return "—"
    return f'<a class="ts" href="#" data-t="{float(t):.3f}">{esc(label or vu.fmt(float(t), prec=2))}</a>'


def link_times(text: str, duration: float) -> str:
    """Escape text and turn moments ("at 12.3 s", "from 7.4 s", "around 3 s") into links that seek the video.
    Durations ("per 10 s", "within 2 s", "0.4 s long") stay plain text."""
    e = esc(text)
    e = re.sub(r"\b(at|from|around|near|until|to) (\d{1,4}(?:\.\d{1,3})?) s\b",
               lambda m: f"{m.group(1)} " + (ts_link(float(m.group(2)), m.group(2) + " s") if float(m.group(2)) <= duration + 0.5 else m.group(2) + " s"), e)
    e = re.sub(r"^(\d{1,4}(?:\.\d{1,3})?) s\b",
               lambda m: ts_link(float(m.group(1)), m.group(0)) if float(m.group(1)) <= duration + 0.5 else m.group(0), e)
    return e


class Images:
    def __init__(self, root: Path):
        self.root, self.used = root, 0.0

    def tag(self, rel: str | None, alt: str = "", cls: str = "") -> str:
        if not rel:
            return ""
        p = self.root / rel
        if not p.exists():
            return ""
        mb = p.stat().st_size / 1e6
        mime = "image/png" if p.suffix == ".png" else "image/jpeg"
        if self.used + mb > BUDGET_MB:
            return f'<a href="{esc(p.as_uri())}">{esc(alt or p.name)}</a>'
        self.used += mb
        return f'<img class="{cls}" loading="lazy" alt="{esc(alt)}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"/>'


def curve_svg(name: str | None, bez=None, w: int = 120, h: int = 80) -> str:
    """The ease as a small curve (time across, progress up), overshoot shown above the top line."""
    if not name and not bez:
        return ""
    try:
        pts = ease.curve_points(name if name and name in ease.EASES else bez, 40)
    except Exception:
        return ""
    lo, hi = min(-0.05, min(p[1] for p in pts)), max(1.05, max(p[1] for p in pts))
    def X(x):
        return 6 + x * (w - 12)
    def Y(y):
        return h - 6 - (y - lo) / (hi - lo) * (h - 12)
    path = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    return (f'<svg class="curve" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(0)}" class="axis"/>'
            f'<line x1="{X(0)}" y1="{Y(1)}" x2="{X(1)}" y2="{Y(1)}" class="axis dash"/><path d="{path}" class="ease"/></svg>')


def kv_table(rows: list[tuple[str, object]]) -> str:
    body = "".join(f"<tr><th>{esc(k)}</th><td>{v if isinstance(v, str) and v.startswith('<') else esc(v)}</td></tr>"
                   for k, v in rows if v not in (None, "", [], {}))
    return f'<table class="kv">{body}</table>' if body else ""


def swatch(hexes: list[str]) -> str:
    return "".join(f'<span class="sw" style="background:{esc(h)}" title="{esc(h)}"></span><code>{esc(h)}</code> ' for h in hexes if h)


def timeline(ctx: Ctx, res: dict) -> str:
    d = max(ctx.duration, 1e-6)
    W = 1000

    def x(t):
        return round(float(t) / d * W, 2)
    lanes, y = [], 0
    labels = []
    edit, motion, sound, deliv, typo = (res.get(k) or {} for k in ("edit", "motion", "sound", "delivery", "typography"))
    fill_col = {"face": "#0a84ff", "face+screen": "#5e5ce6", "face+graphic": "#64d2ff", "screen": "#30d158", "graphic": "#ff9f0a", "broll": "#bf5af2"}
    if edit.get("shots"):
        for s in edit["shots"]:
            lanes.append(f'<rect x="{x(s["a"])}" y="{y}" width="{max(0.8, x(s["b"]) - x(s["a"]))}" height="12" fill="{fill_col.get(s.get("fill"), "#8e8e93")}" opacity=".8"><title>{esc(s.get("fill"))} {s["a"]:.2f}–{s["b"]:.2f} s</title></rect>')
        for c in edit.get("cuts", []):
            lanes.append(f'<rect class="step" data-t="{c["t"]}" x="{x(c["t"])}" y="{y}" width="1.5" height="12" fill="var(--ink)"><title>{esc(c["kind"])} cut {c["t"]:.2f} s</title></rect>')
        labels.append((y, "shots · cuts"))
        y += 18
    if motion.get("elements"):
        rows: list[float] = []
        for e in motion["elements"]:
            k = next((i for i, end in enumerate(rows) if end <= e["t_in"]), len(rows))
            if k == len(rows):
                rows.append(0)
            rows[k] = e["t_out"]
            kind = (e.get("entrance") or {}).get("kind")
            lanes.append(f'<rect class="step" data-t="{e["t_in"]}" x="{x(e["t_in"])}" y="{y + k * 7}" width="{max(1, x(e["t_out"]) - x(e["t_in"]))}" height="6" rx="2" fill="var(--acc)" opacity=".75"><title>element {e["id"]}: {esc(e["kind_guess"])}, {esc(kind)}</title></rect>')
        labels.append((y, "elements"))
        y += 7 * max(1, len(rows)) + 6
    if typo.get("text_items"):
        for it in typo["text_items"]:
            if it["role"] == "caption":
                lanes.append(f'<rect x="{x(it["a"])}" y="{y}" width="{max(0.5, x(it["b"]) - x(it["a"]))}" height="6" fill="var(--muted)" opacity=".5"/>')
            elif it["role"] in ("title", "cta"):
                lanes.append(f'<rect class="step" data-t="{it["a"]}" x="{x(it["a"])}" y="{y + 7}" width="{max(1, x(it["b"]) - x(it["a"]))}" height="5" fill="var(--blue)"><title>{esc(it["role"])}: {esc(it["text"])}</title></rect>')
        labels.append((y, "captions · titles"))
        y += 18
    if deliv.get("pauses", {}).get("list"):
        for t, ln in deliv["pauses"]["list"]:
            lanes.append(f'<rect x="{x(t)}" y="{y}" width="{max(0.5, x(t + ln) - x(t))}" height="6" fill="var(--muted)" opacity=".6"><title>pause {ln:.2f} s</title></rect>')
    for e in deliv.get("emphasis", []):
        lanes.append(f'<circle class="step" data-t="{e["t"]}" cx="{x(e["t"])}" cy="{y + 10}" r="3" fill="var(--blue)"><title>stressed: {esc(e["w"])}</title></circle>')
    if deliv:
        labels.append((y, "pauses · stressed words"))
        y += 18
    for s in sound.get("sfx", []):
        lanes.append(f'<path class="step" data-t="{s["t"]}" d="M{x(s["t"])},{y} l4,10 l-8,0 z" fill="var(--red)"><title>{esc(s["kind"])} {s["t"]:.2f} s</title></path>')
    for b in (sound.get("music") or {}).get("beats", [])[:2000]:
        lanes.append(f'<rect x="{x(b)}" y="{y + 12}" width=".6" height="4" fill="var(--muted)"/>')
    if sound:
        labels.append((y, "sound effects · beats"))
        y += 20
    ticks = "".join(f'<text x="{x(t)}" y="{y + 12}" class="tick">{vu.fmt(t)}</text>' for t in [d * k / 6 for k in range(6)])
    names = "".join(f'<text x="{W + 6}" y="{yy + 9}" class="tick">{esc(n)}</text>' for yy, n in labels)
    return f'<svg viewBox="0 0 {W + 150} {y + 16}" class="tl">{"".join(lanes)}{ticks}{names}</svg>'


def element_card(e: dict, img: Images) -> str:
    en, ex = e.get("entrance") or {}, e.get("exit") or {}
    def move(m, label):
        if not m:
            return ""
        bits = [m.get("kind")]
        if m.get("ease"):
            bits.append(m["ease"])
        if m.get("duration_s"):
            bits.append(f"{m['duration_s']:.2f} s")
        if m.get("scale_from") not in (None, 1.0) and label == "in":
            bits.append(f"from {m['scale_from']}×")
        if m.get("from"):
            bits.append(f"from the {m['from']} {m.get('distance_pct') or ''}%")
        if m.get("overshoot"):
            bits.append(f"{m['overshoot'] * 100:.0f}% overshoot")
        if m.get("match") is not None:
            bits.append(f"re-render match {m['match']}")
        return (f'<div class="mv"><b>{label}</b> {esc(" · ".join(str(b) for b in bits if b))}'
                f'{curve_svg(m.get("ease"), m.get("bezier"))}{"<code>" + esc(m["css"]) + "</code>" if m.get("css") else ""}</div>')
    look = e.get("look") or {}
    return f"""<div class="el">
  <div class="elh"><b>#{e['id']}</b> {ts_link(e['t_in'])}–{ts_link(e['t_out'])} · {esc(e['kind_guess'])} (a guess) · {e['area_pct']}% of the frame
   {'· covers the face' if e.get('covers_face') else ''} · while up: {esc(e.get('secondary'))}</div>
  {'<div class="muted">' + esc(e['text'][:140]) + '</div>' if e.get('text') else ''}
  {img.tag(e.get('strip'), f"element {e['id']} coming in", 'strip')}
  {move(en, 'in')}{move(ex, 'out')}
  {kv_table([('corner radius', f"{look['radius_px']} px" if look.get('radius_px') is not None else None), ('shadow', 'yes' if look.get('shadow') else None), ('border', look.get('border')), ('fill', look.get('fill'))])}
  <div class="code"><button class="copy">copy</button><pre>{esc(e.get('gsap'))}</pre></div>
</div>"""


def html(ctx: Ctx, cross: dict, style: dict, analysis_md: str = "") -> Path:
    res = ctx.results
    img = Images(ctx.out)
    v = ctx.v
    d = ctx.duration
    edit, sound, deliv, design, typo, motion = (res.get(k) or {} for k in ("edit", "sound", "delivery", "design", "typography", "motion"))
    caps = typo.get("captions") or {}
    glance = [
        ("cuts / 10 s", (edit.get("pace") or {}).get("cuts_per_10s")), ("median shot", f"{(edit.get('pace') or {}).get('median_shot_s')} s"),
        ("something new every", f"{(edit.get('rhythm') or {}).get('change_every_s')} s"),
        ("words / min", (deliv.get("pace") or {}).get("wpm")), ("typical pause", f"{(deliv.get('pauses') or {}).get('max_pause_s')} s"),
        ("caption font", f"{caps.get('font_match')} {caps.get('weight') or ''}" if caps.get("present") else "none"),
        ("music", (sound.get("music") or {}).get("verdict")), ("effects / min", sound.get("sfx_per_min")),
        ("elements / min", motion.get("per_min")),
        ("entrance", f"{(motion.get('entrances') or [{}])[0].get('kind')} · {(motion.get('entrances') or [{}])[0].get('ease')}" if motion.get("entrances") else None),
    ]
    glance_html = "".join(f'<div class="g"><div class="gv">{esc(v_)}</div><div class="gk">{esc(k)}</div></div>' for k, v_ in glance if v_ not in (None, "None s"))

    def section(key, title, extra=""):
        r = res.get(key) or {}
        if not r:
            return ""
        lines = "".join(f"<li>{link_times(s, d)}</li>" for s in r.get("summary", []))
        ev = "".join(f'<figure>{img.tag(e.get("img"), e.get("what"))}<figcaption>{ts_link(e.get("t"))} {esc(e.get("what"))}</figcaption></figure>'
                     for e in r.get("evidence", []) if e.get("img"))
        err = f'<p class="warn">This lens failed: {esc(r["error"])}</p>' if r.get("error") else ""
        return f'<section id="{key}"><h2>{esc(title)}</h2>{err}<ul class="sum">{lines}</ul>{extra}<div class="ev">{ev}</div></section>'

    design_extra = ""
    if design.get("palette"):
        pal = design["palette"]
        design_extra = kv_table([("overall", swatch([c["hex"] for c in pal.get("overall", [])[:8]])),
                                 ("canvas", swatch([c["hex"] for c in pal.get("canvas", [])])),
                                 ("accents", swatch([c["hex"] for c in pal.get("accents", [])])),
                                 ("footage", swatch([c["hex"] for c in pal.get("footage", [])])),
                                 ("grade", (design.get("grade") or {}).get("reads"))])
    typo_extra = ""
    if caps.get("present"):
        typo_extra = kv_table([("font", f"{caps.get('font_match')} {caps.get('weight')} ({caps.get('font_style')}, match {caps.get('font_quality')})"),
                               ("size", f"{caps.get('size_pct')}% of the long side"), ("position", f"{caps.get('y_pct')}% down, {caps.get('x_pct')}% across"),
                               ("words / caption", caps.get("words_per_caption")), ("case", caps.get("case")),
                               ("fill", swatch([caps.get("color")])), ("highlight", swatch([caps.get("highlight_color")]) if caps.get("highlight_color") != caps.get("color") else None),
                               ("outline", swatch([caps.get("stroke_color")]) if caps.get("stroke") else None), ("box", swatch([caps.get("box_color")]) if caps.get("box") else None),
                               ("shadow", "yes" if caps.get("shadow") else "no"), ("entrance", f"{caps.get('entrance')} {caps.get('ease') or ''} {caps.get('ease_s') or ''}"),
                               ("sync with the voice", f"{caps['sync_ms']['median']:+d} ms (p10 {caps['sync_ms']['p10']:+d}, p90 {caps['sync_ms']['p90']:+d})" if caps.get("sync_ms") else None),
                               ("each stays", f"{caps.get('hold_s')} s")])
    sound_extra = ""
    if sound.get("sfx"):
        rows = "".join(f"<tr><td>{ts_link(s['t'])}</td><td>{esc(s['kind'])}</td><td>{s.get('dur_s')}</td><td>{'yes' if s.get('in_voice') else ''}</td><td>{s.get('conf')}</td></tr>" for s in sound["sfx"][:60])
        sound_extra = f'<table class="grid"><tr><th>time</th><th>kind</th><th>length s</th><th>over voice</th><th>confidence</th></tr>{rows}</table>'
    motion_extra = "".join(element_card(e, img) for e in motion.get("elements", [])[:24])
    cross_html = ""
    if cross.get("summary"):
        cross_html = '<section id="cross"><h2>Across the lenses</h2><ul class="sum">' + "".join(f"<li>{link_times(s, d)}</li>" for s in cross["summary"]) + "</ul></section>"
    analysis = ""
    if analysis_md.strip():
        analysis = f'<section class="analysis"><h2>Claude’s read</h2>{vu.md_to_html(analysis_md, d)}</section>'
    style_json = json.dumps(style, indent=1, ensure_ascii=False)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Teardown</title>
<style>
:root{{--bg:#fbfaf7;--card:#fff;--ink:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--acc:#ff9f0a;--blue:#0a84ff;--red:#ff375f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#000;--card:#1c1c1e;--ink:#f5f5f7;--muted:#98989d;--line:#38383a}}}}
:root[data-theme="dark"]{{--bg:#000;--card:#1c1c1e;--ink:#f5f5f7;--muted:#98989d;--line:#38383a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:28px 16px 80px}}h1{{font-size:28px;margin:0 0 4px}}h2{{margin:44px 0 10px;font-size:21px}}
.muted{{color:var(--muted)}}.top{{display:grid;grid-template-columns:minmax(0,320px) 1fr;gap:20px;align-items:start}}
@media (max-width:760px){{.top{{grid-template-columns:1fr}}}}
video{{width:100%;max-height:560px;background:#000;border-radius:14px}}
.glance{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}}.g{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}}
.gv{{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}}.gk{{font-size:12px;color:var(--muted)}}
.tl{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px;margin-top:16px}}.tick{{font-size:10px;fill:var(--muted)}}.step{{cursor:pointer}}
section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 20px 18px;margin:16px 0}}section h2{{margin-top:16px}}
.sum li{{margin:4px 0}}a.ts{{color:var(--blue);text-decoration:none;font-variant-numeric:tabular-nums}}
.ev{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:8px 0;max-width:100%}}figure img{{max-width:100%;border-radius:8px;display:block}}figcaption{{font-size:12px;color:var(--muted)}}
table{{border-collapse:collapse;margin:10px 0;font-size:14px}}.kv th{{text-align:left;color:var(--muted);font-weight:500;padding:3px 14px 3px 0;vertical-align:top}}.kv td{{padding:3px 0}}
.grid th,.grid td{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}}
.sw{{display:inline-block;width:16px;height:16px;border-radius:4px;border:1px solid var(--line);vertical-align:-3px;margin-right:4px}}
code{{font:12.5px ui-monospace,SFMono-Regular,monospace}}.el{{border-top:1px solid var(--line);padding:14px 0}}.elh{{margin-bottom:6px}}
.strip{{max-width:100%;border-radius:8px;margin:6px 0}}.mv{{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:4px 0}}
.curve{{background:var(--bg);border-radius:6px}}.curve .axis{{stroke:var(--line);stroke-width:1}}.curve .dash{{stroke-dasharray:3 3}}.curve .ease{{fill:none;stroke:var(--acc);stroke-width:2}}
.code{{position:relative}}.code pre{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font:12.5px ui-monospace,monospace;margin:6px 0}}
.copy{{position:absolute;right:6px;top:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;font-size:12px;padding:2px 8px;cursor:pointer}}
.warn{{color:var(--red)}}.analysis{{border-color:var(--blue)}}details pre{{max-height:420px;overflow:auto}}
</style></head><body><main>
<h1>Teardown</h1>
<p class="muted">{esc(v.meta.get('title') or v.meta['name'])} · {vu.fmt(d)} · {v.meta['width']}×{v.meta['height']} · {v.meta['fps']} fps ·
 {'whole video' if ctx.full else 'sections: ' + ', '.join(f"{s['why']} {vu.fmt(s['a'])}–{vu.fmt(s['b'])}" for s in ctx.sections)} · made {time.strftime('%Y-%m-%d %H:%M')}</p>
<div class="top"><video id="v" src="{esc(Path(v.path).as_uri())}" controls preload="metadata" playsinline></video><div class="glance">{glance_html}</div></div>
{timeline(ctx, res)}
{analysis}
{cross_html}
{section('edit', 'Editing: cuts, shots, camera, rhythm')}
{section('motion', 'Motion: elements and how they move', motion_extra)}
{section('typography', 'Typography: captions and titles', typo_extra)}
{section('design', 'Design: colour, grade, layout, framing', design_extra)}
{section('delivery', 'Delivery: how they talk')}
{section('sound', 'Sound: music, effects, mix', sound_extra)}
<section><h2>style.json</h2><p class="muted">The numbers in the shape an editor reads (creator-teardown contract), with everything else under "deep".</p>
<details><summary>show</summary><div class="code"><button class="copy">copy</button><pre>{esc(style_json)}</pre></div></details></section>
</main>
<script>
const v=document.getElementById('v');
document.addEventListener('click',e=>{{const a=e.target.closest('a.ts,.step');if(a){{e.preventDefault();v.currentTime=parseFloat(a.dataset.t);v.scrollIntoView({{behavior:'smooth',block:'center'}});v.play().catch(()=>{{}});return}}
 const b=e.target.closest('button.copy');if(b){{const t=b.parentElement.querySelector('pre').innerText;navigator.clipboard&&navigator.clipboard.writeText(t).then(()=>{{b.textContent='copied';setTimeout(()=>b.textContent='copy',1200)}}).catch(()=>{{}})}}}});
</script></body></html>"""
    out = ctx.out / "teardown.html"
    out.write_text(page)
    return out


def digest(ctx: Ctx, cross: dict, style: dict) -> Path:
    """teardown.md: everything measured, short, with the evidence to look at for the parts code can't judge."""
    res = ctx.results
    lines = [f"# Teardown: {ctx.v.meta.get('title') or ctx.v.meta['name']}", "",
             f"{vu.fmt(ctx.duration)} · {ctx.W}×{ctx.H} · {ctx.fps} fps · "
             + ("whole video" if ctx.full else "sections: " + ", ".join(f"{s['why']} {vu.fmt(s['a'])}–{vu.fmt(s['b'])}" for s in ctx.sections)), ""]
    order = [("edit", "Editing"), ("motion", "Motion"), ("typography", "Typography"), ("design", "Design"), ("delivery", "Delivery"), ("sound", "Sound")]
    for key, title in order:
        r = res.get(key) or {}
        if not r:
            continue
        lines.append(f"## {title}")
        if r.get("error"):
            lines.append(f"- FAILED: {r['error']}")
        lines += [f"- {s}" for s in r.get("summary", [])]
        if key == "motion":
            for e in r.get("elements", [])[:16]:
                en = e.get("entrance") or {}
                lines.append(f"  - #{e['id']} {e['t_in']:.2f}–{e['t_out']:.2f} s {e['kind_guess']}: in {en.get('kind')} {en.get('ease') or ''} "
                             f"{en.get('duration_s') or ''} s (match {en.get('match')}); while up {e.get('secondary')}")
        if key == "delivery" and r.get("sentences"):
            lines.append("- Script by sentence (time, words/min):")
            for s in r["sentences"][:40]:
                lines.append(f"  - {s['a']:.1f}–{s['b']:.1f} s ({s['wpm']}): {s['text']}")
        ev = [e for e in r.get("evidence", []) if e.get("img")]
        if ev:
            lines.append("- Evidence: " + "; ".join(f"{e['what']} → {ctx.out / e['img']}" for e in ev[:8]))
        lines.append("")
    if cross.get("summary"):
        lines.append("## Across the lenses")
        lines += [f"- {s}" for s in cross["summary"]]
        lines.append("")
    lines += ["## Files", f"- page: {ctx.out / 'teardown.html'}", f"- style.json: {ctx.out / 'style.json'}",
              f"- every measurement: {ctx.out}/<lens>.json", ""]
    out = ctx.out / "teardown.md"
    out.write_text("\n".join(lines))
    return out


def demo():
    assert 'data-t="3.500"' in link_times("at 3.5 s", 10.0)
    assert "a class" not in link_times("5.5 per 10 s, within 2 s", 30.0)
    assert 'data-t="4.200"' in link_times("4.2 s screenshot comes in", 30.0)
    assert "a class" not in link_times("at 99 s", 10.0)
    svg = curve_svg("back.out(1.7)")
    assert svg.startswith("<svg") and "path" in svg
    assert kv_table([("a", 1), ("b", None)]).count("<tr>") == 1
    print("report demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
