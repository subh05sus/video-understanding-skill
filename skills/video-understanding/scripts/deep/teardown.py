"""teardown: measure how a video is made, lens by lens.

    python3 teardown.py <video|url|id> [--lens edit,sound,...] [--full] [--fresh]
    python3 teardown.py demo

Each lens module in this folder has `run(ctx) -> dict` and a `demo()` self-check. They run in LENSES order,
each result is saved to <video cache>/teardown/<lens>.json and passed on in ctx.results, so a later lens
can read an earlier one (but never needs to). Cross-lens findings (a sound effect on a cut, a cut on the
beat, captions against the voice) are worked out afterwards in cross.py.

Smart depth: a video up to SMART_FULL_S long is measured whole. A longer one is mapped whole by the cheap
measures (cuts, loudness, transcript), and the frame-heavy lenses look closely only at its sections: the
hook, the ending, the busiest stretch and a typical one.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import vu  # noqa: E402
from common import Ctx, eprint, json_safe  # noqa: E402

# delivery first: it may fix the transcript (words.json) and finds the stressed words the others line up with
LENSES = ["delivery", "sound", "edit", "design", "typography", "motion"]
SMART_FULL_S = 180.0
WINDOW_S = 30.0


def smart_sections(v: vu.Video) -> tuple[list[dict], bool]:
    d = v.duration
    if d <= SMART_FULL_S:
        return [{"a": 0.0, "b": round(d, 3), "why": "the whole video"}], True
    vis = v.visual()
    au = v.audio() or {}
    nb = len(vis["motion"])
    # how much happens in each half-second bin: cuts count most, then sound hits, then motion
    busy = np.array(vis["motion"], float) * 10
    for c in vis["cuts"]:
        busy[min(nb - 1, int(c / vis["bin"]))] += 1.0
    for h in au.get("hits", []):
        busy[min(nb - 1, int(h / vis["bin"]))] += 0.5
    k = int(WINDOW_S / vis["bin"])
    dens = np.convolve(busy, np.ones(k), mode="valid")  # dens[i] covers bins i .. i+k
    secs = [{"a": 0.0, "b": min(WINDOW_S, d), "why": "the hook"},
            {"a": max(0.0, d - 20.0), "b": d, "why": "the ending"}]

    def free(a, b):
        return all(b <= s["a"] or a >= s["b"] for s in secs)

    order = np.argsort(-dens)
    for i in order:
        a = i * vis["bin"]
        if free(a, a + WINDOW_S):
            secs.append({"a": round(a, 2), "b": round(a + WINDOW_S, 2), "why": "the busiest stretch"})
            break
    med = np.median(dens)
    for i in np.argsort(np.abs(dens - med)):
        a = i * vis["bin"]
        if free(a, a + WINDOW_S):
            secs.append({"a": round(a, 2), "b": round(a + WINDOW_S, 2), "why": "a typical stretch"})
            break
    secs.sort(key=lambda s: s["a"])
    return secs, False


def make_ctx(src: str, full: bool = False) -> Ctx:
    v = vu.resolve(src)
    if full:
        secs, whole = [{"a": 0.0, "b": v.duration, "why": "the whole video (--full)"}], True
    else:
        secs, whole = smart_sections(v)
    return Ctx(v, sections=secs, full=whole)


# what each lens reads from the others; everything else can run side by side
DEPS = {"delivery": [], "edit": [], "sound": ["delivery"], "typography": ["delivery"], "design": ["edit"],
        "motion": ["edit", "typography"]}


def _version(name: str) -> int:
    return getattr(_mod(name), "VERSION", 1)


def _measure(ctx: Ctx, name: str) -> dict:
    t0 = time.time()
    eprint(f"teardown: {name} …")
    try:
        res = _mod(name).run(ctx)
    except Exception as e:  # one lens failing must not lose the others
        traceback.print_exc()
        res = {"lens": name, "error": f"{type(e).__name__}: {e}", "summary": [f"{name} failed: {e}"]}
    res = json_safe({**res, "lens": name, "version": _version(name), "sections": ctx.sections,
                     "seconds": round(time.time() - t0, 1)})
    ctx.save(name, res)
    eprint(f"teardown: {name} done in {res['seconds']}s")
    return res


def _lens_job(src: str, name: str, full: bool) -> tuple[str, dict]:
    """One lens in its own process: the lenses it depends on are read back from their saved results."""
    ctx = make_ctx(src, full)
    for other in LENSES:
        prev = ctx.load(other)
        if prev and other != name:
            ctx.results[other] = prev
    return name, _measure(ctx, name)


def warm(ctx: Ctx) -> None:
    """Build the caches several lenses share, once, before they run side by side (so no two processes write
    the same file): cuts and motion, loudness, the transcript, the samples, the face track, the Vision helper."""
    ctx.v.visual()
    ctx.v.audio()
    ctx.v.transcript()
    ctx.audio(16000)
    vu.vision_helper()
    ctx.faces_track()


def run(src: str, lenses: list[str] | None = None, full: bool = False, fresh: bool = False, jobs: int | None = None) -> Ctx:
    """Measure the lenses that aren't cached. With more than one to do they run in parallel processes, each
    starting as soon as the lenses it reads have finished (DEPS); jobs=1 runs them one after another."""
    ctx = make_ctx(src, full)
    lenses = LENSES if lenses is None else lenses   # [] = measure nothing, only rebuild from what is cached
    ctx.save("sections", {"sections": ctx.sections, "full": ctx.full})
    todo = []
    for name in LENSES:
        prev = ctx.load(name)
        if name in lenses and (fresh or not prev or prev.get("version") != _version(name) or prev.get("sections") != ctx.sections):
            todo.append(name)
        elif prev:
            ctx.results[name] = prev
            if name in lenses:
                eprint(f"teardown: {name} (cached)")
    if not todo:
        return ctx
    workers = jobs or min(len(todo), max(2, (os.cpu_count() or 4) // 2))
    if workers <= 1 or len(todo) == 1:
        for name in todo:
            ctx.results[name] = _measure(ctx, name)
        return ctx
    import concurrent.futures as cf
    import multiprocessing as mp
    warm(ctx)
    t0 = time.time()
    pending, running = list(todo), {}
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        while pending or running:
            for name in list(pending):
                if all(d not in pending and d not in running.values() for d in DEPS.get(name, [])):
                    running[ex.submit(_lens_job, ctx.v.path.as_posix(), name, full)] = name
                    pending.remove(name)
            done, _ = cf.wait(list(running), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                try:
                    _, res = fut.result()
                except Exception as e:   # the worker itself died: keep going with the others
                    res = {"lens": name, "error": f"{type(e).__name__}: {e}", "summary": [f"{name} failed: {e}"]}
                ctx.results[name] = res
    eprint(f"teardown: {len(todo)} lenses in {time.time() - t0:.0f}s with {workers} processes")
    return ctx


def _mod(name: str):
    return importlib.import_module(name)


def finish(ctx: Ctx, analysis: str = "") -> dict:
    """Cross-lens findings, style.json, the digest Claude reads and the page people read."""
    import cross as cross_mod
    import report
    import style as style_mod
    # the edit lens may have run beside delivery: time its zooms against the stressed words now
    edit_r0 = ctx.results.get("edit") or {}
    z = edit_r0.get("zoom") or {}
    if z.get("kind") not in (None, "none"):
        import edit as edit_mod
        zt = [p["t"] for p in z.get("punches", [])] + [p["t"] for p in z.get("pushes", [])]
        stressed = [e["t"] for e in (ctx.results.get("delivery") or {}).get("emphasis", []) if isinstance(e, dict)]
        z["on"] = edit_mod.zoom_on(zt, len(z.get("punches", [])), len(edit_r0.get("cuts", [])), ctx.words(), stressed)
    res = dict(ctx.results)
    res["_words"] = ctx.words()
    res["_faces"] = ctx.faces_track()
    xr = cross_mod.run(res, ctx.duration)
    edit_r = res.get("edit") or {}
    motion_r = res.get("motion") or {}
    # one rhythm across the edit and the graphics: something new on screen every so often
    events = [e["t"] for e in (edit_r.get("rhythm") or {}).get("events", []) if isinstance(e, dict)]
    events += [e["t_in"] for e in motion_r.get("elements", []) if isinstance(e, dict)]
    events = sorted(set(round(t, 2) for t in events))
    if events:
        gaps = np.diff([0.0] + events + [ctx.duration])
        xr["rhythm"] = {"change_every_s": round(float(np.median(gaps)), 3), "p90_s": round(float(np.percentile(gaps, 90)), 3),
                        "longest_static_s": round(float(gaps.max()), 3), "events": len(events)}
        xr["summary"].insert(0, f"Something new on screen (cut, zoom, insert or graphic) every {xr['rhythm']['change_every_s']} s "
                                f"(median; p90 {xr['rhythm']['p90_s']} s; longest stretch {xr['rhythm']['longest_static_s']} s).")
    ctx.save("cross", xr)
    st = style_mod.build(ctx.results, xr, ctx.v.meta)
    if xr.get("rhythm"):
        st.setdefault("deep", {})["rhythm_all"] = xr["rhythm"]
    (ctx.out / "style.json").write_text(json.dumps(json_safe(st), indent=2, ensure_ascii=False))
    md = report.digest(ctx, xr, st)
    page = report.html(ctx, xr, st, analysis)
    return {"cross": xr, "style": st, "digest": md, "page": page}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="teardown")
    p.add_argument("src")
    p.add_argument("--lens", help="comma-separated: " + ",".join(LENSES))
    p.add_argument("--full", action="store_true", help="measure every second, even of a long video")
    p.add_argument("--fresh", action="store_true", help="ignore cached lens results")
    p.add_argument("--jobs", type=int, help="processes to run lenses in (default: half the cores; 1 = one at a time)")
    p.add_argument("--only-report", action="store_true", help="rebuild style.json, teardown.md and the page from cached lenses")
    p.add_argument("--analysis", help="a markdown file with Claude's read of the video, shown at the top of the page")
    a = p.parse_args(argv)
    lenses = [x.strip() for x in a.lens.split(",")] if a.lens else None
    try:
        ctx = run(a.src, [] if a.only_report else lenses, a.full, a.fresh, a.jobs)
    except vu.VuError as e:
        eprint(f"teardown: {e}")
        return 1
    analysis = Path(a.analysis).expanduser().read_text() if a.analysis else ""
    if not analysis and (ctx.out / "analysis.md").exists():
        analysis = (ctx.out / "analysis.md").read_text()
    if a.analysis:
        (ctx.out / "analysis.md").write_text(analysis)
    out = finish(ctx, analysis)
    for name in LENSES:
        r = ctx.results.get(name) or {}
        if r and (lenses is None or name in lenses or a.only_report):
            print(f"\n## {name}")
            for line in r.get("summary", []):
                print(f"- {line}")
    if out["cross"].get("summary"):
        print("\n## across the lenses")
        for line in out["cross"]["summary"]:
            print(f"- {line}")
    print(f"\ndigest   {out['digest']}\npage     {out['page']}\nstyle    {ctx.out / 'style.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
