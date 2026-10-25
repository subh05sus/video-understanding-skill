---
name: video-understanding
description: Watch and understand a video the way a person does, in a Think-Act-Observe loop. Get a map first, jump to the parts that matter, pull frames on demand, re-watch fast parts at a higher fps, zoom in to read, listen to the transcript and the sound, and skip what doesn't matter. Answers come with timestamped evidence. Use it whenever a video file or video URL (YouTube, Instagram, TikTok, X, Loom, a screen recording, a render) is involved and someone wants to know what is in it. That covers "watch this", "what happens in", "when does", "find the moment", "summarize / chapter this video", "critique this reel/hook", "turn this tutorial into steps", "compare these videos", "is there a mistake in this render", or /watch, even when they don't say "video understanding". Also for how a video is made, measured: a deep teardown of its editing (cuts, transitions, camera), motion (every element's entrance with its fitted GSAP ease and duration), typography (caption style and font), design (palette, grade, layout), delivery (pace, pauses, stressed words, gestures) and sound (music, effects), with a style.json and GSAP to recreate it: "break down this reel", "how is this edited", "what font/colours/easing", "make mine like this", "teardown".
---

# Video understanding

You are the viewer. `scripts/vu.py` gives you the senses: a map of the video, eyes at any frame rate,
a zoom, a text reader, ears for speech and for sound. It never decides what the video means. **You
decide where to look next**, look, say what you saw, and decide again. That loop is the whole skill.

Don't do the passive version: pulling 30 evenly spaced frames, reading them once, and answering. It
misses whatever happens between samples, wastes looks on still parts, and can't read small text. Use
evenly spaced frames only as the storyboard, the first glance.

```bash
VU="python3 ~/.claude/skills/video-understanding/scripts/vu.py"   # or wherever this skill lives
```

Find the script relative to this file: `scripts/vu.py` next to this SKILL.md. It needs ffmpeg. Speech needs
whisper.cpp (`whisper-cli`), URLs need yt-dlp, and on-screen text uses macOS Vision (tesseract elsewhere).
Run `$VU doctor` once if something is missing, and tell the person the one-line install it prints.

## The loop

**0. Frame the question.** Before any look, write down in one line what would answer it and what kind of
evidence that is: something *said* (transcript), *shown* (frames, zoom), *written on screen* (ocr), a
*moment* (motion, cuts, sound hits), or a *pattern* across the whole video. Pick a depth:

| depth | for | frame budget |
|---|---|---|
| `quick` | "what is this video", "is it about X" | 40 |
| `normal` | most questions, find-the-moment, summaries | 150 |
| `deep` | "every mistake", step-by-step extraction, frame-exact QA, comparisons | 500 |

**1. Map.** `$VU overview <video|url> --question "…" --depth normal`. You get metadata, a timeline of
segments (motion sparkline, cuts, speech/music, what is said), the fast and still parts, sound hits, notes
from earlier sessions, and a **storyboard** grid. **Read the storyboard image** (Read tool). With several
videos, pass them all: `$VU overview a.mp4 b.mp4 c.mp4`.

**2. Think: where does the answer live?** Say it before acting: "The pricing is probably said around 1:48 and
shown right after; I'll search, then look at 1:44–1:56." Name what you will skip and why ("2:10–6:40 is one
still talking-head shot with no mention of pricing, so I'm skipping it").

**3. Act: pick the cheapest sense that can settle it.**

| need | do | costs |
|---|---|---|
| where a word is said, shown, or noted | `$VU search <v> "pricing" "€"` | 0 frames |
| what is said in a range | `$VU transcript <v> --from 1:40 --to 2:10` (`--words` for word timings) | 0 |
| text on screen | `$VU ocr <v> --at 1:49` or `--from --to --fps 1 --changes` (`--region top-right`) | 0 |
| sound without words, loud hits, silences | `$VU audio <v> --from --to` | 0 |
| where it's fast or still | `$VU motion <v> --from --to` | 0 |
| what is shown over a range | `$VU frames <v> --from 1:44 --to 1:56 --count 9` (a timestamped grid) | n |
| what *exactly* happens in a fast part | `$VU frames <v> --from 0:16.5 --to 0:18 --fps 10` | n |
| specific moments | `$VU frames <v> --at 0:12,0:47.5,1:03` (`--single` for one image each) | n |
| small text or detail | `$VU zoom <v> --at 1:49 --region top-right` (or `x,y,w,h` fractions) | 1 |

Add `--why "…"` to each look. It goes into the watch log and the report, and it makes you state what you
expect to see. **Read every image a look returns** before deciding the next step.

**4. Observe, then decide.** After each look, say in a line what you actually saw, with its time, and
whether it settles the question. If it doesn't, decide what is still unknown, then narrow (a higher fps
over a shorter range), widen (the next segment), or switch senses (if you can't read the text, OCR or zoom
it; if the speech is uncertain, look at what's on screen at the same time). Log the important ones:
`$VU observe <v> "dialog opens at 0:17.6" --at 0:17.6`.

**5. Stop when the evidence is enough**, not when the budget runs out. If a look is refused for the budget,
narrow the question or the range first. Use `--over` only when the question truly needs more, and say so.

**6. Remember.** Durable facts about this video go in `$VU note <v> "…" --at <t>`: what it's about, its
structure, where things are. The next session's overview starts from them, so don't re-watch to re-learn.

## Coarse to fine, and re-watching

- First look at a range: `--count 6–9`. Then zoom in time on what matters: `--fps 4`, then `--fps 10–12`
  over a second or two when the motion map calls it fast, or when the question depends on the order of
  events (what was clicked first, which frame the text appears on, whether a cut lands on the beat).
- A still part needs one frame. A fast part at 1 fps is a lie: you will "see" a jump that was really a
  movement. The overview lists the fast parts. Re-watch them before you claim anything that happens there.
- Text: never quote a number, name, price or URL from a downscaled grid. Confirm it with `ocr` or `zoom`.
- Speech: segments marked `?` are low-confidence. Check them against what's on screen, or say "probably".
  `--model medium` re-transcribes a hard recording. Music is told apart from speech by whether whisper heard
  words, not by listening to the music.

## Answer format

1. **The direct answer**, first, in the person's language.
2. **Evidence**: each claim with its timestamp and how you know it, *seen* / *read on screen* / *heard* /
   *inferred*. Example: `1:49 — read on screen: "Pro $20/mo" (zoom, top right)`.
3. **How I watched**: 3–8 short lines of the loop. What you looked at first, where you jumped and why, what
   you re-watched at what fps, what you skipped.
4. **Confidence**: high, medium or low, and what would raise it.

Optionally, when it helps the person, add `$VU clip <v> --from --to [--gif]` to hand them the moment, and
`$VU report <v> --answer answer.md` for an HTML page showing the watch path: the timeline, where you looked
at what fps, every evidence frame, with clickable timestamps. Make the report for `deep` work, for comparisons,
and whenever someone asks how you reached the answer.

**Never claim what you didn't see or hear.** If the answer is "it isn't in the video", say where you
looked and that you checked the transcript and on-screen text.

## Deep teardown: how it is made

When the question is about craft (editing, animation, type, design, delivery, sound) or about copying a style,
measure it instead of eyeballing it: `$VU teardown <video>` runs six lenses that measure cuts and transitions
at the video's own frame rate, fit GSAP eases to every element's entrance and check them by re-rendering,
match the caption font by rendering candidates, extract the palette and grade, track pitch and pauses, and
judge the music bed and sound effects. It writes `teardown.md` (read it), a page, and a `style.json`.
Then do the part code can't: look at the evidence and write the analysis. Follow
[references/teardown.md](references/teardown.md).

## Recipes

For these jobs, follow [references/recipes.md](references/recipes.md):

- **Summary + chapters**: TL;DR, titled chapters with timestamps, key moments.
- **Reel / hook critique**: the first 3 s, pacing, legibility, caption timing, where people would drop off.
- **Find the moment**: "when does X happen", "every time Y appears", as a precise timestamp list with frame proof.
- **Video → steps**: a tutorial or screen recording turned into numbered steps with screenshots, with the
  commands and code read off the screen.
- **Compare videos**: the same probes at equivalent moments in each.

The full command reference is [references/tools.md](references/tools.md). Worked loops, both a good one
and a passive one, are in [references/loop.md](references/loop.md).
