# Recipes

Each recipe is a way to run the loop, not a fixed script. It still begins with `overview`, still reads
every image, and still stops when the evidence is enough.

## Summary + chapters

Depth: `normal` (use `quick` under 3 minutes, `deep` for lectures you will quote from).

1. `overview`. The timeline segments are candidate chapters, and long pauses (`audio`) and cuts are where
   chapters usually turn.
2. `transcript` over the whole video for talking videos (it's free). For videos with little speech, use the
   storyboard plus `ocr --from 0 --to end --fps 0.2 --changes` to follow the on-screen headings.
3. Draft chapters from what is *said* and *shown*. Then, for each chapter boundary you're not sure of, look at
   `frames --from <b-5s> --to <b+5s> --count 6` and move the boundary to where the topic really changes.
4. Key moments: sound hits, the fastest parts, and what the speaker stresses. Confirm each with a look.
5. Output:
   ```
   TL;DR: 2–3 sentences.
   Chapters
   0:00 Title — one line
   1:42 Title — one line
   …
   Key moments
   3:10 the demo fails and they restart it (seen, 3:08–3:14 @4fps)
   ```
   For YouTube, chapters must start at 0:00 and be at least 10 s apart.
6. `note` the chapter list, so a follow-up ("what did they say about X in chapter 3?") starts from it.

## Reel / hook critique (short-form)

Depth: `deep` (a reel is short, so look at it properly).

1. `overview --depth deep`, then `scenes` for shot lengths and pacing.
2. **The hook, 0:00–0:03, frame by frame**: `frames --from 0 --to 3 --fps 10`. What is the first frame (it's
   the thumbnail in the feed)? Is there motion or a face in the first 0.5 s? Is the promise on screen,
   and *readable* (`zoom`/`ocr` it), by 1.5 s? Is the first spoken word a hook or a throat-clear?
3. **Legibility**: for every text card, `ocr` it and `zoom` it. Is it on screen long enough to read (about 0.3 s
   per word plus 0.5 s)? Is it inside the safe zone (for 9:16, not in the bottom 20% or the right 15%, where the
   UI covers it)? Does it contrast with the ground?
4. **Caption timing**: `transcript --words` against `frames --fps 8` around three spoken lines. Do the words
   land within about 0.15 s of being said?
5. **Pacing**: the median shot and the longest shot. Find the still parts over 2.5 s with no new information.
   Those are where people swipe away.
6. **Sound**: `audio`. Are there hits on the cuts? Silence at the start (bad)? Does music drown the voice
   (sound and speech spans overlapping with high loudness)?
7. **The end**: the last 2 s. Is there a loop point or a call to action?
8. Output: a 1–10 hook score with the reason, then the issues ordered by impact, each with a timestamp and a
   concrete fix ("0:00.0–0:00.8 is an empty cream ground; open on the phone mockup from 0:04 instead"),
   then what works.

## Find the moment

Depth: `normal`. Use `deep` for "every time".

1. `overview`, then `search` with the words for it: the spoken words, the on-screen words, synonyms, and
   both languages if the video is German and the question is English.
2. For each hit, bracket it: `frames --from <hit-3s> --to <hit+3s> --count 6`. Then find the exact frame:
   `frames --from <a> --to <b> --fps 10` over the second where it changes.
3. No hits: walk the timeline segments that could hold it, with `frames --count 6` on each, skipping
   segments the timeline rules out. Say which ones you skipped.
4. Output: a list of `m:ss.ss: what happens (seen/read/heard)`, with the frame grid path as proof. For
   "every time", say how you made sure you didn't miss one (the segments covered, the search terms used).
5. Give in/out points when someone will cut it: `clip --from --to` (pad 0.3 s either side).

## Video → steps / notes

Depth: `deep` for screen recordings, `normal` for talking tutorials.

1. `overview`, then `transcript`: the narration is the skeleton of the steps.
2. `ocr --from 0 --to end --fps 0.5 --changes`. In a screen recording, every change of text on screen is a
   candidate step (a new page, a dialog, a typed command).
3. Each candidate step: `frames --at <t>` to see the screen, and `zoom` on what is clicked or typed.
   Commands, code and URLs **must** come from `ocr`/`zoom`, never from a grid or from the narration alone.
4. Watch the transitions at `--fps 8` where a click happens (the motion map spikes). The menu item that was
   clicked is often only visible for half a second.
5. Output: numbered steps, each one an action plus where it happens (`2:14`), with exact commands in code
   blocks and a screenshot (`frames --at <t> --single`) for the steps that are hard to describe. Close
   with what the video assumed but never showed (a login, an installed tool).

## Compare videos

1. `overview a b c` (the same depth for all).
2. Pick the dimensions the question asks about (hooks, pacing, claims, product shown). Probe the *same* moment
   type in each, for example the first 3 s at 10 fps, or the moment the price appears.
3. A table: one row per video, one column per dimension, a timestamp in every cell. Then the verdict.
4. `report a b c --answer answer.md` puts all the watch paths on one page.
