# Deep teardown: how a video is made

Use this when someone wants to understand or copy the craft of a video, not just its content: "break down
this reel", "how is this edited", "what font / colours / easing", "why does this hook work", "make mine like
this", "recreate this animation", "teardown", "style of this creator".

The teardown measures; you interpret. Every number in it comes from code (cuts at the video's own frame
rate, eases fitted to tracked movement, fonts matched by rendering, pitch tracked from the audio). What code
can't judge (why a design choice works, the script's rhetoric, how the pictures carry the words) is your
part, done by looking at the evidence it hands you.

## 1. Measure

```bash
$VU teardown <video|url>                 # all six lenses, in parallel processes; cached after
$VU teardown <video> --lens motion       # one lens again (delivery, sound, edit, design, typography, motion)
$VU teardown <video> --level max         # a long video: every second instead of hook, ending, busiest, typical
```

First time on a machine: `$VU setup-deep` (makes a small Python environment with numpy, OpenCV, Pillow).
`$VU doctor` shows whether it is ready. The first typography run downloads ~25 MB of free Google Fonts once.

It prints each lens's findings and three paths: `teardown.md` (read this), `teardown.html` (the page for
people), `style.json` (for editors). Everything else is in `<cache>/videos/<id>/teardown/<lens>.json`.

Videos over 3 minutes: the cheap measures (cuts, sound, transcript) cover everything; the frame-heavy lenses
look closely at the hook, the ending, the busiest 30 s and a typical 30 s. Say so when you report.

## 2. Read the digest, then look

Read `teardown.md`. Then open the evidence it lists (Read the images): the filmstrip of each animation
(`motion-el*.jpg`), the shot timeline, the palette, the font comparison, the voice plot, the spectrogram. Add
targeted looks with the normal tools where something matters: `$VU frames <v> --from A --to B --fps 30`
around an entrance, `$VU zoom` on a title, `$VU transcript --words` for the hook.

## 3. Your analysis (the part code can't do)

Write `analysis.md` in the person's language, short sections, every claim with a timestamp:

- **The format in one line.** "Talking head, one-word captions, screenshots popping above the head every ~2 s."
- **Hook (first 3 s):** what is said, shown and written; first cut, first graphic, first word (from the cross
  lens); why it stops the scroll, or doesn't.
- **Script and delivery:** the structure (hook, problem, proof, payoff, CTA) with times; the rhetorical moves
  (open loop, contrast, numbers, you-address, a question); where the stressed words fall and what they carry;
  pace and pauses as a choice ("no pause over 0.2 s: every breath cut out").
- **Visual storytelling:** what is on screen when each claim is made ("show what you say" from the cross lens);
  B-roll and screenshots as proof; how layout changes mark sections.
- **Design system:** palette roles (canvas, ink, accent), type hierarchy, cards (radius, shadow), grade, the
  consistency (or not) across shots.
- **Motion language:** the dominant entrance (kind, ease, duration, overshoot) and what it feels like; how
  exits differ; stagger; secondary motion; transitions between shots and how they're timed to the sound.
- **Editing rhythm:** cuts per 10 s, how often something new happens, where it slows down and why.
- **Sound:** music bed or none (with the evidence), effects and what they're synced to.
- **What to copy, in order of impact**, and **what not to**, each tied to the numbers.

Then put it on the page and hand it over:

```bash
$VU teardown <video> --only-report --analysis analysis.md     # rebuilds teardown.html with your read on top
```

## Reading the numbers honestly

- `match` on an animation is how much of a plain cut's error the fitted move removes when re-rendered (1 = all).
  Under 0.25 the lens calls the entrance `unclear` and leaves it out of the style figures. Quote eases with
  their match.
- A font with `match weak` is the closest free font, not the font itself. Say "closest free match: X"; the
  style class (pixel, bold sans, heavy display) is still reliable.
- Music: `none` and `room tone` come from how far real pauses drop; `unknown` means there were too few pauses
  to tell. Don't invent a bed or a BPM.
- Sound effects over speech are only reported when they are a low boom or louder than the voice, so the
  count is a floor, not a ceiling.
- `kind_guess`, shot fills, body language (eye contact = face turned to the camera, gestures = fast hand
  movement) are proxies: name them as such.
- A transcript replaced by an English run of non-English speech is a translation: its word counts and pace are
  approximate (the digest says so). The captions on screen often hold the real words.

## Recreating it

- **style.json** follows motionmaxxing's creator-teardown contract (pace, zoom, camera, captions, graphics,
  sound, hook, design), measured from this one video, with everything else under `deep`. An editor that reads
  that contract can cut a video in this style from it.
- **GSAP:** every element has a snippet (1080-wide coordinates) in `motion.json` and on the page;
  `deep.animations` in style.json collects them. Pair them with the measured font, colours and radius.
- **Compare two videos** (yours against theirs): run the teardown on both and put the key numbers side by side
  (cuts per 10 s, change every X s, words/min, typical pause, caption font and size, entrance ease and duration,
  palette), then say which differences matter most.
