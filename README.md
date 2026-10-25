# video-understanding

**Agentic video understanding for Claude.** Claude watches a video the way a person does, instead of
squinting at a fixed stack of frames:

- it gets a **map** first: segments, cuts, where it's fast or still, speech vs music, a storyboard, and
  what it already knows about this video;
- it **jumps** to the parts that matter (`search` finds where a word is *said*, *shown on screen* or *noted*);
- it pulls **frames on demand** at any frame rate, and **re-watches fast parts** at 8–12 fps;
- it **zooms** into a frame at full resolution and **reads** on-screen text (OCR);
- it **listens**: a local transcript with word timings, plus loudness, silences, music and sound hits;
- it **skips** what doesn't matter, within a frame **budget** per depth (`quick` / `normal` / `deep`);
- it **remembers** what it learned about each video for the next question;
- it answers with **timestamped evidence** and a "how I watched" trail, and can hand you the **clip/GIF**
  and an **HTML watch-path report**.

It's the Think-Act-Observe loop that Google describes for
[Gemini's agentic video understanding](https://ai.google.dev/gemini-api/docs/video-understanding), with
Claude as the agent and everything run locally. Nothing is uploaded, and no API key is needed.

```
think   "price is probably said and shown around the pricing page; skip the intro"
act     vu search demo.mp4 "pro" "per month"          → said 4:22, shown 4:30
act     vu frames demo.mp4 --from 4:20 --to 4:40 --count 6
act     vu zoom demo.mp4 --at 4:31 --region center    → "Pro €29 / month, billed yearly"
act     vu frames demo.mp4 --from 9:03 --to 9:10 --fps 6   (the fast part)
answer  €29/mo billed yearly (read at 4:31); test-mode Stripe checkout at 9:01–9:12 (seen)
```

## Install

### As a user skill (Claude Code)

```bash
git clone https://github.com/subh05sus/video-understanding-skill ~/video-understanding-skill
ln -s ~/video-understanding-skill/skills/video-understanding ~/.claude/skills/video-understanding
mkdir -p ~/.claude/commands && ln -s ~/video-understanding-skill/commands/watch.md ~/.claude/commands/watch.md
```

### As a plugin

```
/plugin marketplace add subh05sus/video-understanding-skill
/plugin install video-understanding@video-understanding
```

### Dependencies

| | needed for | macOS |
|---|---|---|
| Python 3.9+ (stdlib only) | everything | built in |
| ffmpeg / ffprobe | everything | `brew install ffmpeg` |
| whisper.cpp (`whisper-cli`) | speech | `brew install whisper-cpp` (the `small` model, ~470 MB, and a 1 MB voice-detection model are fetched on first use) |
| yt-dlp | URLs (YouTube, Instagram, TikTok, X, …) | `brew install yt-dlp` |
| swiftc (Xcode command line tools) | on-screen text via macOS Vision | `xcode-select --install` |
| tesseract | on-screen text elsewhere | optional |
| Python 3.10-3.13 with numpy, OpenCV, Pillow | the teardown only | `vu setup-deep` makes its own venv (`brew install python@3.12` if needed) |

`python3 skills/video-understanding/scripts/vu.py doctor` shows what's available.

## Use

Just ask, and the skill triggers on any video file or URL:

> watch ~/Downloads/demo.mp4 and tell me when he opens the settings
>
> summarize this with chapters: https://www.youtube.com/watch?v=…
>
> critique the hook of these three reels: a.mp4 b.mp4 c.mp4
>
> turn this screen recording into step-by-step notes with the exact commands

Or use the command: `/watch <file|url> [question]`. With no question it gives a summary with chapters.

Recipes built in: **summary + chapters**, **reel / hook critique**, **find the moment**, **video → steps**,
and **compare videos**. See [`references/recipes.md`](skills/video-understanding/references/recipes.md).

## Deep teardown: how a video is made

Watching tells you *what* is in a video. The teardown measures *how it is made*, so that "the card pops in
with a bounce" becomes `back.out(1.7)`, 0.39 s, from 0.6× scale, checked by re-rendering it:

```bash
vu teardown reel.mp4          # once per machine first: vu setup-deep  (numpy, OpenCV, Pillow in their own venv)
```

Six lenses run on the video, all locally (whisper.cpp, Apple Vision, OpenCV, numpy; no API key, nothing uploaded):

| lens | what it measures | how |
|---|---|---|
| **edit** | every cut at the video's own frame rate and its kind (hard, jump, match, zoom-punch, dissolve, fade through a colour, wipe/slide/push with its direction, pixel-block mask, zoom transition with its ease), shots and what fills them, punch-ins and pushes, camera shake, montage bursts, "something new every X s" | frame differences + colour histograms, blend fits, region growth, ORB feature matching, sparse optical flow with the speaker masked out |
| **motion** | every element that comes and goes (cards, screenshots, stickers, titles, picture-in-picture): its entrance and exit as GSAP (ease, duration, start scale, slide distance, overshoot), what it does while up, stagger, corner radius and shadow, a ready-to-paste GSAP snippet | multi-scale template tracking at native fps with camera compensation, `ease.fit` against GSAP's eases and a free cubic-bezier, then **re-rendering the fitted move and scoring it against the real frames** |
| **typography** | captions (words per caption, position, size, case, fill, outline, shadow, box, highlight colour, entrance, sync to the voice in ms) and titles, the text hierarchy, the **font** | Apple Vision OCR track at 4 fps, role detection, full-resolution style analysis, and font matching by **rendering ~50 free Google Fonts and system fonts at 400-900 weight** and comparing glyph shapes |
| **design** | palette (canvas, accents, footage), grade (contrast, warmth, black point, vignette), background type (room, flat, gradient with its stops, grid paper with its spacing), layout mix (split, picture-in-picture, cards, canvas), framing (shot size, headroom, eye line), safe zones, card style | Lab k-means with the person mask, plane fits, line periodicity, rectangle detection |
| **delivery** | words per minute, pauses (from the sound), fillers in English/German/Hindi, **which words are stressed**, pitch range and energy, the hook, questions, you-address, CTA, eye contact, gestures | whisper words (re-run when the first pass is poor), YIN pitch tracking, loudness, z-scored prominence, Apple Vision face and hand pose |
| **sound** | loudness (LUFS, range, true peak), whether there is a music bed (and its BPM and beat grid when there is), sound effects (whoosh, pop, riser, impact, ding) and what they land on, mix width and compression | numpy STFT, pause-floor analysis, onset envelopes, Ellis beat tracking, effect shapes |

Then the lenses are laid over each other (effects on cuts, cuts on the beat, graphics on stressed words, "show
what you say", the hook's first cut/graphic/word), and you get:

- **`teardown.html`**: the video with a multi-track timeline, every lens, each animation with its ease curve and GSAP;
- **`teardown.md`**: the digest Claude reads before writing its own analysis (design intent, rhetoric, motion language), which goes on top of the page;
- **`style.json`**: the numbers in the shape an editor reads (motionmaxxing's creator-teardown contract), plus everything else under `deep`.

Long videos (over 3 min) are measured whole by the cheap passes and looked at closely in the hook, the ending,
the busiest 30 s and a typical 30 s (`--full` for every second).

### What it measured on six real reels

Six short-form reels (vertical, 38–77 s; German, English and Hinglish talking heads, split screens and one
motion-graphics piece). Numbers straight from the lenses, no human labelling:

| reel | cuts / 10 s | top cut kinds | something new every | words / min | typical pause | caption font (match) | music | sfx / min | elements | background |
|---|---|---|---|---|---|---|---|---|---|---|
| A · Hinglish, split screens, cards | 6.7 | jump 29%, dissolve 21%, slide 11% | 0.54 s | 185* | 0.37 s | Poppins 700 (fair) | none | 8.6 | 26 | flat colour |
| B · German, split screen, chats | 6.8 | zoom-punch 41%, jump 30% | 0.74 s | 258 | 0.15 s | Raleway 700 (fair) | noise-like floor | 1.8 | 24 | room |
| C · German, split screen, UI animation | 1.4 | hard, jump, zoom | 1.73 s | 211 | 0.23 s | Montserrat 700 (weak) | room tone | 3.7 | 18 | flat colour |
| D · German, floating screenshots | 5.4 | zoom-punch 30%, hard 27% | 0.70 s | 211 | 0.20 s | Montserrat 700 (weak) | room tone | 1.1 | 24 | flat colour |
| E · German, talking head | 2.1 | jump 56% | 1.68 s | 224 | 0.22 s | Montserrat 700 (fair) | none | 0.0 | 24 | gradient |
| F · English, motion graphics | 6.0 | match 26%, hard 22%, wipe 17% | 0.57 s | 153 | 0.19 s | pixel font (weak: closest JetBrains Mono) | room tone | 0.0 | 28 | grid paper |

\* A is Hinglish: its words come from an English re-run of whisper (the first pass was unreadable), so the
pace is marked approximate. "weak" fonts are the closest free font, not the font itself; the style class
(bold sans, pixel) still holds.

Things a plain look misses that showed up here: F opens with an 11-cut montage in 1.2 s and uses a yellow
pixel-block mask, a teal staircase wipe, an iris reveal and a zoom-dissolve; A's footage shrinks into a card
with `back.out(1.7)` and grows back with `power3.out`; B cuts every pause out (the longest pause it keeps is
0.15 s, at 258 words a minute).

### Speed

Everything that isn't Claude thinking runs as ordinary local code, in parallel: the six lenses run in separate
processes as soon as the lenses they read are done (delivery and edit first; sound and typography after
delivery; design after edit; motion last), and the motion lens tracks its elements on threads. `--jobs N`
sets the number of processes (default: half the cores; `--jobs 1` runs them one at a time). Results are cached
per video, so a second run, or `--only-report`, is instant.

## With the skill vs. Claude with no skill

Claude can read images and text, but it can't open a video. Without this skill, given `reel.mp4`, the best it can
do is improvise: pull a handful of still frames with ffmpeg, maybe run a transcription if one is installed, and
describe what those stills show. Everything between the stills, all the sound, and anything that needs
measuring is out of reach.

| | Claude, no skill | with the skill: watching (`vu`) | with the skill: teardown (`vu teardown`) |
|---|---|---|---|
| **Open a file or a URL** | no video input; extracts stills itself if it thinks to | any file, or a URL via yt-dlp, cached by content | same |
| **How it looks** | a few evenly spaced stills, once | a map first, then the parts that matter, re-watched at up to the native frame rate | every frame, at the native rate |
| **Speech** | only if it runs a transcriber itself | local whisper with word timings, searchable | plus a language check: a poor pass is re-run and the better one kept |
| **Text on screen** | read off a downscaled still | OCR with positions, zoom for small text | every caption and title tracked, with its timing and style |
| **Sound** | none | loudness, silences, speech vs music, hits | music bed (from how far pauses drop), BPM and beats when there is one, every sound effect classified |
| **Finding a moment** | guesses between stills | search by what is said, shown or noted, timestamped | — |
| **Cuts and transitions** | what changed between the stills it took | cut list and pacing | every cut with its kind, duration and direction (dissolve, fade through a colour, wipe, slide, pixel mask, zoom) |
| **Animation** | "the card pops in" | watch it at 10–30 fps and describe it | the GSAP ease, duration, start scale, overshoot and slide distance, checked by re-rendering |
| **Font** | "a bold sans, maybe Montserrat" | read it at full size and judge | glyphs rendered in ~50 fonts × weights and compared, with how good the match is |
| **Colour and layout** | colour names | seen in frames | hex palette by role, grade numbers, background type, layout mix, framing |
| **Delivery** | what the words say | what is said and when | words per minute, every pause, which words are stressed (pitch, loudness, length), gestures |
| **Proof** | none | every claim with a timestamp and how it was known, plus a watch-path page | every number with its evidence image, plus a page with a timeline and the video |
| **Something to reuse** | prose | prose, clips, GIFs | `style.json` for an editor and GSAP for every animation |
| **Memory** | starts over every time | notes and caches per video | lens results cached; re-runs are instant |
| **Cost** | image tokens for every still | only the looks it decides to take | measuring is local compute (1–3 min for a 1-min reel); Claude reads a short digest and only the images it needs |

Claude still does the part that needs judgement: why the hook works, the script's structure and rhetoric,
what the design is going for, what to copy first. The teardown hands it measurements to reason from instead of
impressions. See [`references/teardown.md`](skills/video-understanding/references/teardown.md).

## The toolbox (`vu`)

| | |
|---|---|
| `overview` | the map, with the storyboard (one video or several) |
| `search` | said / shown / noted |
| `frames` | timestamped grids at any fps, or exact moments |
| `zoom` | a full-resolution crop |
| `ocr` | on-screen text with positions |
| `transcript` | whisper.cpp, run locally, with word timings |
| `audio` | loudness, silences, speech vs music, hits |
| `motion`, `scenes` | fast/still parts, cuts, pacing |
| `note`, `notes`, `observe` | memory and the watch log |
| `clip` | mp4 or gif |
| `report` | an HTML watch path |
| `teardown`, `setup-deep` | how it is made, measured (see above) |
| `budget`, `doctor`, `fetch`, `cache` | setup and session |

The full reference is in [`references/tools.md`](skills/video-understanding/references/tools.md). Frame grids
are built with a tiny built-in bitmap font, so no ffmpeg `drawtext` or imaging library is needed. The cache
lives in `~/.cache/video-understanding` (set `VU_CACHE` to move it). It is keyed by file content, so a video
keeps its memory when it is moved or renamed.

## Tests

```bash
python3 -m unittest discover -s tests
```

## License

MIT
