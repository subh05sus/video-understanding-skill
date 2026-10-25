# video-understanding

**Give Claude a video. It watches it and answers your questions, with timestamps as proof.**

Claude can read text and look at pictures, but on its own it can't open a video file. This skill gives it
the missing senses: it can look at any moment, read the text on screen, hear what's said, and notice cuts,
music and sound effects. Everything runs on your own computer: nothing is uploaded, and you don't need an
API key.

It can also take a video apart to show **how it was made** (the editing, animations, fonts, colours, how the
person talks), so you can learn from it or recreate the style.

---

## What you can ask

Once it's installed, just talk to Claude Code about a video file or link:

> **watch** ~/Downloads/demo.mp4 and tell me when he opens the settings

> **summarize** this with chapters: https://www.youtube.com/watch?v=…

> **critique the hook** of these three reels: a.mp4 b.mp4 c.mp4

> turn this screen recording into **step-by-step notes** with the exact commands

> **break down** how this reel is edited, so I can make mine like it

Or type the command: `/watch <file or link> [your question]`. With no question you get a summary with chapters.

Every answer points to the exact moments it comes from (for example "at 1:49 the screen reads *Pro €29/month*"),
so you can check it yourself.

---

## Quick start (macOS)

**1. Install the tools it uses** (one time; needs [Homebrew](https://brew.sh)):

```bash
brew install ffmpeg whisper-cpp yt-dlp
xcode-select --install          # lets it read text on screen; skip if you already have Xcode tools
```

**2. Install the skill** (pick one):

In Claude Code, as a plugin:

```
/plugin marketplace add subh05sus/video-understanding-skill
/plugin install video-understanding@video-understanding
```

Or from a copy of this repo:

```bash
git clone https://github.com/subh05sus/video-understanding-skill ~/video-understanding-skill
ln -s ~/video-understanding-skill/skills/video-understanding ~/.claude/skills/video-understanding
mkdir -p ~/.claude/commands && ln -s ~/video-understanding-skill/commands/watch.md ~/.claude/commands/watch.md
```

**3. Try it.** Ask Claude Code: *"watch ~/Downloads/any-video.mp4 and summarize it"*.

Something missing? Run `python3 ~/video-understanding-skill/skills/video-understanding/scripts/vu.py doctor`.
It lists what's installed and the one-line fix for anything that isn't. The first time it transcribes speech,
it downloads a speech model (about 470 MB).

---

## How hard should it look? (effort levels)

You choose how much effort Claude puts in. If you don't say, it uses **normal**. It only asks when your
request could go either way (a long video, several videos, or "analyse this" with no question). One level
covers every video in a request.

| level | in plain words | good for | time for a 1-minute video |
|---|---|---|---|
| **quick** | a glance | "what is this video about?" | seconds |
| **normal** *(default)* | watches the parts that matter | most questions, summaries, finding a moment, hook feedback | under a minute |
| **deep** | watches, then measures how it's made | style breakdowns, "make mine like this", step-by-step guides | a few minutes |
| **max** | measures every second | long videos, "find every mistake" | the longest |

Say it in words ("just a quick look", "do a deep dive") or add it to the command: `/watch reel.mp4 --level deep`.

---

## With this skill vs. Claude on its own

| | Claude on its own | with this skill |
|---|---|---|
| Open a video or a link | ✗ can't open video | ✓ any file, or YouTube/Instagram/TikTok/X links |
| See what happens | only a few still images, if it thinks to grab them | looks at the parts that matter, frame by frame where things move fast |
| Hear speech | ✗ | ✓ a transcript with the time of every word |
| Read text on screen | from a small, blurry still | ✓ sharp, with zoom for small text |
| Music and sound effects | ✗ | ✓ found and labelled |
| Find "when does X happen?" | guesses | ✓ searches what's said and shown, with the time |
| How it's edited (cuts, transitions) | vague | ✓ every cut, its type and length |
| Animations | "the card pops in" | ✓ the exact motion (e.g. `back.out(1.7)`, 0.39 s, starts at 60% size) |
| Font and colours | "a bold font, maybe Montserrat" | ✓ font matched by rendering real fonts; exact hex colours |
| How the person talks | ✗ | ✓ speed, pauses, which words they stress, gestures |
| Proof | ✗ | ✓ every claim has a timestamp and an image |
| Something you can reuse | a description | ✓ a shareable page, editor settings (`style.json`), animation code (GSAP) |
| Remembers the video next time | ✗ | ✓ |

Claude still does the thinking: why a hook works, what the design is going for, what to copy first. The
skill gives it facts to reason from instead of guesses.

---

## "How was this made?" (the deep teardown)

At the **deep** and **max** levels, the skill measures six things about the video:

| | what you learn |
|---|---|
| **Editing** | every cut and transition (jump cut, zoom punch-in, crossfade, wipe, slide…), how often something new happens on screen |
| **Motion** | how each card, screenshot or title comes in and goes out, as copy-paste animation code |
| **Text** | caption style (font, size, colour, outline, position) and how well captions match the voice |
| **Design** | the colour palette, the look of the footage, backgrounds, layouts, how the person is framed |
| **Delivery** | how fast they talk, where they pause, which words they stress, how much they gesture |
| **Sound** | how loud it is, whether there's music, which sound effects play and when |

You get three things:

- **a web page** you can open and share, with the video, a timeline and every finding with its image;
- **a summary** Claude reads, then writes its own analysis on top (the hook, the script, the style, what to copy);
- **`style.json`**, the numbers in a format video editors and tools can use, plus animation code for every element.

To run it yourself: `vu teardown reel.mp4` (the first time, run `vu setup-deep` once to install what it needs).

### Example: six real reels

Measured automatically on six short vertical videos (German, English and Hindi-English):

| reel | cuts per 10 s | something new every | words per minute | longest pause it keeps | caption font | music |
|---|---|---|---|---|---|---|
| A · split screens, cards | 6.7 | 0.54 s | 185* | 0.37 s | Poppins Bold | none |
| B · split screen, chats | 6.8 | 0.74 s | 258 | 0.15 s | Raleway Bold | none clear |
| C · split screen, app animation | 1.4 | 1.73 s | 211 | 0.23 s | Montserrat Bold (closest) | none |
| D · floating screenshots | 5.4 | 0.70 s | 211 | 0.20 s | Montserrat Bold (closest) | none |
| E · talking head | 2.1 | 1.68 s | 224 | 0.22 s | Montserrat Bold | none |
| F · motion graphics | 6.0 | 0.57 s | 153 | 0.19 s | a pixel font | none |

\* A is in Hindi-English. Its transcript came from an English re-run, so its speed is approximate.
"(closest)" means the exact font isn't in the free set; this is the nearest one.

What a quick look would miss: F opens with 11 cuts in 1.2 seconds and uses a pixel-block transition, a
staircase wipe and a circle reveal; in A, the footage shrinks into a card with a springy ease and grows back;
B removes every pause (none longer than 0.15 s, at 258 words a minute).

---

## Common questions

**Is my video uploaded anywhere?** No. Everything runs on your computer. Links are downloaded to your
machine with yt-dlp first.

**Does it cost tokens?** Only for what Claude looks at and writes. The measuring (cuts, sound, fonts,
motion) is ordinary code on your computer. Claude reads a short summary and only the images it needs.

**Which languages?** Speech works in any language whisper supports; it's been tested on English, German and
Hindi-English.

**Windows or Linux?** The watching works anywhere ffmpeg runs. Reading text on screen uses macOS's built-in
text recognition (or tesseract elsewhere), and the face and pose parts of the teardown need macOS.

**It's slow on a long video.** Use **quick** or **normal** for questions. The teardown looks closely at the
key parts of a long video unless you choose **max**. Results are saved, so asking again is instant.

---

<details>
<summary><b>For developers: the toolbox, how it works, tests</b></summary>

### The loop

Claude is the viewer; `vu` gives it senses and never decides what a video means. It gets a map first, then
jumps to the parts that matter, pulls frames on demand, re-watches fast parts at a higher frame rate, zooms
to read, listens, and skips what doesn't matter within a frame budget. It's the Think-Act-Observe loop Google
describes for [Gemini's agentic video understanding](https://ai.google.dev/gemini-api/docs/video-understanding),
with Claude as the agent and everything run locally.

```
think   "price is probably said and shown around the pricing page; skip the intro"
act     vu search demo.mp4 "pro" "per month"          → said 4:22, shown 4:30
act     vu frames demo.mp4 --from 4:20 --to 4:40 --count 6
act     vu zoom demo.mp4 --at 4:31 --region center    → "Pro €29 / month, billed yearly"
answer  €29/mo billed yearly (read at 4:31)
```

### The toolbox (`vu`)

| command | what it does |
|---|---|
| `overview` | the map, with a storyboard (one video or several) |
| `search` | where a word is said, shown or noted |
| `frames` | timestamped grids at any fps, or exact moments |
| `zoom` | a full-resolution crop |
| `ocr` | on-screen text with positions |
| `transcript` | whisper.cpp, run locally, with word timings |
| `audio` | loudness, silences, speech vs music, hits |
| `motion`, `scenes` | fast/still parts, cuts, pacing |
| `note`, `notes`, `observe` | memory and the watch log |
| `clip` | an mp4 or gif of a moment |
| `report` | an HTML page of how it watched |
| `teardown`, `setup-deep` | the deep teardown (`--level max` for every second, `--jobs N` processes) |
| `budget`, `doctor`, `fetch`, `cache` | setup and session |

Full reference: [`references/tools.md`](skills/video-understanding/references/tools.md). Recipes (summary +
chapters, hook critique, find the moment, video → steps, compare videos):
[`references/recipes.md`](skills/video-understanding/references/recipes.md). The teardown method:
[`references/teardown.md`](skills/video-understanding/references/teardown.md).

### How the teardown measures

| lens | how |
|---|---|
| edit | frame differences and colour histograms at the native frame rate, blend fits, region growth, ORB feature matching, sparse optical flow with the speaker masked out |
| motion | multi-scale template tracking with camera compensation; each move fitted to GSAP's eases and a free cubic-bezier, then re-rendered and scored against the real frames |
| typography | Apple Vision OCR at 4 fps, role detection, full-resolution style analysis, font matching by rendering ~50 free Google Fonts and system fonts at 400–900 weight |
| design | Lab k-means with the person mask, plane fits for gradients, line periodicity for grid paper, rectangle detection for cards |
| delivery | whisper words (re-run when the first pass is poor), YIN pitch tracking, loudness, z-scored prominence, Apple Vision face and hand pose |
| sound | numpy STFT, pause-floor analysis for the music bed, onset envelopes, Ellis beat tracking, effect shapes |

The six lenses run in parallel processes as soon as the lenses they depend on are done, and the motion lens
tracks elements on threads. `style.json` follows motionmaxxing's creator-teardown contract.

### Dependencies

| | needed for | macOS |
|---|---|---|
| Python 3.9+ (standard library only) | everything | built in |
| ffmpeg / ffprobe | everything | `brew install ffmpeg` |
| whisper.cpp (`whisper-cli`) | speech | `brew install whisper-cpp` |
| yt-dlp | links | `brew install yt-dlp` |
| swiftc (Xcode command line tools) | on-screen text, faces, pose | `xcode-select --install` |
| tesseract | on-screen text off macOS | optional |
| Python 3.10–3.13 with numpy, OpenCV, Pillow | the teardown only | `vu setup-deep` makes its own environment |

The cache lives in `~/.cache/video-understanding` (set `VU_CACHE` to move it), keyed by file content, so a
video keeps its memory when it's moved or renamed.

### Tests

```bash
python3 -m unittest discover -s tests
```

</details>

## License

MIT
