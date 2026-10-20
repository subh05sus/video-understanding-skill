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
