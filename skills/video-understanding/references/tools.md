# vu: command reference

`vu.py <command> <video> [options]`. `<video>` is a file path, a URL (fetched with yt-dlp and cached), or the
12-character id that `overview` prints. Times are written `83.5`, `1:23.5`, `1:02:03`, `end`, or `-5`
(five seconds before the end).

Everything about a video is cached under `~/.cache/video-understanding/videos/<id>/` (set `VU_CACHE` to move
it): measurements, transcript, on-screen text index, notes, the session, the watch log, and every image a
look produced (`looks/`). The id comes from the file's content, so a renamed or moved file keeps its memory.

## The map

| command | does |
|---|---|
| `overview <v> [<v2> …] [--question "…"] [--depth quick\|normal\|deep\|max] [--rows 30] [--no-transcript] [--no-storyboard] [--model small] [--lang auto]` | Measures cuts, motion and loudness and transcribes, all once and cached. Prints the metadata, the timeline, the fast/still parts, sound hits and notes, and writes a storyboard grid. `--question` or `--depth` starts a new session, which resets the budget and the log. |
| `scenes <v> [--from --to]` | The shot list with lengths: median, shortest and longest shot. |
| `motion <v> [--from --to]` | A motion sparkline (0.5 s bins, each comparing frames 0.25 s apart), the cuts, the fast spans (re-watch these at 8–12 fps) and the still spans (one frame is enough). |
| `audio <v> [--from --to]` | A loudness sparkline (LUFS), the speech spans, the sound-without-speech spans (music, effects), the silences, and the hits (sudden loud). |

## Looking

| command | does |
|---|---|
| `frames <v> --from A --to B [--count 9 \| --fps N]` | A timestamped grid of the range: up to 16 per image, more images if there are more frames, at most 48 per call. |
| `frames <v> --at t1,t2,…` | Exact moments. |
| `… --single` | One image per frame (default 1280 px long edge, `--cell` changes it), for detail. |
| `… --cols N --cell PX` | Grid layout. |
| `zoom <v> --at T [--region R]` | One frame at full resolution. The crop is upscaled when it is small. `R` is `top`, `middle`, `bottom`, `left`, `right`, `center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, or `x,y,w,h` fractions (`0.6,0,0.4,0.25`). |
| `ocr <v> --at T[,T2] \| --from A --to B [--fps N \| --count N] [--region R] [--changes]` | On-screen text with each line's position (`top-left` … `bottom-right`). `--changes` prints only when the text changes. |

`frames` and `zoom` count toward the session budget. `ocr`, `transcript`, `audio`, `motion`, `scenes` and
`search` are free: reach for them first. Every look takes `--why "…"`, which goes into the log.
A look over the budget exits with code 3 and a message. Narrow it, or pass `--over`.

## Listening and reading

| command | does |
|---|---|
| `transcript <v> [--from --to] [--words] [--srt] [--model small\|medium\|large-v3-turbo] [--lang auto\|de\|en…]` | whisper.cpp, run locally, with voice detection first so music isn't "heard" as words. `?` marks a low-confidence segment. The model is fetched once. |
| `search <v> "term" ["term2" …] [--no-ocr]` | Where each term is **said** (with the word time), **shown** (in the on-screen text index, sampled every 20/10/5 s by depth and at every shot), or **noted**. Matching ignores case and accents and also takes all words in any order. The text index is built once per video. |

## Memory and the log

| command | does |
|---|---|
| `note <v> "text" [--at T]` | A durable fact about this video, shown at the top of every later overview. |
| `notes <v> [--clear]` | List them, or clear them. |
| `observe <v> "text" [--at T]` | One line into this session's watch log: what a look showed. It appears in the report. |
| `budget <v> [--depth D]` | Frames used out of the cap, and changes the depth mid-session. |

## Output

| command | does |
|---|---|
| `clip <v> --from A --to B [--gif] [--fps 12] [--width 480] [--out path]` | An mp4 (x264, with sound) or a gif (palette-optimised). It goes to the current directory unless `--out` is given. |
| `report <v> [<v2> …] [--answer answer.md] [--out page.html]` | A self-contained HTML watch path. It shows the answer, the video with clickable timestamps, a timeline (motion, cuts, speech, music, and every look coloured by kind and shaded by fps), each step with its *why*, observations and evidence images, and the notes. |

## Setup

| command | does |
|---|---|
| `doctor` | Lists which senses are available, and the install line for each missing one. |
| `fetch <url> […]` | Downloads with yt-dlp (≤1080p, mp4) into the cache. Other commands call it themselves when given a URL. |
| `cache [--clear <id>]` | The cached videos with their notes, or clears one. |

## Teardown (how it is made)

| command | what |
|---|---|
| `teardown <v>` | all six lenses: delivery, sound, edit, design, typography, motion; prints findings and the paths of `teardown.md`, `teardown.html`, `style.json` |
| `teardown <v> --lens edit,motion` | only these lenses (the others come from the cache) |
| `teardown <v> --level max` (or `--full`) | a long video: measure every second (default `deep`: hook, ending, busiest and a typical 30 s); several videos at once are fine |
| `teardown <v> --fresh` | measure again, ignoring cached lens results |
| `teardown <v> --jobs N` | how many processes the lenses run in (default half the cores; `1` = one at a time) |
| `teardown <v> --only-report --analysis a.md` | rebuild the page with your analysis on top, no measuring |
| `setup-deep` | once per machine: the Python environment the lenses need (numpy, OpenCV, Pillow) |

Each lens writes `<cache>/videos/<id>/teardown/<lens>.json` with a `summary` (plain lines), its measurements and
`evidence` images under `teardown/img/`. See [teardown.md](teardown.md) for how to read and use them.
