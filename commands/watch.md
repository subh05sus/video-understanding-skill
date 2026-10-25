---
description: Watch a video (file or URL) and answer a question about it, in a Think-Act-Observe loop
argument-hint: <video path(s) or URL(s)> [question] [--level quick|normal|deep|max]
---
Use the `video-understanding` skill to watch this video and answer the question about it. Run the skill's
loop: overview first, then targeted looks, and an answer with timestamped evidence.

$ARGUMENTS

Effort level: use `--level` if given (`--depth` means the same), else the level the words ask for, else
`normal`. When it is unclear and matters (a long video, several videos, no question), ask once with the
level menu from the skill. `deep` and `max` add the teardown (`vu teardown`, `--level max`).

If no question was given, give a summary with chapters and key moments (the "Summary + chapters" recipe).
If no video was given, ask for a file path or URL.
