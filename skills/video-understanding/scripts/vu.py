#!/usr/bin/env python3
"""vu: the senses for agentic video understanding.

Claude is the viewer. This tool never decides what a video means. It gives the viewer a map of the
video and lets it look, listen and read wherever it chooses, at whatever frame rate it chooses:

    overview   the map: timeline of segments, motion, sound, speech, a storyboard, what is already known
    search     jump: where a word is said, shown on screen, or noted
    frames     look: a range at any fps (timestamped grids), or exact moments
    zoom       look closer: one frame at full resolution, cropped to a region
    ocr        read: the text on screen at a moment or across a range
    transcript listen: what is said, with word timings
    audio      listen without words: loudness, silences, music/sound vs speech, hits
    motion     where it is fast (re-watch at a higher fps) and where it is still (skip)
    scenes     the cuts, shot lengths and pacing
    note/notes video memory that survives the session
    observe    one line into the watch log: what the last look showed
    clip       cut the moment out as mp4 or gif
    report     an HTML page of the watch path: where it looked, at what fps, and the evidence
    teardown   how the video is made, measured: cuts and camera, sound, delivery, design, type, motion
               (runs the deep/ lenses with numpy and OpenCV; `vu setup-deep` makes their environment once)
    budget     frames looked at in this session, against the depth's cap
    fetch      download a URL (yt-dlp)
    doctor     which senses are available on this machine

Only the standard library is used. Video work is done by ffmpeg/ffprobe, speech by whisper.cpp,
on-screen text by macOS Vision (or tesseract when installed). Everything is cached per video under
~/.cache/video-understanding (VU_CACHE to move it), keyed by the file's content, not its path.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

VERSION = "1.0.0"
INDEX_VERSION = 3
CACHE = Path(os.environ.get("VU_CACHE", Path.home() / ".cache" / "video-understanding"))
HERE = Path(__file__).resolve().parent

# A depth sets how many frames one session may look at, how big the storyboard is, and how densely
# the on-screen text index samples. "quick" answers "what is this", "deep" answers "find every mistake".
DEPTHS = {
    "quick": {"budget": 40, "story": 9, "ocr_every": 20.0},
    "normal": {"budget": 150, "story": 16, "ocr_every": 10.0},
    "deep": {"budget": 500, "story": 25, "ocr_every": 5.0},
}
CUT_SCORE = 0.3          # a cut has to change the picture a lot
CUT_MERGE = 0.25         # two cuts closer than this are one
MOTION_FPS = 4           # motion compares frames a quarter-second apart
BIN = 0.5                # seconds per motion/audio bin
MAX_FRAMES_PER_CALL = 48
SPARK = "▁▂▃▄▅▆▇█"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"


# ─── small helpers ────────────────────────────────────────────────────────────────────────────────

class VuError(Exception):
    pass


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd: list[str], *, input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        tail = p.stderr.decode(errors="replace").strip().splitlines()[-6:]
        raise VuError(f"{Path(cmd[0]).name} failed: " + " | ".join(tail))
    return p


def need(tool: str, hint: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise VuError(f"{tool} is not installed. {hint}")
    return path


def parse_time(s: str | float | None, duration: float | None = None) -> float | None:
    """`83.5`, `83.5s`, `1:23.5`, `1:02:03`, `end`, or `-5` (five seconds before the end)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().lower().rstrip("s")
    if s in ("end", "en"):
        if duration is None:
            raise VuError("`end` needs a known duration")
        return duration
    if s.startswith("-") and duration is not None:
        return max(0.0, duration - parse_time(s[1:]))
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise VuError(f"cannot read time `{s}`: use 83.5, 1:23.5 or 1:02:03")
    t = 0.0
    for n in nums:
        t = t * 60 + n
    return t


def fmt(t: float, long: bool = False, prec: int = 1) -> str:
    t = max(0.0, round(t, prec))
    h, rem = divmod(t, 3600)
    m, sec = divmod(rem, 60)
    w = 3 + prec
    if long or h >= 1:
        return f"{int(h)}:{int(m):02d}:{sec:0{w}.{prec}f}"
    return f"{int(m)}:{sec:0{w}.{prec}f}"


def span(a: float, b: float, long: bool = False) -> str:
    return f"{fmt(a, long)}–{fmt(b, long)}"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w$%€£.]+", " ", s).strip()


def sparkline(values: list[float], lo: float, hi: float) -> str:
    if not values:
        return ""
    out = []
    for v in values:
        v = v if math.isfinite(v) else lo
        x = 0.0 if hi <= lo else (v - lo) / (hi - lo)
        out.append(SPARK[min(len(SPARK) - 1, max(0, int(x * len(SPARK))))])
    return "".join(out)


def resample(values: list[float], n: int) -> list[float]:
    if len(values) <= n:
        return values
    out = []
    for i in range(n):
        a, b = int(i * len(values) / n), int((i + 1) * len(values) / n)
        out.append(max(values[a:max(b, a + 1)]))
    return out


def motion_level(v: float) -> str:
    return "still" if v < 0.004 else "low" if v < 0.02 else "medium" if v < 0.06 else "high"


def motion_spark(values: list[float], n: int = 8) -> str:
    # log scale, so a slow pan and a whip pan both read
    vals = [math.log10(max(v, 1e-4)) for v in resample(values, n)]
    return sparkline(vals, -3.0, -0.7)


# ─── the video and its cache ──────────────────────────────────────────────────────────────────────

def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def content_id(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(2 << 20))
        if size > 4 << 20:
            f.seek(-(2 << 20), os.SEEK_END)
            h.update(f.read(2 << 20))
    return h.hexdigest()[:12]


class Video:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.exists():
            raise VuError(f"no such file: {path}")
        self.id = content_id(self.path)
        self.dir = CACHE / "videos" / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "looks").mkdir(exist_ok=True)
        self.meta = self._meta()

    # ── metadata ──
    def _meta(self) -> dict:
        f = self.dir / "meta.json"
        if f.exists():
            m = json.loads(f.read_text())
            if m.get("v") == INDEX_VERSION:
                if m.get("path") != str(self.path):
                    m["path"] = str(self.path)
                    f.write_text(json.dumps(m, indent=1))
                return m
        need("ffprobe", "Install ffmpeg (macOS: brew install ffmpeg).")
        p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(self.path)])
        info = json.loads(p.stdout)
        vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        if not vs:
            raise VuError(f"{self.path.name} has no video stream")
        aus = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
        w, h = int(vs.get("width", 0)), int(vs.get("height", 0))
        rot = 0
        for sd in vs.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        rot = int(float(vs.get("tags", {}).get("rotate", rot)))
        if abs(rot) % 180 == 90:
            w, h = h, w
        num, _, den = (vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        dur = float(info.get("format", {}).get("duration") or vs.get("duration") or 0)
        m = {
            "v": INDEX_VERSION, "id": self.id, "path": str(self.path), "name": self.path.name,
            "duration": dur, "width": w, "height": h, "fps": round(fps, 3),
            "codec": vs.get("codec_name"), "audio": bool(aus), "size": self.path.stat().st_size,
        }
        src = (CACHE / "urls.json")
        if src.exists():
            for url, rec in json.loads(src.read_text()).items():
                if rec.get("path") == str(self.path):
                    m["url"], m["title"] = url, rec.get("title")
        f.write_text(json.dumps(m, indent=1))
        return m

    @property
    def duration(self) -> float:
        return self.meta["duration"]

    @property
    def long(self) -> bool:
        return self.duration >= 3600

    def t(self, s) -> float | None:
        v = parse_time(s, self.duration)
        return None if v is None else min(max(0.0, v), self.duration)

    def load(self, name: str):
        f = self.dir / name
        return json.loads(f.read_text()) if f.exists() else None

    def save(self, name: str, data) -> None:
        (self.dir / name).write_text(json.dumps(data))

    # ── one pass over the pictures: cuts and motion ──
    def visual(self) -> dict:
        got = self.load("visual.json")
        if got:
            return got
        need("ffmpeg", "Install ffmpeg (macOS: brew install ffmpeg).")
        eprint(f"vu: measuring cuts and motion in {self.meta['name']} …")
        with tempfile.TemporaryDirectory() as td:
            cuts_f, mot_f = Path(td) / "cuts.txt", Path(td) / "motion.txt"
            graph = (
                f"[0:v]scale=160:-2,split[a][b];"
                f"[a]select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={cuts_f}[ca];"
                f"[b]fps={MOTION_FPS},select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={mot_f}[cb]"
            )
            cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "error"]
            if sys.platform == "darwin":
                cmd += ["-hwaccel", "videotoolbox"]
            cmd += ["-i", str(self.path), "-an", "-sn", "-filter_complex", graph,
                    "-map", "[ca]", "-f", "null", "-", "-map", "[cb]", "-f", "null", "-"]
            p = run(cmd, check=False)
            if p.returncode != 0 and sys.platform == "darwin":  # hardware decode refused this file
                cmd.remove("-hwaccel"); cmd.remove("videotoolbox")
                run(cmd)
            elif p.returncode != 0:
                run(cmd)
            raw_cuts = _read_scores(cuts_f)
            raw_mot = _read_scores(mot_f)
        cuts: list[float] = []
        for t, s in raw_cuts:
            if s > CUT_SCORE and t > 0.05:
                if cuts and t - cuts[-1] < CUT_MERGE:
                    continue
                cuts.append(round(t, 3))
        nb = max(1, math.ceil(self.duration / BIN))
        sums, counts = [0.0] * nb, [0] * nb
        for t, s in raw_mot:
            i = min(nb - 1, int(t / BIN))
            if any(abs(t - c) < 0.3 for c in _near(cuts, t)):
                continue  # a cut is not motion
            sums[i] += s
            counts[i] += 1
        motion = [round(sums[i] / counts[i], 5) if counts[i] else None for i in range(nb)]
        for i in range(nb):  # a bin emptied by a cut takes the busier of its neighbours
            if motion[i] is None:
                left = next((motion[j] for j in range(i - 1, -1, -1) if counts[j]), None)
                right = next((motion[j] for j in range(i + 1, nb) if counts[j]), None)
                motion[i] = max([x for x in (left, right) if x is not None], default=0.0)
        data = {"cuts": cuts, "motion": motion, "bin": BIN}
        self.save("visual.json", data)
        return data

    # ── one pass over the sound: loudness and silences ──
    def audio(self) -> dict | None:
        if not self.meta["audio"]:
            return None
        got = self.load("audio.json")
        if got:
            return got
        eprint(f"vu: listening to the loudness of {self.meta['name']} …")
        with tempfile.TemporaryDirectory() as td:
            lf = Path(td) / "loud.txt"
            p = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(self.path), "-vn", "-sn", "-af",
                     f"ebur128=metadata=1:framelog=verbose,ametadata=mode=print:key=lavfi.r128.M:file={lf},"
                     f"silencedetect=n=-42dB:d=0.35", "-f", "null", "-"])
            loud = _read_scores(lf, key="lavfi.r128.M")
        err = p.stderr.decode(errors="replace")
        silences, start = [], None
        for line in err.splitlines():
            m = re.search(r"silence_start: (-?[\d.]+)", line)
            if m:
                start = max(0.0, float(m.group(1)))
            m = re.search(r"silence_end: ([\d.]+)", line)
            if m and start is not None:
                silences.append([round(start, 2), round(float(m.group(1)), 2)])
                start = None
        if start is not None:
            silences.append([round(start, 2), round(self.duration, 2)])
        nb = max(1, math.ceil(self.duration / BIN))
        bins: list[list[float]] = [[] for _ in range(nb)]
        for t, v in loud:
            bins[min(nb - 1, int(t / BIN))].append(max(v, -70.0))
        lufs = [round(statistics.mean(b), 1) if b else -70.0 for b in bins]
        hits = []
        for i in range(2, nb):
            before = statistics.mean(lufs[max(0, i - 3):i])
            if lufs[i] > -32 and lufs[i] - before >= 8 and (not hits or i * BIN - hits[-1] > 0.75):
                hits.append(round(i * BIN, 2))
        data = {"lufs": lufs, "bin": BIN, "silences": silences, "hits": hits}
        self.save("audio.json", data)
        return data

    # ── speech ──
    def transcript(self, model: str = "small", lang: str = "auto", required: bool = False) -> dict | None:
        if not self.meta["audio"]:
            return None
        got = self.load("transcript.json")
        if got and (got.get("model") == model or model == "small"):
            return got
        data = whisper(self.path, model, lang, required=required, name=self.meta["name"])
        if data is not None:
            self.save("transcript.json", data)
        return data

    # ── on-screen text index, built lazily for search ──
    def ocr_index(self, every: float) -> list[dict]:
        got = self.load("ocr_index.json")
        if got and got.get("every", 1e9) <= every:
            return got["frames"]
        vis = self.visual()
        times = set(round(t, 1) for t in frange(0.5, self.duration, every))
        bounds = [0.0] + vis["cuts"] + [self.duration]
        if len(bounds) < 400:
            for a, b in zip(bounds, bounds[1:]):
                times.add(round((a + b) / 2, 1))
        times = sorted(times)
        eprint(f"vu: reading on-screen text at {len(times)} moments (cached after this) …")
        with tempfile.TemporaryDirectory() as td:
            paths = extract_jpegs(self, times, Path(td), long_edge=1600)
            results = ocr_images([p for p in paths])
        frames, last = [], None
        for t, p in zip(times, paths):
            lines = results.get(str(p), [])
            text = " | ".join(l["text"] for l in lines)
            if text and text != last:
                frames.append({"t": t, "text": text})
            last = text
        self.save("ocr_index.json", {"every": every, "frames": frames})
        return frames

    # ── session, budget and the watch log ──
    def session(self) -> dict:
        s = self.load("session.json")
        if not s:
            s = self.begin("normal", None)
        return s

    def begin(self, depth: str, question: str | None) -> dict:
        s = {"id": time.strftime("%Y%m%d-%H%M%S"), "depth": depth, "budget": DEPTHS[depth]["budget"],
             "used": 0, "question": question, "started": time.time()}
        self.save("session.json", s)
        (self.dir / "log.jsonl").write_text("")
        return s

    def spend(self, n: int, over: bool) -> dict:
        s = self.session()
        if n and s["used"] + n > s["budget"] and not over:
            raise VuError(
                f"budget: {s['used']} of {s['budget']} frames used at depth `{s['depth']}`; this look needs {n}. "
                f"Narrow the range, lower the fps, use transcript/ocr/search, or pass --over if the question truly needs it.")
        s["used"] += n
        self.save("session.json", s)
        return s

    def log(self, kind: str, **entry) -> None:
        entry = {"kind": kind, "at": round(time.time(), 1), **{k: v for k, v in entry.items() if v is not None}}
        with (self.dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_log(self) -> list[dict]:
        f = self.dir / "log.jsonl"
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

    def notes(self) -> list[dict]:
        return self.load("notes.json") or []


def _read_scores(path: Path, key: str = "lavfi.scene_score") -> list[tuple[float, float]]:
    rows, t = [], 0.0
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("frame:"):
            m = re.search(r"pts_time:(-?[\d.e+-]+)", line)
            if m:
                t = float(m.group(1))
        elif line.startswith(key + "="):
            try:
                val = float(line.split("=", 1)[1])
            except ValueError:
                continue
            if math.isfinite(val):
                rows.append((t, val))
    return rows


def _near(sorted_list: list[float], t: float) -> list[float]:
    import bisect
    i = bisect.bisect_left(sorted_list, t)
    return sorted_list[max(0, i - 1):i + 1]


def frange(a: float, b: float, step: float):
    t = a
    while t < b:
        yield t
        t += step


def hallucination(text: str) -> bool:
    """whisper's known inventions over music and silence: one repeated character, credits, a lone filler."""
    letters = [c for c in text.lower() if c.isalpha()]
    if len(set(letters)) <= 2:
        return True
    if re.fullmatch(r"(.{1,6}?)\1{4,}", re.sub(r"\s+", "", text)):
        return True
    t = norm(text)
    return any(x in t for x in ("untertitel im auftrag", "untertitelung des zdf", "subtitles by the amara",
                                "thanks for watching", "vielen dank fur ihre aufmerksamkeit")) and len(t) < 60


def find_vad() -> Path | None:
    dest = CACHE / "models" / "ggml-silero-v5.1.2.bin"
    for p in (dest, Path.home() / ".reelcut" / "watch" / "models" / dest.name):  # reelcut keeps the same model
        if p.exists():
            return p
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(VAD_URL, timeout=30) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.rename(dest)
        return dest
    except Exception as e:  # offline: transcribe without voice detection
        eprint(f"vu: voice-detection model unavailable ({e}); music may be transcribed as words")
        return None


def find_model(name: str) -> Path:
    env = os.environ.get("VU_WHISPER_MODEL")
    if env and Path(env).exists():
        return Path(env)
    fname = f"ggml-{name}.bin"
    for d in (CACHE / "models", Path.home() / ".cache" / "hyperframes" / "whisper" / "models",
              Path.home() / ".cache" / "whisper", Path("/opt/homebrew/share/whisper-cpp/models"),
              Path("/usr/local/share/whisper-cpp/models")):
        if (d / fname).exists():
            return d / fname
    dest = CACHE / "models" / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(name=name)
    eprint(f"vu: fetching the whisper `{name}` model once ({url}) …")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total, got, last = int(r.headers.get("Content-Length") or 0), 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total and time.time() - last > 2:
                eprint(f"   {got * 100 // total}%")
                last = time.time()
    tmp.rename(dest)
    return dest


def whisper(path: Path, model: str = "small", lang: str = "auto", required: bool = False,
            name: str | None = None) -> dict | None:
    """Transcribe a file with whisper.cpp: {"model", "language", "segments": [{"a", "b", "text", "conf",
    "words": [{"w", "a", "b", "p"}]}], "seconds"}. None when whisper.cpp is missing (unless required)."""
    binary = next((b for b in ("whisper-cli", "whisper-cpp", "whisper.cpp") if shutil.which(b)), None)
    if not binary:
        if required:
            raise VuError("whisper.cpp is not installed (macOS: brew install whisper-cpp).")
        return None
    model_path = find_model(model)
    eprint(f"vu: transcribing {name or Path(path).name} with whisper {model} (language {lang}) …")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
        out = Path(td) / "out"
        cmd = [binary, "-m", str(model_path), "-f", str(wav), "-l", lang, "-ojf", "-of", str(out), "-np",
               "-t", str(max(2, (os.cpu_count() or 4) - 1))]
        vad = find_vad()
        if vad:  # voice detection first, so music and effects are not "heard" as words
            cmd += ["--vad", "-vm", str(vad)]
        run(cmd)
        raw = json.loads((Path(str(out) + ".json")).read_text(errors="replace"))
    segments = []
    for seg in raw.get("transcription", []):
        words, cur = [], None
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if text.startswith("[_") or text.startswith("<|"):
                continue
            a, b = tok["offsets"]["from"] / 1000, tok["offsets"]["to"] / 1000
            if cur is None or text.startswith(" ") or not cur["w"]:
                if cur and cur["w"].strip():
                    words.append(cur)
                cur = {"w": text.strip(), "a": a, "b": b, "p": [tok.get("p", 1.0)]}
            else:
                cur["w"] += text
                cur["b"] = b
                cur["p"].append(tok.get("p", 1.0))
        if cur and cur["w"].strip():
            words.append(cur)
        words = [{"w": w["w"], "a": round(w["a"], 2), "b": round(w["b"], 2), "p": round(min(w["p"]), 2)} for w in words]
        text = seg.get("text", "").strip()
        if not text or hallucination(text):
            continue
        conf = round(statistics.mean([w["p"] for w in words]), 2) if words else 0.0
        segments.append({"a": seg["offsets"]["from"] / 1000, "b": seg["offsets"]["to"] / 1000,
                         "text": text, "conf": conf, "words": words})
    return {"model": model, "language": raw.get("result", {}).get("language"), "segments": segments,
            "seconds": round(time.time() - t0, 1)}


def resolve(src: str) -> Video:
    if is_url(src):
        return Video(fetch(src))
    p = Path(src).expanduser()
    if p.exists():
        return Video(p)
    if re.fullmatch(r"[0-9a-f]{12}", src) and (CACHE / "videos" / src / "meta.json").exists():
        return Video(Path(json.loads((CACHE / "videos" / src / "meta.json").read_text())["path"]))
    raise VuError(f"no such file, URL or cached video id: {src}")


def fetch(url: str) -> Path:
    urls_f = CACHE / "urls.json"
    urls = json.loads(urls_f.read_text()) if urls_f.exists() else {}
    if url in urls and Path(urls[url]["path"]).exists():
        return Path(urls[url]["path"])
    ytdlp = need("yt-dlp", "Install it to watch URLs (macOS: brew install yt-dlp; elsewhere: pipx install yt-dlp).")
    out = CACHE / "downloads"
    out.mkdir(parents=True, exist_ok=True)
    eprint(f"vu: downloading {url} …")
    p = run([ytdlp, "--no-playlist", "--no-progress", "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
             "--merge-output-format", "mp4", "-o", str(out / "%(extractor)s-%(id)s.%(ext)s"),
             "--print", "after_move:filepath", "--print", "title", url])
    lines = [l for l in p.stdout.decode(errors="replace").splitlines() if l.strip()]
    path = next((l for l in reversed(lines) if Path(l).exists()), None)
    if not path:
        raise VuError("yt-dlp finished but no file was found: " + " | ".join(lines[-3:]))
    title = next((l for l in lines if l != path), None)
    urls[url] = {"path": path, "title": title, "at": time.strftime("%Y-%m-%d")}
    urls_f.write_text(json.dumps(urls, indent=1))
    return Path(path)


# ─── pictures: frame extraction, a tiny bitmap font, grids ───────────────────────────────────────

FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    " ": ["00000"] * 7,
}


class Img:
    """An RGB picture as bytes, enough to stamp labels and tile frames without any imaging library."""

    def __init__(self, w: int, h: int, data: bytearray | None = None, fill=(18, 18, 20)):
        self.w, self.h = w, h
        self.data = data if data is not None else bytearray(bytes(fill) * (w * h))

    @staticmethod
    def from_ppm(b: bytes) -> "Img":
        tokens, i = [], 0
        while len(tokens) < 4:
            while b[i:i + 1].isspace():
                i += 1
            if b[i:i + 1] == b"#":
                while b[i:i + 1] not in (b"\n", b""):
                    i += 1
                continue
            j = i
            while not b[j:j + 1].isspace():
                j += 1
            tokens.append(b[i:j])
            i = j
        i += 1
        if tokens[0] != b"P6" or int(tokens[3]) != 255:
            raise VuError("unexpected frame format from ffmpeg")
        w, h = int(tokens[1]), int(tokens[2])
        return Img(w, h, bytearray(b[i:i + w * h * 3]))

    def to_ppm(self) -> bytes:
        return f"P6\n{self.w} {self.h}\n255\n".encode() + bytes(self.data)

    def rect(self, x: int, y: int, w: int, h: int, rgb) -> None:
        x2, y2 = min(self.w, x + w), min(self.h, y + h)
        x, y = max(0, x), max(0, y)
        if x2 <= x:
            return
        row = bytes(rgb) * (x2 - x)
        for yy in range(y, y2):
            o = (yy * self.w + x) * 3
            self.data[o:o + len(row)] = row

    def paste(self, other: "Img", x: int, y: int) -> None:
        for yy in range(other.h):
            if 0 <= y + yy < self.h:
                o = ((y + yy) * self.w + x) * 3
                self.data[o:o + other.w * 3] = other.data[yy * other.w * 3:(yy + 1) * other.w * 3]

    def label(self, text: str, scale: int) -> None:
        pad = scale * 2
        cw = 6 * scale
        self.rect(0, 0, len(text) * cw + pad * 2 - scale, 7 * scale + pad * 2, (0, 0, 0))
        for n, ch in enumerate(text):
            glyph = FONT.get(ch, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(pad + n * cw + gx * scale, pad + gy * scale, scale, scale, (255, 255, 255))


def fit(w: int, h: int, long_edge: int) -> tuple[int, int]:
    if max(w, h) <= long_edge:
        tw, th = w, h
    elif w >= h:
        tw, th = long_edge, round(h * long_edge / w)
    else:
        tw, th = round(w * long_edge / h), long_edge
    return max(2, tw // 2 * 2), max(2, th // 2 * 2)


def grab_ppm(v: Video, t: float, w: int, h: int, crop: str | None = None) -> Img:
    t = min(t, max(0.0, v.duration - 0.05))
    vf = (crop + "," if crop else "") + f"scale={w}:{h}:flags=lanczos"
    p = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(v.path), "-frames:v", "1", "-an", "-sn",
             "-vf", vf, "-f", "image2pipe", "-c:v", "ppm", "-"])
    if not p.stdout:
        raise VuError(f"no frame at {fmt(t)}")
    return Img.from_ppm(p.stdout)


def write_jpeg(img: Img, out: Path, q: int = 3) -> Path:
    run(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-c:v", "ppm", "-i", "-", "-q:v", str(q), str(out)],
        input=img.to_ppm())
    return out


def extract_jpegs(v: Video, times: list[float], out_dir: Path, long_edge: int) -> list[Path]:
    w, h = fit(v.meta["width"], v.meta["height"], long_edge)

    def one(i_t):
        i, t = i_t
        out = out_dir / f"f{i:05d}.jpg"
        t = min(t, max(0.0, v.duration - 0.05))
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(v.path), "-frames:v", "1", "-an",
             "-vf", f"scale={w}:{h}", "-q:v", "2", str(out)])
        return out

    with cf.ThreadPoolExecutor(max_workers=min(8, len(times) or 1)) as ex:
        return list(ex.map(one, enumerate(times)))


def label_prec(times: list[float]) -> int:
    return 2 if any(abs(t * 10 - round(t * 10)) > 0.01 for t in times) else 1


def make_grids(v: Video, times: list[float], tag: str, cell: int | None, cols: int | None) -> list[tuple[Path, list[float]]]:
    n = len(times)
    prec = label_prec(times)
    per = 16 if n > 9 else max(n, 1)
    portrait = v.meta["height"] > v.meta["width"]
    cell = cell or (480 if per <= 9 else 400)
    if portrait:
        cell = int(cell * 1.2)
    cw, ch = fit(v.meta["width"], v.meta["height"], cell)
    scale = max(2, min(cw, ch) // 120)
    out = []
    chunks = [times[i:i + per] for i in range(0, n, per)]
    for k, chunk in enumerate(chunks):
        c = cols or (len(chunk) if len(chunk) <= (5 if portrait else 3) else (5 if portrait and len(chunk) > 9 else 4 if len(chunk) > 9 else 3))
        c = max(1, min(c, len(chunk)))
        rows = math.ceil(len(chunk) / c)
        gap = 4
        grid = Img(c * cw + (c - 1) * gap, rows * ch + (rows - 1) * gap)
        with cf.ThreadPoolExecutor(max_workers=min(8, len(chunk))) as ex:
            frames = list(ex.map(lambda t: grab_ppm(v, t, cw, ch), chunk))
        for i, (t, img) in enumerate(zip(chunk, frames)):
            img.label(fmt(t, v.long, prec), scale)
            grid.paste(img, (i % c) * (cw + gap), (i // c) * (ch + gap))
        seq = len(list((v.dir / "looks").glob("*.jpg")))
        path = v.dir / "looks" / f"{seq:03d}-{tag}{'-' + str(k + 1) if len(chunks) > 1 else ''}.jpg"
        write_jpeg(grid, path)
        out.append((path, chunk))
    return out


def legend(chunk: list[float], cols: int, long: bool) -> str:
    rows, prec = [], label_prec(chunk)
    for r in range(0, len(chunk), cols):
        rows.append("   " + " | ".join(fmt(t, long, prec) for t in chunk[r:r + cols]))
    return "\n".join(rows)


# ─── on-screen text ──────────────────────────────────────────────────────────────────────────────

def vision_helper() -> Path | None:
    if sys.platform != "darwin":
        return None
    out = CACHE / "bin" / "vu-vision"
    src = HERE / "vision.swift"
    if not src.exists():   # a packaged copy may ship the helper built, with no source
        return HERE / "vu-vision" if (HERE / "vu-vision").exists() else None
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    if not shutil.which("swiftc"):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    eprint("vu: building the macOS Vision text reader once …")
    p = run(["swiftc", "-O", str(src), "-o", str(out)], check=False)
    return out if p.returncode == 0 else None


def ocr_images(paths: list[Path]) -> dict[str, list[dict]]:
    if not paths:
        return {}
    helper = vision_helper()
    results: dict[str, list[dict]] = {}
    if helper:
        for i in range(0, len(paths), 200):
            p = run([str(helper)] + [str(x) for x in paths[i:i + 200]])
            for line in p.stdout.decode(errors="replace").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    results[rec["path"]] = rec.get("lines", [])
        return results
    if shutil.which("tesseract"):
        for path in paths:
            p = run(["tesseract", str(path), "-", "--psm", "11"], check=False)
            results[str(path)] = [{"text": l.strip(), "conf": None, "box": None}
                                  for l in p.stdout.decode(errors="replace").splitlines() if l.strip()]
        return results
    raise VuError("no text reader: on macOS install Xcode command line tools (xcode-select --install) for Vision, "
                  "elsewhere install tesseract. Or zoom and read the frame yourself.")


REGIONS = {
    "full": (0, 0, 1, 1), "top": (0, 0, 1, .34), "middle": (0, .33, 1, .34), "bottom": (0, .66, 1, .34),
    "left": (0, 0, .5, 1), "right": (.5, 0, .5, 1), "center": (.25, .25, .5, .5),
    "top-left": (0, 0, .5, .5), "top-right": (.5, 0, .5, .5), "bottom-left": (0, .5, .5, .5), "bottom-right": (.5, .5, .5, .5),
}


def region(s: str | None) -> tuple[float, float, float, float]:
    if not s:
        return REGIONS["full"]
    if s in REGIONS:
        return REGIONS[s]
    try:
        x, y, w, h = (float(p) for p in s.split(","))
    except ValueError:
        raise VuError(f"region `{s}`: use one of {', '.join(REGIONS)} or x,y,w,h as fractions (0.5,0,0.5,0.3)")
    return x, y, min(w, 1 - x), min(h, 1 - y)


# ─── shared readings over a range ────────────────────────────────────────────────────────────────

def speech_in(tr: dict | None, a: float, b: float) -> list[dict]:
    if not tr:
        return []
    return [s for s in tr["segments"] if s["b"] > a and s["a"] < b]


def words_in(tr: dict | None, a: float, b: float) -> str:
    out = []
    for s in speech_in(tr, a, b):
        ws = [w["w"] for w in s["words"] if a <= (w["a"] + w["b"]) / 2 < b] if s["words"] else [s["text"]]
        out.extend(ws)
    return " ".join(out)


def sound_spans(v: Video, au: dict | None, tr: dict | None) -> dict:
    """speech (from the transcript), sound without speech (music, effects), quiet."""
    if not au:
        return {"speech": [], "sound": [], "quiet": []}
    speech = [[s["a"], s["b"]] for s in (tr or {}).get("segments", [])]
    sound, cur = [], None
    for i, l in enumerate(au["lufs"]):
        t = i * au["bin"]
        loud = l > -45
        spoken = any(a - 0.2 <= t <= b + 0.2 for a, b in speech)
        if loud and not spoken:
            if cur and t - cur[1] <= au["bin"] + 1e-6:
                cur[1] = t + au["bin"]
            else:
                cur = [t, t + au["bin"]]
                sound.append(cur)
    sound = [s for s in sound if s[1] - s[0] >= 1.0]
    return {"speech": speech, "sound": sound, "quiet": au["silences"]}


def overlap(spans: list[list[float]], a: float, b: float) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def segments_for(v: Video, vis: dict, au: dict | None, rows: int = 30) -> list[tuple[float, float]]:
    d = v.duration
    bounds = [0.0] + [c for c in vis["cuts"] if 0 < c < d] + [d]
    scenes = list(zip(bounds, bounds[1:]))
    minlen = d / rows
    merged: list[list[float]] = []
    for a, b in scenes:
        if merged and merged[-1][1] - merged[-1][0] < minlen:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < minlen / 3:
        merged[-2][1] = merged.pop()[1]
    maxlen = max(minlen * 2, min(20.0, d / 8))
    silences = (au or {}).get("silences", [])
    out = []
    for a, b in merged:
        while b - a > maxlen:
            # break a long shot (a screen recording, a talking head) at a pause near the ideal length
            ideal = a + maxlen * 0.75
            pauses = [(x + y) / 2 for x, y in silences if a + maxlen * 0.4 < (x + y) / 2 < a + maxlen]
            cut = min(pauses, key=lambda p: abs(p - ideal)) if pauses else a + maxlen * 0.75
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out


# ─── commands ────────────────────────────────────────────────────────────────────────────────────

def cmd_overview(args) -> None:
    for i, src in enumerate(args.src):
        if i:
            print("\n" + "═" * 100 + "\n")
        overview_one(resolve(src), args)


def overview_one(v: Video, args) -> None:
    depth = args.depth or (v.load("session.json") or {}).get("depth", "normal")
    if args.question or args.depth or not v.load("session.json"):
        v.begin(depth, args.question)
    s = v.session()
    vis = v.visual()
    au = v.audio()
    tr = None if args.no_transcript else v.transcript(args.model, args.lang)
    m, d = v.meta, v.duration
    cuts = vis["cuts"]
    shots = [b - a for a, b in zip([0.0] + cuts, cuts + [d])]
    med = statistics.median(shots) if shots else d
    pacing = "fast" if med <= 1.6 else "medium" if med <= 3.5 else "slow"
    print(f"VIDEO  {m.get('title') or m['name']}   id {v.id}")
    if m.get("url"):
        print(f"       {m['url']}")
    print(f"       {fmt(d, v.long)} · {m['width']}×{m['height']} {'portrait' if m['height'] > m['width'] else 'square' if m['height'] == m['width'] else 'landscape'}"
          f" · {m['fps']:g} fps · {'sound' if m['audio'] else 'NO SOUND'}")
    print(f"       {len(cuts)} cuts · median shot {med:.1f}s · pacing {pacing} · {len(cuts) / max(d, 1) * 60:.0f} cuts/min")
    if tr and not tr["segments"]:
        print("       no speech heard (voice detection found no words)")
    elif tr:
        spoken = sum(seg["b"] - seg["a"] for seg in tr["segments"])
        low = sum(1 for seg in tr["segments"] if seg["conf"] < 0.5)
        print(f"       speech {spoken / max(d, 1) * 100:.0f}% of the time · language {tr['language']} · "
              f"{len(tr['segments'])} segments" + (f" ({low} low-confidence, marked ?)" if low else ""))
    elif m["audio"]:
        print("       transcript: " + ("skipped (--no-transcript)" if args.no_transcript else "unavailable (install whisper-cpp)"))
    snd = sound_spans(v, au, tr)
    if au:
        music = sum(b - a for a, b in snd["sound"])
        print(f"       sound without speech {music / max(d, 1) * 100:.0f}% · {len(au['silences'])} silences · {len(au['hits'])} hits")
    print(f"       session depth `{s['depth']}` · budget {s['used']}/{s['budget']} frames"
          + (f" · question: {s['question']}" if s.get("question") else ""))

    notes = v.notes()
    if notes:
        print("\nWHAT IS ALREADY KNOWN (notes from earlier watching)")
        for n in notes[-15:]:
            print(f"   {('@' + fmt(n['t'], v.long)) if n.get('t') is not None else '      '}  {n['text']}")

    segs = segments_for(v, vis, au, rows=args.rows)
    lo_all = [x for x in vis["motion"]]
    print(f"\nTIMELINE  ({len(segs)} segments; motion ▁ still … █ very fast; audio S speech · M music/sound · · quiet)")
    for a, b in segs:
        i0, i1 = int(a / BIN), max(int(a / BIN) + 1, int(b / BIN))
        mot = lo_all[i0:i1]
        mean_m = statistics.mean(mot) if mot else 0.0
        ncut = sum(1 for c in cuts if a < c < b)
        L = max(b - a, 1e-6)
        sp, so = overlap(snd["speech"], a, b) / L, overlap(snd["sound"], a, b) / L
        aud = ("S" if sp > 0.4 else "s" if sp > 0.1 else " ") + ("M" if so > 0.4 else "m" if so > 0.1 else " ")
        if not m["audio"]:
            aud = "--"
        elif aud == "  ":
            aud = "· "
        said = words_in(tr, a, b)
        conf = [seg["conf"] for seg in speech_in(tr, a, b)]
        q = "?" if conf and min(conf) < 0.5 else " "
        said = (said[:64] + "…") if len(said) > 65 else said
        print(f"  {span(a, b, v.long):>15}  {motion_spark(mot):<8} {motion_level(mean_m):<6} cuts {ncut:>2}  {aud} {q}{('“' + said + '”') if said else ''}")

    fast = fast_spans(v, vis)
    if fast:
        print("\nFAST PARTS (re-watch at 8–12 fps if they matter): " + ", ".join(span(a, b, v.long) for a, b in fast[:12])
              + (" …" if len(fast) > 12 else ""))
    still = still_spans(v, vis)
    if still:
        print("STILL PARTS (one frame tells all): " + ", ".join(span(a, b, v.long) for a, b in still[:10])
              + (" …" if len(still) > 10 else ""))
    if au and au["hits"]:
        print("SOUND HITS: " + ", ".join(fmt(h, v.long) for h in au["hits"][:20]) + (" …" if len(au["hits"]) > 20 else ""))

    if not args.no_storyboard:
        n = DEPTHS[s["depth"]]["story"]
        times = storyboard_times(segs, d, n)
        v.spend(len(times), over=True)
        grids = make_grids(v, times, "storyboard", None, None)
        v.log("storyboard", frm=0, to=d, frames=len(times), files=[str(p) for p, _ in grids], times=times)
        print("\nSTORYBOARD (one frame from the middle of each segment; Read these images):")
        for p, chunk in grids:
            print(f"   {p}")
    print("\nNEXT: think about where the answer lives, then search / frames / transcript / audio there. "
          "Skip what the timeline shows is irrelevant.")


def storyboard_times(segs: list[tuple[float, float]], d: float, n: int) -> list[float]:
    """The middle of each segment; evenly spread when there are more segments than frames; topped up with
    evenly spaced moments when there are fewer (one long shot still deserves a storyboard)."""
    picks = segs if len(segs) <= n else [segs[round(i * (len(segs) - 1) / (n - 1))] for i in range(n)]
    times = [round((a + b) / 2, 2) for a, b in picks]
    for i in range(n):
        if len(times) >= n:
            break
        t = round((i + 0.5) * d / n, 2)
        if all(abs(t - x) > d / (n * 2) for x in times):
            times.append(t)
    return sorted(times)


def fast_spans(v: Video, vis: dict) -> list[tuple[float, float]]:
    out, cur = [], None
    cuts = vis["cuts"]
    for i, x in enumerate(vis["motion"]):
        t = i * BIN
        dense = sum(1 for c in _near(cuts, t + BIN / 2) if abs(c - t) < 1.0) >= 2
        if x >= 0.06 or dense:
            if cur and t - cur[1] <= BIN:
                cur[1] = t + BIN
            else:
                cur = [t, t + BIN]
                out.append(cur)
    return [(a, min(b, v.duration)) for a, b in out if b - a >= 0.5]


def still_spans(v: Video, vis: dict) -> list[tuple[float, float]]:
    out, cur = [], None
    for i, x in enumerate(vis["motion"]):
        t = i * BIN
        if x < 0.004:
            if cur and t - cur[1] <= BIN:
                cur[1] = t + BIN
            else:
                cur = [t, t + BIN]
                out.append(cur)
        else:
            cur = None
    return [(a, min(b, v.duration)) for a, b in out if b - a >= 3]


def cmd_frames(args) -> None:
    v = resolve(args.src)
    if args.at:
        times = [v.t(x) for x in args.at.split(",") if x.strip()]
        a, b = min(times), max(times)
    else:
        if args.frm is None:
            raise VuError("give --at t1,t2,… or a range with --from/--to")
        a, b = v.t(args.frm), v.t(args.to if args.to is not None else "end")
        if b <= a:
            raise VuError("--to must be after --from")
        if args.fps:
            step = 1.0 / args.fps
            times = [round(t, 3) for t in frange(a, b + 1e-6, step)]
        else:
            n = args.count or 9
            times = [round(a + (b - a) * (i + 0.5) / n, 3) for i in range(n)] if n > 1 else [round((a + b) / 2, 3)]
    if len(times) > MAX_FRAMES_PER_CALL:
        raise VuError(f"{len(times)} frames in one look is too many (max {MAX_FRAMES_PER_CALL}). "
                      f"Narrow the range or lower the fps; look coarse first, then fine where it matters.")
    s = v.spend(len(times), args.over)
    tag = f"{fmt(a).replace(':', 'm')}-{fmt(b).replace(':', 'm')}"
    if args.single:
        paths = []
        w, h = fit(v.meta["width"], v.meta["height"], args.cell or 1280)
        scale = max(2, min(w, h) // 160)
        for t in times:
            img = grab_ppm(v, t, w, h)
            img.label(fmt(t, v.long), scale)
            seq = len(list((v.dir / "looks").glob("*.jpg")))
            paths.append(write_jpeg(img, v.dir / "looks" / f"{seq:03d}-frame-{fmt(t).replace(':', 'm')}.jpg"))
        v.log("frames", frm=a, to=b, fps=args.fps, frames=len(times), files=[str(p) for p in paths], times=times, why=args.why)
        for p, t in zip(paths, times):
            print(f"{fmt(t, v.long)}  {p}")
    else:
        grids = make_grids(v, times, tag, args.cell, args.cols)
        v.log("frames", frm=a, to=b, fps=args.fps, frames=len(times), files=[str(p) for p, _ in grids], times=times, why=args.why)
        for p, chunk in grids:
            cols = args.cols or _cols_used(v, chunk)
            print(f"{p}\n{legend(chunk, cols, v.long)}")
    rate = f"{args.fps:g} fps" if args.fps else f"{len(times)} frames"
    print(f"({rate} over {span(a, b, v.long)} · budget {s['used']}/{s['budget']})")
    tr = v.load("transcript.json")
    said = words_in(tr, a, b)
    if said:
        print(f"said here: “{said[:400]}{'…' if len(said) > 400 else ''}”")


def _cols_used(v: Video, chunk: list[float]) -> int:
    portrait = v.meta["height"] > v.meta["width"]
    n = len(chunk)
    return n if n <= (5 if portrait else 3) else (5 if portrait and n > 9 else 4 if n > 9 else 3)


def cmd_zoom(args) -> None:
    v = resolve(args.src)
    t = v.t(args.at)
    x, y, w, h = region(args.region)
    W, H = v.meta["width"], v.meta["height"]
    cw, ch = max(2, int(W * w) // 2 * 2), max(2, int(H * h) // 2 * 2)
    crop = f"crop={cw}:{ch}:{int(W * x)}:{int(H * y)}"
    long_edge = max(cw, ch)
    tw, th = (cw, ch) if long_edge >= 1000 else fit(cw * 4, ch * 4, 1000)
    tw, th = fit(tw, th, 2000)
    s = v.spend(1, args.over)
    img = grab_ppm(v, t, tw, th, crop=crop)
    seq = len(list((v.dir / "looks").glob("*.jpg")))
    path = write_jpeg(img, v.dir / "looks" / f"{seq:03d}-zoom-{fmt(t).replace(':', 'm')}-{args.region or 'full'}.jpg", q=2)
    v.log("zoom", frm=t, to=t, frames=1, files=[str(path)], region=args.region or "full", why=args.why)
    print(f"{path}\n({fmt(t, v.long)} · {args.region or 'full'} · {tw}×{th} · budget {s['used']}/{s['budget']})")


def cmd_ocr(args) -> None:
    v = resolve(args.src)
    if args.at:
        times = [v.t(x) for x in args.at.split(",")]
    else:
        if args.frm is None:
            raise VuError("give --at t1,t2,… or --from/--to (with --fps or --count)")
        a, b = v.t(args.frm), v.t(args.to or "end")
        if args.fps:
            times = list(frange(a, b + 1e-6, 1 / args.fps))
        else:
            n = args.count or 6
            times = [a + (b - a) * (i + 0.5) / n for i in range(n)]
    if len(times) > 120:
        raise VuError("too many moments for one read; narrow the range")
    x, y, w, h = region(args.region)
    W, H = v.meta["width"], v.meta["height"]
    crop = None if args.region in (None, "full") else f"crop={int(W * w)}:{int(H * h)}:{int(W * x)}:{int(H * y)}"
    with tempfile.TemporaryDirectory() as td:
        paths = []
        tw, th = fit(int(W * w), int(H * h), 2000)
        if max(tw, th) < 1000:
            tw, th = fit(tw * 3, th * 3, 1600)
        for i, t in enumerate(times):
            img = grab_ppm(v, t, tw, th, crop=crop)
            paths.append(write_jpeg(img, Path(td) / f"o{i:04d}.jpg", q=2))
        res = ocr_images(paths)
    v.log("ocr", frm=min(times), to=max(times), frames=0, times=times, region=args.region, why=args.why)
    last = None
    for t, p in zip(times, paths):
        lines = res.get(str(p), [])
        text = "\n      ".join(f"{l['text']}" + (f"   [{_where(l['box'])}]" if l.get("box") else "") for l in lines)
        if args.changes and text == last:
            continue
        print(f"{fmt(t, v.long)}  " + (text if text else "(no text)"))
        last = text


def _where(box) -> str:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return ("top" if cy < .33 else "middle" if cy < .66 else "bottom") + "-" + ("left" if cx < .33 else "center" if cx < .66 else "right")


def cmd_transcript(args) -> None:
    v = resolve(args.src)
    tr = v.transcript(args.model, args.lang, required=True)
    if not tr:
        raise VuError("this video has no sound")
    a = v.t(args.frm) if args.frm is not None else 0.0
    b = v.t(args.to) if args.to is not None else v.duration
    segs = speech_in(tr, a, b)
    v.log("transcript", frm=a, to=b, frames=0, why=args.why)
    if args.srt:
        for i, s in enumerate(segs, 1):
            print(f"{i}\n{_srt(s['a'])} --> {_srt(s['b'])}\n{s['text']}\n")
        return
    print(f"(language {tr['language']} · whisper {tr['model']} · ? = low confidence, check by looking or listening around it)")
    for s in segs:
        q = "?" if s["conf"] < 0.5 else " "
        if args.words:
            print(f"{span(s['a'], s['b'], v.long)} {q}")
            print("   " + " ".join(f"{w['w']}@{fmt(w['a'], v.long)}" for w in s["words"] if w["b"] > a and w["a"] < b))
        else:
            print(f"{span(s['a'], s['b'], v.long)} {q} {s['text']}")
    if not segs:
        print("(nothing said in this range)")


def _srt(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def cmd_search(args) -> None:
    v = resolve(args.src)
    tr = v.transcript(args.model, "auto")
    s = v.session()
    ocr = [] if args.no_ocr else _safe_ocr_index(v, DEPTHS[s["depth"]]["ocr_every"])
    notes = v.notes()
    v.log("search", frames=0, query=args.query)
    for q in args.query:
        nq = norm(q)
        qwords = nq.split()
        print(f"── “{q}”")
        hits = 0
        for seg in (tr or {}).get("segments", []):
            nt = norm(seg["text"])
            if nq in nt or (len(qwords) > 1 and all(w in nt for w in qwords)):
                at = seg["a"]
                for w in seg["words"]:
                    if norm(w["w"]).startswith(qwords[0][:max(3, len(qwords[0]) - 2)]):
                        at = w["a"]
                        break
                print(f"   said   {fmt(at, v.long)}  “{seg['text'][:110]}”")
                hits += 1
        for fr in ocr:
            nt = norm(fr["text"])
            if nq in nt or (len(qwords) > 1 and all(w in nt for w in qwords)):
                print(f"   shown  {fmt(fr['t'], v.long)}  {fr['text'][:110]}")
                hits += 1
        for n in notes:
            if nq in norm(n["text"]):
                print(f"   noted  {fmt(n['t'], v.long) if n.get('t') is not None else '   —  '}  {n['text'][:110]}")
                hits += 1
        if not hits:
            print("   no match in speech" + ("" if args.no_ocr else ", on-screen text") + " or notes. "
                  "Try a synonym, or look at the segments the timeline suggests.")
    if ocr:
        print(f"(on-screen text sampled every {DEPTHS[s['depth']]['ocr_every']:g}s and at each shot; text between samples can be missed)")


def _safe_ocr_index(v: Video, every: float) -> list[dict]:
    try:
        return v.ocr_index(every)
    except VuError as e:
        eprint(f"vu: on-screen text skipped: {e}")
        return []


def cmd_audio(args) -> None:
    v = resolve(args.src)
    au = v.audio()
    if not au:
        print("this video has no sound")
        return
    tr = v.load("transcript.json")
    a = v.t(args.frm) if args.frm is not None else 0.0
    b = v.t(args.to) if args.to is not None else v.duration
    i0, i1 = int(a / BIN), max(int(a / BIN) + 1, math.ceil(b / BIN))
    vals = au["lufs"][i0:i1]
    width = min(80, len(vals))
    print(f"LOUDNESS {span(a, b, v.long)}  (-60 … -10 LUFS, {(b - a) / max(width, 1):.1f}s per mark)")
    print("   " + sparkline(resample(vals, width), -60, -10))
    loud = max(vals) if vals else -70
    print(f"   loudest {loud:.0f} LUFS at {fmt(i0 * BIN + vals.index(loud) * BIN, v.long) if vals else '-'} · mean {statistics.mean(vals):.0f} LUFS")
    snd = sound_spans(v, au, tr)
    clip = lambda spans: [(max(a, x), min(b, y)) for x, y in spans if y > a and x < b]
    for name, key in (("speech", "speech"), ("music / sound without speech", "sound"), ("silence", "quiet")):
        got = clip(snd[key])
        if key == "speech" and not tr:
            print("   speech: unknown until a transcript exists (vu transcript)")
            continue
        print(f"   {name}: " + (", ".join(span(x, y, v.long) for x, y in got[:20]) + (" …" if len(got) > 20 else "") if got else "none"))
    hits = [h for h in au["hits"] if a <= h <= b]
    print("   hits (sudden loud): " + (", ".join(fmt(h, v.long) for h in hits[:30]) if hits else "none"))
    print("   (music vs speech is told apart by whether whisper heard words there, not by listening to the music itself)")
    v.log("audio", frm=a, to=b, frames=0, why=args.why)


def cmd_motion(args) -> None:
    v = resolve(args.src)
    vis = v.visual()
    a = v.t(args.frm) if args.frm is not None else 0.0
    b = v.t(args.to) if args.to is not None else v.duration
    i0, i1 = int(a / BIN), max(int(a / BIN) + 1, math.ceil(b / BIN))
    vals = vis["motion"][i0:i1]
    width = min(80, len(vals))
    print(f"MOTION {span(a, b, v.long)}  ({(b - a) / max(width, 1):.1f}s per mark, ▁ still … █ very fast)")
    print("   " + motion_spark(vals, width))
    cuts = [c for c in vis["cuts"] if a <= c <= b]
    print(f"   cuts ({len(cuts)}): " + (", ".join(fmt(c, v.long) for c in cuts[:40]) + (" …" if len(cuts) > 40 else "") if cuts else "none"))
    fast = [(x, y) for x, y in fast_spans(v, vis) if y > a and x < b]
    still = [(x, y) for x, y in still_spans(v, vis) if y > a and x < b]
    print("   fast: " + (", ".join(span(x, y, v.long) for x, y in fast) if fast else "none")
          + ("   → re-watch these at 8–12 fps" if fast else ""))
    print("   still: " + (", ".join(span(x, y, v.long) for x, y in still) if still else "none")
          + ("   → one frame each is enough" if still else ""))
    v.log("motion", frm=a, to=b, frames=0, why=args.why)


def cmd_scenes(args) -> None:
    v = resolve(args.src)
    vis = v.visual()
    a = v.t(args.frm) if args.frm is not None else 0.0
    b = v.t(args.to) if args.to is not None else v.duration
    bounds = [0.0] + vis["cuts"] + [v.duration]
    shots = [(x, y) for x, y in zip(bounds, bounds[1:]) if y > a and x < b]
    lens = [y - x for x, y in shots]
    med = statistics.median(lens) if lens else 0
    print(f"{len(shots)} shots in {span(a, b, v.long)} · median {med:.2f}s · shortest {min(lens):.2f}s · longest {max(lens):.2f}s")
    for i, (x, y) in enumerate(shots[:200], 1):
        print(f"  #{i:<3} {span(x, y, v.long)}  {y - x:5.2f}s")
    if len(shots) > 200:
        print("  …")
    v.log("scenes", frm=a, to=b, frames=0, why=args.why)


def cmd_note(args) -> None:
    v = resolve(args.src)
    notes = v.notes()
    notes.append({"t": v.t(args.at) if args.at else None, "text": args.text, "on": time.strftime("%Y-%m-%d")})
    v.save("notes.json", notes)
    v.log("note", frm=notes[-1]["t"], text=args.text)
    print(f"noted ({len(notes)} notes on this video)")


def cmd_notes(args) -> None:
    v = resolve(args.src)
    notes = v.notes()
    if args.clear:
        v.save("notes.json", [])
        print("notes cleared")
        return
    for n in notes:
        print(f"{fmt(n['t'], v.long) if n.get('t') is not None else '   —  '}  {n['text']}   ({n.get('on', '')})")
    if not notes:
        print("no notes yet")


def cmd_observe(args) -> None:
    v = resolve(args.src)
    v.log("observe", frm=v.t(args.at) if args.at else None, text=args.text)
    print("logged")


def cmd_clip(args) -> None:
    v = resolve(args.src)
    a, b = v.t(args.frm), v.t(args.to)
    if b <= a:
        raise VuError("--to must be after --from")
    ext = "gif" if args.gif else "mp4"
    out = Path(args.out).expanduser() if args.out else Path.cwd() / f"{Path(v.meta['name']).stem}-{fmt(a).replace(':', 'm')}-{fmt(b).replace(':', 'm')}.{ext}"
    if args.gif:
        w = args.width or 480
        vf = f"fps={args.fps or 12},scale={w}:-2:flags=lanczos,split[x][y];[x]palettegen=stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(v.path), "-an",
             "-filter_complex", vf, str(out)])
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(v.path),
               "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if args.width:
            cmd += ["-vf", f"scale={args.width}:-2"]
        cmd += ["-c:a", "aac", "-b:a", "160k", str(out)] if v.meta["audio"] else ["-an", str(out)]
        run(cmd)
    v.log("clip", frm=a, to=b, file=str(out))
    print(out)


def cmd_budget(args) -> None:
    v = resolve(args.src)
    if args.depth:
        s = v.session()
        s["depth"], s["budget"] = args.depth, DEPTHS[args.depth]["budget"]
        v.save("session.json", s)
    s = v.session()
    print(f"depth `{s['depth']}` · {s['used']} of {s['budget']} frames looked at · {max(0, s['budget'] - s['used'])} left")


def cmd_fetch(args) -> None:
    for url in args.url:
        print(fetch(url))


def cmd_doctor(args) -> None:
    def row(name, ok, detail):
        print(f"  {'✓' if ok else '✗'} {name:<14} {detail}")
    print(f"vu {VERSION} · cache {CACHE}")
    row("ffmpeg", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "required: brew install ffmpeg")
    row("ffprobe", bool(shutil.which("ffprobe")), shutil.which("ffprobe") or "required (comes with ffmpeg)")
    wb = next((b for b in ("whisper-cli", "whisper-cpp") if shutil.which(b)), None)
    row("whisper.cpp", bool(wb), shutil.which(wb) if wb else "for speech: brew install whisper-cpp")
    try:
        mp = None
        fname = "ggml-small.bin"
        for d in (CACHE / "models", Path.home() / ".cache" / "hyperframes" / "whisper" / "models", Path.home() / ".cache" / "whisper",
                  Path("/opt/homebrew/share/whisper-cpp/models")):
            if (d / fname).exists():
                mp = d / fname
        row("whisper model", bool(mp), str(mp) if mp else "small model is fetched on first transcript (~470 MB)")
    except Exception as e:  # pragma: no cover
        row("whisper model", False, str(e))
    row("yt-dlp", bool(shutil.which("yt-dlp")), shutil.which("yt-dlp") or "for URLs: brew install yt-dlp")
    vh = vision_helper() if sys.platform == "darwin" else None
    row("text (Vision)", bool(vh), str(vh) if vh else ("needs swiftc: xcode-select --install" if sys.platform == "darwin" else "macOS only"))
    row("text (tesseract)", bool(shutil.which("tesseract")), shutil.which("tesseract") or "optional fallback")
    py = deep_python()
    row("teardown", bool(py), py or "deep teardown mode needs numpy + OpenCV + Pillow: run `vu setup-deep` once")


# ─── deep teardown (numpy, OpenCV, Pillow in their own environment) ──────────────────────────────

DEEP_VENV = CACHE / "venv"
DEEP_PKGS = ["numpy", "opencv-python-headless", "pillow"]


def _has_deep(py: str) -> bool:
    p = subprocess.run([py, "-c", "import numpy, cv2, PIL"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0


def deep_python() -> str | None:
    """A Python that can run the teardown lenses: $VU_DEEP_PY, vu's own venv, motionmaxxing's, or this one."""
    for cand in (os.environ.get("VU_DEEP_PY"), str(DEEP_VENV / "bin" / "python"),
                 str(Path(os.environ.get("MM_HOME", Path.home() / ".motionmaxxing")) / "venv" / "bin" / "python"), sys.executable):
        if cand and Path(cand).exists() and _has_deep(cand):
            return cand
    return None


def cmd_setup_deep(args) -> None:
    """Make vu's venv for the teardown lenses with a Python 3.10-3.13 (wheels exist for those)."""
    base = next((p for p in (shutil.which(f"python3.{v}") for v in (12, 13, 11, 10)) if p), None)
    if not base:
        raise VuError("no Python 3.10-3.13 found for the teardown venv (macOS: brew install python@3.12)")
    if not (DEEP_VENV / "bin" / "python").exists():
        eprint(f"vu: making {DEEP_VENV} with {base} …")
        run([base, "-m", "venv", str(DEEP_VENV)])
    eprint("vu: installing " + ", ".join(DEEP_PKGS) + " …")
    p = subprocess.run([str(DEEP_VENV / "bin" / "pip"), "install", "-q", *DEEP_PKGS])
    if p.returncode != 0:
        raise VuError("pip install failed; see the messages above")
    print(f"teardown ready: {DEEP_VENV / 'bin' / 'python'}")


def cmd_teardown(args) -> None:
    py = deep_python()
    if not py:
        raise VuError("the teardown needs numpy, OpenCV and Pillow: run `vu setup-deep` once (or set VU_DEEP_PY)")
    v = resolve(args.src)
    cmd = [py, str(HERE / "deep" / "teardown.py"), str(v.path)]
    if args.lens:
        cmd += ["--lens", args.lens]
    if args.full:
        cmd.append("--full")
    if args.fresh:
        cmd.append("--fresh")
    if args.only_report:
        cmd.append("--only-report")
    p = subprocess.run(cmd)
    if p.returncode:
        raise VuError(f"teardown stopped (exit {p.returncode})")


def cmd_cache(args) -> None:
    root = CACHE / "videos"
    if args.clear:
        target = root / args.clear
        if target.exists() and target.parent == root:
            shutil.rmtree(target)
            print(f"cleared {args.clear}")
        else:
            print("no such cached video")
        return
    for d in sorted(root.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []:
        m = json.loads(d.read_text())
        notes = len(json.loads((d.parent / "notes.json").read_text())) if (d.parent / "notes.json").exists() else 0
        print(f"{m['id']}  {fmt(m['duration']):>8}  {notes:>2} notes  {m.get('title') or m['name']}")


# ─── the watch-path report ───────────────────────────────────────────────────────────────────────

def md_to_html(md: str, duration: float | None) -> str:
    out, in_list = [], False
    for line in md.splitlines():
        e = html.escape(line)
        e = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", e)
        e = re.sub(r"`(.+?)`", r"<code>\1</code>", e)
        if duration:  # only times inside the video become links; "posts at 17:00" stays text
            e = re.sub(r"(?<![\d:])((?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)(?![\d])",
                       lambda m: f'<a class="ts" href="#" data-t="{m.group(1)}">{m.group(1)}</a>'
                       if parse_time(m.group(1)) <= duration + 0.5 else m.group(1), e)
        if re.match(r"^\s*[-*] ", line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + re.sub(r"^\s*[-*] ", "", e) + "</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,4}) (.*)", e)
        if m:
            out.append(f"<h{len(m.group(1)) + 1}>{m.group(2)}</h{len(m.group(1)) + 1}>")
        elif e.strip():
            out.append(f"<p>{e}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def cmd_report(args) -> None:
    answer = Path(args.answer).expanduser().read_text() if args.answer else ""
    sections = []
    budget_mb = 14.0
    used_mb = 0.0
    for vi, src in enumerate(args.src):
        v = resolve(src)
        vis, au = v.visual(), v.audio()
        tr = v.load("transcript.json")
        s = v.session()
        logs = v.read_log()
        d = v.duration
        W = 1000

        def x(t):
            return round(t / max(d, 1e-6) * W, 2)
        lanes = []
        # motion heat
        mot = vis["motion"]
        for i, val in enumerate(mot):
            lvl = motion_level(val)
            if lvl != "still":
                op = {"low": .25, "medium": .55, "high": .95}[lvl]
                lanes.append(f'<rect x="{x(i * BIN)}" y="0" width="{max(.6, x(BIN))}" height="14" class="mot" opacity="{op}"/>')
        for c in vis["cuts"]:
            lanes.append(f'<rect x="{x(c)}" y="18" width="1" height="12" class="cut"/>')
        snd = sound_spans(v, au, tr)
        for a, b in snd["speech"]:
            lanes.append(f'<rect x="{x(a)}" y="34" width="{max(.6, x(b) - x(a))}" height="10" class="sp"/>')
        for a, b in snd["sound"]:
            lanes.append(f'<rect x="{x(a)}" y="34" width="{max(.6, x(b) - x(a))}" height="10" class="mu"/>')
        looks = [l for l in logs if l["kind"] in ("frames", "zoom", "ocr", "storyboard", "transcript", "audio", "motion")]
        y = 52
        for n, l in enumerate(looks):
            a, b = l.get("frm", 0) or 0, l.get("to", d) if l.get("to") is not None else d
            fps = l.get("fps")
            cls = {"frames": "lk", "storyboard": "sb", "zoom": "zm", "ocr": "oc"}.get(l["kind"], "ot")
            op = min(1, .35 + (fps or 1) / 12) if l["kind"] == "frames" else .8
            lanes.append(f'<g class="step" data-t="{a}"><rect x="{x(a)}" y="{y}" width="{max(3, x(b) - x(a))}" height="8" rx="2" class="{cls}" opacity="{op:.2f}"/>'
                         f'<title>step {n + 1}: {l["kind"]} {span(a, b, v.long)}{(" @ " + format(fps, "g") + " fps") if fps else ""}{(" — " + html.escape(l["why"])) if l.get("why") else ""}</title></g>')
            y += 11
        svg_h = y + 4
        ticks = "".join(f'<text x="{x(t)}" y="{svg_h + 12}" class="tick">{fmt(t, v.long)}</text>'
                        for t in [d * k / 6 for k in range(7)][:-1])
        svg = (f'<svg viewBox="0 0 {W} {svg_h + 16}" class="tl" data-v="{vi}">'
               f'<text x="0" y="-2" class="tick"></text>{"".join(lanes)}{ticks}</svg>')
        steps = []
        for n, l in enumerate(logs):
            a = l.get("frm")
            head = {"frames": "look", "storyboard": "storyboard", "zoom": "zoom", "ocr": "read text", "transcript": "listen (transcript)",
                    "audio": "listen (audio)", "motion": "motion map", "scenes": "cut list", "search": "search", "note": "note",
                    "observe": "observed", "clip": "clip"}.get(l["kind"], l["kind"])
            when = ""
            if a is not None:
                b = l.get("to")
                when = f'<a class="ts" href="#" data-v="{vi}" data-t="{fmt(a, v.long)}">{fmt(a, v.long)}</a>' + (f"–{fmt(b, v.long)}" if b is not None and b != a else "")
            detail = []
            if l.get("fps"):
                detail.append(f"{l['fps']:g} fps")
            if l.get("frames"):
                detail.append(f"{l['frames']} frames")
            if l.get("region"):
                detail.append(l["region"])
            if l.get("query"):
                detail.append("“" + "”, “".join(html.escape(q) for q in l["query"]) + "”")
            why = f'<div class="why">think: {html.escape(l["why"])}</div>' if l.get("why") else ""
            text = f'<div class="obs">{html.escape(l["text"])}</div>' if l.get("text") else ""
            imgs = ""
            for f in l.get("files", []) or []:
                p = Path(f)
                if p.exists() and p.suffix == ".jpg":
                    mb = p.stat().st_size / 1e6
                    if used_mb + mb > budget_mb:
                        imgs += f'<div class="muted">(image left out to keep the page small: {html.escape(str(p))})</div>'
                        continue
                    used_mb += mb
                    imgs += f'<img loading="lazy" src="data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{head} {when}"/>'
            steps.append(f'<li class="k-{l["kind"]}"><div class="h"><span class="n">{n + 1}</span> <b>{head}</b> {when} '
                         f'<span class="muted">{" · ".join(detail)}</span></div>{why}{text}{imgs}</li>')
        notes = "".join(f"<li>{('<a class=ts href=# data-v=%d data-t=%s>%s</a> ' % (vi, fmt(n['t'], v.long), fmt(n['t'], v.long))) if n.get('t') is not None else ''}{html.escape(n['text'])}</li>"
                        for n in v.notes())
        uri = Path(v.path).as_uri()
        sections.append(f"""
<section>
  <h2>{html.escape(v.meta.get('title') or v.meta['name'])}</h2>
  <p class="muted">{fmt(d, v.long)} · {v.meta['width']}×{v.meta['height']} · depth {s['depth']} · {s['used']}/{s['budget']} frames looked at · {len(looks)} looks</p>
  <video id="v{vi}" src="{uri}" controls preload="metadata"></video>
  <div class="legend"><i class="mot"></i>motion <i class="cut"></i>cuts <i class="sp"></i>speech <i class="mu"></i>music/sound
     <i class="sb"></i>storyboard <i class="lk"></i>look (darker = higher fps) <i class="zm"></i>zoom <i class="oc"></i>read text <i class="ot"></i>listen</div>
  {svg}
  <h3>Watch path</h3>
  <ol class="steps">{''.join(steps) or '<li>no steps logged</li>'}</ol>
  {('<h3>Notes kept on this video</h3><ul>' + notes + '</ul>') if notes else ''}
</section>""")
    first = resolve(args.src[0])
    q = first.session().get("question")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watch Path</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#e5e5ea;--mot:#ff9f0a;--cut:#1d1d1f;--sp:#0a84ff;--mu:#bf5af2;--lk:#30d158;--sb:#8e8e93;--zm:#ff375f;--oc:#64d2ff;--ot:#ffd60a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#000;--fg:#f5f5f7;--muted:#98989d;--card:#1c1c1e;--line:#38383a;--cut:#f5f5f7}}}}
:root[data-theme="dark"]{{--bg:#000;--fg:#f5f5f7;--muted:#98989d;--card:#1c1c1e;--line:#38383a;--cut:#f5f5f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 80px}}h1{{font-size:28px;margin:0 0 4px}}h2{{margin:40px 0 4px}}h3{{margin:28px 0 8px}}
.muted{{color:var(--muted)}}.answer{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:4px 20px;margin:20px 0}}
video{{width:100%;max-height:420px;background:#000;border-radius:12px;margin:12px 0}}
.tl{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;overflow:visible}}
.mot{{fill:var(--mot)}}.cut{{fill:var(--cut)}}.sp{{fill:var(--sp)}}.mu{{fill:var(--mu)}}.lk{{fill:var(--lk)}}.sb{{fill:var(--sb)}}.zm{{fill:var(--zm)}}.oc{{fill:var(--oc)}}.ot{{fill:var(--ot)}}
.step{{cursor:pointer}}.tick{{font-size:10px;fill:var(--muted)}}
.legend{{font-size:12px;color:var(--muted);margin:4px 0 8px;display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px;background:currentColor}}
.legend i.mot{{background:var(--mot)}}.legend i.cut{{background:var(--cut)}}.legend i.sp{{background:var(--sp)}}.legend i.mu{{background:var(--mu)}}.legend i.lk{{background:var(--lk)}}.legend i.sb{{background:var(--sb)}}.legend i.zm{{background:var(--zm)}}.legend i.oc{{background:var(--oc)}}.legend i.ot{{background:var(--ot)}}
.steps{{list-style:none;padding:0;margin:0}}.steps li{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:8px 0}}
.steps .n{{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:11px;background:var(--line);font-size:12px}}
.steps img{{max-width:100%;border-radius:8px;margin-top:8px;display:block}}.why{{color:var(--muted);margin-top:4px}}.obs{{margin-top:4px}}
a.ts{{color:var(--sp);text-decoration:none;font-variant-numeric:tabular-nums}}code{{font:13px ui-monospace,SFMono-Regular,monospace}}
</style></head><body><main>
<h1>Watch path</h1>
<p class="muted">{html.escape(q) if q else 'How the video was watched'} · made by vu {VERSION} on {time.strftime('%Y-%m-%d %H:%M')}</p>
{('<div class="answer">' + md_to_html(answer, max(resolve(x).duration for x in args.src)) + '</div>') if answer else ''}
{''.join(sections)}
</main>
<script>
function secs(s){{return s.split(':').reduce((a,b)=>a*60+parseFloat(b),0)}}
document.addEventListener('click',e=>{{const a=e.target.closest('a.ts,.step');if(!a)return;e.preventDefault();
 const vi=a.dataset.v??a.closest('svg')?.dataset.v??'0';const v=document.getElementById('v'+vi);if(!v)return;
 const t=a.dataset.t;v.currentTime=t.includes(':')?secs(t):parseFloat(t);v.scrollIntoView({{behavior:'smooth',block:'center'}});v.play().catch(()=>{{}})}});
</script></body></html>"""
    out = Path(args.out).expanduser() if args.out else first.dir / "report.html"
    out.write_text(page)
    print(out)


# ─── cli ─────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vu", description="Senses for agentic video understanding. Claude decides where to look.")
    p.add_argument("--version", action="version", version=f"vu {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, src=True, many=False):
        sp = sub.add_parser(name, help=help_)
        if src:
            sp.add_argument("src", nargs="+" if many else None, help="video file, URL, or cached id")
        sp.set_defaults(fn=fn)
        return sp

    def rng(sp, why=True):
        sp.add_argument("--from", dest="frm")
        sp.add_argument("--to")
        if why:
            sp.add_argument("--why", help="one line: what you are looking for (goes into the watch log)")

    sp = add("overview", cmd_overview, "the map of a video (or several): timeline, storyboard, notes", many=True)
    sp.add_argument("--question", "-q")
    sp.add_argument("--depth", choices=DEPTHS)
    sp.add_argument("--model", default="small")
    sp.add_argument("--lang", default="auto")
    sp.add_argument("--rows", type=int, default=30, help="max timeline segments")
    sp.add_argument("--no-transcript", action="store_true")
    sp.add_argument("--no-storyboard", action="store_true")

    sp = add("frames", cmd_frames, "look at a range at any fps, or at exact moments")
    rng(sp)
    sp.add_argument("--at", help="exact moments: 1:02,1:04.5")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--count", type=int)
    sp.add_argument("--single", action="store_true", help="one image per frame instead of a grid")
    sp.add_argument("--cell", type=int, help="long edge of each frame in px")
    sp.add_argument("--cols", type=int)
    sp.add_argument("--over", action="store_true", help="allow going over the session budget")

    sp = add("zoom", cmd_zoom, "one frame at full resolution, optionally cropped")
    sp.add_argument("--at", required=True)
    sp.add_argument("--region", help=f"{', '.join(REGIONS)} or x,y,w,h fractions")
    sp.add_argument("--why")
    sp.add_argument("--over", action="store_true")

    sp = add("ocr", cmd_ocr, "read on-screen text at moments or across a range")
    rng(sp)
    sp.add_argument("--at")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--count", type=int)
    sp.add_argument("--region")
    sp.add_argument("--changes", action="store_true", help="only print when the text changes")

    sp = add("transcript", cmd_transcript, "what is said, with timings")
    rng(sp)
    sp.add_argument("--words", action="store_true")
    sp.add_argument("--srt", action="store_true")
    sp.add_argument("--model", default="small")
    sp.add_argument("--lang", default="auto")

    sp = add("search", cmd_search, "where a word is said, shown, or noted")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--no-ocr", action="store_true")
    sp.add_argument("--model", default="small")

    sp = add("audio", cmd_audio, "loudness, silences, speech vs music/sound, hits")
    rng(sp)
    sp = add("motion", cmd_motion, "fast and still parts, cuts")
    rng(sp)
    sp = add("scenes", cmd_scenes, "cut list and shot lengths")
    rng(sp)

    sp = add("note", cmd_note, "remember something about this video for later sessions")
    sp.add_argument("text")
    sp.add_argument("--at")
    sp = add("notes", cmd_notes, "what is remembered about this video")
    sp.add_argument("--clear", action="store_true")
    sp = add("observe", cmd_observe, "log what the last look showed (for the report)")
    sp.add_argument("text")
    sp.add_argument("--at")

    sp = add("clip", cmd_clip, "cut a moment out as mp4 or gif")
    sp.add_argument("--from", dest="frm", required=True)
    sp.add_argument("--to", required=True)
    sp.add_argument("--gif", action="store_true")
    sp.add_argument("--fps", type=float)
    sp.add_argument("--width", type=int)
    sp.add_argument("--out")

    sp = add("report", cmd_report, "HTML page of the watch path (one or several videos)", many=True)
    sp.add_argument("--answer", help="markdown file with the answer, shown at the top")
    sp.add_argument("--out")

    sp = add("budget", cmd_budget, "frames used in this session; --depth to change it")
    sp.add_argument("--depth", choices=DEPTHS)

    sp = add("teardown", cmd_teardown, "measure how the video is made: edit, sound, delivery, design, type, motion")
    sp.add_argument("--lens", help="only these lenses, comma-separated (delivery,sound,edit,design,typography,motion)")
    sp.add_argument("--full", action="store_true", help="look closely at every second, even of a long video")
    sp.add_argument("--fresh", action="store_true", help="measure again instead of using cached lens results")
    sp.add_argument("--only-report", action="store_true", help="rebuild style.json, teardown.md and the page from cached lenses")
    add("setup-deep", cmd_setup_deep, "make the Python environment the teardown needs (once)", src=False)

    sp = add("fetch", cmd_fetch, "download URLs with yt-dlp", src=False)
    sp.add_argument("url", nargs="+")
    add("doctor", cmd_doctor, "which senses are available here", src=False)
    sp = add("cache", cmd_cache, "list cached videos, or --clear <id>", src=False)
    sp.add_argument("--clear")

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except VuError as e:
        eprint(f"vu: {e}")
        return 3 if str(e).startswith("budget") else 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
