"""Shared plumbing for the teardown lenses.

Every lens gets one `Ctx`: the video (vu.Video, so the cache, cuts, loudness and transcript are shared with
the rest of the skill), the sections to look at closely, the results of the lenses that ran before it, and
helpers that hand back numpy arrays: frames over a range at any fps and size, single frames, the audio as
samples, Apple Vision reads (text, faces, body and hand pose, the person's mask) on arrays.

Conventions every lens follows, so the orchestrator, style.json and the report can read any of them:
  - times are seconds (float, 3 decimals), ranges are [a, b];
  - positions and boxes are fractions of the frame, origin top left: box = [x, y, w, h];
  - sizes that matter to an editor also come in source pixels (`px`) and in % of the frame's height;
  - colours are "#RRGGBB";
  - every lens returns a dict with "lens", "summary" (plain lines a person reads), its measurements,
    and "evidence": [{"t", "what", "img"?}] pointing at the moments that show each finding.

    python3 common.py demo
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import vu  # noqa: E402  (the stdlib half of the skill: cache, cuts, loudness, transcript, OCR helper)

TEARDOWN_VERSION = 1


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def r3(x) -> float:
    return round(float(x), 3)


def hexcolor(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(c))))) for c in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def json_safe(o):
    """numpy scalars and arrays -> plain JSON."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def ffmpeg_dims(w: int, h: int, long_edge: int | None) -> tuple[int, int]:
    if not long_edge:
        return w // 2 * 2, h // 2 * 2
    return vu.fit(w, h, long_edge)


@dataclass
class Ctx:
    v: vu.Video
    sections: list[dict] = field(default_factory=list)   # [{"a", "b", "why"}]: where frame-heavy work looks
    full: bool = True                                    # the sections cover the whole video
    results: dict = field(default_factory=dict)          # lens name -> its result, in run order
    out: Path | None = None                              # <video cache>/teardown

    def __post_init__(self):
        if self.out is None:
            self.out = self.v.dir / "teardown"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "img").mkdir(exist_ok=True)
        if not self.sections:
            self.sections = [{"a": 0.0, "b": self.duration, "why": "the whole video"}]

    # ── facts ──
    @property
    def W(self) -> int:
        return self.v.meta["width"]

    @property
    def H(self) -> int:
        return self.v.meta["height"]

    @property
    def fps(self) -> float:
        return self.v.meta["fps"] or 30.0

    @property
    def duration(self) -> float:
        return self.v.duration

    @property
    def portrait(self) -> bool:
        return self.H > self.W

    def vis(self) -> dict:
        return self.v.visual()

    def cuts(self) -> list[float]:
        return list(self.vis()["cuts"])

    def loud(self) -> dict | None:
        return self.v.audio()

    def transcript(self) -> dict | None:
        return self.v.transcript()

    def words(self) -> list[dict]:
        """Every word with its timing: [{"w", "a", "b", "p"}], in order. The delivery lens may write a better
        transcript (a re-run in the right language or model) to teardown/words.json; it wins when present."""
        better = self.load("words")
        if better and better.get("words"):
            return better["words"]
        tr = self.transcript() or {}
        return [w for s in tr.get("segments", []) for w in s.get("words", [])]

    def in_sections(self, t: float) -> bool:
        return any(s["a"] <= t <= s["b"] for s in self.sections)

    # ── pictures ──
    def frames(self, a: float, b: float, fps: float | None = None, long_edge: int | None = 480,
               gray: bool = False):
        """Yield (t, frame) from a to b at `fps` (default: the video's own), scaled so the long edge is
        `long_edge` px (None = source size). Frames are uint8 RGB (H, W, 3), or (H, W) when gray."""
        a = max(0.0, a)
        b = min(self.duration, b)
        if b <= a:
            return
        native = self.fps
        w, h = ffmpeg_dims(self.W, self.H, long_edge)
        pix = "gray" if gray else "rgb24"
        ch = 1 if gray else 3
        if not fps or abs(fps - native) < 0.01:
            # the video's own frames, passed through: seek half a frame early so ffmpeg starts exactly on frame
            # `first` (it starts at the first frame at or after the seek time), and name each frame by its index
            fps = native
            first = math.ceil(a * native - 1e-6)
            a = first / native
            seek, vf = max(0.0, (first - 0.5) / native), f"scale={w}:{h}:flags=area"
        else:
            seek = max(0.0, a - 0.5 / native)
            vf = f"fps={fps},scale={w}:{h}:flags=area"
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-t", f"{b - seek:.6f}", "-i", str(self.v.path),
               "-an", "-sn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix, "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        size = w * h * ch
        i = 0
        try:
            while True:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                arr = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
                yield a + i / fps, arr
                i += 1
        finally:
            p.stdout.close()
            p.kill()
            p.wait()

    def clip(self, a: float, b: float, fps: float | None = None, long_edge: int | None = 480, gray: bool = False):
        """frames() as arrays: (times, stack) with stack shaped (N, H, W[, 3])."""
        ts, fs = [], []
        for t, f in self.frames(a, b, fps, long_edge, gray):
            ts.append(t)
            fs.append(f)
        if not fs:
            return np.zeros(0), np.zeros((0, 1, 1, 3), np.uint8)
        return np.array(ts), np.stack(fs)

    def frame(self, t: float, long_edge: int | None = None) -> np.ndarray:
        """One RGB frame at t (source size by default)."""
        t = min(max(0.0, t), max(0.0, self.duration - 0.05))
        w, h = ffmpeg_dims(self.W, self.H, long_edge)
        # the frame showing at t is frame floor(t·fps); seek half a frame before it so rounding can't skip it
        idx = math.floor(t * self.fps + 1e-6)
        # the picture often stops a little before the sound does: step back to the last frame that exists
        for back in (0, 1, 2, 4, 8, 15, 30):
            seek = max(0.0, (idx - back - 0.5) / self.fps)
            p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-i", str(self.v.path), "-frames:v", "1",
                                "-an", "-sn", "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if len(p.stdout) >= w * h * 3:
                return np.frombuffer(p.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()
            if seek == 0.0:
                break
        raise vu.VuError(f"no frame at {vu.fmt(t)}")

    def frames_at(self, times: list[float], long_edge: int | None = 480) -> list[np.ndarray]:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=min(8, max(1, len(times)))) as ex:
            return list(ex.map(lambda t: self.frame(t, long_edge), times))

    # ── sound ──
    def audio(self, sr: int = 16000) -> np.ndarray:
        """The soundtrack as mono float32 samples in [-1, 1] at `sr` (cached). Empty when silent/no audio."""
        if not self.v.meta.get("audio"):
            return np.zeros(0, np.float32)
        f = self.out / f"audio-{sr}.npy"
        if f.exists():
            return np.load(f)
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.v.path), "-vn", "-ac", "1", "-ar", str(sr),
                            "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        x = np.frombuffer(p.stdout, np.float32).copy()
        np.save(f, x)
        return x

    # ── Apple Vision on arrays ──
    def vision(self, mode: str, images: list[np.ndarray]) -> list[dict]:
        """mode: "text" | "faces" | "pose" | "person". One result per image, in order ({} when unavailable).
        "person" results carry "mask" as a float array (H, W) in 0..1 at the image's size."""
        helper = vu.vision_helper()
        if not helper or not images:
            return [{} for _ in images]
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, im in enumerate(images):
                pth = Path(td) / f"i{i:05d}.png"
                Image.fromarray(im).save(pth)
                paths.append(str(pth))
            out: dict[str, dict] = {}
            for k in range(0, len(paths), 200):
                cmd = [str(helper)] + ([] if mode == "text" else [f"--{mode}"]) + paths[k:k + 200]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                for line in p.stdout.decode(errors="replace").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        out[rec["path"]] = rec
            res = []
            for pth, im in zip(paths, images):
                rec = out.get(pth, {})
                if mode == "person" and rec.get("mask"):
                    m = Image.open(rec["mask"]).convert("L").resize((im.shape[1], im.shape[0]))
                    rec = {**rec, "mask": np.asarray(m, np.float32) / 255.0}
                res.append(rec)
            return res

    # ── shared tracks (cached, so lenses don't run Vision twice) ──
    def faces_track(self, fps: float = 2.0) -> list[dict]:
        """[{"t", "faces": [{"box", "conf", "yaw", "roll", "pitch"}]}] over the whole video at `fps`,
        biggest face first. Cached in teardown/faces.json."""
        got = self.load("faces")
        if got and got.get("fps", 0) >= fps:
            return got["track"]
        times = [round(t, 3) for t in np.arange(0.25, self.duration, 1 / fps)]
        eprint(f"teardown: finding faces at {len(times)} moments …")
        frames = self.frames_at(times, long_edge=640)
        recs = self.vision("faces", frames)
        track = []
        for t, r in zip(times, recs):
            faces = sorted(r.get("faces", []), key=lambda f: -f["box"][2] * f["box"][3])
            track.append({"t": t, "faces": faces})
        self.save("faces", {"fps": fps, "track": track})
        return track

    # ── output ──
    def save(self, name: str, data) -> Path:
        f = self.out / f"{name}.json"
        f.write_text(json.dumps(json_safe(data), indent=1, ensure_ascii=False))
        return f

    def load(self, name: str):
        f = self.out / f"{name}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def img(self, name: str, arr: np.ndarray, q: int = 88) -> str:
        """Write an evidence image (RGB array) under teardown/img/; returns the path relative to teardown/."""
        from PIL import Image
        p = self.out / "img" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(arr)
        if p.suffix.lower() in (".jpg", ".jpeg"):
            im.convert("RGB").save(p, quality=q)
        else:
            im.save(p)
        return str(p.relative_to(self.out))

    def strip(self, name: str, times: list[float], long_edge: int = 360, label: bool = True) -> str:
        """A horizontal filmstrip of frames at `times` (labelled), for showing an animation's steps."""
        frames = self.frames_at(times, long_edge)
        if not frames:
            return ""
        h, w = frames[0].shape[:2]
        gap = 4
        out = np.full((h, len(frames) * (w + gap) - gap, 3), 18, np.uint8)
        for i, (t, f) in enumerate(zip(times, frames)):
            out[:, i * (w + gap): i * (w + gap) + w] = f[:h, :w]
            if label:
                _stamp(out, i * (w + gap) + 4, 4, vu.fmt(t, prec=2), max(1, w // 160))
        return self.img(name, out, q=85)


def _stamp(arr: np.ndarray, x: int, y: int, text: str, scale: int) -> None:
    """Draw `text` with vu's bitmap font onto an RGB array (white on a black box)."""
    cw = 6 * scale
    pad = scale * 2
    arr[y:y + 7 * scale + pad * 2, x:x + len(text) * cw + pad * 2] = 0
    for n, chx in enumerate(text):
        glyph = vu.FONT.get(chx, vu.FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    yy, xx = y + pad + gy * scale, x + pad + n * cw + gx * scale
                    arr[yy:yy + scale, xx:xx + scale] = 255


# ─── small numeric helpers the lenses share ──────────────────────────────────────────────────────

def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return np.asarray(x, float)
    w = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), w, mode="valid")


def peaks(x: np.ndarray, min_gap: int, thresh: float) -> list[int]:
    """Indices of local maxima above thresh, at least min_gap apart (greedy, biggest first)."""
    x = np.asarray(x, float)
    cand = [i for i in range(1, len(x) - 1) if x[i] >= thresh and x[i] >= x[i - 1] and x[i] >= x[i + 1]]
    cand.sort(key=lambda i: -x[i])
    taken: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_gap for j in taken):
            taken.append(i)
    return sorted(taken)


def near(t: float, times, tol: float) -> bool:
    return any(abs(t - u) <= tol for u in times)


def pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


# ─── self-check ──────────────────────────────────────────────────────────────────────────────────

def demo():
    """A synthetic 3 s clip: a white square slides in over grey, with a tone."""
    if not shutil.which("ffmpeg"):
        print("common demo skipped (no ffmpeg)")
        return
    with tempfile.TemporaryDirectory() as td:
        os.environ["VU_CACHE"] = str(Path(td) / "cache")
        vu.CACHE = Path(os.environ["VU_CACHE"])
        src = Path(td) / "s.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x568:r=30:d=3",
                        "-f", "lavfi", "-i", "sine=f=440:d=3", "-vf",
                        "drawbox=x='min(200,-80+t*400)':y=200:w=80:h=80:c=white:t=fill",
                        "-shortest", "-pix_fmt", "yuv420p", str(src)], check=True)
        ctx = Ctx(vu.Video(src))
        assert ctx.portrait and abs(ctx.duration - 3) < 0.1
        ts, st = ctx.clip(0, 1, fps=10, long_edge=None)
        assert len(ts) == 10 and st.shape[1:] == (568, 320, 3), st.shape
        f = ctx.frame(2.0)
        assert f[240, 240].mean() > 200 and f[50, 50].mean() < 200   # the square has arrived
        x = ctx.audio()
        assert 2.9 * 16000 < len(x) < 3.1 * 16000 and np.abs(x).max() > 0.1
        p = ctx.strip("demo.jpg", [0.2, 0.5, 1.0], long_edge=160)
        assert (ctx.out / p).exists()
        ctx.save("demo", {"x": np.float32(1.5), "a": np.arange(3)})
        assert ctx.load("demo") == {"x": 1.5, "a": [0, 1, 2]}
        assert hexcolor((255, 128, 0)) == "#FF8000" and hex_to_rgb("#FF8000") == (255, 128, 0)
        assert peaks(np.array([0, 1, 0, 0, 3, 0, 2, 0]), 2, 0.5) == [1, 4, 6]
        if vu.vision_helper():
            r = ctx.vision("text", [f])[0]
            assert "lines" in r
    print("common demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        print(__doc__)
