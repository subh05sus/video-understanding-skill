"""End-to-end checks on a synthetic video: three shots (moving, still, moving) with a tone, so cuts,
motion, sound, grids, budget and memory can be tested without any real footage.

    python3 -m unittest discover -s tests
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skills" / "video-understanding" / "scripts"))

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def run_vu(*argv) -> tuple[int, str]:
    import vu
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = vu.main(list(argv))
    return code, out.getvalue()


class TimeParsing(unittest.TestCase):
    def test_forms(self):
        import vu
        self.assertEqual(vu.parse_time("83.5"), 83.5)
        self.assertEqual(vu.parse_time("1:23.5"), 83.5)
        self.assertEqual(vu.parse_time("1:02:03"), 3723)
        self.assertEqual(vu.parse_time("end", 60), 60)
        self.assertEqual(vu.parse_time("-5", 60), 55)
        self.assertEqual(vu.fmt(83.46), "1:23.5")
        self.assertEqual(vu.fmt(83.456, prec=2), "1:23.46")
        self.assertEqual(vu.fmt(3723, long=True), "1:02:03.0")

    def test_hallucinations(self):
        import vu
        self.assertTrue(vu.hallucination("ლლლლლლლლ"))
        self.assertTrue(vu.hallucination("Untertitel im Auftrag des ZDF, 2021"))
        self.assertFalse(vu.hallucination("Then we look at pricing."))


class Pictures(unittest.TestCase):
    def test_label_and_paste(self):
        import vu
        img = vu.Img(40, 20, fill=(200, 200, 200))
        img.label("1:0", 1)
        self.assertEqual(bytes(img.data[0:3]), b"\x00\x00\x00")       # label box
        grid = vu.Img(100, 30)
        grid.paste(img, 10, 5)
        o = (5 * 100 + 10) * 3
        self.assertEqual(bytes(grid.data[o:o + 3]), b"\x00\x00\x00")
        back = vu.Img.from_ppm(grid.to_ppm())
        self.assertEqual((back.w, back.h, back.data), (grid.w, grid.h, grid.data))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class OnASyntheticVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        os.environ["VU_CACHE"] = str(cls.tmp / "cache")
        import vu
        vu.CACHE = Path(os.environ["VU_CACHE"])
        cls.video = cls.tmp / "three-shots.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "color=c=0x2050c0:size=640x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(cls.video)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_overview_finds_the_cuts_and_the_still_shot(self):
        code, out = run_vu("overview", str(self.video), "--no-transcript", "--depth", "quick")
        self.assertEqual(code, 0, out)
        self.assertIn("2 cuts", out)
        self.assertIn("STORYBOARD", out)
        import vu
        v = vu.Video(self.video)
        cuts = v.visual()["cuts"]
        self.assertEqual(len(cuts), 2)
        self.assertAlmostEqual(cuts[0], 4.0, delta=0.1)
        self.assertAlmostEqual(cuts[1], 8.0, delta=0.1)
        self.assertTrue(any(a <= 5.0 and b >= 7.5 for a, b in vu.still_spans(v, v.visual())))

    def test_frames_grid_budget_and_log(self):
        run_vu("overview", str(self.video), "--no-transcript", "--depth", "quick", "--no-storyboard")
        code, out = run_vu("frames", str(self.video), "--from", "0", "--to", "2", "--fps", "4", "--why", "test")
        self.assertEqual(code, 0, out)
        grid = Path(out.splitlines()[0])
        self.assertTrue(grid.exists() and grid.stat().st_size > 1000)
        self.assertIn("0:00.00 | 0:00.25", out)
        code, out = run_vu("frames", str(self.video), "--from", "0", "--to", "10", "--fps", "4")
        self.assertEqual(code, 3)  # 41 frames on top of 9 is over the quick budget of 40
        import vu
        log = vu.Video(self.video).read_log()
        self.assertEqual(log[-1]["why"], "test")

    def test_zoom_note_search_report_clip(self):
        run_vu("overview", str(self.video), "--no-transcript", "--no-storyboard", "-q", "what is in it")
        code, out = run_vu("zoom", str(self.video), "--at", "1", "--region", "top-right")
        self.assertEqual(code, 0, out)
        self.assertTrue(Path(out.splitlines()[0]).exists())
        run_vu("note", str(self.video), "the blue still shot is the middle", "--at", "6")
        code, out = run_vu("search", str(self.video), "blue still", "--no-ocr")
        self.assertIn("noted", out)
        code, out = run_vu("report", str(self.video), "--out", str(self.tmp / "r.html"))
        self.assertEqual(code, 0, out)
        page = (self.tmp / "r.html").read_text()
        self.assertIn("Watch path", page)
        self.assertIn("data:image/jpeg;base64", page)
        code, out = run_vu("clip", str(self.video), "--from", "3", "--to", "5", "--out", str(self.tmp / "c.mp4"))
        self.assertEqual(code, 0, out)
        self.assertTrue((self.tmp / "c.mp4").stat().st_size > 1000)

    def test_audio_and_motion(self):
        code, out = run_vu("audio", str(self.video))
        self.assertEqual(code, 0, out)
        self.assertIn("LOUDNESS", out)
        code, out = run_vu("motion", str(self.video))
        self.assertIn("cuts (2)", out)


if __name__ == "__main__":
    unittest.main()
