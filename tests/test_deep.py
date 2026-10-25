"""The teardown lenses' own self-checks, run with the Python that has numpy, OpenCV and Pillow (vu setup-deep).
Each module's demo builds synthetic media (no network, no real footage) and asserts what it measures.

    python3 -m unittest discover -s tests
"""
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "skills" / "video-understanding" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vu  # noqa: E402

DEEP_PY = vu.deep_python()
HAVE_FFMPEG = bool(shutil.which("ffmpeg"))
MODULES = ["ease", "common", "cross", "style", "report", "fonts", "edit", "sound", "delivery", "design", "typography", "motion"]


@unittest.skipUnless(DEEP_PY and HAVE_FFMPEG, "teardown needs `vu setup-deep` and ffmpeg")
class Lenses(unittest.TestCase):
    def test_demos(self):
        for name in MODULES:
            with self.subTest(module=name):
                p = subprocess.run([DEEP_PY, str(SCRIPTS / "deep" / f"{name}.py"), "demo"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
                out = p.stdout.decode(errors="replace")
                self.assertEqual(p.returncode, 0, f"{name}: {p.stderr.decode(errors='replace')[-1500:]}")
                self.assertIn(f"{name} demo ok", out)


class Wiring(unittest.TestCase):
    def test_cli_has_teardown(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            vu.main(["teardown", "--help"])
        self.assertIn("--only-report", out.getvalue())


if __name__ == "__main__":
    unittest.main()
