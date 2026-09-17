"""Exercise actual child-process failures and timeout cleanup without a GPU."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.calibration_droid import run_child


class CalibrationProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_nonzero_child_preserves_log_and_cannot_succeed(self):
        path = self.root / "failed.log"
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            run_child(
                [sys.executable, "-c", "print('partial training',flush=True);raise SystemExit(7)"],
                output=path,
                timeout=5,
            )
        self.assertEqual(failure.exception.returncode, 7)
        self.assertIn("partial training", path.read_text())

    def test_timeout_stops_the_child_and_retains_partial_output(self):
        path = self.root / "timeout.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            run_child(
                [
                    sys.executable,
                    "-c",
                    "import os,time; print(os.getpid(),flush=True);time.sleep(60)",
                ],
                output=path,
                timeout=0.5,
            )
        pid = int(path.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_success_interval_and_existing_log_are_not_overwritten(self):
        path = self.root / "success.log"
        start, end = run_child([sys.executable, "-c", "print('complete')"], output=path, timeout=5)
        self.assertLess(start, end)
        self.assertIn("complete", path.read_text())
        with self.assertRaises(FileExistsError):
            run_child([sys.executable, "-c", "raise SystemExit(1)"], output=path, timeout=5)

    def test_interrupt_unwinds_process_group(self):
        # Use a separate driver so sending SIGTERM cannot affect the test runner.
        child_log = self.root / "child.log"
        source = (
            "import signal\nfrom scripts.calibration_droid import run_child\n"
            "def stop(signum,frame): raise InterruptedError('stop')\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            f"run_child({[sys.executable, '-c', 'import os,time;print(os.getpid(),flush=True);time.sleep(60)']!r},"
            f"output={str(child_log)!r},timeout=60)\n"
        )
        driver = subprocess.Popen([sys.executable, "-c", source], stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while (
                not child_log.exists() or not child_log.read_text().strip()
            ) and time.monotonic() < deadline:
                time.sleep(0.02)
            child = int(child_log.read_text().strip())
            driver.send_signal(signal.SIGTERM)
            self.assertNotEqual(driver.wait(timeout=5), 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait()


if __name__ == "__main__":
    unittest.main()
