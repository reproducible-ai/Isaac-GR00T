"""CPU contract tests for the ordinary trainer calibration callback."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from gr00t.experiment.calibration import (
    PointTiming,
    make_calibration_callback,
    validate_calibration,
)


class CalibrationTimingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "point.json"
        self.training = SimpleNamespace(
            calibration_stop_steps=4,
            calibration_warmup_updates=2,
            calibration_timing_path=str(self.path),
            max_steps=10000,
            num_gpus=1,
            resume_from_checkpoint=False,
            eval_strategy="no",
            save_steps=1000,
            enable_profiling=False,
        )

    def test_real_callback_keeps_full_schedule_and_requests_only_bounded_final_save(self):
        ticks = iter([0, 8, 12, 14, 16, 25])
        synchronizations = []
        callback = make_calibration_callback(
            self.training,
            callback_base=object,
            synchronize=lambda: synchronizations.append(True),
            clock=lambda: next(ticks),
        )
        args = SimpleNamespace(max_steps=10000, get_warmup_steps=lambda count: int(count * 0.05))
        state = SimpleNamespace(global_step=0)
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        callback.on_train_begin(args, state, control)
        for step in range(1, 5):
            state.global_step = step
            callback.on_step_end(args, state, control)
            self.assertEqual(control.should_training_stop, step == 4)
            self.assertEqual(control.should_save, step == 4)
        callback.on_save(args, state, control)
        callback.on_train_end(args, state, control)
        result = json.loads(self.path.read_text())
        self.assertEqual(args.max_steps, 10000)
        self.assertEqual(result["fullSteps"], 10000)
        self.assertEqual(result["completedSteps"], 4)
        self.assertEqual(result["steadyUpdates"], 2)
        self.assertEqual(result["steadySeconds"], 4)
        self.assertEqual(result["checkpointSeconds"], 9)
        self.assertEqual(result["trainingFinishedMonotonicSeconds"], 16)
        self.assertEqual(len(synchronizations), 7)
        events = [
            json.loads(line) for line in Path(str(self.path) + ".jsonl").read_text().splitlines()
        ]
        self.assertEqual(events[0]["schedulerWarmupSteps"], 500)
        self.assertEqual(events[-1]["event"], "training-complete")

    def test_incomplete_training_retains_raw_events_without_success_summary(self):
        timing = PointTiming(
            self.path, stop=4, warmup=2, full_steps=10000, clock=iter([0, 1]).__next__
        )
        self.addCleanup(timing.stream.close)
        timing.begin(0, max_steps=10000, warmup_steps=500)
        timing.step(1)
        with self.assertRaisesRegex(ValueError, "did not finish"):
            timing.finish(1)
        self.assertFalse(self.path.exists())
        self.assertIn('"completedSteps": 1', timing.journal.read_text())

    def test_complete_updates_without_checkpoint_cannot_be_claimed_complete(self):
        timing = PointTiming(
            self.path, stop=2, warmup=1, full_steps=10000, clock=iter([0, 1, 2]).__next__
        )
        self.addCleanup(timing.stream.close)
        timing.begin(0, max_steps=10000, warmup_steps=500)
        timing.step(1)
        timing.step(2)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            timing.finish(2)
        self.assertFalse(self.path.exists())

    def test_schedule_reset_and_discontinuous_updates_fail(self):
        timing = PointTiming(self.path, stop=4, warmup=2, full_steps=10000, clock=lambda: 0)
        self.addCleanup(timing.stream.close)
        for step, max_steps in [(1, 10000), (0, 4)]:
            with self.assertRaises(ValueError):
                timing.begin(step, max_steps=max_steps, warmup_steps=500)
        timing.begin(0, max_steps=10000, warmup_steps=500)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            timing.step(2)

    def test_unsupported_recipes_fail_before_training(self):
        for field, value in [
            ("calibration_stop_steps", 2),
            ("calibration_stop_steps", True),
            ("calibration_stop_steps", 10000),
            ("calibration_timing_path", None),
            ("calibration_warmup_updates", 0),
            ("num_gpus", 2),
            ("resume_from_checkpoint", True),
            ("eval_strategy", "steps"),
            ("save_steps", 4),
            ("enable_profiling", True),
        ]:
            with self.subTest(field=field, value=value):
                training = SimpleNamespace(**vars(self.training))
                setattr(training, field, value)
                with self.assertRaises(ValueError):
                    validate_calibration(training)

    def test_repeated_point_cannot_overwrite_prior_receipt_or_failure_journal(self):
        self.path.write_text("retained")
        with self.assertRaisesRegex(ValueError, "reuse"):
            validate_calibration(self.training)
        self.path.unlink()
        Path(str(self.path) + ".jsonl").write_text("partial")
        with self.assertRaisesRegex(ValueError, "reuse"):
            validate_calibration(self.training)

    def test_ordinary_training_has_no_calibration_dependency(self):
        validate_calibration(
            SimpleNamespace(calibration_stop_steps=None, calibration_timing_path=None)
        )


if __name__ == "__main__":
    unittest.main()
