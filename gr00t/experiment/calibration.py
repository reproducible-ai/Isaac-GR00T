"""Independent early-stop timing without changing the training schedule.

This first adapter supports single-GPU points shorter than the first periodic save
and recipes without evaluation. It refuses unsupported cases before training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time


def validate_calibration(training) -> None:
    stop = training.calibration_stop_steps
    path = training.calibration_timing_path
    if stop is None and path is None:
        return
    if (
        isinstance(stop, bool)
        or not isinstance(stop, int)
        or stop <= training.calibration_warmup_updates
        or stop >= training.max_steps
        or not path
    ):
        raise ValueError("calibration requires warmup < stop < full max_steps and a timing path")
    if training.calibration_warmup_updates < 1:
        raise ValueError("calibration needs at least one excluded warmup update")
    if training.num_gpus != 1:
        raise ValueError("this calibration adapter supports one GPU only")
    if training.resume_from_checkpoint:
        raise ValueError("independent calibration points cannot resume checkpoints")
    if training.eval_strategy != "no" or stop >= training.save_steps:
        raise ValueError("calibration points must end before periodic save/evaluation work")
    if training.enable_profiling:
        raise ValueError("profiling changes calibration timings")
    if Path(path).exists() or Path(str(path) + ".jsonl").exists():
        raise ValueError("refusing to reuse calibration timing evidence")


class PointTiming:
    """Durable raw observations; injectable clocks allow deterministic CPU tests."""

    def __init__(self, path, *, stop, warmup, full_steps, clock=time.monotonic):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.path.with_name(self.path.name + ".jsonl")
        self.stream = self.journal.open("x")
        self.clock = clock
        self.stop = stop
        self.warmup = warmup
        self.full_steps = full_steps
        self.ends = []
        self.checkpoint_seconds = None
        self.started = None
        self.finished = None

    def record(self, event, **fields):
        self.stream.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
        self.stream.flush()

    def begin(self, global_step, *, max_steps, warmup_steps):
        if global_step != 0 or max_steps != self.full_steps:
            raise ValueError("calibration must start at zero with the full scheduler unchanged")
        self.started = self.clock()
        self.record(
            "training-start",
            monotonicSeconds=self.started,
            fullSteps=max_steps,
            schedulerWarmupSteps=warmup_steps,
        )

    def step(self, global_step):
        if self.started is None or global_step != len(self.ends) + 1 or global_step > self.stop:
            raise ValueError("calibration optimizer updates must be contiguous and bounded")
        now = self.clock()
        previous = self.ends[-1] if self.ends else self.started
        if now <= previous:
            raise ValueError("calibration clock did not advance")
        self.ends.append(now)
        self.record("optimizer-update", completedSteps=global_step, monotonicSeconds=now)
        if global_step == self.stop:
            self.finished = now
            return True
        return False

    def saved(self, global_step):
        if global_step != self.stop or self.finished is None:
            raise ValueError("unexpected checkpoint during calibration timing interval")
        self.checkpoint_seconds = self.clock() - self.finished
        if self.checkpoint_seconds < 0:
            raise ValueError("checkpoint clock moved backwards")
        self.record("checkpoint-saved", completedSteps=global_step, seconds=self.checkpoint_seconds)

    def finish(self, global_step):
        if global_step != self.stop or self.finished is None or self.checkpoint_seconds is None:
            raise ValueError("calibration did not finish its requested updates and checkpoint")
        document = {
            "schema": "gr00t.calibration-point/v1",
            "fullSteps": self.full_steps,
            "completedSteps": global_step,
            "trainingStartedMonotonicSeconds": self.started,
            "trainingFinishedMonotonicSeconds": self.finished,
            "warmupUpdates": self.warmup,
            "steadyUpdates": self.stop - self.warmup,
            "steadySeconds": self.ends[-1] - self.ends[self.warmup - 1],
            "checkpointSeconds": self.checkpoint_seconds,
        }
        self.record("training-complete", **document)
        os.fsync(self.stream.fileno())
        self.stream.close()
        with self.path.open("x") as stream:
            json.dump(document, stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return document


def make_calibration_callback(
    training, *, callback_base=None, synchronize=None, clock=time.monotonic
):
    """Keep framework imports at the GPU boundary; ordinary recipes need no extra dependency."""
    validate_calibration(training)
    if callback_base is None:
        from transformers import TrainerCallback

        callback_base = TrainerCallback
    if synchronize is None:
        import torch

        synchronize = torch.cuda.synchronize

    timing = PointTiming(
        training.calibration_timing_path,
        stop=training.calibration_stop_steps,
        warmup=training.calibration_warmup_updates,
        full_steps=training.max_steps,
        clock=clock,
    )

    class CalibrationCallback(callback_base):
        def on_train_begin(self, args, state, control, **kwargs):
            synchronize()
            timing.begin(
                state.global_step,
                max_steps=args.max_steps,
                warmup_steps=args.get_warmup_steps(args.max_steps),
            )

        def on_step_end(self, args, state, control, **kwargs):
            synchronize()
            if timing.step(state.global_step):
                control.should_training_stop = True
                # The calibration checkpoint is outside the measured training interval.
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            synchronize()
            timing.saved(state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            synchronize()
            timing.finish(state.global_step)

    return CalibrationCallback()
