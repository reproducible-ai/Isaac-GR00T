"""Ordinary, isolated GR00T calibration points using a committed resolved recipe.

This program knows nothing about orchestration or lineage services. The caller
supplies immutable input files, and collects the raw timing and checkpoint outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_digest(recipe):
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_inputs(plan_path, config_path):
    plan = json.loads(Path(plan_path).read_bytes())
    data = Path(config_path).read_bytes()
    config = json.loads(data)
    if hashlib.sha256(data).hexdigest() != plan["recipe"]["configSha256"]:
        raise ValueError("resolved configuration differs from the pinned plan")
    recipe = plan["recipe"]
    training = config["training"]
    if (
        training["max_steps"] != recipe["fullSteps"]
        or training["global_batch_size"] != recipe["perDeviceBatchSize"]
        or training["gradient_accumulation_steps"] != recipe["gradientAccumulationSteps"]
        or training["num_gpus"] != 1
        or recipe["hardware"]["gpuCount"] != 1
        or config["data"]["seed"] != plan["protocol"]["seed"]
        or training["calibration_stop_steps"] is not None
        or training["calibration_timing_path"] is not None
        or training["resume_from_checkpoint"]
    ):
        raise ValueError("resolved runtime configuration disagrees with the calibration recipe")
    precision = "bf16" if training["bf16"] else ("fp16" if training["fp16"] else "fp32")
    modules = {
        name
        for name, field in (
            ("language_model", "tune_llm"),
            ("vision_model", "tune_visual"),
            ("projector", "tune_projector"),
            ("diffusion_model", "tune_diffusion_model"),
        )
        if config["model"][field]
    }
    warmup = training["warmup_steps"] or math.ceil(training["max_steps"] * training["warmup_ratio"])
    if (
        precision != recipe["precision"]
        or modules != set(recipe["trainableModules"])
        or training["lr_scheduler_type"] != recipe["scheduler"]["name"]
        or warmup != recipe["scheduler"]["warmupSteps"]
    ):
        raise ValueError("precision, trainable modules or scheduler disagree with the recipe")
    if training["save_steps"] != recipe["checkpoint"]["everySteps"]:
        raise ValueError("checkpoint cadence disagrees with plan")
    if training["eval_strategy"] != "no" or recipe["evaluation"]["everySteps"] is not None:
        raise ValueError("this recipe does not support periodic evaluation")
    if max(point["steps"] for point in plan["protocol"]["points"]) >= training["save_steps"]:
        raise ValueError("points may not cross periodic checkpoint boundaries")
    return plan, config, data


def run_child(command, *, output, timeout, env=None):
    """Stop the entire process group on timeout; retain output and propagate failure."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("x") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
            errors="replace",
            bufsize=1,
        )

        def copy_output():
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

        pump = threading.Thread(target=copy_output, daemon=True)
        pump.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
        finally:
            pump.join(timeout=5)
            if pump.is_alive():
                # A descendant holding the pipe must not outlive a point.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pump.join(timeout=5)
                if pump.is_alive():
                    raise RuntimeError("child log pipe did not close")
            process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return started, time.monotonic()


def run_point(args):
    # Capture process startup separately in the parent; these imports are timed.
    from gr00t.experiment.experiment import run
    from scripts.resolve_droid_calibration import build_config, resolve_config
    import torch

    plan, raw, config_bytes = load_inputs(args.plan, args.config)
    if encoded(resolve_config()) != config_bytes:
        raise ValueError("executable configuration differs from reviewed resolved configuration")
    point = next(point for point in plan["protocol"]["points"] if point["id"] == args.point)
    hardware = plan["recipe"]["hardware"]
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != hardware["accelerator"]:
        raise RuntimeError("runtime accelerator differs from the pinned plan")
    config = build_config()
    config.training.output_dir = str(args.output / point["id"])
    config.training.calibration_stop_steps = point["steps"]
    config.training.calibration_timing_path = str(args.output / f"{point['id']}.json")
    run(config)


def record_event(journal, event):
    """Persist diagnostics locally and in captured stdout before proceeding."""
    line = json.dumps(event, allow_nan=False)
    journal.write(line + "\n")
    journal.flush()
    os.fsync(journal.fileno())
    print("CALIBRATION_EVENT=" + line, flush=True)


def run_points(args):
    plan, _, config_bytes = load_inputs(args.plan, args.config)
    if any(p.is_file() and p.name != ".gitkeep" for p in args.output.rglob("*")):
        raise ValueError("refusing to reuse a calibration output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resolved-config.json").write_bytes(config_bytes)
    journal_path = args.output / "processes.jsonl"
    with journal_path.open("x") as journal:
        for point in plan["protocol"]["points"]:
            # Make the declared cache policy repeatable: warm source files, but give
            # each fresh training process its own initially empty compiler caches.
            inputs = json.loads(
                Path("artifacts/droid-calibration/input-manifest.json").read_bytes()
            )
            for entry in inputs["files"]:
                if sha256_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("calibration input bytes changed between independent points")
            command = [
                sys.executable,
                "-m",
                "scripts.calibration_droid",
                "point",
                "--plan",
                str(args.plan),
                "--config",
                str(args.config),
                "--output",
                str(args.output),
                "--point",
                point["id"],
            ]
            attempt = {
                "pointId": point["id"],
                "requestedSteps": point["steps"],
                "startedUnixSeconds": time.time(),
                "command": command,
            }
            record_event(journal, {"event": "point-start", **attempt})
            try:
                started, exited = run_child(
                    command,
                    output=args.output / f"{point['id']}.log",
                    timeout=plan["protocol"]["maxPointSeconds"],
                    env=dict(
                        os.environ,
                        PYTHONUNBUFFERED="1",
                        HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1",
                        TRITON_CACHE_DIR=str((args.output / f"{point['id']}-triton").resolve()),
                        CUDA_CACHE_PATH=str((args.output / f"{point['id']}-cuda").resolve()),
                    ),
                )
                result = json.loads((args.output / f"{point['id']}.json").read_bytes())
                if (
                    result["completedSteps"] != point["steps"]
                    or result["fullSteps"] != plan["recipe"]["fullSteps"]
                ):
                    raise ValueError("child result disagrees with requested point or full schedule")
                finished = result["trainingFinishedMonotonicSeconds"]
                if not started < finished < exited:
                    raise ValueError("child monotonic training interval is outside its process")
                record_event(
                    journal,
                    {
                        "event": "point-complete",
                        **attempt,
                        "startedMonotonicSeconds": started,
                        "processExitedMonotonicSeconds": exited,
                        "finishedUnixSeconds": time.time(),
                        "timing": result,
                    },
                )
            except BaseException as exc:
                record_event(
                    journal,
                    {
                        "event": "point-failed",
                        **attempt,
                        "finishedUnixSeconds": time.time(),
                        "failureType": type(exc).__name__,
                    },
                )
                raise
            print(
                f"Completed independent calibration point {point['id']}: {point['steps']} updates",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "point"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "point":
        if not args.point:
            parser.error("point requires --point")
        run_point(args)
    else:
        # Workflow timeout must unwind run_child and stop its GPU process group.
        def interrupted(signum, frame):
            raise InterruptedError(f"calibration interrupted by signal {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            run_points(args)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
