"""Package real calibration measurements and the last load-verified checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from scripts.calibration_droid import encoded, load_inputs, recipe_digest, sha256_file


ROOT = Path("artifacts/droid-calibration")
RELEASE = ROOT / "release"
POINTS = ROOT / "points"


def measured_points(plan, events, *, initial_state_sha, final_sha):
    completed = [event for event in events if event["event"] == "point-complete"]
    expected = plan["protocol"]["points"]
    if any(event["event"] == "point-failed" for event in events):
        raise ValueError("failed points cannot be packaged as a successful calibration")
    if [event["pointId"] for event in completed] != [point["id"] for point in expected]:
        raise ValueError("missing, duplicate or out-of-order calibration point completion")
    result = []
    for point, event in zip(expected, completed, strict=True):
        timing = event["timing"]
        if (
            timing["completedSteps"] != point["steps"]
            or timing["fullSteps"] != plan["recipe"]["fullSteps"]
        ):
            raise ValueError("point used another update count or full schedule")
        start, finish = event["startedMonotonicSeconds"], timing["trainingFinishedMonotonicSeconds"]
        if (
            not math.isfinite(start)
            or not math.isfinite(finish)
            or not start < finish < event["processExitedMonotonicSeconds"]
        ):
            raise ValueError("invalid synchronized point interval")
        result.append(
            {
                "id": point["id"],
                "requestedSteps": point["steps"],
                "completedSteps": timing["completedSteps"],
                "outcome": "completed",
                "exitCode": 0,
                "recipeSha256": recipe_digest(plan["recipe"]),
                "hardware": plan["recipe"]["hardware"],
                "seed": plan["protocol"]["seed"],
                "cachePolicy": plan["protocol"]["cachePolicy"],
                "initialStateSha256": initial_state_sha,
                "startedMonotonicSeconds": start,
                "finishedMonotonicSeconds": finish,
                "steadyUpdates": timing["steadyUpdates"],
                "steadySeconds": timing["steadySeconds"],
                "checkpointSha256": final_sha
                if point["id"] == plan["protocol"]["finalPointId"]
                else None,
            }
        )
    return result


def main():
    from verify_droid_canary import verify_checkpoint

    started = time.time()
    plan_path = Path(".treqs/calibration/plan.json")
    config_path = Path(".treqs/calibration/resolved-config.json")
    plan, config, config_bytes = load_inputs(plan_path, config_path)
    if RELEASE.exists():
        raise RuntimeError("refusing to replace a packaged calibration")
    final = plan["protocol"]["points"][-1]
    checkpoint = POINTS / final["id"] / f"checkpoint-{final['steps']}"
    evaluation_path = ROOT / "evaluation.json"
    verify_checkpoint(checkpoint, final["steps"], ROOT / "input-manifest.json", evaluation_path)
    evaluation = json.loads(evaluation_path.read_bytes())
    # Preserve the final model and complete optimizer/scheduler/RNG checkpoint.
    shutil.copytree(checkpoint, RELEASE)
    index = RELEASE / "model.safetensors.index.json"
    final_sha = sha256_file(index)
    manifest = json.loads((ROOT / "input-manifest.json").read_bytes())
    identity = {
        "baseFiles": [
            {key: entry[key] for key in ("input", "sha256", "sizeBytes")}
            for entry in manifest["files"]
            if entry["input"] in {"base", "backbone"}
        ],
        "seed": plan["protocol"]["seed"],
        "optimizer": "fresh AdamW; no checkpoint resume",
        "configurationSha256": sha256_file(config_path),
    }
    initial_sha = hashlib.sha256(encoded(identity)).hexdigest()
    events = [json.loads(line) for line in (POINTS / "processes.jsonl").read_text().splitlines()]
    points = measured_points(plan, events, initial_state_sha=initial_sha, final_sha=final_sha)
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common = {"planSha256": sha256_file(plan_path), "candidateCommit": candidate}
    raw = {
        "schema": "reproai.calibration-timings/v1",
        **common,
        "points": points,
        "processEvents": events,
        "initialStateBasis": identity,
        "trainerEvents": {},
    }
    for point in plan["protocol"]["points"]:
        raw["trainerEvents"][point["id"]] = [
            json.loads(line)
            for line in (POINTS / f"{point['id']}.json.jsonl").read_text().splitlines()
        ]
    (RELEASE / "timings.json").write_bytes(encoded(raw))
    # Late upload and allocation shutdown cannot truthfully be known in this payload.
    # Preserve clock anchors for later host reconciliation instead of inventing zeros.
    components = {
        "coldSetupSeconds": None,
        "finalizationSeconds": None,
        "checkpointSeconds": sum(
            event["timing"]["checkpointSeconds"]
            for event in events
            if event["event"] == "point-complete"
        )
        / len(points),
        "evaluationSeconds": None,
        "otherCostUsd": None,
    }
    overhead = {
        "schema": "reproai.calibration-overhead/v1",
        **common,
        "components": components,
        "inputPreparation": {
            key: manifest[key] for key in ("startedUnixSeconds", "finishedUnixSeconds")
        },
        "packageStartedUnixSeconds": started,
        "setupClock": json.loads((ROOT / "setup-clock.json").read_bytes()),
    }
    (RELEASE / "overhead.json").write_bytes(encoded(overhead))
    final_checkpoint = {
        "pointId": final["id"],
        "sha256": final_sha,
        "optimizerSteps": final["steps"],
        "loadVerified": True,
    }
    measurements = {
        "schema": "reproai.calibration-measurements/v1",
        "planSha256": common["planSha256"],
        "rawLogSha256": sha256_file(RELEASE / "timings.json"),
        "coverage": {"fullRecipe": True, "representativeData": True, "fixedWork": False},
        "overhead": {**components, "evidenceSha256": sha256_file(RELEASE / "overhead.json")},
        "points": points,
        "finalCheckpoint": final_checkpoint,
    }
    (RELEASE / "calibration.json").write_bytes(encoded(measurements))
    (RELEASE / "resolved-config.json").write_bytes(config_bytes)
    (RELEASE / "trainer-state.json").write_bytes(
        encoded(
            {
                "schema": "reproai.calibration-trainer-state/v1",
                **common,
                "finalCheckpoint": final_checkpoint,
                "initialStateSha256": initial_sha,
                "trainerStateSha256": sha256_file(RELEASE / "trainer_state.json"),
                "initialStateBasis": "Pinned model input bytes and reset configuration; source and process journal establish fresh loading and optimizer construction.",
            }
        )
    )
    shutil.copy2(ROOT / "input-manifest.json", RELEASE / "input-manifest.json")
    shutil.copy2(evaluation_path, RELEASE / "evaluation.json")
    shutil.copy2(plan_path, RELEASE / "calibration-plan.json")
    # Preserve upstream license and attribution for the private derived checkpoint.
    shutil.copy2(ROOT / "inputs/base/LICENSE", RELEASE / "LICENSE")
    shutil.copy2(
        ".treqs/assets/NVIDIA_OPEN_MODEL_LICENSE.md", RELEASE / "NVIDIA_OPEN_MODEL_LICENSE.md"
    )
    (RELEASE / "NOTICE").write_text(
        "Derived from NVIDIA Isaac GR00T N1.7. Built on NVIDIA Cosmos.\n"
    )
    (RELEASE / "README.md").write_text(
        "# GR00T N1.7 DROID calibration checkpoint\n\n"
        f"Private {final['steps']}-update calibration, with the full 10,000-update scheduler preserved. "
        "This is not a full run, certification or policy-quality evaluation.\n"
        f"Source: {candidate}\n"
    )
    files = [
        {
            "path": str(path.relative_to(RELEASE)),
            "sha256": sha256_file(path),
            "sizeBytes": path.stat().st_size,
        }
        for path in sorted(RELEASE.rglob("*"))
        if path.is_file()
    ]
    artifact_manifest = {
        "schema": "reproai.artifact-manifest/v1",
        "format": "gr00t-n1.7-safetensors-checkpoint",
        "loadVerified": True,
        "files": files,
    }
    (RELEASE / "artifact-manifest.json").write_bytes(encoded(artifact_manifest))
    artifact = {
        **artifact_manifest,
        "schema": "reproai.artifact/v1",
        "path": str(index),
        "sha256": final_sha,
        "sizeBytes": index.stat().st_size,
        "files": [{**entry, "path": str(RELEASE / entry["path"])} for entry in files],
    }
    result = {
        "schema": "reproai.result/v1",
        "checkpoint": str(index),
        "artifactSha256": final_sha,
        "artifactSizeBytes": index.stat().st_size,
        "loadVerified": True,
        "optimizerSteps": final["steps"],
        "taskMetric": {"metric": "finiteLoss", "minimum": 1},
        "finiteLoss": 1,
        "finalLoss": evaluation["final_loss"],
    }
    (RELEASE / "result.json").write_bytes(encoded(result))
    print("E2E_ARTIFACT=" + json.dumps(artifact, sort_keys=True, allow_nan=False), flush=True)
    print("E2E_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
