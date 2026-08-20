"""Verify the one-step GR00T checkpoint and write a compact result record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from droid_canary_contract import ARTIFACT_ROOT, INPUT_MANIFEST_PATH, RESULT_PATH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing input manifest: {INPUT_MANIFEST_PATH}")

    checkpoint = ARTIFACT_ROOT / "checkpoint-1"
    if not checkpoint.is_dir():
        raise RuntimeError(f"Missing one-step checkpoint: {checkpoint}")
    model_files = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("pytorch_model*.bin")
    )
    if not model_files:
        raise RuntimeError(f"No model weights found in {checkpoint}")

    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise RuntimeError(f"Missing trainer state: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text())
    if trainer_state.get("global_step") != 1:
        raise RuntimeError(f"Expected global_step=1; found {trainer_state.get('global_step')}")

    log_history = trainer_state.get("log_history", [])
    result = {
        "schema_version": 1,
        "status": "passed",
        "global_step": 1,
        "checkpoint": str(checkpoint),
        "model_files": [
            {
                "path": str(path.relative_to(ARTIFACT_ROOT)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_files
        ],
        "final_log": log_history[-1] if log_history else None,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"DROID canary passed at global_step=1; result={RESULT_PATH}")


if __name__ == "__main__":
    main()
