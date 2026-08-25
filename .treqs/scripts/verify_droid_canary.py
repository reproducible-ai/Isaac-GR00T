"""Verify the one-step GR00T checkpoint and write a compact result record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from droid_canary_contract import CHECKPOINT_PATH, INPUT_MANIFEST_PATH, RESULT_PATH
from safetensors.torch import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> dict[str, object]:
    """Validate the tensor table without materializing the full checkpoint."""
    with safe_open(path, framework="pt", device="cpu") as archive:
        tensor_names = list(archive.keys())
        if not tensor_names:
            raise RuntimeError(f"No tensors found in {path}")

        first_tensor = tensor_names[0]
        for tensor_name in tensor_names:
            archive.get_slice(tensor_name).get_shape()

        return {
            "tensor_count": len(tensor_names),
            "first_tensor": {
                "name": first_tensor,
                "shape": list(archive.get_slice(first_tensor).get_shape()),
            },
        }


def main() -> None:
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing input manifest: {INPUT_MANIFEST_PATH}")

    checkpoint = CHECKPOINT_PATH
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
    model_records = []
    for path in model_files:
        record = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".safetensors":
            record.update(inspect_safetensors(path))
        model_records.append(record)

    result = {
        "schema_version": 1,
        "status": "passed",
        "global_step": 1,
        "checkpoint": str(checkpoint),
        "model_files": model_records,
        "final_log": log_history[-1] if log_history else None,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"DROID canary passed at global_step=1; result={RESULT_PATH}")


if __name__ == "__main__":
    main()
