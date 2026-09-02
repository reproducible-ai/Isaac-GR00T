"""Verify the 100-step GR00T checkpoint and write a compact result record."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from droid_canary_contract import CHECKPOINT_PATH, INPUT_MANIFEST_PATH, RESULT_PATH, TRAINING_STEPS
from safetensors.torch import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> tuple[dict[str, object], set[str]]:
    """Validate the tensor table without materializing the full checkpoint."""
    with safe_open(path, framework="pt", device="cpu") as archive:
        tensor_names = list(archive.keys())
        if not tensor_names:
            raise RuntimeError(f"No tensors found in {path}")

        first_tensor = tensor_names[0]
        for tensor_name in tensor_names:
            archive.get_slice(tensor_name).get_shape()

        return (
            {
                "tensor_count": len(tensor_names),
                "first_tensor": {
                    "name": first_tensor,
                    "shape": list(archive.get_slice(first_tensor).get_shape()),
                },
            },
            set(tensor_names),
        )


def main() -> None:
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing input manifest: {INPUT_MANIFEST_PATH}")

    checkpoint = CHECKPOINT_PATH
    if not checkpoint.is_dir():
        raise RuntimeError(f"Missing {TRAINING_STEPS}-step checkpoint: {checkpoint}")
    model_files = sorted(checkpoint.glob("*.safetensors"))
    if not model_files:
        raise RuntimeError(f"No safetensors model weights found in {checkpoint}")
    if list(checkpoint.glob("pytorch_model*.bin")):
        raise RuntimeError("Legacy PyTorch checkpoint weights are not accepted")

    safetensors_index = None
    weight_map = None
    index_path = checkpoint / "model.safetensors.index.json"
    safetensors_files = model_files
    if not index_path.is_file():
        raise RuntimeError("Missing required safetensors shard index")
    index = json.loads(index_path.read_text())
    raw_weight_map = index.get("weight_map")
    if (
        not isinstance(raw_weight_map, dict)
        or not raw_weight_map
        or not all(
            isinstance(tensor_name, str) and isinstance(shard_name, str)
            for tensor_name, shard_name in raw_weight_map.items()
        )
    ):
        raise RuntimeError(f"Invalid safetensors weight map: {index_path}")
    weight_map = raw_weight_map
    shards = sorted(set(weight_map.values()))
    discovered = sorted(path.name for path in safetensors_files)
    if shards != discovered:
        raise RuntimeError(
            f"Safetensors index shard mismatch: expected {shards}; found {discovered}"
        )
    if len(safetensors_files) < 2:
        raise RuntimeError("Checkpoint must contain at least two safetensors shards")
    safetensors_index = {
        "path": str(index_path),
        "size": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
        "weight_count": len(weight_map),
        "shards": shards,
    }

    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise RuntimeError(f"Missing trainer state: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text())
    if trainer_state.get("global_step") != TRAINING_STEPS:
        raise RuntimeError(
            f"Expected global_step={TRAINING_STEPS}; found {trainer_state.get('global_step')}"
        )

    log_history = trainer_state.get("log_history", [])
    losses = [
        float(entry["loss"])
        for entry in log_history
        if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float))
    ]
    if not losses or not math.isfinite(losses[-1]):
        raise RuntimeError("Training did not record a finite loss")
    final_loss = losses[-1]
    model_records = []
    for path in model_files:
        record = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".safetensors":
            metadata, tensor_names = inspect_safetensors(path)
            record.update(metadata)
            if weight_map is not None:
                expected_tensors = {
                    tensor_name
                    for tensor_name, shard_name in weight_map.items()
                    if shard_name == path.name
                }
                if tensor_names != expected_tensors:
                    raise RuntimeError(
                        f"Safetensors tensor map mismatch for {path.name}: "
                        f"expected {sorted(expected_tensors)}; found {sorted(tensor_names)}"
                    )
        model_records.append(record)

    result = {
        "schema_version": 1,
        "status": "passed",
        "global_step": TRAINING_STEPS,
        "final_loss": final_loss,
        "checkpoint": str(checkpoint),
        "model_files": model_records,
        "safetensors_index": safetensors_index,
        "final_log": log_history[-1] if log_history else None,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"DROID canary passed at global_step={TRAINING_STEPS}; result={RESULT_PATH}")


if __name__ == "__main__":
    main()
