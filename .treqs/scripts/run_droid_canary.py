"""Run 100 GR00T optimizer steps against the pinned DROID sample."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from droid_canary_contract import (
    ARTIFACT_ROOT,
    BACKBONE_MODEL_ID,
    BACKBONE_MODEL_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHECKPOINT_PATH,
    DATASET_PATH,
    INPUT_MANIFEST_PATH,
    TRAINING_STEPS,
)


def cached_snapshot(repo_id: str, revision: str, token: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_files_only=True,
    )


def validate_placeholder_scaffold(path: Path) -> None:
    """Allow committed placeholders without allowing stale output reuse."""
    existing = [item for item in path.rglob("*") if item.is_file() and item.name != ".gitkeep"]
    if existing:
        raise RuntimeError(f"Refusing to reuse existing training output: {path}")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated VLM backbone")
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing fetch manifest: {INPUT_MANIFEST_PATH}")

    base_model_path = cached_snapshot(BASE_MODEL_ID, BASE_MODEL_REVISION, token)
    cached_snapshot(BACKBONE_MODEL_ID, BACKBONE_MODEL_REVISION, token)

    # Leave the committed .gitkeep files in place so every named step starts
    # from a clean Git tree. The package step explicitly rewrites them, making
    # them recorded outputs rather than orphaned files in the publication.
    validate_placeholder_scaffold(CHECKPOINT_PATH)

    triton_cache = ARTIFACT_ROOT / "triton-cache"
    triton_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "NUM_GPUS": "1",
            "MAX_STEPS": str(TRAINING_STEPS),
            "SAVE_STEPS": str(TRAINING_STEPS),
            "GLOBAL_BATCH_SIZE": "1",
            "DATALOADER_NUM_WORKERS": "0",
            "SHARD_SIZE": "8",
            "NUM_SHARDS_PER_EPOCH": "8",
            "EPISODE_SAMPLING_RATE": "1.0",
            "USE_WANDB": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "TRITON_CACHE_DIR": str(triton_cache.resolve()),
        }
    )
    subprocess.run(
        [
            "bash",
            "examples/finetune.sh",
            "--base-model-path",
            base_model_path,
            "--dataset-path",
            str(DATASET_PATH),
            "--embodiment-tag",
            "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "--output-dir",
            str(ARTIFACT_ROOT),
            "--save-only-model",
            "--",
            "--no-use-flash-attention",
            "--backbone-model-revision",
            BACKBONE_MODEL_REVISION,
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
