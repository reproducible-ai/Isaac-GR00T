"""Run one GR00T optimizer step against the pinned DROID sample."""

from __future__ import annotations

import os
import subprocess

from droid_canary_contract import (
    ARTIFACT_ROOT,
    BACKBONE_MODEL_ID,
    BACKBONE_MODEL_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_PATH,
    INPUT_MANIFEST_PATH,
)


def cached_snapshot(repo_id: str, revision: str, token: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_files_only=True,
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated VLM backbone")
    if not INPUT_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing fetch manifest: {INPUT_MANIFEST_PATH}")

    base_model_path = cached_snapshot(BASE_MODEL_ID, BASE_MODEL_REVISION, token)
    cached_snapshot(BACKBONE_MODEL_ID, BACKBONE_MODEL_REVISION, token)

    env = os.environ.copy()
    env.update(
        {
            "NUM_GPUS": "1",
            "MAX_STEPS": "1",
            "SAVE_STEPS": "1",
            "GLOBAL_BATCH_SIZE": "1",
            "DATALOADER_NUM_WORKERS": "0",
            "SHARD_SIZE": "8",
            "NUM_SHARDS_PER_EPOCH": "8",
            "EPISODE_SAMPLING_RATE": "1.0",
            "USE_WANDB": "0",
            "CUDA_VISIBLE_DEVICES": "0",
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
