"""Validate the runtime and fetch every immutable DROID canary input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from droid_canary_contract import (
    ARTIFACT_ROOT,
    BACKBONE_MODEL_ID,
    BACKBONE_MODEL_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_ID,
    DATASET_PATH,
    DATASET_REVISION,
    EPISODE_COUNT,
    INPUT_MANIFEST_PATH,
    MIN_GPU_MEMORY_MIB,
)


DATASET_SCAFFOLD_RELATIVE_PATHS = (
    Path("meta/.gitkeep"),
    Path("data/chunk-000/.gitkeep"),
    Path("videos/chunk-000/observation.images.exterior_1_left/.gitkeep"),
    Path("videos/chunk-000/observation.images.wrist_left/.gitkeep"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime() -> dict[str, object]:
    import torch

    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    match = re.search(r"ffmpeg version\s+(\d+)", ffmpeg)
    if match is None or int(match.group(1)) not in range(4, 8):
        raise RuntimeError(f"FFmpeg 4-7 is required; found: {ffmpeg}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    memory_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    if memory_mib < MIN_GPU_MEMORY_MIB:
        raise RuntimeError(
            f"At least {MIN_GPU_MEMORY_MIB} MiB GPU memory is required; found {memory_mib} MiB"
        )

    free_disk_gib = shutil.disk_usage("/tmp").free // (1024**3)
    if free_disk_gib < 20:
        raise RuntimeError(
            f"At least 20 GiB free disk is required after setup; found {free_disk_gib}"
        )

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": memory_mib,
        "ffmpeg": ffmpeg,
        "free_disk_gib_before_fetch": free_disk_gib,
    }


def download_models(token: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Fetch a small repository file from both gated inputs before transferring
    # multi-gigabyte snapshots. Model metadata can remain public even when the
    # token cannot read gated files, so model_info() is not a sufficient check.
    hf_hub_download(
        repo_id=BASE_MODEL_ID,
        filename=".gitattributes",
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    hf_hub_download(
        repo_id=BACKBONE_MODEL_ID,
        filename=".gitattributes",
        revision=BACKBONE_MODEL_REVISION,
        token=token,
    )
    base_path = snapshot_download(
        repo_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    backbone_path = snapshot_download(
        repo_id=BACKBONE_MODEL_ID,
        revision=BACKBONE_MODEL_REVISION,
        token=token,
    )
    return {
        "base_model_snapshot": Path(base_path).name,
        "backbone_model_snapshot": Path(backbone_path).name,
    }


def download_dataset() -> None:
    existing = [
        path for path in DATASET_PATH.rglob("*") if path.is_file() and path.name != ".gitkeep"
    ]
    if existing:
        raise RuntimeError(f"Refusing to reuse existing canary dataset: {DATASET_PATH}")
    # The repository tracks this otherwise-empty directory with .gitkeep, but
    # download_droid_sample.py deliberately refuses any pre-existing output
    # directory. Remove the validated placeholder scaffold before invoking it.
    if DATASET_PATH.exists():
        shutil.rmtree(DATASET_PATH)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.download_droid_sample",
            "--revision",
            DATASET_REVISION,
            "--num-episodes",
            str(EPISODE_COUNT),
            "--cache-dir",
            f"/tmp/droid-download-{DATASET_REVISION}",
            "--output-dir",
            str(DATASET_PATH),
        ],
        check=True,
    )
    # Restore the committed placeholders after conversion so the next named
    # Roar step still starts from a clean Git tree. The generated data remains
    # ignored, while each output parent is known to exist on a fresh checkout.
    for relative_path in DATASET_SCAFFOLD_RELATIVE_PATHS:
        scaffold_path = DATASET_PATH / relative_path
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text("\n")


def dataset_manifest() -> dict[str, object]:
    files = []
    for path in sorted(DATASET_PATH.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            files.append(
                {
                    "path": str(path.relative_to(DATASET_PATH)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise RuntimeError("DROID conversion produced no files")
    return {
        "path": str(DATASET_PATH),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated VLM backbone")

    runtime = validate_runtime()
    snapshots = download_models(token)
    download_dataset()
    manifest = {
        "schema_version": 1,
        "inputs": {
            "base_model": {
                "repo_id": BASE_MODEL_ID,
                "revision": BASE_MODEL_REVISION,
                "url": f"https://huggingface.co/{BASE_MODEL_ID}/tree/{BASE_MODEL_REVISION}",
            },
            "backbone_model": {
                "repo_id": BACKBONE_MODEL_ID,
                "revision": BACKBONE_MODEL_REVISION,
                "url": (
                    f"https://huggingface.co/{BACKBONE_MODEL_ID}/tree/{BACKBONE_MODEL_REVISION}"
                ),
            },
            "dataset": {
                "repo_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "url": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}",
            },
        },
        "resolved_snapshots": snapshots,
        "runtime": runtime,
        "dataset": dataset_manifest(),
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Prepared {EPISODE_COUNT} pinned DROID episodes and both pinned model snapshots; "
        f"manifest={INPUT_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
