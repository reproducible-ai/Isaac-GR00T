"""Package the verified canary with reproducibility and license documents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from droid_canary_contract import (
    BACKBONE_MODEL_ID,
    BACKBONE_MODEL_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    CHECKPOINT_PATH,
    DATASET_ID,
    DATASET_REVISION,
    PUBLICATION_REPO_ID,
    PUBLICATION_VERSION,
    RESULT_PATH,
)


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / ".treqs" / "assets"
UPSTREAM_NOTICE_FILES = (
    "README.md",
    "EXPLAINABILITY.md",
    "PRIVACY.md",
    "SAFETY_and_SECURITY.md",
)


def cached_snapshot(repo_id: str, revision: str, token: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_files_only=True,
        )
    )


def source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def render_model_card(commit: str) -> str:
    card = (ASSET_ROOT / "droid-canary-model-card.md").read_text()
    replacements = {
        "{{SOURCE_COMMIT}}": commit,
        "{{BASE_MODEL_REVISION}}": BASE_MODEL_REVISION,
        "{{BACKBONE_MODEL_REVISION}}": BACKBONE_MODEL_REVISION,
        "{{DATASET_REVISION}}": DATASET_REVISION,
        "{{PUBLICATION_VERSION}}": PUBLICATION_VERSION,
    }
    for marker, value in replacements.items():
        card = card.replace(marker, value)
    if "{{" in card or "}}" in card:
        raise RuntimeError("Unresolved model-card template marker")
    return card


def copy_required_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Missing required upstream notice: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def produce_checkpoint_scaffolds() -> None:
    """Record tracked path placeholders as package outputs without dirtying Git."""
    scaffold_paths = sorted(CHECKPOINT_PATH.rglob(".gitkeep"))
    if not scaffold_paths:
        raise RuntimeError(f"Missing tracked checkpoint scaffolds: {CHECKPOINT_PATH}")
    for path in scaffold_paths:
        path.write_bytes(path.read_bytes())


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to resolve the pinned model notices")
    if not RESULT_PATH.is_file():
        raise RuntimeError(f"Missing checkpoint evaluation: {RESULT_PATH}")
    evaluation = json.loads(RESULT_PATH.read_text())
    if evaluation.get("status") != "passed":
        raise RuntimeError("Refusing to package a checkpoint that did not pass evaluation")

    base_snapshot = cached_snapshot(BASE_MODEL_ID, BASE_MODEL_REVISION, token)
    backbone_snapshot = cached_snapshot(BACKBONE_MODEL_ID, BACKBONE_MODEL_REVISION, token)

    copy_required_file(base_snapshot / "LICENSE", CHECKPOINT_PATH / "LICENSE")
    base_notices = CHECKPOINT_PATH / "upstream" / "gr00t-n1.7"
    for filename in UPSTREAM_NOTICE_FILES:
        copy_required_file(base_snapshot / filename, base_notices / filename)
    copy_required_file(
        backbone_snapshot / "README.md",
        CHECKPOINT_PATH / "upstream" / "cosmos-reason2-2b" / "README.md",
    )
    copy_required_file(
        ASSET_ROOT / "NVIDIA_OPEN_MODEL_LICENSE.md",
        CHECKPOINT_PATH / "NVIDIA_OPEN_MODEL_LICENSE.md",
    )

    commit = source_commit()
    (CHECKPOINT_PATH / "README.md").write_text(render_model_card(commit))
    (CHECKPOINT_PATH / "NOTICE").write_text(
        "Isaac GR00T N1.7 DROID reproducibility canary\n\n"
        "This checkpoint is derived from NVIDIA Isaac GR00T N1.7 and uses "
        "NVIDIA Cosmos Reason2.\n"
        "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n"
        "Built on NVIDIA Cosmos\n"
    )
    publication = {
        "schema_version": 1,
        "repository": PUBLICATION_REPO_ID,
        "version": PUBLICATION_VERSION,
        "source_commit": commit,
        "base_model": {"repo_id": BASE_MODEL_ID, "revision": BASE_MODEL_REVISION},
        "backbone_model": {
            "repo_id": BACKBONE_MODEL_ID,
            "revision": BACKBONE_MODEL_REVISION,
        },
        "dataset": {"repo_id": DATASET_ID, "revision": DATASET_REVISION},
    }
    (CHECKPOINT_PATH / "publication.json").write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n"
    )
    produce_checkpoint_scaffolds()
    print(f"Packaged checkpoint for hf://{PUBLICATION_REPO_ID}/{PUBLICATION_VERSION}")


if __name__ == "__main__":
    main()
