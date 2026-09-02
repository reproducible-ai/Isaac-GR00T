"""Immutable input and output contract for the reproducible DROID canary."""

from pathlib import Path


BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BACKBONE_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
BACKBONE_MODEL_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
DATASET_ID = "lerobot/droid_1.0.1"
DATASET_REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
TRAINING_STEPS = 100
ARTIFACT_ROOT = Path("artifacts/droid-canary")
DATASET_PATH = ARTIFACT_ROOT / "dataset"
CHECKPOINT_PATH = ARTIFACT_ROOT / f"checkpoint-{TRAINING_STEPS}"
INPUT_MANIFEST_PATH = ARTIFACT_ROOT / "input-manifest.json"
RESULT_PATH = CHECKPOINT_PATH / "evaluation.json"
PUBLICATION_REPO_ID = "reproducible-ai/harness-test-gr00t-droid100-issue-30"
PUBLICATION_VERSION = "artifacts/gr00t-droid-100step"
EPISODE_COUNT = 3
MIN_GPU_MEMORY_MIB = 40 * 1024
