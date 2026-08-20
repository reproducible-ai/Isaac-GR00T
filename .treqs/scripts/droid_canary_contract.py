"""Immutable input and output contract for the reproducible DROID canary."""

from pathlib import Path


BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BACKBONE_MODEL_ID = "nvidia/Cosmos-Reason2-2B"
BACKBONE_MODEL_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
DATASET_ID = "lerobot/droid_1.0.1"
DATASET_REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
DATASET_PATH = Path("demo_data/droid_sample_canary")
ARTIFACT_ROOT = Path("artifacts/droid-canary")
INPUT_MANIFEST_PATH = ARTIFACT_ROOT / "input-manifest.json"
RESULT_PATH = ARTIFACT_ROOT / "result.json"
EPISODE_COUNT = 3
MIN_GPU_MEMORY_MIB = 40 * 1024
