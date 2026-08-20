from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from gr00t.configs.finetune_config import FinetuneConfig
import huggingface_hub
from scripts import download_droid_sample


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / ".treqs" / "scripts"


def load_contract():
    spec = importlib.util.spec_from_file_location(
        "droid_canary_contract", SCRIPTS_DIR / "droid_canary_contract.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canary_revisions_are_immutable_hashes():
    contract = load_contract()
    revisions = (
        contract.BASE_MODEL_REVISION,
        contract.BACKBONE_MODEL_REVISION,
        contract.DATASET_REVISION,
    )
    assert all(len(revision) == 40 for revision in revisions)
    assert all(set(revision) <= set("0123456789abcdef") for revision in revisions)


def test_finetune_config_accepts_a_pinned_backbone_revision():
    config = FinetuneConfig(
        base_model_path="/tmp/base-model",
        dataset_path="/tmp/dataset",
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        backbone_model_revision="a" * 40,
    )
    assert config.backbone_model_revision == "a" * 40


def test_droid_download_forwards_the_dataset_revision(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    download_droid_sample.download_droid_files(tmp_path, revision="b" * 40)

    assert calls
    assert all(call["repo_id"] == "lerobot/droid_1.0.1" for call in calls)
    assert all(call["repo_type"] == "dataset" for call in calls)
    assert all(call["revision"] == "b" * 40 for call in calls)


def test_canary_setup_waits_for_the_fresh_instance_dpkg_lock():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "timeout --signal=TERM 420" in workflow
    assert "DPkg::Lock::Timeout=360" in workflow


def test_canary_setup_installs_the_test_runner_before_validation():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "uv sync --locked --python 3.12 --no-editable --extra dev" in workflow
    assert "uv run --no-sync pytest" in workflow
