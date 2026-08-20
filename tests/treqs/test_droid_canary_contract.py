from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from gr00t.configs.finetune_config import FinetuneConfig
import huggingface_hub
from scripts import download_droid_sample
import yaml


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


def load_workflow():
    return yaml.safe_load((ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text())


def load_canary_script(filename: str):
    load_contract()
    module_name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
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


def test_model_access_is_checked_before_snapshot_transfers(monkeypatch):
    contract = load_contract()
    prepare = load_canary_script("prepare_droid_canary.py")
    events = []

    class FakeApi:
        def model_info(self, *, repo_id, revision=None, token=None):
            events.append(("info", repo_id, revision, token))

    def fake_snapshot_download(*, repo_id, revision, token):
        events.append(("snapshot", repo_id, revision, token))
        return f"/tmp/{revision}"

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    prepare.download_models("read-write-token")

    assert events[:3] == [
        ("info", contract.BASE_MODEL_ID, contract.BASE_MODEL_REVISION, "read-write-token"),
        (
            "info",
            contract.BACKBONE_MODEL_ID,
            contract.BACKBONE_MODEL_REVISION,
            "read-write-token",
        ),
        ("info", contract.PUBLISH_REPO_ID, None, "read-write-token"),
    ]
    assert [event[0] for event in events[3:]] == ["snapshot", "snapshot"]


def test_canary_setup_waits_for_the_fresh_instance_dpkg_lock():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "timeout --signal=TERM 420" in workflow
    assert "DPkg::Lock::Timeout=360" in workflow


def test_canary_setup_installs_the_test_runner_before_validation():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "uv sync --locked --python 3.12 --no-install-project --extra dev" in workflow
    assert "uv run --no-sync pytest" in workflow


def test_canary_setup_pins_an_isolated_roar_runtime():
    setup = load_workflow()["setup"]["command"]

    assert "command -v uv" in setup
    assert "include-system-site-packages = false" in setup
    assert "roar-cli==0.4.4" in setup
    assert "--with huggingface-hub" in setup
    assert "env PATH=/usr/local/bin:/usr/bin:/bin roar --version" in setup
    assert "roar tracer use preload" in setup
    assert "roar init" in setup


def test_workload_stages_are_named_roar_runs_without_nested_tracing():
    workflow = load_workflow()

    for stage_name in ("fetch_droid", "train", "evaluate"):
        stage = workflow[stage_name]
        assert stage["trace"] == "off"
        assert f"roar run -n {stage_name} --wandb-to-trackio --" in stage["command"]
        assert "PYTHONPATH=.:.treqs/scripts" in stage["command"]
        assert "TRACKIO_SPACE_ID=reproducible-ai/experiments" in stage["command"]


def test_publish_is_labeled_gated_and_delegated():
    workflow = load_workflow()
    label = workflow["label"]
    publish = workflow["publish"]

    assert label["trace"] == "off"
    assert "roar label set artifact" in label["command"]
    assert "LicenseRef-NVIDIA-Open-Model-License" in label["command"]
    assert publish["trace"] == "off"
    assert publish["glaas_creds"] is True
    assert "roar status --untracked-dirs" in publish["command"]
    assert (
        "roar put artifacts/droid-canary/checkpoint-1 hf://reproducible-ai/isaac-groot/droid-canary"
    ) in publish["command"]
    assert "--public --yes --no-tag" in publish["command"]


def test_generated_inputs_and_outputs_use_tracked_directories():
    contract = load_contract()

    assert contract.DATASET_PATH == contract.ARTIFACT_ROOT / "dataset"
    assert contract.CHECKPOINT_PATH == contract.ARTIFACT_ROOT / "checkpoint-1"
    assert contract.RESULT_PATH == contract.CHECKPOINT_PATH / "evaluation.json"
    assert (ROOT / contract.DATASET_PATH / ".gitkeep").is_file()
    assert (ROOT / contract.CHECKPOINT_PATH / ".gitkeep").is_file()


def test_hugging_face_sdk_is_read_only_in_canary_scripts():
    scripts = "\n".join(path.read_text() for path in sorted(SCRIPTS_DIR.glob("*droid_canary.py")))

    assert "snapshot_download" in scripts
    for upload_call in ("upload_file", "upload_folder", "push_to_hub", "create_repo"):
        assert upload_call not in scripts


def test_canary_explicitly_requests_the_gated_model_secret():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "secrets:\n  - HF_TOKEN\n" in workflow
