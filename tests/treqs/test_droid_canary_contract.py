from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tomllib
from urllib.error import HTTPError

from gr00t.configs.finetune_config import FinetuneConfig
import huggingface_hub
import pytest
from safetensors.torch import save_file
from scripts import download_droid_sample
import torch
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


def test_canary_launches_repo_modules_without_replacing_pythonpath():
    workflow = load_workflow()
    prepare = (SCRIPTS_DIR / "prepare_droid_canary.py").read_text()
    finetune = (ROOT / "examples" / "finetune.sh").read_text()

    for stage_name in ("fetch_droid", "train", "evaluate", "package"):
        assert "PYTHONPATH=" not in workflow[stage_name]["command"]
    assert '"-m",\n            "scripts.download_droid_sample"' in prepare
    assert "    -m\n    gr00t.experiment.launch_finetune" in finetune


def test_model_access_is_checked_before_snapshot_transfers(monkeypatch):
    contract = load_contract()
    prepare = load_canary_script("prepare_droid_canary.py")
    events = []

    def fake_hf_hub_download(*, repo_id, filename, revision, token):
        events.append(("access", repo_id, filename, revision, token))
        return f"/tmp/{revision}/{filename}"

    def fake_snapshot_download(*, repo_id, revision, token):
        events.append(("snapshot", repo_id, revision, token))
        return f"/tmp/{revision}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    prepare.download_models("read-token")

    assert events[:2] == [
        (
            "access",
            contract.BASE_MODEL_ID,
            ".gitattributes",
            contract.BASE_MODEL_REVISION,
            "read-token",
        ),
        (
            "access",
            contract.BACKBONE_MODEL_ID,
            ".gitattributes",
            contract.BACKBONE_MODEL_REVISION,
            "read-token",
        ),
    ]
    assert [event[0] for event in events[2:]] == ["snapshot", "snapshot"]


def test_setup_preflights_hf_access_before_installing_the_environment(monkeypatch):
    check = load_canary_script("check_hf_access.py")
    requests = []

    def fake_urlopen(request, *, timeout):
        assert timeout == 30
        requests.append(request)
        if request.get_method() == "POST":
            return BytesIO(b'{"files":[{"uploadMode":"regular"}]}')
        return BytesIO(b"ok")

    monkeypatch.setattr(check, "urlopen", fake_urlopen)
    monkeypatch.setenv("HF_TOKEN", "scoped-token")

    check.main()

    assert len(requests) == 3
    assert all(
        request.get_header("Authorization") == "Bearer scoped-token"
        for request in requests
    )
    assert any("Cosmos-Reason2-2B" in request.full_url for request in requests)
    write_request = next(request for request in requests if request.get_method() == "POST")
    assert "reproducible-ai/GR00T/preupload/main" in write_request.full_url
    assert json.loads(write_request.data)["files"][0]["path"] == (
        ".hf-write-permission-check"
    )
    setup = load_workflow()["setup"]["command"]
    assert setup.index("check_hf_access.py") < setup.index("pip install")


def test_hf_preflight_explains_the_gated_repo_permission(monkeypatch):
    check = load_canary_script("check_hf_access.py")

    def forbidden(request, *, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(check, "urlopen", forbidden)

    with pytest.raises(RuntimeError, match="fine-grained token permission"):
        check.check_access("gated Cosmos backbone", "https://example.test/model", "token")


def test_hf_write_preflight_is_non_mutating_and_explains_missing_permission(monkeypatch):
    check = load_canary_script("check_hf_access.py")

    def forbidden(request, *, timeout):
        assert request.full_url.endswith("/preupload/main")
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(check, "urlopen", forbidden)

    with pytest.raises(RuntimeError, match="write access to reproducible-ai/GR00T"):
        check.check_write_access("token")

    script = (SCRIPTS_DIR / "check_hf_access.py").read_text()
    for mutation in ("upload_file", "create_commit", "delete_file", "create_repo"):
        assert mutation not in script


def test_diagnostic_hf_preflight_is_untraced_and_does_not_publish():
    workflow = yaml.safe_load(
        (ROOT / ".treqs" / "workflows" / "hf-access-preflight.yaml").read_text()
    )

    assert workflow["secrets"] == ["HF_TOKEN"]
    assert workflow["check_hf_access"]["trace"] == "off"
    assert "check_hf_access.py" in workflow["check_hf_access"]["command"]
    assert "roar" not in workflow["check_hf_access"]["command"]
    assert "put" not in workflow["check_hf_access"]["command"]


def test_dataset_placeholder_is_removed_before_download(monkeypatch, tmp_path):
    prepare = load_canary_script("prepare_droid_canary.py")
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / ".gitkeep").touch()
    calls = []

    def fake_run(command, *, check):
        assert check is True
        assert not dataset_path.exists()
        calls.append(command)

    monkeypatch.setattr(prepare, "DATASET_PATH", dataset_path)
    monkeypatch.setattr(prepare.subprocess, "run", fake_run)

    prepare.download_dataset()

    assert len(calls) == 1
    assert calls[0][-1] == str(dataset_path)


def test_canary_setup_waits_for_the_fresh_instance_dpkg_lock():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "timeout --signal=TERM 420" in workflow
    assert "DPkg::Lock::Timeout=360" in workflow


def test_canary_setup_supports_sudo_and_root_only_images():
    setup = load_workflow()["setup"]["command"]

    assert "if command -v sudo" in setup
    assert 'elif [ "$(id -u)" -eq 0 ]' in setup
    assert '"${PRIVILEGE[@]}"' in setup
    assert "timeout --signal=TERM 180 sudo" not in setup
    assert "sudo -n ln -sf" not in setup


def test_canary_setup_installs_the_test_runner_before_validation():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "uv python install 3.12.13" in workflow
    assert "UV_PYTHON_PREFERENCE=only-managed uv python find 3.12.13" in workflow
    assert 'uv sync --locked --python "$MANAGED_PYTHON"' in workflow
    assert "uv run --no-sync pytest" in workflow


def test_canary_setup_validates_the_torchcodec_native_runtime():
    pyproject = (ROOT / "pyproject.toml").read_text()
    setup = load_workflow()["setup"]["command"]

    assert "torchcodec==0.8.1; platform_machine == 'x86_64'" in pyproject
    assert "from torchcodec.decoders import VideoDecoder" in setup


def test_canary_evaluation_opens_the_safetensors_checkpoint(monkeypatch, tmp_path):
    verify = load_canary_script("verify_droid_canary.py")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text("{}\n")
    result_path = checkpoint / "evaluation.json"
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 1, "log_history": [{"loss": 0.125}]}\n'
    )
    save_file(
        {
            "action_head.bias": torch.zeros(2),
            "action_head.weight": torch.zeros((2, 3)),
        },
        checkpoint / "model.safetensors",
    )

    monkeypatch.setattr(verify, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(verify, "INPUT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(verify, "RESULT_PATH", result_path)

    verify.main()

    result = json.loads(result_path.read_text())
    assert result["status"] == "passed"
    assert result["model_files"][0]["tensor_count"] == 2
    assert result["model_files"][0]["first_tensor"] == {
        "name": "action_head.bias",
        "shape": [2],
    }


def test_canary_records_pypi_reproducible_gpu_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    sources = pyproject["tool"]["uv"]["sources"]
    run_script = (SCRIPTS_DIR / "run_droid_canary.py").read_text()

    for package in ("torch", "torchvision"):
        assert sources[package] == [
            {
                "index": "pytorch-cu128",
                "marker": "sys_platform == 'linux' and platform_machine == 'aarch64'",
            }
        ]
    assert '"--no-use-flash-attention"' in run_script


def test_canary_setup_pins_an_isolated_roar_runtime():
    setup = load_workflow()["setup"]["command"]

    assert "command -v uv" in setup
    assert "include-system-site-packages = false" in setup
    assert "roar-cli==0.4.5" in setup
    assert 'roar --version)" = "roar, version 0.4.5"' in setup
    assert "--with huggingface-hub" in setup
    assert "env PATH=/usr/local/bin:/usr/bin:/bin roar --version" in setup
    assert "roar tracer use preload" in setup
    assert "roar init" in setup


def test_workload_stages_are_named_roar_runs_without_nested_tracing():
    workflow = load_workflow()

    for stage_name in ("fetch_droid", "train", "evaluate", "package"):
        stage = workflow[stage_name]
        assert stage["trace"] == "off"
        assert f"roar run -n {stage_name} --" in stage["command"]
        assert "PYTHONPATH=" not in stage["command"]
        assert "--wandb-to-trackio" not in stage["command"]
        assert "TRACKIO_SPACE_ID" not in stage["command"]


def test_checkpoint_is_labeled_and_published_to_the_precreated_model_repo():
    workflow = load_workflow()
    label = workflow["label"]
    publish = workflow["publish"]
    model_card = (ROOT / ".treqs" / "assets" / "droid-canary-model-card.md").read_text()
    front_matter = yaml.safe_load(model_card.split("---", 2)[1])

    assert label["trace"] == "off"
    assert "roar label set artifact" in label["command"]
    assert "LicenseRef-NVIDIA-License" in label["command"]
    assert "non-commercial research/evaluation" in label["command"]
    assert publish["trace"] == "off"
    assert publish["glaas_creds"] is True
    assert "hf://reproducible-ai/GR00T/droid-canary-0.0.1" in publish["command"]
    assert "--public --yes --no-tag" in publish["command"]
    assert "artifacts/droid-canary/dataset" not in publish["command"]
    assert "/tmp/isaac-groot-hf" not in publish["command"]
    assert front_matter["license"] == "other"
    assert front_matter["license_name"] == "nvidia-license"


def test_package_copies_upstream_notices_and_writes_release_metadata(
    monkeypatch, tmp_path
):
    package = load_canary_script("package_droid_canary.py")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result_path = checkpoint / "evaluation.json"
    result_path.write_text('{"status": "passed"}\n')
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "droid-canary-model-card.md").write_text(
        "source={{SOURCE_COMMIT}}\n"
        "base={{BASE_MODEL_REVISION}}\n"
        "backbone={{BACKBONE_MODEL_REVISION}}\n"
        "dataset={{DATASET_REVISION}}\n"
        "version={{PUBLICATION_VERSION}}\n"
    )
    (assets / "NVIDIA_OPEN_MODEL_LICENSE.md").write_text("cosmos license\n")
    base_snapshot = tmp_path / "base"
    backbone_snapshot = tmp_path / "backbone"
    base_snapshot.mkdir()
    backbone_snapshot.mkdir()
    (base_snapshot / "LICENSE").write_text("base license\n")
    for filename in package.UPSTREAM_NOTICE_FILES:
        (base_snapshot / filename).write_text(f"base {filename}\n")
    (backbone_snapshot / "README.md").write_text("cosmos readme\n")

    def fake_snapshot(repo_id, revision, token):
        assert revision
        assert token == "write-token"
        if repo_id == package.BASE_MODEL_ID:
            return base_snapshot
        if repo_id == package.BACKBONE_MODEL_ID:
            return backbone_snapshot
        raise AssertionError(repo_id)

    monkeypatch.setattr(package, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(package, "RESULT_PATH", result_path)
    monkeypatch.setattr(package, "ASSET_ROOT", assets)
    monkeypatch.setattr(package, "cached_snapshot", fake_snapshot)
    monkeypatch.setattr(package, "source_commit", lambda: "c" * 40)
    monkeypatch.setenv("HF_TOKEN", "write-token")

    package.main()

    assert (checkpoint / "LICENSE").read_text() == "base license\n"
    assert "source=" + "c" * 40 in (checkpoint / "README.md").read_text()
    assert "Built on NVIDIA Cosmos" in (checkpoint / "NOTICE").read_text()
    assert (checkpoint / "NVIDIA_OPEN_MODEL_LICENSE.md").read_text() == (
        "cosmos license\n"
    )
    publication = json.loads((checkpoint / "publication.json").read_text())
    assert publication["repository"] == "reproducible-ai/GR00T"
    assert publication["version"] == "droid-canary-0.0.1"


def test_generated_paths_preserve_a_clean_checkout():
    contract = load_contract()

    assert contract.DATASET_PATH == contract.ARTIFACT_ROOT / "dataset"
    assert contract.CHECKPOINT_PATH == contract.ARTIFACT_ROOT / "checkpoint-1"
    assert contract.RESULT_PATH == contract.CHECKPOINT_PATH / "evaluation.json"
    assert not (ROOT / contract.DATASET_PATH).exists()
    assert (ROOT / contract.CHECKPOINT_PATH / ".gitkeep").is_file()


def test_hugging_face_sdk_uploads_are_owned_by_roar_not_workload_scripts():
    scripts = "\n".join(path.read_text() for path in sorted(SCRIPTS_DIR.glob("*droid_canary.py")))

    assert "snapshot_download" in scripts
    for upload_call in ("upload_file", "upload_folder", "push_to_hub", "create_repo"):
        assert upload_call not in scripts


def test_canary_explicitly_requests_the_gated_model_secret():
    workflow = (ROOT / ".treqs" / "workflows" / "droid-canary.yaml").read_text()

    assert "secrets:\n  - HF_TOKEN\n" in workflow
