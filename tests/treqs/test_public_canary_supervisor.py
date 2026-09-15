"""Host-only regression tests; no GPU or remote mutation is needed."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".treqs" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from public_canary_supervisor import Actions, Supervisor
from public_canary_verify import verify_lineage, log_receipts


class Backend:
    def __init__(self):
        self.request = self.queued = None
        self.instance = {"id": "instance", "status": "running", "launchedAt": "1970-01-01T00:00:00Z"}
        self.status = "IN_PROGRESS"
        self.calls = []
        self.disconnect = None

    def instances(self):
        return [dict(self.instance)] if self.queued else []

    def prepare(self):
        self.calls.append("prepare")

    def find(self, kind, **kwargs):
        return self.request if kind == "tr" else self.queued

    def create(self):
        self.calls.append("create")
        self.request = {"id": "request"}
        if self.disconnect == "create":
            raise ConnectionError("request succeeded but reply was lost")
        return self.request

    def queue(self, request):
        self.calls.append("queue")
        self.queued = {"id": "job"}
        if self.disconnect == "queue":
            raise ConnectionError("queue succeeded but reply was lost")
        return {"jobId": "job"}

    def job(self, job):
        return {"id": job, "status": self.status, "onDemandInstanceId": "instance"}

    def stop(self, job):
        self.calls.append("stop")
        self.status = "STOPPED"

    def verify(self, state):
        self.calls.append("verify")
        return {"verified": True}

    def publish(self, state):
        self.calls.append("publish")
        return {"notesPr": "18"}


@pytest.fixture
def plan():
    return {"title": "unique public canary", "conservativeHourlyUsd": 4,
            "stopAtUsd": 4, "budgetUsd": 5, "maxJobSeconds": 3000}


@pytest.mark.parametrize("operation", ["create", "queue"])
def test_restart_after_lost_mutation_reply_never_duplicates_paid_job(tmp_path, plan, operation):
    backend = Backend()
    backend.disconnect = operation
    runner = Supervisor(plan, tmp_path, backend, clock=lambda: 10)
    with pytest.raises(ConnectionError):
        for _ in range(4):
            runner.tick()
    restarted = Supervisor(plan, tmp_path, backend, clock=lambda: 20)
    for _ in range(4):
        restarted.tick()
    assert restarted.state["jobId"] == "job"
    assert backend.calls.count("create") == backend.calls.count("queue") == 1


def launch(tmp_path, plan):
    backend = Backend()
    runner = Supervisor(plan, tmp_path, backend, clock=lambda: 0)
    for _ in range(3):
        runner.tick()
    return runner, backend


def test_completion_waits_for_allocation_shutdown_then_publishes_once(tmp_path, plan):
    runner, backend = launch(tmp_path, plan)
    backend.status = "COMPLETED"
    runner.tick()
    assert runner.state["phase"] == "monitor"
    assert "verify" not in backend.calls
    backend.instance.update(status="stopped", totalCostCents=152)
    for _ in range(3):
        runner.tick()
    assert runner.state["phase"] == "complete"
    assert runner.state["observedCostUsd"] == 1.52
    restarted = Supervisor(plan, tmp_path, backend)
    assert restarted.tick() is False
    assert backend.calls.count("queue") == backend.calls.count("publish") == 1


def test_budget_includes_provisioning_time_and_stops_before_publication(tmp_path, plan):
    runner, backend = launch(tmp_path, plan)
    runner.clock = lambda: 3601
    runner.tick()
    assert "stop" in backend.calls
    assert runner.state["phase"] == "monitor"
    backend.instance.update(status="stopped", totalCostCents=401)
    runner.tick()
    assert runner.state["phase"] == "failed"
    assert "verify" not in backend.calls


def test_network_failure_keeps_elapsed_time_stop_guard(tmp_path, plan):
    runner, backend = launch(tmp_path, plan)
    runner.clock = lambda: 3001
    runner.error(ConnectionError("API unavailable"))
    assert "stop" in backend.calls
    assert runner.state["stopReason"]


def test_plan_changes_are_rejected_after_restart(tmp_path, plan):
    runner, _ = launch(tmp_path, plan)
    with pytest.raises(RuntimeError, match="Saved plan differs"):
        Supervisor({**plan, "budgetUsd": 500}, tmp_path, Backend())


def test_real_cli_adapter_sends_public_pinned_launch(tmp_path, monkeypatch):
    executable = tmp_path / "treqs"
    executable.write_text("#!/usr/bin/env python3\nimport json,sys,os\n"
                          "assert 'TREQS_API_TOKEN' not in os.environ\n"
                          "a=sys.argv[1:]\nassert a[:3] == ['--json','tr','create']\n"
                          "assert a[a.index('--lineage-mode')+1] == 'public'\n"
                          "assert a[a.index('--source-commit')+1] == 'a'*40\n"
                          "print(json.dumps({'id':'request'}))\n")
    executable.chmod(0o755)
    monkeypatch.setenv("TREQS_API_TOKEN", "stale-test-token")
    action = Actions({"treqs": str(executable), "controlWorkspace": str(tmp_path),
                      "title": "test", "targetId": "target", "sourceBranch": "tb/public",
                      "sourceCommit": "a" * 40}, tmp_path)
    assert action.create() == {"id": "request"}


def test_split_log_chunks_bind_exactly_one_training_receipt(tmp_path):
    row = {"result": {"chunks": [{"content": 'E2E_ARTI'},
                                 {"content": 'FACT={"files":[]}\nE2E_RESULT={"optimizerSteps":100}\n'}]}}
    (tmp_path / "workload.jsonl").write_text(json.dumps(row) + "\n")
    assert log_receipts(tmp_path)["RESULT"]["optimizerSteps"] == 100
    with (tmp_path / "workload.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        log_receipts(tmp_path)


def test_public_graph_requires_actual_trained_shards_not_only_packaging():
    entries = [{"path": f"model-{i}.safetensors", "sizeBytes": 10} for i in range(3)]
    relations = [{"artifactHash": str(i), "path": f"artifacts/droid-canary/checkpoint-100/{e['path']}",
                  "artifact": {"size": "10"}} for i, e in enumerate(entries)]
    payload = {"total": 2, "jobs": [
        {"jobUid": "train", "command": "python .treqs/scripts/run_droid_canary.py", "inputs": [], "outputs": relations},
        {"jobUid": "put", "command": "roar put checkpoint hf://repo --public", "inputs": relations, "outputs": []}]}
    job = {"lineagePublishedSessionHash": "a" * 64,
           "lineagePublicationCounts": {"jobs": 2, "artifacts": 3, "links": 6}}
    assert len(verify_lineage(job, payload, entries)["trainingToPut"]) == 3
    payload["jobs"][0]["command"] = "python package_droid_canary.py"
    with pytest.raises(ValueError, match="actual training"):
        verify_lineage(job, payload, entries)


@pytest.mark.parametrize("corrupt", [False, True])
def test_anonymous_full_readback_binds_bytes_tensors_source_and_training(tmp_path, monkeypatch, corrupt):
    import hashlib
    import shutil
    from types import SimpleNamespace
    import numpy as np
    from safetensors.numpy import save_file
    import public_canary_verify as verify

    remote, root = tmp_path / "remote", tmp_path / "run"
    remote.mkdir()
    root.mkdir()
    weights = {}
    for shard in range(3):
        name = f"model-{shard}.safetensors"
        data = {f"tensor-{i}": np.ones(1, dtype=np.float32) for i in range(shard, 1030, 3)}
        save_file(data, remote / name)
        weights.update({key: name for key in data})
    (remote / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    (remote / "publication.json").write_text(json.dumps({"repository": "owner/model", "source_commit": "b" * 40}))
    files = [{"path": p.name, "sizeBytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(remote.iterdir())]
    manifest = {"schema": "reproai.artifact-manifest/v1", "loadVerified": True, "files": files}
    (remote / "artifact-manifest.json").write_text(json.dumps(manifest))
    result = {"optimizerSteps": 100, "loadVerified": True, "finiteLoss": 1, "finalLoss": .08,
              "artifactSha256": hashlib.sha256((remote / "model.safetensors.index.json").read_bytes()).hexdigest()}
    (remote / "result.json").write_text(json.dumps(result))
    artifact = {"files": [{**e, "path": f"{verify.PREFIX}/{e['path']}"} for e in files]}
    log = f"E2E_ARTIFACT={json.dumps(artifact)}\nE2E_RESULT={json.dumps(result)}\n"
    (root / "workload.jsonl").write_text(json.dumps({"result": {"chunks": [{"content": log}]}}) + "\n")
    info = SimpleNamespace(private=False, sha="c" * 40,
                           siblings=[SimpleNamespace(rfilename=f"{verify.PREFIX}/{p.name}") for p in remote.iterdir()])

    class PublicApi:
        def __init__(self, *, token):
            assert token is False

        def model_info(self, repo, **kwargs):
            assert kwargs["token"] is False
            return info

    def download(repo, revision, name, destination):
        assert revision == "c" * 40
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / Path(name).name, destination)
        if corrupt and name.endswith("model-1.safetensors"):
            destination.write_bytes(destination.read_bytes() + b"corruption")
        return destination

    relations = [{"artifactHash": e["sha256"], "path": f"{verify.PREFIX}/{e['path']}",
                  "artifact": {"size": str(e["sizeBytes"])}} for e in files if e["path"].endswith(".safetensors")]
    topology = {"total": 2, "jobs": [
        {"jobUid": "train", "command": "python run_droid_canary.py", "inputs": [], "outputs": relations},
        {"jobUid": "put", "command": "roar put checkpoint hf://owner/model --public", "inputs": relations, "outputs": []}]}
    monkeypatch.setattr(verify, "HfApi", PublicApi)
    monkeypatch.setattr(verify, "download_public", download)
    monkeypatch.setattr(verify, "public_json", lambda url: topology if "/jobs?" in url else {})
    job = {"lineagePublishedSessionHash": "a" * 64, "lineagePublicationCounts": {"jobs": 2, "artifacts": 3, "links": 6}}
    plan = {"hfRepository": "owner/model", "sourceCommit": "b" * 40}
    if corrupt:
        with pytest.raises(ValueError, match="Downloaded bytes differ"):
            verify.verify_release(plan, root, job)
    else:
        receipt = verify.verify_release(plan, root, job)
        assert receipt["verified"] and receipt["tensorCount"] == 1030
        assert len(receipt["files"]) == len(files)
        assert not list((root / "readback").rglob("*.safetensors"))
