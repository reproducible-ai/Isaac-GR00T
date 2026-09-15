"""Anonymous public-release checks and an idempotent notes PR addendum."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen

from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError
from safetensors import safe_open

from reproai_harness.artifact_verifier import _parse_manifest, _sha256, _safe_relative
from reproai_harness.lineage import normalize_lineage_topology
from public_canary_supervisor import atomic_json

PREFIX = "artifacts/droid-canary/checkpoint-100"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_repository(plan, root):
    from check_hf_access import READ_CHECKS, check_access, check_write_access
    token = get_token()
    require(bool(token), "Saved Hugging Face token is required")
    api = HfApi(token=token)
    repo = plan["hfRepository"]
    try:
        info = api.model_info(repo)
    except RepositoryNotFoundError as exc:
        if exc.response.status_code != 404:
            raise
        api.create_repo(repo, private=False, exist_ok=True)
        info = api.model_info(repo)
    require(not info.private, "Refusing to change an existing private repository")
    files = {x.rfilename for x in info.siblings}
    # Restart after our own create/upload is safe; never overwrite an existing model.
    require(files <= {".gitattributes", "README.md"}, "Publication repository already contains artifacts")
    marker = f"<!-- public-canary:{plan['runId']} -->"
    if "README.md" in files:
        path = hf_hub_download(repo, "README.md", revision=info.sha, token=token,
                               local_dir=str(root / "preflight"))
        require(marker in Path(path).read_text(), "Repository belongs to a different publication")
    else:
        api.upload_file(repo_id=repo, path_in_repo="README.md", path_or_fileobj=(
            "---\nlicense: other\nlicense_name: nvidia-license\nlicense_link: "
            "https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/"
            "2fc962b973bccdd5d8ce4f67cc63b264d6886495/LICENSE\n---\n\n"
            "# GR00T N1.7 public canary\n\n"
            "A fresh 100-step DROID fine-tuning run is being prepared. "
            "No verified checkpoint has been released yet.\n\n"
            "Non-commercial research and evaluation only.\n\n" + marker + "\n"
        ).encode(), commit_message="docs: prepare authorized public canary")
    for label, url in READ_CHECKS:
        check_access(label, url, token)
    check_write_access(token)
    public = HfApi(token=False).model_info(repo, token=False)
    require(public.private is False, "Repository is not anonymously visible")
    atomic_json(root / "preflight.json", {"repository": repo, "public": True,
                                          "pinnedInputAccess": True, "writeAccess": True})


def public_json(url):
    # Deliberately no Authorization header, cookies, or credential-bearing SDK.
    with urlopen(url, timeout=30) as response:
        body = response.read(5_000_001)
    require(len(body) <= 5_000_000, "Public response exceeds verification limit")
    envelope = json.loads(body)
    require(envelope.get("success") is True, "Public GLaaS read was not successful")
    return envelope["data"]


def log_receipts(root):
    content = "".join(chunk["content"] for line in (root / "workload.jsonl").read_text().splitlines()
                      for chunk in json.loads(line)["result"]["chunks"])
    receipts = {}
    for name in ("ARTIFACT", "RESULT"):
        values = [json.loads(line.split(f"E2E_{name}=", 1)[1])
                  for line in content.splitlines() if f"E2E_{name}=" in line]
        require(len(values) == 1, f"Expected exactly one E2E_{name} receipt")
        receipts[name] = values[0]
    return receipts


def verify_lineage(job, payload, entries):
    dag = job["lineagePublishedSessionHash"]
    shards = [entry for entry in entries if entry["path"].endswith(".safetensors")]
    require(len(shards) == 3, "Expected the GR00T three-shard checkpoint")
    proofs = []
    for entry in shards:
        path = f"{PREFIX}/{entry['path']}"
        graph = normalize_lineage_topology(dag_hash=dag, payload=payload,
                    publication_counts=job["lineagePublicationCounts"],
                    artifact_path=path, artifact_size_bytes=entry["sizeBytes"])
        require(graph["verified"], "Public graph does not connect checkpoint to PUT")
        trained = {relation["artifactHash"] for j in payload["jobs"]
                   if "run_droid_canary.py" in j["command"] for relation in j["outputs"]
                   if (relation.get("path") or "").endswith(path)}
        published = {relation["artifactHash"] for j in payload["jobs"]
                     if "roar put " in j["command"] for relation in j["inputs"]
                     if (relation.get("path") or "").endswith(path)}
        require(bool(trained & published), "PUT does not consume the actual training output")
        proofs.append({"path": path, "artifactHashes": sorted(trained & published)})
    return {"dagHash": dag, "counts": job["lineagePublicationCounts"], "trainingToPut": proofs}


def download_public(repo, revision, name, destination):
    # Stream without the SDK cache: only one ~5 GB shard occupies disk at a time.
    # An interrupted transfer is discarded and the same immutable revision retried.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urlopen(hf_hub_url(repo, name, revision=revision), timeout=60) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
    temporary.replace(destination)
    return destination


def verify_release(plan, root, job):
    repo = plan["hfRepository"]
    info = HfApi(token=False).model_info(repo, token=False, files_metadata=True)
    require(info.private is False and re.fullmatch(r"[0-9a-f]{40}", info.sha), "HF is not public and immutable")
    revision_file = root / "artifact-revision.json"
    if not revision_file.exists():
        atomic_json(revision_file, {"revision": info.sha})
    revision = json.loads(revision_file.read_text())["revision"]
    info = HfApi(token=False).model_info(repo, revision=revision, token=False, files_metadata=True)
    inventory = {x.rfilename for x in info.siblings}
    directory = root / "readback" / revision

    def get(name):
        return download_public(repo, revision, f"{PREFIX}/{name}", directory / name)

    manifest_path = get("artifact-manifest.json")
    manifest, entries = _parse_manifest(manifest_path)
    require(manifest.get("loadVerified") is True, "Manifest has no load verification")
    expected = {f"{PREFIX}/{e['path']}" for e in entries} | {
        f"{PREFIX}/artifact-manifest.json", f"{PREFIX}/result.json"}
    require({p for p in inventory if p.startswith(PREFIX + "/")} == expected,
            "Remote checkpoint inventory differs from manifest")
    receipts = log_receipts(root)
    expected_files = [{**e, "path": f"{PREFIX}/{e['path']}"} for e in entries]
    require(receipts["ARTIFACT"]["files"] == expected_files, "HF manifest differs from captured training receipt")
    result = json.loads(get("result.json").read_text())
    require(result == receipts["RESULT"], "Remote result differs from captured training result")
    require(result.get("optimizerSteps") == 100 and result.get("loadVerified") is True
            and result.get("finiteLoss") == 1 and math.isfinite(result["finalLoss"]), "Training contract failed")
    checked, tensors = [], {}
    for entry in entries:
        name = _safe_relative(entry["path"], label="manifest path")
        path = get(name)
        digest = _sha256(path)
        require(path.stat().st_size == entry["sizeBytes"] and digest == entry["sha256"],
                f"Downloaded bytes differ from manifest: {name}")
        record = dict(entry)
        if name.endswith(".safetensors"):
            with safe_open(path, framework="numpy", device="cpu") as handle:
                names = list(handle.keys())
                require(bool(names), f"Empty weight shard: {name}")
                for key in names:
                    require(key not in tensors, "Duplicate tensor across shards")
                    handle.get_slice(key).get_shape()
                    tensors[key] = name
                record["tensorCount"] = len(names)
            # Only this new temporary readback is removed; original private evidence is retained.
            path.unlink()
        checked.append(record)
        atomic_json(root / "readback-progress.json", {"revision": revision, "files": checked})
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    require(index["weight_map"] == tensors and len(tensors) == 1030, "Index does not exactly map all 1,030 tensors")
    publication = json.loads((directory / "publication.json").read_text())
    require(publication["source_commit"] == plan["sourceCommit"] and publication["repository"] == repo,
            "Checkpoint source or destination differs from launch plan")
    require(_sha256(directory / "model.safetensors.index.json") == result["artifactSha256"], "Result index digest differs")
    dag = job["lineagePublishedSessionHash"]
    require(bool(re.fullmatch(r"[0-9a-f]{64}", dag)), "Invalid canonical public DAG hash")
    public_json(f"https://api.glaas.ai/api/v1/public/sessions/{dag}")
    topology = public_json(f"https://api.glaas.ai/api/v1/public/sessions/{dag}/jobs?includeArtifacts=true")
    proof = verify_lineage(job, topology, entries)
    atomic_json(root / "public-lineage.json", topology)
    return {"schema": "reproai.public-canary/v1", "verified": True, "anonymousReadback": True,
            "repository": repo, "revision": revision, "path": PREFIX, "sourceCommit": plan["sourceCommit"],
            "result": result, "files": checked, "tensorCount": len(tensors), "lineage": proof,
            "hfUrl": f"https://huggingface.co/{repo}/tree/{revision}/{PREFIX}",
            "glaasUrl": f"https://glaas.ai/dag/{dag}"}


def publish_notes(plan, root, state, actions):
    receipt = state["verification"]
    require(receipt["verified"] and receipt["anonymousReadback"], "No verified public release to publish")
    card = (root / "readback" / receipt["revision"] / "README.md").read_text()
    card += (f"\n## Public verification\n\n"
             f"- [Immutable checkpoint]({receipt['hfUrl']})\n"
             f"- [Public training lineage]({receipt['glaasUrl']})\n"
             f"- [Run notes](https://github.com/reproducible-ai/notes/pull/18)\n\n"
             f"All checkpoint files were downloaded anonymously and SHA-256 verified; "
             f"all {receipt['tensorCount']:,} tensors matched the shard index.\n"
             f"<!-- public-canary:{plan['runId']} -->\n")
    api = HfApi(token=get_token())
    api.upload_file(repo_id=plan["hfRepository"], path_in_repo="README.md", path_or_fileobj=card.encode(),
                    commit_message="docs: link verified public checkpoint and training lineage")
    # Explicitly use the saved account for this child process only.
    actions.env["GH_TOKEN"] = subprocess.run(
        ["gh", "auth", "token", "--hostname", "github.com", "--user", "TrevorBasinger"],
        capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    pr = json.loads(actions.command(["gh", "pr", "view", "18", "--repo", "reproducible-ai/notes",
                                      "--json", "state,headRefName,headRefOid,url"]))
    require(pr["state"] == "OPEN" and pr["headRefName"] == "tb/notes-issue-39", "Notes PR identity changed")
    checkout = root / "notes"
    if not checkout.exists():
        actions.command(["git", "clone", "--single-branch", "--branch", pr["headRefName"],
                         "git@github-trev-public:reproducible-ai/notes.git", str(checkout)], timeout=180)
    actions.command(["git", "fetch", "origin"], cwd=checkout)
    upstream = actions.command(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], cwd=checkout).strip()
    require(upstream == "origin/tb/notes-issue-39", "Notes upstream is not the same-named branch")
    actions.command(["git", "merge", "--ff-only", upstream], cwd=checkout)
    directory = checkout / "023-robotics-gr00t"
    require(directory.is_dir(), "Notes model directory is missing")
    public_receipt = {**receipt, "runId": plan["runId"], "jobId": state["jobId"],
                      "requestId": state["requestId"], "costUsd": state["observedCostUsd"],
                      "computeStopped": not state["allocationActive"], "automationSource": plan["sourceCommit"]}
    atomic_json(directory / "evidence" / "public-release.json", public_receipt)
    (directory / "PUBLIC-RELEASE.md").write_text(
        "# Public GR00T canary\n\nA fresh public run completed 100 optimizer steps on the same "
        "pinned inputs as the private capture. This addendum preserves the original capture and its audit.\n\n"
        f"- [Hugging Face checkpoint, pinned revision]({receipt['hfUrl']})\n"
        f"- [Public GLaaS training lineage]({receipt['glaasUrl']})\n"
        f"- [Source and supervisor](https://github.com/reproducible-ai/Isaac-GR00T/tree/{plan['sourceCommit']}/.treqs)\n"
        f"- TReqs job: `{state['jobId']}`\n"
        f"- Full allocation cost: **${state['observedCostUsd']:.2f}**; compute stopped.\n"
        f"- Final training loss: `{receipt['result']['finalLoss']}`.\n\n"
        "Anonymous downloads matched every manifest size and SHA-256. All 1,030 tensors matched the "
        "three-shard index. Anonymous GLaaS reads confirmed each trained weight shard is consumed by PUT. "
        "The supervisor performed these checks and published this addendum automatically. "
        "This is a training-path canary, with no new model-quality or independent-auditor claim.\n\n"
        "Use is limited to non-commercial research/evaluation under the packaged NVIDIA license. "
        "See [verification evidence](evidence/public-release.json).\n")
    readme = directory / "README.md"
    link = "\n## Public artifacts and lineage\n\nSee the [fresh public canary](PUBLIC-RELEASE.md) for verified downloads and public GLaaS lineage.\n"
    if "(PUBLIC-RELEASE.md)" not in readme.read_text():
        readme.write_text(readme.read_text().rstrip() + "\n" + link)
    actions.command(["git", "add", "023-robotics-gr00t/README.md", "023-robotics-gr00t/PUBLIC-RELEASE.md",
                     "023-robotics-gr00t/evidence/public-release.json"], cwd=checkout)
    diff = actions.command(["git", "diff", "--cached", "--name-only"], cwd=checkout)
    if diff.strip():
        actions.command(["git", "commit", "-m", "docs(gr00t): record verified public canary and lineage"], cwd=checkout)
    actions.command(["git", "push", "origin", "HEAD:refs/heads/tb/notes-issue-39"], cwd=checkout, timeout=120)
    commit = actions.command(["git", "rev-parse", "HEAD"], cwd=checkout).strip()
    current = json.loads(actions.command(["gh", "pr", "view", "18", "--repo", "reproducible-ai/notes",
                                          "--json", "headRefOid"]))
    require(current["headRefOid"] == commit, "Notes PR does not contain the published commit")
    return {"notesPr": pr["url"], "notesCommit": commit, "hfUrl": receipt["hfUrl"], "glaasUrl": receipt["glaasUrl"]}
