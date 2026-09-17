"""Fetch pinned calibration inputs and retain their byte inventory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.calibration_droid import encoded, load_inputs, sha256_file


ROOT = Path("artifacts/droid-calibration")
INPUTS = ROOT / "inputs"


def storage_inventory(block_root=Path("/sys/block")):
    """Nitro exposes EBS identity and capacity through ordinary kernel block devices."""
    volumes = []
    for device in sorted(block_root.glob("nvme*n1")):
        model = (device / "device/model").read_text().strip()
        if model != "Amazon Elastic Block Store":
            continue
        size = int((device / "size").read_text()) * 512
        if size <= 0:
            raise RuntimeError("invalid attached EBS capacity")
        volumes.append({"device": device.name, "model": model, "sizeBytes": size})
    if not volumes or sum(volume["sizeBytes"] for volume in volumes) > 1024**4:
        raise RuntimeError("allocation storage is outside the reviewed 1 TiB EBS allowance")
    return volumes


def main():
    from huggingface_hub import snapshot_download
    import torch

    started = time.time()
    plan, config, _ = load_inputs(
        ".treqs/calibration/plan.json", ".treqs/calibration/resolved-config.json"
    )
    expected = plan["recipe"]["hardware"]
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != expected["accelerator"]:
        raise RuntimeError("calibration hardware differs from the reviewed recipe")
    if torch.cuda.get_device_properties(0).total_memory < 90 * 1024**3:
        raise RuntimeError("calibration requires a 96 GB GPU")
    if any(p.is_file() and p.name != ".gitkeep" for p in INPUTS.rglob("*")):
        raise RuntimeError("refusing to reuse previously prepared calibration inputs")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned gated models")
    volumes = storage_inventory()
    INPUTS.mkdir(parents=True, exist_ok=True)
    base = snapshot_download(
        "nvidia/GR00T-N1.7-3B",
        revision=plan["recipe"]["baseModel"]["revision"],
        token=token,
        local_dir=INPUTS / "base",
    )
    backbone = snapshot_download(
        config["model"]["model_name"],
        revision=config["model"]["model_revision"],
        token=token,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.download_droid_sample",
            "--revision",
            plan["recipe"]["dataset"]["revision"],
            "--num-episodes",
            "32",
            "--cache-dir",
            "/tmp/droid-calibration-download",
            "--output-dir",
            str(INPUTS / "dataset"),
        ],
        check=True,
    )
    dataset_info = json.loads((INPUTS / "dataset/meta/info.json").read_bytes())
    if dataset_info["total_episodes"] != 32:
        raise RuntimeError("DROID conversion did not produce the reviewed 32 episodes")
    files = []
    for label, root in [
        ("base", Path(base)),
        ("backbone", Path(backbone)),
        ("dataset", INPUTS / "dataset"),
    ]:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".cache" in path.parts or path.name == ".gitkeep":
                continue
            files.append(
                {
                    "input": label,
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema": "gr00t.calibration-inputs/v1",
        "startedUnixSeconds": started,
        "finishedUnixSeconds": time.time(),
        "files": files,
        "episodes": 32,
        "planSha256": sha256_file(".treqs/calibration/plan.json"),
        "hardware": expected,
        "attachedEbsVolumes": volumes,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    (ROOT / "input-manifest.json").write_bytes(encoded(manifest))
    print(f"Prepared {len(files)} pinned input files and 32 DROID episodes", flush=True)


if __name__ == "__main__":
    main()
