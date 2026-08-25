"""Fail fast when HF_TOKEN cannot read inputs or publish the canary."""

from __future__ import annotations

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from droid_canary_contract import (
    BACKBONE_MODEL_ID,
    BACKBONE_MODEL_REVISION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    PUBLICATION_REPO_ID,
)


READ_CHECKS = (
    (
        "GR00T base model",
        f"https://huggingface.co/{BASE_MODEL_ID}/resolve/"
        f"{BASE_MODEL_REVISION}/.gitattributes",
    ),
    (
        "gated Cosmos backbone",
        f"https://huggingface.co/{BACKBONE_MODEL_ID}/resolve/"
        f"{BACKBONE_MODEL_REVISION}/.gitattributes",
    ),
)


def check_access(label: str, url: str, token: str) -> None:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "reproducible-ai-gr00t-canary/1.0",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            response.read(1)
    except HTTPError as error:
        hint = ""
        if label == "gated Cosmos backbone" and error.code in (401, 403):
            hint = (
                " Accept the model terms with the HF_TOKEN account and enable the "
                "fine-grained token permission to read public gated repositories."
            )
        raise RuntimeError(
            f"Hugging Face preflight failed for {label} (HTTP {error.code}).{hint}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Hugging Face preflight failed for {label}: {error.reason}") from error


def check_write_access(token: str) -> None:
    """Ask HF how it would store a tiny file without creating a commit."""
    sample = b"permission-check"
    payload = json.dumps(
        {
            "files": [
                {
                    "path": ".hf-write-permission-check",
                    "sample": base64.b64encode(sample).decode("ascii"),
                    "size": len(sample),
                }
            ]
        }
    ).encode()
    request = Request(
        f"https://huggingface.co/api/models/{PUBLICATION_REPO_ID}/preupload/main",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "reproducible-ai-gr00t-canary/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            result = json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(
            "Hugging Face preflight failed for publication repository write access "
            f"(HTTP {error.code}). Grant HF_TOKEN write access to "
            f"{PUBLICATION_REPO_ID}."
        ) from error
    except (URLError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Hugging Face publication write preflight failed: {error}"
        ) from error
    files = result.get("files", [])
    if len(files) != 1 or files[0].get("uploadMode") not in ("regular", "lfs"):
        raise RuntimeError("Hugging Face publication write preflight returned an invalid response")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the Hugging Face preflight")
    for label, url in READ_CHECKS:
        check_access(label, url, token)
    check_write_access(token)
    print("Hugging Face gated-read and publication-write preflight passed")


if __name__ == "__main__":
    main()
