"""Standard-library-only receipt and workflow checks; no model or API calls."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".treqs/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("package", SCRIPTS / "package_droid_canary.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class OfflineContract(unittest.TestCase):
    def test_workflow_and_bound_publication(self):
        text = (ROOT / ".treqs/workflows/droid-canary.yaml").read_text()
        for repository in ("reproducible-ai/harness-test-pending", "org/attempt-123"):
            bound = text.replace(package.publication_repository(), repository)
            stages = {}
            current = None
            for line in bound.splitlines():
                if re.match(r"^[a-z_]+:", line):
                    current = line.split(":", 1)[0]
                    stages[current] = []
                elif current:
                    stages[current].append(line)
            self.assertEqual(
                set(stages),
                {
                    "name",
                    "secrets",
                    "setup",
                    "fetch_droid",
                    "train",
                    "evaluate",
                    "package",
                    "label",
                    "publish",
                },
            )
            for name in set(stages) - {"name", "secrets"}:
                lines = stages[name]
                keys = [
                    line.strip().split(":", 1)[0] for line in lines if re.match(r"^  \w+:", line)
                ]
                self.assertEqual(keys.count("command"), 1)
                self.assertTrue(set(keys) <= {"command", "trace", "glaas_creds"})
                command = "\n".join(line[4:] for line in lines if line.startswith("    "))
                subprocess.run(["bash", "-n"], input=command, text=True, check=True)
                if name == "train":
                    self.assertIn('  trace: "run"', lines)
                    self.assertNotIn("roar", command)
            commands = re.findall(r"^    roar put .+$", bound, re.M)
            self.assertEqual(len(commands), 1)
            args = shlex.split(commands[0])
            self.assertEqual(args[:3], ["roar", "put", "artifacts/droid-canary/checkpoint-100"])
            self.assertIn(f"hf://{repository}/artifacts/droid-canary/checkpoint-100", args)
            self.assertEqual(args.count("-m"), 1)
            self.assertTrue(args[args.index("-m") + 1].strip())
            self.assertTrue({"--public", "--yes", "--no-tag"} <= set(args))
            self.assertFalse({"--private", "--anonymous"} & set(args))
            self.assertIn('  trace: "off"', stages["publish"])
            self.assertIn("  glaas_creds: true", stages["publish"])

    def test_receipts_and_stale_checkpoint_rejection(self):
        # Synthetic bytes exercise receipt binding only; they are not model weights.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            for name in (
                "model-1.safetensors",
                "model-2.safetensors",
                "LICENSE",
                "NOTICE",
                "README.md",
            ):
                (checkpoint / name).write_bytes(b"synthetic contract fixture")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
            )
            (checkpoint / "trainer_state.json").write_text('{"global_step":100}')

            def record(name):
                return {
                    "path": str(checkpoint / name),
                    "sha256": package.sha256_file(checkpoint / name),
                }

            evaluation = {
                "status": "passed",
                "loadVerified": True,
                "global_step": 100,
                "final_loss": 0.125,
                "trainer_state_sha256": package.sha256_file(checkpoint / "trainer_state.json"),
                "safetensors_index": record("model.safetensors.index.json"),
                "model_files": [record("model-1.safetensors"), record("model-2.safetensors")],
            }
            with patch.object(package, "CHECKPOINT_PATH", checkpoint):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    package.write_training_receipts(evaluation)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                artifact = json.loads(lines[0].removeprefix("E2E_ARTIFACT="))
                result = json.loads(lines[1].removeprefix("E2E_RESULT="))
                self.assertEqual(result, json.loads((checkpoint / "result.json").read_text()))
                self.assertEqual(result["schema"], "reproai.result/v1")
                self.assertEqual(artifact["schema"], "reproai.artifact/v1")
                self.assertEqual(result["taskMetric"], {"metric": "finiteLoss", "minimum": 1})
                self.assertEqual(result["optimizerSteps"], 100)
                self.assertEqual(result["finiteLoss"], 1)
                self.assertTrue(result["loadVerified"])
                index = (checkpoint / "model.safetensors.index.json").read_bytes()
                self.assertEqual(artifact["sha256"], hashlib.sha256(index).hexdigest())
                self.assertEqual(artifact["sha256"], result["artifactSha256"])
                self.assertEqual(artifact["sizeBytes"], len(index))
                self.assertEqual(artifact["sizeBytes"], result["artifactSizeBytes"])
                manifest = json.loads((checkpoint / "artifact-manifest.json").read_text())
                self.assertEqual(manifest["schema"], "reproai.artifact-manifest/v1")
                self.assertEqual(manifest["format"], "gr00t-n1.7-safetensors-checkpoint")
                self.assertTrue(manifest["loadVerified"])
                expected = {p.name for p in checkpoint.iterdir()} - {
                    "artifact-manifest.json",
                    "result.json",
                }
                self.assertEqual({entry["path"] for entry in manifest["files"]}, expected)
                for entry in manifest["files"]:
                    self.assertNotIn("..", Path(entry["path"]).parts)
                    self.assertFalse(Path(entry["path"]).is_absolute())
                    data = (checkpoint / entry["path"]).read_bytes()
                    self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                    self.assertEqual(entry["sizeBytes"], len(data))
                for changes in (
                    {"global_step": 99},
                    {"loadVerified": False},
                    {"final_loss": float("nan")},
                    {"model_files": []},
                ):
                    with self.assertRaises(RuntimeError):
                        package.write_training_receipts({**evaluation, **changes})
                for name in (
                    "model-1.safetensors",
                    "model.safetensors.index.json",
                    "trainer_state.json",
                ):
                    path = checkpoint / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"changed")
                    with self.assertRaises(RuntimeError):
                        package.write_training_receipts(evaluation)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
