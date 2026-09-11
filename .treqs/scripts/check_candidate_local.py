"""Offline preparation checks; no model execution, credentials, or API calls."""

import ast
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import droid_canary_contract as contract
import run_droid_canary as runner


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".treqs/workflows/droid-canary.yaml"


def stages(text):
    # Deliberately limited to this recipe's flat, literal-block stage format.
    return dict(re.findall(r"(?m)^([a-z_]+):\n((?:[ \t]+.*\n|\n)*)", text))


def command(block):
    match = re.search(r"(?m)^  command: \|\n((?:    .*\n|\n)*)", block)
    if match is None:
        raise AssertionError("Expected one literal command block")
    return "\n".join(line[4:] for line in match[1].splitlines())


class CandidateChecks(unittest.TestCase):
    def test_workflow_and_shell_syntax(self):
        text = WORKFLOW.read_text()
        recipe = stages(text)
        self.assertEqual(
            set(re.findall(r"(?m)^([a-z_]+):", text)),
            {"name", "secrets", "setup", "fetch_droid", "train", "evaluate", "package", "label", "publish"},
        )
        for name in ("setup", "fetch_droid", "train", "evaluate", "package", "label", "publish"):
            block = recipe[name]
            self.assertEqual(block.count("  command:"), 1)
            self.assertNotIn("  argv:", block)
            subprocess.run(["bash", "-n"], input=command(block), text=True, check=True)
        self.assertIn('  trace: "run"', recipe["train"])
        self.assertNotIn("roar", command(recipe["train"]))

    def test_publication_contract_survives_binding(self):
        original = WORKFLOW.read_text()
        for repository in (contract.publication_repository(), "reproducible-ai/harness-test-attempt-abc"):
            text = original.replace(contract.publication_repository(), repository)
            block = stages(text)["publish"]
            self.assertIn("  glaas_creds: true", block)
            self.assertIn('  trace: "off"', block)
            puts = [shlex.split(line) for line in command(block).splitlines() if line.startswith("roar put ")]
            self.assertEqual(len(puts), 1)
            args = puts[0][2:]
            destination = f"hf://{repository}/artifacts/droid-canary/checkpoint-100"
            self.assertEqual(args[:2], [str(contract.CHECKPOINT_PATH), destination])
            self.assertEqual(args[2:6], ["--private", "--yes", "--no-tag", "-m"])
            self.assertEqual(len(args), 7)
            self.assertTrue(args[6].strip())
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = Path(directory) / "bound.yaml"
                path.write_text(text)
                self.assertEqual(contract.publication_repository(path), repository)

    def test_training_schedule_without_running_training(self):
        with patch.object(runner, "cached_snapshot", return_value="/unused/model"), patch.object(
            runner, "validate_placeholder_scaffold"
        ), patch.object(runner.Path, "is_file", return_value=True), patch.object(
            runner.Path, "mkdir"
        ), patch.dict(runner.os.environ, {"HF_TOKEN": "synthetic-test-value"}, clear=True), patch.object(
            runner.subprocess, "run"
        ) as launch:
            runner.main()
        launch.assert_called_once()
        env = launch.call_args.kwargs["env"]
        for key, value in {"NUM_GPUS": "1", "MAX_STEPS": "100", "SAVE_STEPS": "100", "GLOBAL_BATCH_SIZE": "1"}.items():
            self.assertEqual(env[key], value)
        self.assertEqual(contract.EPISODE_COUNT, 3)
        self.assertEqual(contract.DATASET_REVISION, "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4")
        self.assertIn("--no-use-flash-attention", launch.call_args.args[0])

    def test_python_syntax(self):
        paths = list((ROOT / ".treqs/scripts").glob("*.py"))
        paths.append(ROOT / "tests/treqs/test_droid_canary_contract.py")
        for path in paths:
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
