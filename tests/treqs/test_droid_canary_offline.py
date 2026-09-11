"""Dependency-free local checks; no downloads, credentials, or GPU execution."""

import ast
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".treqs/scripts"
WORKFLOW = ROOT / ".treqs/workflows/droid-canary.yaml"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CanaryOfflineTests(unittest.TestCase):
    def test_shell_syntax_and_publication_binding(self):
        contract = load_script("droid_canary_contract")
        text = WORKFLOW.read_text()
        commands = []
        for line in text.splitlines():
            if line == "  command: |":
                commands.append([])
            elif line.startswith("    "):
                commands[-1].append(line[4:])
        self.assertEqual(len(commands), 7)
        for command in commands:
            subprocess.run(["bash", "-n"], input="\n".join(command), text=True, check=True)
        original = contract.publication_repository()
        for repository in (original, "reproducible-ai/another-attempt-123"):
            bound = text.replace(original, repository)
            with tempfile.TemporaryDirectory(dir=ROOT / ".treqs/evidence") as directory:
                workflow = Path(directory) / "workflow.yaml"
                workflow.write_text(bound)
                self.assertEqual(contract.publication_repository(workflow), repository)
            puts = [shlex.split(line) for line in bound.splitlines()
                    if line.strip().startswith("roar put ")]
            self.assertEqual(len(puts), 1)
            args = puts[0]
            self.assertEqual(args[2], "artifacts/droid-canary/checkpoint-100")
            self.assertEqual(args[-1], f"hf://{repository}/artifacts/droid-canary/checkpoint-100")
            for flag in ("--private", "--yes", "--no-tag", "-m"):
                self.assertEqual(args.count(flag), 1)
            self.assertTrue(args[args.index("-m") + 1].strip())
            self.assertFalse({"--public", "--anonymous"}.intersection(args))

    def test_training_schedule_without_launching(self):
        contract = load_script("droid_canary_contract")
        run = load_script("run_droid_canary")
        with patch.dict(run.os.environ, {"HF_TOKEN": "synthetic-test-value"}, clear=True), \
             patch.object(run, "cached_snapshot", return_value="/unused/model"), \
             patch.object(run, "validate_placeholder_scaffold"), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "mkdir"), \
             patch.object(run.subprocess, "run") as launch:
            run.main()
        launch.assert_called_once()
        env = launch.call_args.kwargs["env"]
        for key, value in {"NUM_GPUS": "1", "GLOBAL_BATCH_SIZE": "1",
                           "MAX_STEPS": "100", "SAVE_STEPS": "100", "USE_WANDB": "0"}.items():
            self.assertEqual(env[key], value)
        self.assertEqual(contract.EPISODE_COUNT, 3)
        self.assertEqual(contract.DATASET_REVISION, "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4")
        self.assertIn("--save-only-model", launch.call_args.args[0])
        config = (ROOT / "gr00t/configs/training/training_config.py").read_text()
        self.assertIn("bf16: bool = True", config)

    def test_candidate_python_syntax(self):
        paths = list(SCRIPTS.glob("*.py")) + list((ROOT / "tests/treqs").glob("*.py"))
        for path in paths:
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
