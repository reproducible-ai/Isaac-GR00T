"""CPU checkpoint packaging through the ordinary verifier and receipt writer."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.calibration_droid import load_inputs, sha256_file


SOURCE = Path(__file__).resolve().parents[2]
SCRIPTS = SOURCE / ".treqs/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "package_droid_calibration", SCRIPTS / "package_droid_calibration.py"
)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class CalibrationPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(SOURCE / ".treqs/calibration", self.root / ".treqs/calibration")
        self.plan_path = self.root / ".treqs/calibration/plan.json"
        self.config_path = self.root / ".treqs/calibration/resolved-config.json"
        self.plan = json.loads(self.plan_path.read_bytes())

    def events(self):
        return [
            {
                "event": "point-complete",
                "pointId": point["id"],
                "startedMonotonicSeconds": index * 10000,
                "processExitedMonotonicSeconds": index * 10000 + 50 + point["steps"] * 0.5,
                "timing": {
                    "fullSteps": 10000,
                    "completedSteps": point["steps"],
                    "trainingFinishedMonotonicSeconds": index * 10000 + 10 + point["steps"] * 0.5,
                    "steadyUpdates": point["steps"] - 20,
                    "steadySeconds": (point["steps"] - 20) * 0.5,
                    "checkpointSeconds": 20,
                },
            }
            for index, point in enumerate(self.plan["protocol"]["points"])
        ]

    def test_configuration_edit_is_rejected_before_importing_training_runtime(self):
        with self.config_path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "pinned plan"):
            load_inputs(self.plan_path, self.config_path)

    def test_recipe_metadata_cannot_disagree_with_pinned_runtime(self):
        for field, value in (
            ("precision", "fp32"),
            ("trainableModules", ["language"]),
            ("scheduler", {"name": "linear", "warmupSteps": 500}),
            ("scheduler", {"name": "cosine", "warmupSteps": 5}),
        ):
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(self.plan)
                plan["recipe"][field] = value
                self.plan_path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "recipe"):
                    load_inputs(self.plan_path, self.config_path)

    def test_missing_failed_duplicate_and_changed_schedule_are_rejected(self):
        cases = [
            self.events()[:-1],
            self.events() + [self.events()[0]],
            self.events() + [{"event": "point-failed"}],
        ]
        changed = self.events()
        changed[0]["timing"]["fullSteps"] = 100
        cases.append(changed)
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                package.measured_points(
                    self.plan, events, initial_state_sha="a" * 64, final_sha="b" * 64
                )

    def test_real_checkpoint_readback_produces_one_final_receipt_and_honest_unknown_costs(self):
        from safetensors.torch import save_file
        import torch

        checkpoint = self.root / package.POINTS / "p3/checkpoint-400"
        checkpoint.mkdir(parents=True)
        save_file({"a": torch.ones(2)}, str(checkpoint / "model-00001.safetensors"))
        save_file({"b": torch.zeros(2)}, str(checkpoint / "model-00002.safetensors"))
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}
            )
        )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 400, "log_history": [{"loss": 1.0}]})
        )
        root = self.root / package.ROOT
        (root / "inputs/base").mkdir(parents=True)
        (root / "inputs/base/LICENSE").write_text("Synthetic fixture license\n")
        assets = self.root / ".treqs/assets"
        assets.mkdir()
        (assets / "NVIDIA_OPEN_MODEL_LICENSE.md").write_text("Synthetic fixture notice\n")
        (root / "setup-clock.json").write_text('{"startedUnixSeconds":1}')
        (root / "input-manifest.json").write_text(
            json.dumps(
                {
                    "startedUnixSeconds": 2,
                    "finishedUnixSeconds": 3,
                    "files": [{"input": "base", "sha256": "a" * 64, "sizeBytes": 16}],
                }
            )
        )
        (self.root / package.POINTS / "processes.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in self.events())
        )
        for point in self.plan["protocol"]["points"]:
            (self.root / package.POINTS / f"{point['id']}.json.jsonl").write_text(
                '{"event":"synthetic-point"}\n'
            )
        old = Path.cwd()
        output = io.StringIO()
        try:
            os.chdir(self.root)
            with (
                patch.object(package.subprocess, "check_output", return_value="b" * 40 + "\n"),
                contextlib.redirect_stdout(output),
            ):
                package.main()
        finally:
            os.chdir(old)
        release = self.root / package.RELEASE
        lines = output.getvalue().splitlines()
        artifacts = [
            json.loads(line.removeprefix("E2E_ARTIFACT="))
            for line in lines
            if line.startswith("E2E_ARTIFACT=")
        ]
        results = [
            json.loads(line.removeprefix("E2E_RESULT="))
            for line in lines
            if line.startswith("E2E_RESULT=")
        ]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["optimizerSteps"], 400)
        self.assertTrue(results[0]["loadVerified"])
        measurements = json.loads((release / "calibration.json").read_bytes())
        self.assertEqual(sum(point["completedSteps"] for point in measurements["points"]), 700)
        self.assertIsNone(measurements["overhead"]["finalizationSeconds"])
        self.assertIsNone(measurements["overhead"]["otherCostUsd"])
        self.assertFalse(measurements["coverage"]["fixedWork"])
        self.assertEqual(measurements["rawLogSha256"], sha256_file(release / "timings.json"))
        manifest = json.loads((release / "artifact-manifest.json").read_bytes())
        for entry in manifest["files"]:
            self.assertEqual(entry["sha256"], sha256_file(release / entry["path"]))
        self.assertEqual(
            (release / "resolved-config.json").read_bytes(), self.config_path.read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
