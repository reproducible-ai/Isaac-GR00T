"""Real Transformers training on CPU verifies the calibration callback's integration."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class CalibrationTrainerTests(unittest.TestCase):
    def test_optimizer_updates_stop_early_with_full_scheduler_and_real_checkpoint(self):
        from gr00t.experiment.calibration import make_calibration_callback
        import torch
        from transformers import Trainer, TrainingArguments

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, input_ids, labels):
                prediction = self.linear(input_ids.float())
                return {"loss": (prediction - labels).square().mean(), "logits": prediction}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = SimpleNamespace(
                calibration_stop_steps=4,
                calibration_timing_path=str(root / "timing.json"),
                calibration_warmup_updates=2,
                max_steps=10000,
                num_gpus=1,
                resume_from_checkpoint=False,
                eval_strategy="no",
                save_steps=1000,
                enable_profiling=False,
            )
            args = TrainingArguments(
                output_dir=str(root / "checkpoints"),
                max_steps=10000,
                warmup_steps=500,
                save_steps=1000,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                logging_steps=1,
                use_cpu=True,
                report_to=[],
                disable_tqdm=True,
            )
            dataset = [
                {"input_ids": torch.tensor([1.0, 2.0]), "labels": torch.tensor([3.0])}
                for _ in range(32)
            ]
            trainer = Trainer(model=TinyModel(), args=args, train_dataset=dataset)
            trainer.add_callback(make_calibration_callback(training, synchronize=lambda: None))
            trainer.train()
            self.assertEqual(trainer.state.global_step, 4)
            self.assertEqual(trainer.state.max_steps, 10000)
            self.assertEqual(args.get_warmup_steps(args.max_steps), 500)
            checkpoint = root / "checkpoints/checkpoint-4"
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())
            state = json.loads((checkpoint / "trainer_state.json").read_bytes())
            self.assertEqual(state["global_step"], 4)
            self.assertEqual(state["max_steps"], 10000)
            timing = json.loads((root / "timing.json").read_bytes())
            self.assertEqual(timing["completedSteps"], 4)
            self.assertGreater(timing["steadySeconds"], 0)
            self.assertGreater(timing["checkpointSeconds"], 0)

    def test_executable_resolved_configuration_matches_committed_recipe(self):
        from scripts.calibration_droid import encoded, load_inputs
        from scripts.resolve_droid_calibration import build_config, resolve_config

        root = Path(__file__).resolve().parents[2]
        plan, _, data = load_inputs(
            root / ".treqs/calibration/plan.json", root / ".treqs/calibration/resolved-config.json"
        )
        self.assertEqual(encoded(resolve_config()), data)
        config = build_config()
        config.validate()
        self.assertEqual(config.training.max_steps, plan["recipe"]["fullSteps"])
        self.assertEqual(config.training.warmup_ratio * config.training.max_steps, 500)
        self.assertIsNone(config.training.calibration_stop_steps)


if __name__ == "__main__":
    unittest.main()
