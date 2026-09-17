"""Resolve the explicit DROID fine-tuning scenario before authorizing compute.

Writes the complete ordinary training configuration, not a list of overrides.
Projection pricing is supplied from a separately retained provider quote.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gr00t.configs.base_config import _build_safe_tree, get_default_config
from scripts.calibration_droid import encoded


def build_config():
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": ["artifacts/droid-calibration/inputs/dataset"],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
                    }
                ],
            }
        }
    )
    model = config.model
    for key, value in {
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "use_flash_attention": False,
        "state_dropout_prob": 0.2,
        "color_jitter_params": {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        },
        "use_percentiles": True,
        "load_bf16": False,
        "reproject_vision": False,
        "model_name": "nvidia/Cosmos-Reason2-2B",
        "model_revision": "9ce19a195e423419c349abfc86fd07178b230561",
        "backbone_trainable_params_fp32": True,
        "use_relative_action": True,
    }.items():
        setattr(model, key, value)
    training = config.training
    for key, value in {
        "start_from_checkpoint": "artifacts/droid-calibration/inputs/base",
        "output_dir": "artifacts/droid-calibration/full-scenario",
        "experiment_name": None,
        "optim": "adamw_torch",
        "global_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 4,
        "learning_rate": 1e-4,
        "max_steps": 10000,
        "weight_decay": 1e-5,
        "warmup_ratio": 0.05,
        "save_steps": 1000,
        "save_total_limit": 1,
        "save_only_model": False,
        "resume_from_checkpoint": False,
        "num_gpus": 1,
        "use_wandb": False,
        "calibration_stop_steps": None,
        "calibration_timing_path": None,
        "calibration_warmup_updates": 20,
    }.items():
        setattr(training, key, value)
    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 100000
    config.data.seed = 42
    return config


def resolve_config():
    return _build_safe_tree(asdict(build_config()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = encoded(resolve_config())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
