# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.base_config import get_default_config
from gr00t.model.gr00t_n1d7 import setup as pipeline_setup_module
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.modules.qwen3_backbone import _attention_loading_kwargs
import torch


def test_disabled_flash_attention_explicitly_overrides_checkpoint_config():
    assert _attention_loading_kwargs(use_flash_attention=False) == {"attn_implementation": "sdpa"}


def test_checkpoint_load_preserves_disabled_flash_attention(monkeypatch, tmp_path):
    config = get_default_config()
    config.training.start_from_checkpoint = "checkpoint"
    config.training.skip_weight_loading = False
    config.model.use_flash_attention = False
    captured_kwargs = {}

    class FakeConfig:
        @staticmethod
        def to_filtered_json():
            return "{}"

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = FakeConfig()

    def fake_from_pretrained(_checkpoint, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeModel(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    monkeypatch.setattr(pipeline_setup_module.AutoModel, "from_pretrained", fake_from_pretrained)

    pipeline = Gr00tN1d7Pipeline(config, tmp_path)
    pipeline._create_model()

    assert captured_kwargs["use_flash_attention"] is False
