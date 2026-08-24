# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.model.modules.qwen3_backbone import _attention_loading_kwargs


def test_disabled_flash_attention_explicitly_overrides_checkpoint_config():
    assert _attention_loading_kwargs(use_flash_attention=False) == {"attn_implementation": "sdpa"}
