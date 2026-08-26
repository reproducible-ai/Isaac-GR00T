---
license: other
license_name: nvidia-license
license_link: https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/LICENSE
base_model: nvidia/GR00T-N1.7-3B
base_model_relation: finetune
datasets:
- lerobot/droid_1.0.1
tags:
- robotics
- vision-language-action
- reproducible-ai
---

# GR00T N1.7 DROID reproducibility canary

This is a one-optimizer-step fine-tuning canary for NVIDIA Isaac GR00T N1.7.
It demonstrates that the pinned source, models, three-episode DROID sample, and
Reproducible AI workflow can produce and verify a readable checkpoint. It is not
a quality, convergence, or deployment-safety claim.

## Reproducible inputs

- Source: [`reproducible-ai/Isaac-GR00T@{{SOURCE_COMMIT}}`](https://github.com/reproducible-ai/Isaac-GR00T/tree/{{SOURCE_COMMIT}})
- Base model: [`nvidia/GR00T-N1.7-3B@{{BASE_MODEL_REVISION}}`](https://huggingface.co/nvidia/GR00T-N1.7-3B/tree/{{BASE_MODEL_REVISION}})
- VLM backbone: [`nvidia/Cosmos-Reason2-2B@{{BACKBONE_MODEL_REVISION}}`](https://huggingface.co/nvidia/Cosmos-Reason2-2B/tree/{{BACKBONE_MODEL_REVISION}})
- Dataset: [`lerobot/droid_1.0.1@{{DATASET_REVISION}}`](https://huggingface.co/datasets/lerobot/droid_1.0.1/tree/{{DATASET_REVISION}}) (three episodes)
- Training: one optimizer step, global batch size 1
- Release path: `{{PUBLICATION_VERSION}}`

The checkpoint's `evaluation.json` records the step count, training log, byte
size, SHA-256 digest, tensor count, and first tensor shape for each weight shard.

## License and safety

The checkpoint is distributed under the NVIDIA License copied into `LICENSE`.
That license restricts this work and derivatives to non-commercial research or
evaluation. The upstream GR00T and Cosmos notices are retained under `upstream/`;
Cosmos attribution and its separate license are included in `NOTICE` and
`NVIDIA_OPEN_MODEL_LICENSE.md`.

This model is not tested or intended for mission-critical applications that
require functional safety. Users are responsible for use-case-specific testing,
guardrails, and compliance before deployment.
