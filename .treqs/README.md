# Reproducible DROID canary

This tracer bullet proves that the pinned Isaac GR00T N1.7 source, model inputs,
and three-episode DROID sample can complete one optimizer step on the existing
Reproducible AI L40S target.

## Immutable inputs

- Fork base commit: `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Base model: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- VLM backbone: `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`
- Dataset: `lerobot/droid_1.0.1@0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`
- Python dependencies: the committed `uv.lock`, installed with `uv sync --locked`
- Compute target: `e4d609eb-db96-40d2-bc74-7e13d6e75e8b`
- Required target secret: `HF_TOKEN`, explicitly declared by name in the workflow

The workflow downloads model snapshots by immutable revision into an ephemeral
Hugging Face cache. The DROID converter records hashes for every generated
sample file in `artifacts/droid-canary/input-manifest.json`.
It also materializes the lockfile's architecture-specific Git LFS wheel before
asking `uv` to validate the cross-platform lock.

## Canary contract

The canary succeeds only if it:

1. sees one CUDA GPU with at least 40 GiB VRAM and FFmpeg major version 4-7;
2. downloads the exact base-model, gated-backbone, and dataset revisions;
3. converts exactly three DROID episodes;
4. performs exactly one optimizer step without tuning the LLM or visual encoder;
5. writes a step-1 checkpoint and `artifacts/droid-canary/result.json`.

The setup, fetch, and train commands have an aggregate timeout below two hours,
keeping the run below the approved $5 cap at the target's observed hourly rate.
This is not a quality or convergence run, and it does not publish a model.
