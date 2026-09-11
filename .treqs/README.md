# Reproducible DROID 100-step private canary

This workflow is designed to test whether the pinned Isaac GR00T N1.7 source, model inputs,
and three-episode DROID sample can complete 100 optimizer steps on the
Reproducible AI 96 GB RTX PRO 6000 Blackwell target.

## Immutable inputs

- Fork base commit: `9124d64bc9b19f118285939e8ffd29ce5f094733`
- Base model: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- VLM backbone: `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`
- Dataset: `lerobot/droid_1.0.1@0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`
- Python dependencies: the committed `uv.lock`, installed without replacing the
  checked-out `gr00t` source with a separately built project package
- Lineage runtime: `roar-cli==0.4.5` with the pinned `preload` tracer
- Compute target: `c33842f4-f374-4ca5-845f-7e0c0dd502f7`
- Required target secret: `HF_TOKEN`, explicitly declared by name in the workflow

`HF_TOKEN` needs read access to both NVIDIA model repositories and write access
to the private attempt repository bound into the workflow by the harness.
The unbound `harness-test-pending` destination is a placeholder; do not launch it directly.
Workload scripts use the Hugging Face SDK only to read pinned upstream inputs;
publication is handled by `roar put`, and no metrics are synchronized to a
Hugging Face Space.

Before installing the multi-gigabyte environment, setup checks that the token can
read both pinned model revisions and see the pre-created publication repository.
The check uses Hugging Face's non-mutating pre-upload negotiation endpoint to
verify write permission without creating a file or commit. The same check can be
run by itself with `.treqs/workflows/hf-access-preflight.yaml` before launching
the paid canary.
The fetch step checks model access again immediately before transferring model
bytes, then downloads the snapshots into an ephemeral Hugging Face cache. The
DROID converter records hashes for every generated sample file in
`artifacts/droid-canary/input-manifest.json`.
It also materializes the lockfile's architecture-specific Git LFS wheel before
asking `uv` to validate the cross-platform lock.

## Reproducible AI pipeline

The paid workload is one clean, named ROAR DAG:

1. `fetch_droid` downloads and converts the pinned inputs;
2. `train` performs the bounded optimizer step with external experiment logging
   disabled;
3. `evaluate` opens every generated safetensors shard, validates its tensor
   metadata without materializing the full model, and writes its evaluation
   record inside the checkpoint directory;
4. `package` copies the pinned upstream license and safety notices, adds the
   required Cosmos attribution, and writes a reproducibility model card;
5. `label` attaches model, version, license, description, and documentation
   metadata to every model-weight shard locally;
6. `publish` uses one broker-scoped operation to upload the checkpoint, including
   its model card and license notices, to
   the bound repository under `artifacts/droid-canary/checkpoint-100`.

The four workload stages use TReqs `trace: run`; setup, labeling and publication
use `trace: off`. Workload commands contain no nested tracer wrappers.

```bash
roar reproduce <lineage-hash> --lineage --run --no-puts
```

## Canary contract

The canary succeeds only if it:

1. sees one CUDA GPU with at least 40 GiB VRAM and FFmpeg major version 4-7;
2. downloads the exact base-model, gated-backbone, and dataset revisions;
3. converts exactly three DROID episodes;
4. performs exactly 100 optimizer steps without tuning the LLM or visual encoder
   and records a finite loss;
5. writes a step-100 checkpoint whose safetensors index names every shard and an
   `artifacts/droid-canary/checkpoint-100/evaluation.json` record containing
   their hashes and tensor metadata;
6. packages the license, notices, model card, and complete file inventory with
   the checkpoint;
7. publishes the verified checkpoint to the pre-created private Hugging Face
   repository while keeping GLaaS lineage private.

The attempt has a $5 NTE ceiling, including provisioning. The supervisor must
enforce that ceiling across the entire run; stage timeouts alone do not bound
cost. No automatic retry or larger schedule is authorized. The one-hour
training timeout is an additional upper bound, subject to the $5 ceiling.
This is a training-path canary, not a quality or convergence claim.

## Harness evidence contract

Workload stages use TReqs `trace: run` around ordinary commands. Setup, labeling,
and publication remain untraced orchestration. Packaging emits exactly one
`E2E_ARTIFACT` and one `E2E_RESULT` receipt and writes `result.json` beside the
manifest. The declared artifact is `model.safetensors.index.json`; the manifest
includes all checkpoint shards and supporting files. Success requires exactly
100 optimizer steps, finite loss, immutable private HF read-back, canonical
lineage verification, and consistent terminal TReqs authorities.

The access checker and publication metadata read the actual workflow destination
so per-attempt binding also changes the repository checked before downloads.
This recipe never uploads to the public notes repository.
