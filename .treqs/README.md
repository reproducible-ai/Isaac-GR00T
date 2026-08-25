# Reproducible DROID canary

This tracer bullet proves that the pinned Isaac GR00T N1.7 source, model inputs,
and three-episode DROID sample can complete one optimizer step on the existing
Reproducible AI L40S target.

## Immutable inputs

- Fork base commit: `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Base model: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- VLM backbone: `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`
- Dataset: `lerobot/droid_1.0.1@0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`
- Python dependencies: the committed `uv.lock`, installed without replacing the
  checked-out `gr00t` source with a separately built project package
- Lineage runtime: `roar-cli==0.4.4` with the pinned `preload` tracer
- Compute target: `5ad26838-4267-402d-b8aa-0bd271041be3`
- Required target secret: `HF_TOKEN`, explicitly declared by name in the workflow

`HF_TOKEN` needs read access to both NVIDIA model repositories. The workflow
uses the Hugging Face SDK only to read the pinned upstream inputs; it does not
upload a model or synchronize metrics to a Hugging Face Space.

The workflow checks access to both model revisions before transferring model
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
4. `label` attaches model, version, license, description, and documentation
   metadata to every model-weight shard locally.

All workflow stages use `trace: off`; the three workload stages invoke
`roar run -n ...` explicitly so the captured commands and tracer ABI are stable.
This canary deliberately stops before artifact publication.

```bash
roar reproduce <lineage-hash> --lineage --run
```

## Canary contract

The canary succeeds only if it:

1. sees one CUDA GPU with at least 40 GiB VRAM and FFmpeg major version 4-7;
2. downloads the exact base-model, gated-backbone, and dataset revisions;
3. converts exactly three DROID episodes;
4. performs exactly one optimizer step without tuning the LLM or visual encoder;
5. writes a step-1 checkpoint whose safetensors shards can be opened and an
   `artifacts/droid-canary/checkpoint-1/evaluation.json` record containing their
   hashes and tensor metadata;
6. labels the model weights locally without uploading them.

The setup, fetch, and train commands have an aggregate timeout below two hours,
keeping the run below the approved $5 cap at the target's observed hourly rate.
This is a training-path canary, not a quality or convergence claim.
