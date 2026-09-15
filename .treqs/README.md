# Reproducible DROID 100-step public canary

This candidate is designed to check whether the pinned Isaac GR00T N1.7 source, model inputs,
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
to `reproducible-ai/gr00t-n1-7`, the public destination pinned in this workflow.
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
   contents one tensor at a time on CPU, and writes its evaluation
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
7. publishes the verified checkpoint to the public Hugging Face repository and
   registers attributed public GLaaS lineage.

The supervisor must enforce a $5 NTE ceiling across provisioning and all stages;
stage timeouts alone do not enforce a dollar budget. No retry allowance is authorized. The one-hour training timeout bounds the paid training stage.
This is a training-path canary, not a quality or convergence claim.

## Harness evidence contract

Workload stages use TReqs `trace: run` around ordinary commands. Setup, labeling,
and publication remain untraced orchestration. Packaging emits exactly one
`E2E_ARTIFACT` and one `E2E_RESULT` receipt and writes `result.json` beside the
manifest. The declared artifact is `model.safetensors.index.json`; the manifest
includes all checkpoint shards and supporting files. Success requires exactly
100 optimizer steps, finite loss, immutable anonymous HF read-back, canonical
lineage verification, and consistent terminal TReqs authorities.

The access checker and publication metadata read the actual workflow destination
so per-attempt binding also changes the repository checked before downloads.
This recipe never uploads to the public notes repository.

## Unattended public canary

The host supervisor is `.treqs/scripts/public_canary_supervisor.py`. It uses the
existing TReqs CLI and the pinned harness verification helpers; it does not run
an LLM or modify the private campaign attempt. Human authorization for the public
run and its $5 compute budget is recorded in the external launch plan.

The supervisor creates one TReqs request with `--lineage-mode public`, queues it,
monitors the full allocation cost, waits for automatic idle shutdown, checks all
seven tasks, verifies anonymous HF downloads and public training-to-PUT lineage,
and adds `PUBLIC-RELEASE.md` plus evidence to notes PR #18. The original private
capture and audit stay in that PR as historical evidence.

The plan pins source and harness commits, target/AMI, repository, source branch,
and an isolated bound control checkout. It has `budgetUsd: 5`, `stopAtUsd: 4`,
`conservativeHourlyUsd: 4`, and `maxJobSeconds: 3000`. The hourly rate is a
conservative monitoring estimate above the prior allocation's effective rate.
The target has a 15-minute idle shutdown; the early stop reserves shutdown cost.
This is a supervised stop policy, not a provider-enforced dollar limit. It needs
host/API availability; launchd restarts crashes and caffeinate prevents idle sleep.

Run the frozen host environment under a user launchd service:

```bash
/path/to/host/python .treqs/scripts/public_canary_supervisor.py --plan /absolute/run/plan.json
```

The service persists `state.json`, `events.jsonl`, `workload.jsonl`, anonymous
verification receipts, and the exact notes commit beside the plan. `complete`
means compute is stopped, every public check passed, and the notes PR update was
pushed and read back. `failed` includes a reason and never launches another paid
job. A lost create/queue reply is reconciled by request title and request ID;
an ambiguous operation is not repeated. Re-running the same plan resumes it.
Do not delete its state or edit its immutable plan to retry a paid run.

Read `state.json` for status or `events.jsonl` for transitions. Operational TReqs
commands run only from the isolated control checkout. Post-run verification and
notes publication retry for up to three hours, without additional GPU jobs.
Safetensors shards are downloaded and checked one at a time to limit disk use;
the byte hashes, sizes, and tensor inventory are retained in the public receipt.

Host regression checks:

```bash
python -m pytest -q tests/treqs/test_public_canary_supervisor.py
python -m unittest discover -s tests/treqs -p test_droid_canary_offline.py -v
```

The worker also runs `tests/treqs/test_droid_canary_contract.py` against its pinned
training environment before downloading the model inputs.
