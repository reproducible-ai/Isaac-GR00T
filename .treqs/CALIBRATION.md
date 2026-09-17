# GR00T N1.7 DROID calibration

## Restricted operator checks

The harness operator has no network access. Run these standard-library commands from the frozen candidate checkout:

```bash
python3 -S -c 'from scripts.calibration_droid import load_inputs; load_inputs(".treqs/calibration/plan.json", ".treqs/calibration/resolved-config.json"); print("PASS: actual runtime recipe checks")'
python3 -S -m unittest tests.treqs.test_calibration_timing tests.treqs.test_calibration_process tests.treqs.test_calibration_storage
```

Retain check output and temporary files only under the ignored `artifacts/operator-checks/` directory. Compare the plan SHA to the task packet and inspect the prepared workflow. After these checks pass, return the frozen candidate with a `treqs-run` request for the supervisor. The complete dependency-bearing tests below belong to the supervisor/CI and worker environment; do not attempt package downloads or unittest discovery of those tests inside the restricted operator. Report the local subset honestly and do not claim GPU fit or a completed run.

## Recipe and evidence

The private workflow is `.treqs/workflows/droid-calibration.yaml`. It collects three
independent 100/200/400-update points on one 96 GB RTX PRO 6000 Blackwell GPU. The
reviewed full scenario is **10,000-update DROID fine-tuning**, not original model
pretraining: batch 32, BF16, frozen language/vision backbone, trainable projector and
diffusion model, cosine scheduling with 500 warmup updates, and a complete checkpoint
every 1,000 updates. It uses the first 32 pinned DROID episodes and both model cameras.
This data selection bounds the scenario; it does not establish performance across
the full DROID distribution or model quality.

`.treqs/calibration/resolved-config.json` contains the complete configuration produced
by `scripts/resolve_droid_calibration.py`. The runtime rebuilds that typed configuration
and checks its exact serialized bytes before training. `.treqs/calibration/plan.json`
binds it by SHA-256 and declares the measurement/projection assumptions. Changing the
recipe requires regenerating and reviewing those pins and the harness task digest.

Each point starts a fresh Python process and loads the same base weights, optimizer,
full scheduler and seed. Input bytes are checked and pre-read before each point;
compiler caches start empty in distinct directories. The callback stops training
without changing the full scheduler, synchronizes CUDA before recording update times,
then requests the final checkpoint. The parent includes process startup in the fit
interval. The first 20 updates are excluded from the separate steady-state average.
Periodic save/evaluation crossings, resume, multi-GPU execution and profiling are
rejected by this first adapter.

Raw journals survive incomplete points. Flushed `CALIBRATION_EVENT=` stdout records
preserve point start, completion and failure diagnostics in the host log capture. Nonzero exits, timeouts and termination
signals fail the workload and stop the training process group. Successful packaging
reads every final safetensors tensor, checks trainer state and finite loss, and emits
exactly one `E2E_ARTIFACT` / `E2E_RESULT` pair for the **400-update** checkpoint. The
700 total measured updates are independent points, not one accumulated checkpoint
or separately billed jobs.

The release includes optimizer/scheduler state, raw timing/process records, resolved
configuration, input hashes, a plan-bound trainer receipt, licenses and the usual
artifact manifest. It publishes privately to the harness-bound HF destination.
Existing public model releases are separate.

## Accounting and deployment

The worker leaves cold allocation setup, completed upload/shutdown and non-compute
charges unknown. Their full evidence exists only after allocation reconciliation;
the host must retain and bind that evidence before a numeric all-in estimate can be
published. Without it, the current contract yields an honest null estimate. The
plan's $4/allocation-hour value is a conservative allowance based on the retained
Ohio EC2 price of $3.36312/hour plus disk/IPv4 allowance, not a finalized charge.
The plan also opts into allocation reconciliation: the host validates worker/process
clocks against the completed training task and allocation boundaries, derives setup
and finalization once, and applies a conservative $0.15/GiB transfer allowance to the
verified release inventory. This forecast is separate from actual finalized billing.
It requires the harness reader from PR #10 (or its merged successor).

Launch through a pinned harness calibration task and scoped policy with a separate
all-in budget, shutdown reserve, explicit notes baseline and draft-PR publication
choice. This workflow alone does not enforce a dollar ceiling or authorize a launch.
The live acceptance checkpoint remains the GPU run, artifact readback, audit,
shutdown/cost reconciliation and automatic notes PR.

## Supervisor and CI validation

The supervising maintainer runs these checks before authorizing a source pin, with the pinned training dependencies available. The full calibration/loader CPU suite passed for the reviewed implementation; worker setup independently runs its prepared checks before training:

```sh
python -m unittest discover -s tests/treqs -p 'test_calibration_*.py' -v
python -m unittest tests.treqs.test_offline_processor
pytest -q tests/treqs/test_droid_canary_contract.py \
  tests/gr00t/configs/test_batch_size_invariant.py \
  tests/gr00t/configs/test_base_config_safe_yaml.py \
  tests/gr00t/experiment/test_resume_compatibility.py
```

Tests include real CPU Transformers training with the early-stop callback and real
safetensors checkpoint packaging. They do not establish GPU memory fit or throughput.

Child stdout/stderr is streamed into host-captured workload logs and retained in each point log. Workflow path additions preserve injected Python bootstrap paths.


## Packaging diagnostics and disk readiness

Packaging records each phase and file-copy start/completion in
`artifacts/droid-calibration/package-events.jsonl` and emits single-line
`CALIBRATION_PACKAGE_EVENT=` records to stdout. Failures retain their phase,
exception type/message and errno, print the original traceback to stdout, and
exit unsuccessfully without emitting an artifact/result success pair. A journal
write/close failure does not replace the original packaging exception. Existing
release directories and journals are never overwritten.

Before copying, the packager compares available space on the release filesystem
with the measured complete checkpoint size plus a 64 MiB metadata reserve. This
includes optimizer/scheduler state as well as model weights. It is a point-in-time
check; concurrent disk writers can still exhaust space afterward. The host also
needs enough free space for immutable artifact readback across concurrent runs.
These records improve diagnosis but do not guarantee delivery of a failed job's
last log chunk by the deployed compute agent.


After a successful nonfinal point's process/timing receipt is flushed and synced,
the wrapper removes that point's generated weight/checkpoint directory before
launching the next point. Its timing JSON, raw journal and stdout/stderr log remain
outside that directory. Failed-point files and the entire final-point directory
are retained. This reduces peak training/package storage without changing the
independent point resets, full scheduler, measured checkpoint durations or final
artifact contract.
