# Publication correction — iteration 1

## Findings

- publication-incomplete (P1, verified): supplied candidateValidationFailure says
  the sharded checkpoint requires complete directory publication. Inspection
  confirmed the working workflow selected only index, manifest, and result.
  This omitted weight shards and supporting manifest entries.
- publication-directory-restored (decision): supersedes the historical
  publication-source conclusion in iteration-1.md. Restored the sole source to
  artifacts/droid-canary/checkpoint-100. Kept the existing bound destination,
  private/yes/no-tag flags, nonempty message, trace off, and GLaaS credentials
  declaration. No publication command was executed.
- publication-regression-check (decision): offline checks now require the whole
  directory for both the actual and substituted attempt repositories. The pytest
  suite also enforces the directory source and exact option/message shape.
- local-validation-limits (observation): all four dependency-free checks passed.
  Full pytest and pre-commit are unavailable in the local environment. Remote
  training, finite loss, checkpoint readability, budget compliance, and independent
  audit remain unverified and belong to the supervisor. No model quality claim.

## Commands actually run for this correction

- pwd; git status --short; rg --files: inspected workspace and candidate files.
- git diff, cat, sed: read workflow, checks, historical notes, and training recipe.
- python3 heredoc: edited workflow, documentation, and publication assertions.
- python3 .treqs/scripts/check_candidate_local.py: PASS, four checks; transcript
  publication-fix-checks.txt. Shell syntax only; training subprocess and snapshot
  lookups mocked with a synthetic environment value; no credentials used.
- git diff --check: PASS.
- python3 -m pytest -q tests/treqs/test_droid_canary_contract.py: unavailable,
  /usr/local/bin/python3: No module named pytest.
- pre-commit run --all-files: unavailable, command not found.

## Parameters

No training parameters changed in this correction. Existing differences from
examples/finetune.sh remain documented in iteration-1.md: 100 versus 10000 steps,
100 versus 1000 save interval, batch 1 versus 32, workers 0 versus 4, shard size
8 versus 1024, shards/epoch 8 versus 100000, sampling 1.0 versus 0.1, three pinned
DROID episodes, external logging off, and Flash Attention disabled. One GPU and
BF16 remain unchanged. These choices do not restore the full schedule.

Supervisor handoff: request the declared workflow with the $14.16 total NTE
ceiling, then retrieve the complete checkpoint and arrange the required
independent audit. No remote actions or model audit performed locally.
