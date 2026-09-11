# Local candidate preparation, iteration 1

Historical preparation notes below. The publication-source conclusion is incorrect
and superseded by publication-fix.md; its check transcript predates the correction.

Source supplied by task: 9124d64bc9b19f118285939e8ffd29ce5f094733.
No compute, API calls, credential use, publication, or model audit performed.

Verified findings:
- Publication used a directory source; changed to the three explicit contract
  sources (index, artifact-manifest.json, result.json).
- Four existing assertions and a binding fixture assumed the placeholder repo.
  They now derive the repository from the actual workflow.
- README had a $15 ceiling and an older source commit. Aligned to the task's
  $14.16 ceiling and source. Supervisor must enforce total spend; timeouts do
  not establish cost compliance.

Validation commands:
- `python3 .treqs/scripts/check_candidate_local.py`: four tests pass; transcript
  in iteration-1-checks.txt. Training launch is mocked; shell commands receive
  syntax checks only. No real secrets are accessed by these checks.
- `git diff --check`: passed.
- `python3 -c 'import pytest, yaml, torch, huggingface_hub; print("test dependencies available")'`:
  failed at pytest import (ModuleNotFoundError); remaining imports untested.
- `pre-commit run --all-files`: unavailable (command not found).
- Local reads used pwd, git status, rg, cat, sed, and ls; no .venv exists.
- File edits used a local Python script and apply_patch; no commit created.

The workflow's existing pytest suite remains a remote setup gate. Remote
training viability and checkpoint readability are unverified. The independent
audit remains the supervisor's responsibility.

Recipe comparison (examples/finetune.sh defaults -> existing canary choices;
no training parameter was changed in this iteration):
- max steps 10000 -> 100; save interval 1000 -> 100.
- global batch 32 -> 1; GPUs 1 -> 1; BF16 true -> true (training config default).
- workers 4 -> 0; shard size 1024 -> 8; shards/epoch 100000 -> 8.
- episode sampling 0.1 -> 1.0; dataset restricted to three pinned episodes.
- Flash attention disabled by the existing canary launcher.
- Learning rate 1e-4, warmup ratio 0.05, weight decay 1e-5 unchanged.
These choices implement a viability canary and do not restore a full run.
