# Training operator iteration 1

Source packet: 9124d64bc9b19f118285939e8ffd29ce5f094733.

- binding-tests (verified issue, fixed): four assertions depended on the literal
  placeholder repository, and the replacement test assumed that placeholder was
  still present. All now derive the destination from the actual workflow.
- publication-order (verified issue, fixed): the publication destination preceded
  option arguments. It is now the final argument. The complete checkpoint
  directory remains the sole source; private flags and one message are retained.
- budget-docs (verified issue, fixed): README stated $15 and an older source base.
  It now records the packet's $5 ceiling and source commit. The supervisor must
  enforce total spend; elapsed-time limits do not guarantee dollar cost.
- local-validation (observation): three dependency-free checks passed, covering
  original and rebound publication destinations, all seven shell blocks, Python
  syntax, and the training launch with subprocess and snapshot access mocked.
  Full pytest and pre-commit could not start because tools are absent.
- remote-evidence (decision): request one supervised run under $5. No optimizer
  steps or artifact quality have been verified here. Independent audit remains
  the supervisor's responsibility after execution.

Recipe parameters are preserved: 100 steps versus examples/finetune.sh's 10000;
1 batch versus 32; 1 GPU unchanged; BF16 unchanged; save at 100 versus 1000;
3 pinned DROID episodes; workers 0 versus 4; shard size 8 versus 1024;
shards per epoch 8 versus 100000; episode sampling 1.0 versus 0.1;
W&B disabled versus enabled; flash attention disabled; save-only-model enabled.
These are existing canary reductions, not newly changed training settings.
No full reproduction or convergence claim is made. A larger schedule needs a
separate decision. This iteration only changes tests, publication argument order,
and documentation.

Commands and outcomes: see iteration-1-checks.txt. Inspection used pwd, rg, cat,
sed, head, ls, command -v, git status/diff, and Python import discovery. File
updates used local Python and shell heredocs. No external actions were executed.
