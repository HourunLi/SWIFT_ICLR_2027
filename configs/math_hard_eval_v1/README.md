# MATH hard protected evaluation v1

This is a one-shot external evaluation of the already frozen 57-checkpoint
CLIR grid. It uses 500 official MATH test questions: 250 Level 4 and 250
Level 5, each with 16 Phi-3.5-mini candidates. No training, checkpoint
selection, weight tuning, threshold tuning, or post-result subset selection is
allowed.

Before the first test-row access, `protocol.json`, all runner code, and tests
must be committed with a clean worktree. Selection is based only on level,
predeclared Asymptote/reference/prompt validity rules, train-template exclusion,
and a stable hash. Candidate correctness and CLIR scores cannot remove a query.

The checker is the expression-equivalence implementation from the official
SWIFT repository pinned at commit `41f7c9f7`. This makes the answer semantics
closer to SWIFT than the earlier numeric-only internal evaluations. Published
SWIFT accuracies are still not directly comparable because the generator,
training population, difficulty mix, and maximum candidate count differ.

The execution order is:

1. `prepare_clir_math_hard_eval.py materialize`, then `verify`.
2. `run_clir_math_hard_eval_rollout.py rollout` for `hard-000` first, then the
   other nine shards; verify and merge.
3. `check_clir_math_hard_eval.py fetch`, `materialize`, then `verify`.
4. `extract_clir_math_hard_eval_features.py prepare`, `verify-plan`,
   `preflight`, eight extraction workers, eight verification workers, then
   `finalize`.
5. `score_clir_math_hard_eval.py worker` on eight GPUs, merge, then run
   `summarize_clir_math_hard_eval.py` exactly once.

Large protected rows, rollouts, features, scores, and reports stay under
`run_artifacts/math_hard_eval_v1` and are not committed.
