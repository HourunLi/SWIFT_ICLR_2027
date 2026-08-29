# Consistency v6.1 C0/C1 matched replication

This directory freezes one mechanism-only comparison on the expanded
Consistency data.

- `c0_correctness_only.json`: final-correctness BCE only.
- `c1_consistency.json`: the same model, data, batches, optimizer, seeds and
  epochs, with `consistency_weight=1.0`.
- seeds: 42, 43, 44 (CLI seed override only).
- epochs: 3. Seed 42 first runs one epoch as an execution pilot and then resumes
  the same full-state checkpoint to epoch 3 after the finite/runtime gate.

The shared train view contains the historical 3,968 correctness rows plus 400
new two-view Consistency relations (800 trajectories). Historical Consistency,
H and Prior annotations are omitted from the constructed view. New relation
views receive `relative_compact` / `relative_expanded` style IDs solely from
their saved output-token lengths; this supports the existing positive and
same-style in-batch negative loss without adding AI labels.

The frozen 150 positive and 150 hard-negative relations are evaluation-only.
This experiment can show whether C1 learns the held-out relation geometry. It
cannot establish Best-of-N or reward-ranking improvement because no new
independent ranking population is part of this stage.
