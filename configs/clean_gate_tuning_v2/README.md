# Clean prior-to-gate weight tuning v2

Status: frozen before inspecting any new `0.25 / 1 / 4 / 10` training,
mechanism, or ranking metrics. The user has made a method-identity decision that
the original `main` shared-gradient prior-to-reward-gate path must remain
enabled by default. This protocol selects an engineering default on the current
development population; it is not an independent efficacy test.

## Fixed implementation

Every cell keeps the original `main` coupling unchanged:

```text
gate_attention = normalize(sigmoid(gate_logits))
fused_prior = normalize(0.5 * key_prior + 0.5 * complete_prior).detach()
L_gate = squared-L2(gate_attention, fused_prior) on shared prior coverage
```

The same SWIFT-style gate continues to weight token reward in the scalar score.
Only `gate_prior_weight` changes. There is no KL replacement, head-only routing,
runtime score fusion, gradient rescaling, or architecture change.

## Cells and reused anchors

The new logarithmic grid is:

| cell | `gate_prior_weight` | rationale |
|---|---:|---|
| `g025_main_inner` | `0.25` | exact inner coefficient declared by `origin/main`; clean outer prior weight is `1` |
| `g100_balanced` | `1.0` | gate loss has the same outer coefficient as each direct prior loss |
| `g400_intermediate` | `4.0` | log-scale bridge toward the historical strong setting |
| `g1000_historical_strong` | `10.0` | historical strong shared-gradient calibration point |

The already completed `P0=0` and `PG0=.0625` artifacts are reused as immutable
anchors. They are not retrained because the three rerun P0 checkpoints were
already tensor-for-tensor identical to the original clean P0 runs.

## Frozen data and budget

- train: `run_artifacts/joint_training_pilot_v1/data_v1/train3968_joint_v1.jsonl`;
- mechanism dev: `run_artifacts/joint_training_pilot_v1/data_v1/mechanism_dev16_joint_v1.jsonl`;
- ranking development pool: `run_artifacts/stage1b_v2/manifests/validation_extracted.v4.local.jsonl`;
- seeds: `42,43,44`;
- epochs: `3` for every new cell;
- output root: `run_artifacts/clean_gate_tuning_v2/`;
- P0/PG0 anchors: `run_artifacts/clean_gate_ablation_v1/`.

All cells retain correctness plus direct key/complete BCE. Mutual distillation,
consistency, hallucination objectives, progress, and reconstruction remain off,
so the sweep isolates coupling strength.

## Frozen eligibility and selection rule

A positive weight is eligible when all three seeds finish with finite losses
and scores, no normalized-gate collapse (`mean normalized entropy >= .50` and
`mean effective-token fraction >= .20`), and neither mean key AP nor mean
complete AP drops by more than `.05` relative to P0 on mechanism dev.

Among eligible positive weights, select the largest three-seed mean BoN@16 on
the frozen 500x16 ranking development pool. If multiple candidates are within
`.002` absolute BoN@16 (0.2 percentage points) of the best, select the smallest
weight. An exact remaining tie is broken by lower gate-to-fused-prior squared
L2, then higher within-query pairwise accuracy, then lower weight.

Because the same 500-query population is used for selection, the selected
weight is labeled `dev-tuned engineering default`. Bootstrap intervals and
per-seed deltas remain descriptive and cannot establish held-out efficacy.
Expanded independent prior supervision and a new query-disjoint evaluation are
required before making a scientific gain claim.

After selection, the chosen positive value is written to
`configs/best_current.json`. A matched three-seed integrated full-cell run may
be used as an engineering interaction check, but it does not reopen selection
or upgrade the evidence tier.
