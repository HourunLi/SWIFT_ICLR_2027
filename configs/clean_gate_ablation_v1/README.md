# Clean prior-to-gate ablation v1

Status: pre-registered before inspecting any new training, mechanism, or ranking
metrics. This is a `small-scale real` screening experiment, not a formal
efficacy protocol.

## Question and single changed factor

Does direct key/complete prior supervision improve held-out ranking when the
detached fused prior also supervises the reward gate that appears in the scalar
score?

The two cells are:

1. `p0_direct_prior`: correctness plus direct key/complete BCE;
2. `pg0_direct_prior_gate`: P0 plus gate-to-detached-fused-prior alignment.

The cells have identical architecture, initialization policy, sampler,
optimizer, batch size, BF16 policy, three-epoch budget, direct prior losses, and
data. Mutual distillation, consistency, every hallucination objective, progress,
and reconstruction remain disabled. The only changed value is
`gate_prior_weight: 0.0 -> 0.0625`.

The weight is frozen from method identity rather than selected on the current
16-row mechanism dev. In `origin/main`, `prior_weight=0.25` and
`gate_prior_weight=0.25`, so the effective total-loss coefficient is `0.0625`.
Clean integration uses `prior_weight=1.0`; therefore `0.0625` reproduces the
original effective scale. Setting it to `0.25` here would be four times the
original scale and is not part of this experiment.

## Frozen data, seeds, and outputs

- train: `run_artifacts/joint_training_pilot_v1/data_v1/train3968_joint_v1.jsonl`;
- mechanism dev: `run_artifacts/joint_training_pilot_v1/data_v1/mechanism_dev16_joint_v1.jsonl`;
- ranking validation: `run_artifacts/stage1b_v2/manifests/validation_extracted.v4.local.jsonl`;
- seeds: `42,43,44`;
- epochs: `3` for every cell and seed;
- output root: `run_artifacts/clean_gate_ablation_v1/`.

P0 is rerun from scratch rather than copied from the earlier clean matrix so
both cells bind the same code commit and runtime. Seed 42 is a health stage:
both cells must finish with finite losses/gradients, identical data contracts,
nonzero applicable direct-prior counts, and a nonzero PG0 gate loss. If that
passes, both cells expand to seeds 43/44 regardless of seed-42 ranking. Ranking
metrics are not used to select which seeds to run.

The train manifest contains 48 independently labeled prior trajectories, while
the mechanism dev contains only 16 trajectories. More epochs do not add
independent evidence, so this experiment is not eligible for a three-to-five
epoch extension or a post-result weight sweep on the same data.

## Pre-registered metrics and decisions

Primary task metric:

- paired query-level BoN@16 delta `PG0 - P0` on the frozen 500x16 population.

Secondary task metrics:

- BoN@4/8;
- within-query correct-vs-wrong pairwise accuracy;
- fixed-seed query bootstrap and exploratory seed+query hierarchical bootstrap.

Mechanism and protection metrics on the query-disjoint mechanism dev:

- full-trajectory gate-to-fused-prior squared L2, matching the training loss
  definition (lower is better);
- gate attention entropy/effective support and raw sigmoid-gate mean as collapse
  diagnostics;
- key/complete AP and AUROC;
- key-to-complete map discrepancy.

Interpretation is frozen as follows:

- If gate squared L2 improves but ranking does not, the coupling is learnable
  but reward-selection efficacy is not established.
- A mean BoN@16 improvement with at least two of three seeds nonnegative is only
  a candidate for larger-data replication when either paired interval crosses
  zero; it is not a stable efficacy claim.
- A negative mean BoN@16 delta with at least two negative seeds rejects this gate
  at the current scale even if alignment improves.
- An absolute mean drop greater than `.05` in either key or complete AP is a
  prior-protection failure.
- Non-finite values or pathological gate collapse invalidate the run rather than
  count as a scientific result.

No result from this screen changes `configs/best_current.json` or the Full cell
without a separate, larger-data replication decision.

## Completion status

The full P0/PG0 matrix completed for seeds 42/43/44 at three epochs from clean
commit `649747f3605e820430d4c93d788e368676ff37ea`. All data/provenance/finite and
candidate-parity gates passed; rerun P0 state dicts were bit-exact to the
earlier clean matrix.

PG0 did not pass its mechanism or ranking gate. Mean full-trajectory
gate-to-fused-prior squared L2 worsened from `.01195` to `.01335` and improved
in only one of three seeds. Key/complete AP protection passed, but BoN@16 moved
from `.9180` to `.9167` (`-.13` points); fixed-seed query and seed+query
intervals both crossed zero. The gate changed the selected candidate for 53–76%
of queries, yet most changes preserved correctness and the three-seed net was
two fewer correct selections over 1500 seed-query units.

The decision is to keep gate alignment disabled, avoid a post-result scale or
epoch sweep on the same 16-row dev, and expand independent prior supervision
before testing a newly pre-registered coupling. Full results are in
[`docs/clean_gate_ablation_v1_results.md`](../../docs/clean_gate_ablation_v1_results.md).
