# Clean ablation v1

This directory freezes the first matched ablation matrix for
`clir-clean-integration`. It is a small-scale real screening protocol, not a
formal efficacy protocol.

All cells keep the same layer-axis encoder, condition path, residual/value
scorer, initialization policy, optimizer, batch size, sampler, and three-epoch
budget. Only loss-family weights change. Inactive heads remain instantiated so
the seed consumes the same model-initialization RNG stream in every cell.

## Frozen data and evaluation

- train: `run_artifacts/joint_training_pilot_v1/data_v1/train3968_joint_v1.jsonl`
  (`3968` rows, `496 x 8`, checker v5);
- mechanism dev: `run_artifacts/joint_training_pilot_v1/data_v1/mechanism_dev16_joint_v1.jsonl`
  (`16` query-disjoint rows);
- ranking validation: `run_artifacts/stage1b_v2/manifests/validation_extracted.v4.local.jsonl`
  (`500 x 16`, checker v4, frozen candidate prefixes);
- primary ranking metric: paired BoN@16 delta versus `c0_correctness_only`;
- secondary ranking metrics: BoN@4/8 and within-query correct-vs-wrong pairwise
  accuracy;
- uncertainty unit: query, never candidate row.

The train/validation checker-version mismatch is frozen and applies equally to
every cell. It is acceptable for this matched screening matrix but must be
resolved before a formal protocol.

## Cells

1. `c0_correctness_only`: correctness-only CLIR backbone;
2. `c1_consistency`: correctness plus semantic/style consistency;
3. `h0_onset_bce`: correctness plus explicit-onset H-head BCE only;
4. `h1_onset_tail`: H BCE plus direct onset-to-negative-tail reward shaping;
5. `p0_direct_prior`: correctness plus direct key/complete prior targets;
6. `p1_mutual_prior`: direct priors plus bidirectional stop-gradient distillation;
7. `full_integration`: all currently active clean modules.

`h0` versus `h1` isolates diagnostic localization from direct scalar-reward
coupling. `p0` versus `p1` isolates mutual distillation. Path MIL, pseudo-tail,
progress, gate-prior alignment, and reconstruction stay disabled in every cell.

## Staging rule

Run all seven cells at seed 42 for three epochs first. If the matrix is healthy
(finite training/scoring, identical data/candidate hashes, and no protocol
violation), expand every cell—not a result-selected subset—to seeds 43 and 44.
Extend every cell uniformly to five epochs only if the three-epoch train/dev
curves have not saturated and do not show clear mechanism-dev deterioration.

The available supervision is too small for a formal mechanism claim: 27
consistency positive pairs with no held-out relation set, 17 positive plus 31
clean onset rows, and 48 prior trajectories. Mechanism-dev contains only 6
positive onset and 10 clean rows. More epochs do not increase those independent
sample counts.

## Completion status

The full seven-cell matrix completed at seeds 42, 43, and 44 for three epochs.
All run/candidate parity gates passed. The uniform five-epoch extension gate did
not pass because several auxiliary mechanism-dev curves deteriorated while
training loss decreased. Do not resume a result-selected subset. Ranking,
mechanism, uncertainty, and component decisions are frozen in
[`docs/clean_ablation_v1_results.md`](../../docs/clean_ablation_v1_results.md).
