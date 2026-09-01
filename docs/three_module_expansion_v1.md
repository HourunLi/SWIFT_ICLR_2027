# Three-module expanded factorial v1

Status: unified data materialized and independently verified; full-width GPU
preflight passed; all 24 authorized runs completed three epochs and every
checkpoint plus optimizer state passed finite-state validation. The separate
mechanism evaluation passed its execution contract, and the hash-bound ranking
authorization is now frozen. Ranking results are not yet available.

## What is being combined

This stage tests the three current CLIR modules on one matched training
manifest:

- **C — Consistency:** compact and expanded answers from the same query are
  trained as a positive semantic relation;
- **H — H0 hallucination localization:** token-level BCE learns whether the
  reasoning has entered the first-bad-unit tail. H1 value-tail shaping remains
  off;
- **P — Dual Prior plus the original Gate:** direct Key/Complete token targets
  are trained with weight `1`, and the unchanged main-style shared-gradient
  Gate is fixed at `.25`.

The P factor is intentionally the user-requested method identity. The preceding
standalone Gate screen learned internal alignment but failed BoN@16; carrying
the same `.25` into this interaction test does not reclassify that result as a
pass and is not a new tuning choice.

## Why there are eight cells

The complete `2×2×2` grid is used:

| Cell | C | H0 | Prior + Gate |
|---|---:|---:|---:|
| U0 | off | off | off |
| C | on | off | off |
| H | off | on | off |
| P | off | off | on |
| CH | on | on | off |
| CP | on | off | on |
| HP | off | on | on |
| Full | on | on | on |

This costs 24 runs at three seeds, but it avoids an ambiguous Full-only result.
If Full changes, the matched grid can distinguish a main effect from C×H,
C×P, H×P, or three-way interaction. Every cell sees the exact same rows in the
same order and uses three epochs, batch size 4, learning rate `1e-4`, and seeds
42/43/44. H1, mutual Prior, MIL, pseudo-tail, progress, and reconstruction are
off everywhere.

## Unified data construction

The two parent train files cannot be concatenated:

- both contain the same 3,968 historical trajectories;
- the Consistency/H0 file adds 800 Consistency endpoints and 400 H rows;
- the Prior file adds direct targets to 48 historical rows and appends 202 new
  Prior rows.

The deterministic merge therefore keeps the richer Consistency/H0 copy of the
3,968 shared rows, copies the 48 legacy Key/Complete targets onto their matching
IDs, and appends only the 202 new Prior rows. The expected unified inventory is
5,370 unique trajectories across 1,493 queries, with 400 Consistency relations,
400 H rows (200 positive and 200 clean), and 250 Prior rows.

The six core fields `id`, `query_id`, candidate index, correctness, prompt token
IDs, and output token IDs must match on every shared historical row. Feature
paths are rebased without changing their targets. No feature is re-extracted.

## Cross-module dev leakage fix

Each parent experiment was query-disjoint internally, but combining their train
sets exposes two cross-task overlaps in each direction: two H-dev queries occur
in Prior train, and two Prior-dev queries occur in Consistency/H0 train. Before
training, the materializer removes every dev row whose query appears anywhere
in unified train. This fixed mechanical rule leaves:

- H dev: 198 rows;
- Prior dev: 49 rows;
- Consistency held-out relations: unchanged 150 positive +150 hard negative;
- ranking: unchanged 892 queries ×16 candidates.

The Consistency held-out endpoints and ranking population both have zero query
overlap with unified train. The 892-query ranking set is nevertheless reused
exploratory data, not a new protected or confirmatory test.

## Evaluation and evidence boundary

Mechanism diagnostics are reported separately for C, H0, and Prior/Gate. The
ranking primary contrast is paired Full−U0 BoN@16; secondary contrasts include
Full against every single/pair cell and the frozen factorial interactions.
K=1/2/4/8/16, within-query pairwise accuracy, stable ties, selection changes,
and 10,000 paired bootstrap replicates remain fixed.

Materialization alone did not authorize training. The published manifests now
contain exactly 5,370 train rows, 198 clean H-dev rows, and 49 clean Prior-dev
rows, and an independent recomputation passed. Their exact file, ordered-row,
and sidecar hashes are bound in
`configs/three_module_expansion_v1/training_authorization.json`. The clean-commit,
full-width preflight covered all eight cells using separate representative
batches for Consistency, positive/clean H0, and Prior. All enabled objectives
produced finite losses and gradients in their intended heads. The subsequent
8 cells × 3 seeds completed as authorized, and the validation report records
`PASS_THREE_MODULE_COMPLETE_2X2X2_24_RUN_TRAINING`.

Mechanism and ranking scoring use row sharding rather than checkpoint sharding.
Each worker owns a disjoint set of source rows, reads each large hidden-state
payload once, and applies all 24 small reward checkpoints to that in-memory
batch. Eight shard manifests are hash checked before merge; merge restores the
exact source order and verifies complete row coverage, source identity, finite
scores, and the checkpoint hash for every output. This avoids rereading the
roughly 1 TB ranking payload 24 times without changing any model computation.

## Mechanism result before ranking

The query-disjoint Silver mechanism evaluation used all 24 checkpoints. The
factorial main effects show that Consistency improves held-out positive versus
hard-negative relation separation, H0 improves tail-token AP and BCE, and the
P factor improves both Key and Complete token AP/BCE. H0 path discrimination is
clear, but first-onset placement at the fixed `.5` threshold remains imprecise:
the mean Full within-5-token rate is about `3.3%`. H0 should therefore be
described as learning the bad-tail signal, not as reliably identifying the
exact first bad token.

The raw absolute Gate-to-Prior L2 is larger for P-on cells because direct Prior
training changes the fused Prior from almost uniform to concentrated; comparing
that value against P-off's two nearly uniform untrained maps is not a meaningful
alignment baseline. A separately recorded, mechanical scale-aware diagnostic
compares the learned Gate with uniform attention against the same learned fused
Prior. The learned Gate is closer in every P-on cell and seed (`12/12`). This
does not alter the original raw L2 result and is marked posthoc/no-retuning.

`configs/three_module_expansion_v1/ranking_evaluation_authorization.json` binds
the 24 validated checkpoints through the training completion report, the
mechanism report, the reused 892×16 ranking manifest, scorer and summarizer
source hashes, eight row shards, BF16, K=`1/2/4/8/16`, stable ties, and 10,000
paired bootstrap replicates. No ranking output may be used to change a weight,
epoch, cell, seed, or data subset.

No result may be called Gold, human-verified, fresh confirmation, protected-test
evidence, or a repair of the original v7/v12/v13 failures. No inspected dev or
ranking result may be used to change epochs, weights, subsets, or cells.
