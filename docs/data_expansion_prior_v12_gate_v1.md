# Prior v12-posthoc fixed-.25 Gate replication

Status: completed post-hoc exploratory screen. The protocol/config/evaluator
were committed before training, and the ranking authorization was committed
after the mechanism guards passed but before any PG0 ranking score was
produced.

## Question

The completed v12-posthoc P0 run learned Key/Complete strongly, but those
targets affected the scalar reward only indirectly through the shared encoder
and did not improve final ranking. This stage asks one narrow question:

> Does the unchanged `origin/main` shared-gradient prior-to-reward Gate make
> the learned Prior useful to final candidate selection?

## Matched cells

- `P0`: the already completed direct Key/Complete cell, with
  `gate_prior_weight=0`;
- `PG0`: the same config, data, initialization policy, optimizer, seeds, and
  three-epoch budget, changing only `gate_prior_weight=0.25`.

The `.25` value is fixed before this run. It is the existing clean-branch
engineering default and will not be tuned on the 51-row Prior dev set or the
reused 892-query ranking population. Mutual distillation, Consistency, H0/H1,
MIL, pseudo-tail, progress, reconstruction, and Full stay disabled.

The implementation is unchanged:

```text
gate_attention = normalize(sigmoid(gate_logits))
fused_prior = normalize(0.5 * key_prior + 0.5 * complete_prior).detach()
L_gate = squared-L2(gate_attention, fused_prior)
```

The same Gate weights token values in the scalar score. Gradients from
`L_gate` therefore update the Gate and shared encoder, while the detached fused
Prior is not pulled toward the Gate by this loss.

## Frozen data and budget

- shared train: 4,170 rows =3,968 historical rows +202 new v12-posthoc Prior
  rows;
- direct Prior coverage: 250 rows =48 legacy +202 new;
- Prior dev: the existing 51 post-hoc exact dual-AI Silver rows;
- ranking: the existing 892 queries ×16 candidates from v7.4;
- seeds: `42,43,44`;
- epochs: `3`;
- batch size: `4`, learning rate: `1e-4`, BF16.

P0 checkpoints are reused because model/data/training code has not changed
since their run. PG0 is trained from scratch with the same seed-specific
initialization. The ranking set is reused exploratory data, not a fresh
confirmatory or protected test population.

## Frozen evaluation

Mechanism and health checks on the 51-row Prior dev set:

- full-trajectory Gate-to-fused-Prior squared L2, lower is better;
- Key and Complete AUROC/AP/BCE;
- normalized Gate-attention entropy and effective-token fraction;
- correctness AUROC/BCE;
- finite checkpoints, gradients, losses, and scores.

Prior protection fails if mean Key AP or Complete AP drops by more than `.05`
from P0. Gate collapse fails if mean normalized entropy is below `.50` or mean
effective-token fraction is below `.20`. Alignment is called learned only if
mean squared L2 falls and at least two of three seeds improve.

The primary ranking contrast is paired `PG0-P0` Best-of-N accuracy at `K=16`.
Secondary metrics are `K=1/2/4/8`, within-query correct-vs-wrong pairwise
accuracy, selection-change counts, and fixed-seed plus hierarchical paired
query bootstrap intervals with 10,000 replicates.

An exploratory ranking benefit requires a positive BoN@16 mean, at least two
positive seeds, and a fixed-seed query interval above zero. Anything weaker is
reported as a candidate or null result, not efficacy. A negative mean with at
least two negative seeds rejects this Gate on the present exploratory screen.
No outcome permits weight, epoch, subset, or `K` tuning on these inspected
populations.

## Evidence boundary and next stage

This experiment cannot turn the original v12/v13 failures into passes and
cannot make the Silver labels Gold or human-verified. It tests direct Prior plus
fixed Gate only. The user's requested three-module combination is a later,
separately frozen stage after this result; it is not silently included here.

## Completed result

All three PG0 seeds completed three epochs with finite, loadable checkpoints.
On the 51-row Prior dev set, the mean full-trajectory Gate-to-fused-Prior
squared L2 fell from `.03114` to `.02584`; two of three seed deltas were lower.
Key AP dropped only `.01071`, Complete AP dropped `.00018`, normalized Gate
entropy was `.87191`, and effective-token fraction was `.41419`. Thus the
predeclared alignment, Prior-protection, and anti-collapse guards all passed:
the Gate learned to look more like the fused Key/Complete map without erasing
the two Prior heads or collapsing onto a tiny token set.

That mechanism result did not become stable ranking benefit on the reused
892-query ×16-candidate population:

| K | P0 Gate-off | PG0 Gate=.25 | PG0−P0 |
|---:|---:|---:|---:|
| 1 | `82.74%` | `82.74%` | `0.00` point |
| 2 | `85.13%` | `84.90%` | `-0.22` point |
| 4 | `85.76%` | `85.72%` | `-0.04` point |
| 8 | `84.60%` | `85.05%` | `+0.45` point |
| 16 | `85.54%` | `85.01%` | `-0.52` point |

At K=8 all three seed deltas were positive, but both frozen intervals crossed
zero. At the primary K=16 all three deltas were negative
(`-.34/-.78/-.45` points); the fixed-seed query interval was
`[-1.20,+.11]` points and the seed+query interval was `[-1.42,+.45]` points.
Within-query correct-vs-wrong pairwise accuracy was essentially unchanged
(`.65937 -> .65907`).

The Gate was not inert: at K=16 it changed the selected candidate for
`57.2%/30.4%/39.3%` of queries by seed. It simply changed slightly more
correct selections into wrong ones than the reverse. The frozen decision is
therefore: **mechanism alignment learned, but fixed `.25` standalone Gate
rejected on this exploratory ranking screen.** Per the user's method-identity
constraint, the unchanged `.25` route may still be carried into the separately
frozen three-module interaction experiment; this is not a claim that `.25`
passed, and it must not be retuned on these 51/892 inspected populations.
