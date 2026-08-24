# Clean prior→reward-gate 权重选择 v2

状态：2026-08-24 完成 `0/.0625/.25/1/4/10` 六个权重点的
3-seed × 3-epoch 配对比较。`0` 与 `.0625` 复用已验证为不可变的 v1 anchor，
`.25/1/4/10` 在查看新结果前提交配置并完整训练。最终默认固定为
`gate_prior_weight=.25`。

证据等级是 **dev-tuned engineering default**，不是独立 held-out efficacy result。
本轮按用户的方法身份裁决保留 `main` 原始 coupling，并使用同一 500-query ranking
development population 选择强度；因此这些数字只能决定工程默认值，不能证明 gate 带来
可泛化提升。

## 结论先行

- 六个权重都完成 3 个 seed，loss/score finite；所有正权重都通过预设的 prior protection、
  gate entropy 和有效支持宽度门。
- `10.0` 的三-seed BoN@16 点估计最高，为 `.92067`；`.25` 为 `.91867`，两者差值恰好
  是预设 near-tie 边界 `.002`（0.20 percentage point）。冻结规则要求此时选择更小
  权重，因此选中 `.25`。
- `.25` 相对无 coupling 的 P0 只有 `+.00067`（+0.07 point）；fixed-seed query 95% CI
  `[-.0080,+.00867]`，seed+query hierarchical CI `[-.0120,+.01533]`，都跨 0。
  这不是 efficacy 证据。
- `.25` 保留 `origin/main` 声明的内部系数和完全相同的 shared-gradient 路径。clean 的
  `prior_weight=1`，所以它在 clean 总 loss 中的绝对系数为 `.25`；原 main 是
  `prior_weight=.25 × gate_prior_weight=.25 = .0625`。本轮调的是强度，没有修改公式、
  detach 位置、mask、梯度路由或推理 score。
- 强权重 `10` 的确显著降低 gate↔fused-prior L2（`.000934`），但也把 effective-token
  fraction 压到 `.364`；它没有超过 `.25` 足够多以越过保守选择阈值，因此不作为默认。

## 保持不变的 main coupling

每个有成对 key/complete 标签的 trajectory 上：

```text
gate_attention = normalize(sigmoid(gate_logits))
fused_prior = normalize(0.5 * key_prior + 0.5 * complete_prior).detach()
L_gate = squared-L2(gate_attention, fused_prior) on shared prior coverage
L_total += prior_weight * gate_prior_weight * L_gate
```

`fused_prior.detach()` 使 alignment loss 不通过该项反向修改两个 prior map，但它会通过
reward gate head 以及共享 encoder 改变 gate。推理时 scalar score 仍是：

```text
score = sum_t(gate_t * value_t) / sum_t(gate_t) + trajectory_residual
```

因此 dual prior 不是在推理时额外加一项分数，而是在训练时约束已经参与 score 的 gate，
从而直接改变 token 权重和最终候选选择。没有引入 KL、head-only gradient、runtime prior
fusion 或新的 gate。

## 冻结协议

| 项目 | 固定值 |
|---|---|
| train | 3968 rows = 496 queries ×8；其中 48 条 paired-prior trajectory |
| mechanism dev | 16 条 query-disjoint trajectory |
| ranking development | 500 queries ×16 frozen candidates |
| seeds / budget | `42/43/44`；3 epochs；batch 4；BF16；LR `1e-4` |
| 隔离因素 | correctness + direct priors；C/H/mutual/progress/reconstruction 全关 |
| 新训练 commit | `d1896aee6ab85d851f0a53aa55f6da487a4b602e`，`dirty=false` |
| ranking candidate signature | `93f5d1bafcc81c109c8f2d7f8672e8e233baa45ff57eb88a888d0a7090e5a040` |
| paired summary SHA-256 | `3aaf0d67378c2bb189bf1794e6d0f683822f1b5aca2836037dedd11084db277c` |

资格门要求：三个 seed 均 finite；平均 normalized gate entropy `>=.50`；平均
effective-token fraction `>=.20`；key AP 与 complete AP 各自相对 P0 下降不超过 `.05`。
所有正权重均通过。

选择规则要求：在合格正权重中先最大化三-seed mean BoN@16；与最高点相差不超过
`.002` 时选更小权重；若仍精确同分，再依次比较更低 gate L2、更高 pairwise、更低权重。

## Ranking 结果

| gate weight | BoN@16 mean ± SD | seed 42 / 43 / 44 | 相对 P0 | pairwise mean |
|---:|---:|---:|---:|---:|
| `0` P0 | `.9180 ± .0072` | `.910/.924/.920` | — | `.68913` |
| `.0625` | `.9167 ± .0050` | `.912/.916/.922` | `-.13` pt | `.69267` |
| **`.25` selected** | **`.9187 ± .0042`** | **`.920/.922/.914`** | **`+.07` pt** | **`.68926`** |
| `1` | `.9180 ± .0040` | `.922/.914/.918` | `.00` pt | `.68867` |
| `4` | `.9173 ± .0031` | `.918/.914/.920` | `-.07` pt | `.69307` |
| `10` raw best | `.9207 ± .0064` | `.918/.916/.928` | `+.27` pt | `.69339` |

各 K 的三-seed mean：

| gate weight | BoN@1 | BoN@2 | BoN@4 | BoN@8 | BoN@16 |
|---:|---:|---:|---:|---:|---:|
| `0` | `.8840` | `.9027` | `.9087` | `.9107` | `.9180` |
| `.0625` | `.8840` | `.8987` | `.9107` | `.9133` | `.9167` |
| `.25` | `.8840` | `.9033` | `.9120` | `.9180` | `.9187` |
| `1` | `.8840` | `.8987` | `.9080` | `.9107` | `.9180` |
| `4` | `.8840` | `.9000` | `.9087` | `.9100` | `.9173` |
| `10` | `.8840` | `.9013` | `.9113` | `.9153` | `.9207` |

`.25` 的 BoN@8 在三个 seed 都高于各自 P0，平均 `+.73` point；但选择指标在协议中是
BoN@16，不能事后改主指标。BoN@16 的逐-seed `.25−P0` 为 `+1.0/-.2/-.6` points，
方向不一致；`10−P0` 为 `+.8/-.8/+.8` points，区间同样跨 0。故不能把任一正点估计
写成稳定性能提升。

## Mechanism 与保护门

以下均为 16-row mechanism dev 上的三-seed mean：

| gate weight | gate L2 ↓ | normalized entropy | effective fraction | key AP | complete AP | eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| `0` | `.011953` | `.9687` | `.8149` | `.2969` | `.9210` | anchor |
| `.0625` | `.013347` | `.9658` | `.7907` | `.2926` | `.9251` | yes |
| **`.25`** | **`.012798`** | **`.9669`** | **`.7952`** | **`.2897`** | **`.9256`** | **yes** |
| `1` | `.010270` | `.9664` | `.7722` | `.2705` | `.9152` | yes |
| `4` | `.005609` | `.9472` | `.7290` | `.2633` | `.9152` | yes |
| `10` | `.000934` | `.8592` | `.3645` | `.3073` | `.9270` | yes |

这个表说明两个不同问题：

1. 更强权重从 `1→4→10` 明显能把 gate 拉向 fused prior，说明 objective 在足够强时可优化；
2. alignment 更好并没有随权重单调变成更好的 BoN，说明“学会对齐”与“对最终选择有益”
   仍是两个问题。

选中的 `.25` 没有比 P0 获得更低 held-out gate L2，因此不能声称它已经验证了 prior→gate
机制收益。它被保留，是因为用户要求方法默认包含 main 原始 coupling，而在当前开发集上
它是冻结规则选出的最保守近优强度。

## 当前裁决

1. `RewardConfig` 与 `configs/best_current.json` 默认都设为 `.25`；coupling 默认开启。
2. v1 与 v2 的冻结实验配置保持原样，历史 P0/PG0/Full 结果不会因默认值改变而被重写。
3. 当前不再在这 16 条 mechanism dev / 500-query ranking dev 上继续扫描权重、epoch 或
   routing；该 population 已经被用于选择。
4. 下一轮先扩 prior train 至约 300–500 条独立 trajectory、建立 100–200 条
   query-disjoint mechanism dev，并把 ranking validation 扩到约 1500–2000 queries；
   然后用固定 `.25` 复测 `gate off/on`，才能判断可泛化增益。
5. 当前 `best_current` 同时还有 C、H/tail、mutual 等模块；本轮只隔离选择了 gate 强度，
   尚未重新跑新的 integrated Full 交互。因此“默认接入”是工程整合决定，不是 Full 已验证。

结构化选择记录见
[`configs/clean_gate_tuning_v2/selection.json`](../configs/clean_gate_tuning_v2/selection.json)。
完整本地产物位于 `run_artifacts/clean_gate_tuning_v2/`（约 5.5 GB，Git ignored）。
