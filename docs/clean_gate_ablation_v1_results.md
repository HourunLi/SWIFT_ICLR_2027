# Clean prior→reward-gate 消融结果

状态：2026-08-24 完成 `P0` 与 `PG0` 的 2-cell × 3-seed × 3-epoch
matched screen。证据等级为 `small-scale real screening`，不是正式 efficacy result。

> 当前状态更新：本文是不可改写的 `.0625` 单点历史结论。用户随后明确要求保留并默认
> 开启 `main` 原始 coupling；v2 冻结网格按近优规则选出 `.25` 作为
> `dev-tuned engineering default`。当前默认与完整新证据见
> [`clean_gate_tuning_v2_results.md`](clean_gate_tuning_v2_results.md)。这项后续工程裁决
> 不把 v1 的负结果改写成 efficacy 成功。

## 结论先行

当前 `main` 原始有效强度的 prior→gate coupling **工程上能运行，也确实会改变最终选择，
但没有通过机制门或 ranking 门**：

- PG0 没有让 reward gate 更接近 detached fused prior。16-row mechanism dev 上，与训练
  loss 同定义的 full-trajectory squared L2 从 P0 的 `.01195` 变为 PG0 的 `.01335`
  （越低越好），平均恶化约 `11.7%`，只有 seed 42 改善，seeds 43/44 都恶化。
- key/complete prior 没有被明显破坏：key AP 只下降 `.0044`，complete AP 上升
  `.0041`，都未触发预注册的 `.05` protection guard；gate 也没有熵或有效支持塌缩。
- BoN@16 从 P0 `.9180` 变为 PG0 `.9167`，平均 `-.13` points；逐 seed 是
  `+.2/-.8/+.2` points。fixed-seed query 95% CI 为 `[-.87,+.60]` points，探索性
  seed+query CI 为 `[-1.20,+1.00]` points，均跨 0。
- pairwise accuracy 点估计从 `.6891` 变为 `.6927`，约 `+.35` points，但没有转化为
  top-1-of-16 增益。
- gate 不是“完全没起作用”：三个 seed 分别有 `53.2%/75.6%/57.6%` 的 query 更换了
  最终候选。绝大多数变化发生在相同 correctness 的候选之间；净正确性变化为
  `+1/-4/+1` 个 query，总计正好是 `-2/1500 = -.13` points。

因此本文在 v1 完成时的裁决是 `gate_prior_weight` 保持默认 `0`，不把 PG0 接入
`best_current` 或 Full。该裁决后来被用户的方法身份决定与预先冻结的 v2 工程调参取代；
当前 `.25` 默认仍不是独立 efficacy 证据，不能反向抹去本轮 `.0625` 的失败。

## 冻结协议与 provenance

两份配置在查看任何新指标前提交并推送，训练 commit 为
`649747f3605e820430d4c93d788e368676ff37ea`，所有 checkpoint 都记录
`branch=clir-clean-integration`、`dirty=false`、完整命令、配置、数据、环境和 RNG/optimizer
状态。完整测试使用固定 SWIFT Python 环境，结果为 `44 passed`。

| 项目 | 冻结值 |
|---|---|
| P0 | correctness + direct key/complete BCE；gate `0` |
| PG0 | P0 + detached fused-prior→reward-gate squared-L2；gate `.0625` |
| 为什么是 `.0625` | `origin/main` 的总有效系数为 `prior .25 × gate .25 = .0625`；clean 外层 prior 为 `1` |
| train | 3968 rows = 496 queries ×8；48 条 paired-prior trajectory |
| mechanism dev | 16 条 query-disjoint trajectory |
| ranking | 500 queries ×16 frozen candidates |
| seeds / budget | `42/43/44`，每 cell 3 epochs，batch 4，BF16，LR `1e-4` |
| train SHA-256 | `dc868b60494c0417902f4f4679d24771a75cd5c9f51107104bb365579a0f4d48` |
| mechanism-dev SHA-256 | `22b4dadc0c75cfe9422bc9805697028dd31e19fe129ac100ab5a10b69a16678d` |
| ranking candidate signature | `93f5d1bafcc81c109c8f2d7f8672e8e233baa45ff57eb88a888d0a7090e5a040` |
| P0 config SHA-256 | `f330de67dfe35636d14c0287a299cb1f24b9ca19fbc3921bfd8631333aed919b` |
| PG0 config SHA-256 | `7e435a0f2eff9c892c210594aa97e92c9a893fc700506a905cdf42d79ba70368` |
| paired summary SHA-256 | `b378893b2877f79932c6f77aa867fdad1132c7ea812950e2a51f3217d4f035fc` |

训练前的真实全宽 gate batch 使用 BF16 `[T,101376]` feature，loss 与 60 个 gradient
tensor 全部 finite，PG0 gate loss 非零，峰值 CUDA memory 约 `1.30 GiB`。六个训练 run
都完成；重新训练的三个 P0 checkpoint state dict 与原 clean-ablation-v1 的 P0 逐 tensor
bit-exact，说明新 commit、并行运行和诊断扩展没有改变 baseline 训练语义。

## Gate loss 是否真的被优化

PG0 的 raw gate squared-L2 确实进入 total loss，但在固定的低系数下没有呈现稳定下降：

| seed | train gate loss，epoch 1→2→3 | dev gate loss，epoch 1→2→3 |
|---:|---:|---:|
| 42 | `.000294→.000514→.000717` | `.007143→.014008→.011374` |
| 43 | `.000166→.000483→.000546` | `.012315→.007146→.012627` |
| 44 | `.000343→.000499→.000686` | `.001506→.012039→.016039` |

这不是 mask 失效：P0 的对应项三轮严格为 0，PG0 每轮都非零。更合理的解释是，当前
attention-distribution L2 的数值尺度很小，乘 `.0625` 后相对 correctness/direct-prior
gradient 太弱，且 correctness 所需要的 reward gate 不一定与 key/complete fused prior
一致。该解释是从 loss 尺度与趋势作出的推断，不是已经证明的唯一因果机制。

## Mechanism dev

以下为三 seed mean ± sample SD。gate L2 使用完整 trajectory 的归一化 gate/fused-prior
分布，定义与训练的 `attention_mse` 一致；越低越好。

| 指标 | P0 | PG0 | PG0−P0 |
|---|---:|---:|---:|
| gate↔fused prior squared L2 | `.011953 ± .002251` | `.013347 ± .002414` | `+.001393`，更差 |
| gate/fused dot product | `.001698 ± .001353` | `.001069 ± .000096` | `-.000629` |
| raw sigmoid gate mean | `.6925 ± .1835` | `.6167 ± .0567` | `-.0759` |
| normalized gate entropy | `.9687 ± .0177` | `.9658 ± .0075` | `-.0029` |
| effective-token fraction | `.8149 ± .1003` | `.7907 ± .0341` | `-.0242` |
| key AP / AUROC | `.2969/.6629` | `.2926/.6589` | `-.0044/-.0040` |
| complete AP / AUROC | `.9210/.8688` | `.9251/.8749` | `+.0041/+.0061` |
| key↔complete squared-L2 | `.001746` | `.001987` | `+.000241` |

gate squared-L2 的逐 seed P0→PG0 为：

- seed 42：`.012525→.011374`，改善；
- seed 43：`.009471→.012627`，恶化；
- seed 44：`.013863→.016039`，恶化。

所以本轮不能报告“gate alignment target learnable”。更准确的状态是：代码路径工作，
objective 被计算并反向传播，但当前 main-scale coupling 没有在 held-out mechanism dev 上
实现它声称的对齐效果。prior protection guard 通过，只能说明失败不是由明显摧毁 prior
head 引起。

## Ranking

下表为三 seed mean ± sample SD；delta 是 PG0−P0。

| 指标 | P0 | PG0 | delta | seed 42/43/44 delta |
|---|---:|---:|---:|---:|
| BoN@1 | `.8840 ± .0000` | `.8840 ± .0000` | `.00` pt | `.0/.0/.0` pt |
| BoN@2 | `.9027 ± .0023` | `.8987 ± .0031` | `-.40` pt | `-.6/+.2/-.8` pt |
| BoN@4 | `.9087 ± .0031` | `.9107 ± .0031` | `+.20` pt | `-.4/+.2/+.8` pt |
| BoN@8 | `.9107 ± .0050` | `.9133 ± .0083` | `+.27` pt | `-.2/+.4/+.6` pt |
| BoN@16 | `.9180 ± .0072` | `.9167 ± .0050` | `-.13` pt | `+.2/-.8/+.2` pt |
| query 内 pairwise | `.6891 ± .0113` | `.6927 ± .0192` | `+.35` pt | `-.49/+1.08/+.47` pt |

主指标 BoN@16 的 paired uncertainty：

- fixed-seed query bootstrap 95% CI：`[-.87,+.60]` points；
- exploratory seed+query hierarchical 95% CI：`[-1.20,+1.00]` points。

两个区间都跨 0，点估计也为负，因此没有 ranking efficacy。K=4/8 和 pairwise 的小幅正
点估计说明 gate 不是让所有排序统一变差；它更像是重新排列了候选，但没有稳定改善
每个 16-candidate pool 的最高分选择。

## Gate 如何改变最终选择

| seed | 更换最终候选 | 错→对 | 对→错 | 换了但 correctness 相同 | 净变化 |
|---:|---:|---:|---:|---:|---:|
| 42 | `266/500 = 53.2%` | 9 | 8 | 249 | +1 |
| 43 | `378/500 = 75.6%` | 4 | 8 | 366 | -4 |
| 44 | `288/500 = 57.6%` | 5 | 4 | 279 | +1 |
| 合计 | `932/1500 = 62.1%` | 18 | 20 | 894 | -2 |

P0 与 PG0 的 candidate score Pearson correlation 仍为 `.960/.937/.955`，但 mean absolute
score delta 为 `.428/.560/.523`。这解释了表面上的矛盾：整体分数高度相关，细小到中等
的相对变化却足以让很多接近的 top candidates 互换；由于 validation pool 正确率很高，
绝大多数互换不改变 correctness，少量 `18 vs 20` 的净差决定了最终 `-.13` points。

ranking 全池的 gate squared-L2 也复现了 mechanism 的 seed pattern：seed 42
`.01463→.01355` 改善，seed 43 `.01045→.01403` 和 seed 44 `.01571→.01889` 恶化。

## 裁决与下一步

1. **工程实现通过。** `fused_prior.detach() → gate loss → score gate` 路径、mask、梯度、
   checkpoint、scoring 和 paired evaluation 全部闭环。
2. **main-scale alignment 未通过。** PG0 的 held-out gate L2 平均更差，只有 1/3 seed
   改善；不能说 Dual 已成功通过 gate 直接帮助 reward。
3. **prior protection 通过。** key/complete AP 没有超过 `.05` 的下降，gate 没有塌缩。
4. **ranking 未通过。** BoN@16 点估计 `-.13` points、区间跨 0；pairwise 的小正点估计
   不足以替代主指标。
5. **v1 当时的默认裁决是关闭。** `configs/best_current.json` 和 Full 当时不改，当前实验
   不续到 5 epochs；后续 v2 的用户授权、冻结规则与当前默认另见新结果文档。
6. **重开的最早条件。** 先把 prior train 扩到约 300–500 条独立、覆盖不同长度/推理结构
   的 trajectory，并建立 100–200 条 query-disjoint mechanism dev；重新跑 P0 direct
   learnability 后，才比较新的 gate coupling。
7. **若改变方法，要另立假设。** 可研究对长序列尺度更稳定的 KL/cross-entropy、显式
   gradient balancing、gate-head-only routing，或推理时显式 prior fusion；这些都不是
   本轮 `main` shared-gradient MSE，必须单独预注册，不能拿本轮结果当作已验证修复。

完整本地产物位于 `run_artifacts/clean_gate_ablation_v1/`（约 2.8 GB，Git ignored）。
`paired_summary.json` 绑定所有 6 份 scored/checkpoint hash；每个 run 目录还包含 3-epoch
checkpoint/metrics、mechanism scored/metrics 和 8000-row ranking scored/metrics。
