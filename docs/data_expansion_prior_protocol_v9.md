# CLIR Dual Prior 数据扩量协议 v9：直接集合标注与局部共识

## 1. 为什么 v8 停止、v9 另起一轮

v8 让两位 AI 标依赖图，再由程序对图做传递闭包来生成 Key/Complete。它已经完成标注和 raw 评估，不是“尚未标注”：eligibility 60/60、path agreement `.95`，但 Key/Complete F1 只有 `.7667/.8040`，可直接训练的完整共识仅 8/60，控制题 A/B 也只有 4/6、5/6。最终状态固定为 `STOP_PRIOR_DEPENDENCY_SMOKE_V8_RAW_GATE_FAILURE`，不裁决、不抽 feature、不训练，也不从失败包里事后挑行。

失败诊断显示主要问题是把一个本来直观的“哪些步骤重要”任务改成了更难、更脆弱的图闭包任务。v3 的旧 direct-set 标注反而显示：Key 55/60 行完全一致，Key 的 unit decision agreement 为 `.9909`；Complete 虽只有 26/60 行完全相同，但 unit agreement 仍有 `.9341`，分歧 unit 仅 `.0659`，两边正集合的交并比为 `.8503`。也就是说 Complete 的大部分 unit 有共识，少数边界 unit 不一致。

因此 v9 回到直接标 Key/Complete，但不再要求“整行两个集合完全一样才有用”。双方一致选择或一致排除的 unit 可以监督；只屏蔽双方意见不同的 Complete unit。v8 失败证据原样保留，v9 使用全新题目。

## 2. 本轮只做新鲜 60 条 smoke

本轮复用 v6 已完成 checker v3、exact-token unitizer v2 的 16,000 条候选，不重新 rollout，不需要 GPU。固定选择 60 个不同 query、60 个不同 template cluster，GSM8K/MATH × numeric match/mismatch 四格各 15 条。

机械要求：

- 仅 `train_acquisition`、normal stop、可监督、unitization=ok 的候选；
- material claim 数为 6–40；
- 每题、每 cluster 最多一条；
- 排除 v6.1 Consistency 已使用的全部 query/cluster；
- 排除 v8 Prior smoke 的全部 query/cluster；
- 与 v7 H 子集和 v7 ranking population 的 query/cluster 必须零重合；
- 选择只使用冻结的 SHA-256 优先级，不看 CLIR 分数或 AI 标签。

这 60 题是 prompt-development/train-side smoke：一经发布，永久不能进入后续 Prior train、heldout、ranking 或 protected test。

## 3. 两位 AI 直接标什么

每位标注者独立输出：

- `eligibility`：usable、no_auditable_reasoning 或 insufficient_unitization；
- `key_unit_indices`：最决定候选结论成立与否的最小步骤集合；错误链通常选择最早的决定性错误；
- `complete_unit_indices`：审计候选实际结论所需的最小因果主线，包括被后续使用的错误步骤；
- confidence 和简短理由。

usable 行必须有非空 Key/Complete 且 `Key ⊆ Complete`。题面复述、计划话术、未使用旁枝、重复等式、同义重复和 final wrapper 默认不选；但不能把候选重算成另一条更短解法。

本轮仍是两个不同非 Phi 模型系列：A 为用户报告的 GPT-5.6-sol xhigh，B 为 Claude Opus 5 high。每边 60 natural +6 隐藏控制，A 另有 12 条盲重复，所以 A/B 各一个 78/66 行 shard。`PRIVATE_package_index.jsonl` 绝不能发送给任何标注者。

## 4. 局部共识怎样变成训练监督

只在双方都判 usable 且都不是 low confidence 时考虑训练 materialization。

### Key

Key 保持严格：只有 A/B 的非空 Key 集合完全一致，整行 Key target 才可训练。此时共识 Key 为正，其余有效 output token 为负；只要 Key 集合不同，这一行不提供 Key target，也不能把分歧位置当负样本。

### Complete

设两边集合为 `C_A` 与 `C_B`：

```text
positive = C_A ∩ C_B
masked   = C_A △ C_B
negative = valid units − (C_A ∪ C_B)
covered  = valid units − masked
```

只有 `positive` 非空时该行 Complete 可训练。双方都选的是正样本，双方都不选的是负样本，只有一边选的 unit 使用显式 `complete_prior_mask=0`。这不是多数投票，也不是把争议 unit 判成负例。

模型的 Key/Complete attention map 仍在完整有效 trajectory 上归一化；loss 只在 coverage mask 为 1 的 token 上计算，禁止在已覆盖子集上重新 softmax。网络结构、mutual loss 和 main 的 `.25` gate coupling 路径均不因 v9 改动。

## 5. 冻结的 raw 门

60 条 natural 全部保留在分母；没有第三模型或事后裁决。必须同时满足：

| 指标 | 门槛 |
|---|---:|
| eligibility agreement | ≥.95 |
| 双方 usable / 双方非低置信 usable | 各 ≥50 |
| Key / Complete macro F1 | 各 ≥.90 |
| 非低置信 exact Key 行 | ≥50 |
| Complete 有非空共识正 unit 的行 | ≥50 |
| 同时可训练 Key 与 Complete 的行 | ≥50 |
| Complete unit decision agreement | ≥.90 |
| Complete 分歧 unit 比例 | ≤.10 |
| Complete 正集合交并比 | ≥.80 |
| Complete 每行平均 coverage | ≥.90 |
| 隐藏控制 | A/B 各 6/6 |
| A self-repeat | ≥.95 |
| Complete=全部 material units | A/B 各 ≤.25 |

完整 Key+Complete 的 exact joint 行数仍报告，但不再作为 gate；因为 v9 正是在检验“少量 Complete 边界分歧可否被显式屏蔽”。任一门失败，状态为 `STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE`：不裁决、不扩量、不抽 feature、不训练。

通过只说明这套 direct-set 定义和局部共识规则具有足够的操作稳定性。无人类复核时，后续标签只能叫 `silver_dual_ai_partial_consensus_prior_v9`，不能叫 Gold、verified、高质量人工标签，也不证明 Prior 或最终排序有效。

## 6. 通过后怎样扩量

smoke 全过后才另冻一份未检查的 scale 协议：目标 train 400、query/cluster-disjoint heldout 150，允许范围分别为 300–500、100–200，并排除所有 smoke query/cluster。Key 继续只接收 exact non-low A/B 共识；Complete 使用同一局部共识和显式 mask。

随后严格分步：

1. P0：只验证 direct Key/Complete 在新 heldout 上是否可学；
2. P1：在完全相同数据上增加 mutual distillation，验证是否有增量；
3. gate：只比较 gate-off 与固定 `gate_prior_weight=.25`，不重新扫权重；
4. gate 的最终选择效果用新的 query-disjoint ranking population；
5. Full 必须等 Consistency、H、Prior 各自机制和数据门都明确后再讨论。

## 7. 当前权限与运行入口

当前只允许只读 parent audit、确定性选样、建包和独立复算。准备入口为：

```bash
python prepare_clir_prior_v9.py prepare-smoke
python prepare_clir_prior_v9.py verify-smoke
```

准备和复算成功后，用户分别把 `launch_prompt_a.txt`、`launch_prompt_b.txt` 交给对应模型。AI 标注完成后才运行 `evaluate-labels`。在 raw gate 通过以前不使用 GPU。
