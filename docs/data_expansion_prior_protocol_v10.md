# CLIR Dual Prior 数据扩量协议 v10：统一回溯口径

## 1. v9 为什么失败，v10 改什么

v9 已经终止在 `STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE`。它不能重标翻盘，也不能从已经看过的 39/60 行里挑训练子集。v9 的只读诊断很明确：Claude 的 Complete 是 GPT Complete 的严格子集 53/60，平均集合大小约 `6.03` 对 `10.87`。这不是零散噪声，而是两位标注者分别采用了“最短可验证骨架”和“候选实际完整计算链”两种都能被旧提示词解释的口径。

`origin/main` 只规定 Key 较窄、Complete 是更完整的支持 span，没有给出唯一的人工标注算法。v10 不改变模型、loss、局部 mask 或 `.25` gate 路径，只把标注自由度冻结成同一个简单算法：

- usable 行的 Key 恰好一个；错误主线取最早致命错误，否则取首次完成答案的最后一个非包装步骤；
- Complete 从最终实质结论向前回溯候选实际使用的依赖；
- 分开的“代入算式 + 计算结果”都保留，单行已自包含时不再收重复结果；
- 题面复述、计划、通用公式、未用旁枝、重复和 final wrapper 排除；
- 不允许把候选改写成另一条更短解法。

这仍保持 main 的方法身份：Key 是窄锚点，Complete 是较宽的实际支持链。

## 2. 全新 60 条定义 smoke

本轮继续复用 v6 已完成 checker v3 和 exact-token unitizer v2 的 16,000 条候选，不重新 rollout，不需要 GPU。固定选择 60 个不同 query、60 个不同 template cluster，GSM8K/MATH × numeric match/mismatch 四格各 15 条。

选择前统一排除：

- v6.1 Consistency 使用的所有 query/cluster；
- v7 H 子集与 v7 ranking population 的所有 query/cluster；
- v8、v9 Prior smoke 的所有 query/cluster。

机械条件仍为 `train_acquisition`、normal stop、可监督、unitization=ok、6–40 个 material claim。每个 query/cluster 最多一条，SHA-256 优先级在标签前固定，不看 CLIR 分数、checker 分层身份或 AI 标签。v10 的 60 题也是永久烧掉的 prompt-development/train-side smoke，不能进入未来 Prior train、heldout、ranking 或 protected test。

## 3. 双 AI 包

A 为用户报告的 GPT-5.6-sol xhigh，B 为 Claude Opus 5 high；两边必须使用互相独立的新上下文，且不能看另一方结果、checker、参考答案、历史标签或 `PRIVATE_package_index.jsonl`。

每边收到一个 80 行 shard：

- 60 条全新 natural；
- 8 条隐藏控制，覆盖拆分计算、题面复述、未用旁枝、最早错误、晚发语义错误、重复结果和 answer-only；
- 12 条各自独立盲重复。

两边都测 self-repeat，避免只知道一位标注者是否稳定。格式不合法、Key 不为单元素、索引越界、Key 不属于 Complete、漏行或多行都 fail closed。

## 4. 局部共识仍怎样变成监督

只有双方都判 usable 且都不是 low confidence 时，才考虑未来 materialization。

- Key：双方单元素集合完全一致才整行训练；否则该行不提供 Key target。
- Complete：交集为正、并集外为负、对称差 unit 使用 `complete_prior_mask=0`；交集必须非空。
- Key/Complete attention 仍在完整有效 trajectory 上归一化，禁止只在 coverage 子集上重新 softmax。

v10 smoke 本身无论通过与否都不发布训练行。通过后另冻 scale 协议，使用全新 query/cluster。

## 5. 冻结的 raw 门与“扩量取子集”规则

60 条 natural 全部留在原始分母，不裁决、不用第三模型修复。

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
| 隐藏控制 | A/B 各 8/8 |
| self-repeat | A/B 各 ≥.95 |
| Complete=全部 material units | A/B 各 ≤.25 |

结果分三类：

1. 全门通过：允许准备一份独立 scale 协议。
2. 只有数量/yield 门失败，而 F1、unit agreement、controls、self-repeat 和反退化门都通过：允许按用户授权准备“先扩量、再按事先固定规则取严格共识子集”的 scale 协议。proposal 数、train/heldout split、shard、接收条件和 first-N/hash 顺序必须在 scale 标签出现前冻结；所有未入选行仍留在原始分母。
3. 任一语义一致性、控制题或稳定性门失败：说明提示词仍没有统一定义，禁止直接烧大预算扩标。

第二类不是把 v10 smoke 的容易行拿来训练，也不是事后手挑；它只允许下一轮在一份预先冻结的大池上做前瞻性的严格共识筛选。目标仍为 400 train +150 query/cluster-disjoint heldout。实际 proposal 数要等 v10 的冻结 yield 出来后按公式另行固定。

## 6. 证据边界和后续训练

通过只说明双 AI 在这套统一回溯算法下具有足够的 target operability。无人类复核时，未来标签只能叫 `silver_dual_ai_canonical_prior_v10`，不能叫 Gold、人工验证、事实准确或高质量真值。

scale 成功后仍按以下顺序做独立实验：

1. P0：direct Key/Complete heldout learnability；
2. P1：完全相同数据上增加 mutual distillation；
3. gate-off 对固定 `gate_prior_weight=.25`，不重新调权重；
4. gate 的选择效果用 query-disjoint ranking population；
5. Full 不由本 smoke 自动解锁。

## 7. 当前入口

```bash
python prepare_clir_prior_v10.py prepare-smoke
python prepare_clir_prior_v10.py verify-smoke
```

准备和独立复算通过后，用户才分别把 `launch_prompt_a.txt`、`launch_prompt_b.txt` 交给两位模型。两份标签都完成后运行 `evaluate-labels`。raw gate 前不用 GPU，也不允许抽 feature 或训练。
