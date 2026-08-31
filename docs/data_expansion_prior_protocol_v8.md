# CLIR Dual Prior 数据扩量协议 v8：先标依赖图，再机械生成 Key/Complete

## 1. 为什么不能直接放大旧 Prior 标注

历史可训练 Prior 只有 48 条。v3 的 60 条新 smoke 中，两位 AI 的 Key/Complete macro F1 已经达到 `.9167/.9267`，说明大方向接近；但只有 25/60 条 Key 和 Complete 同时完全一致，需要裁决的最低比例是 35/60=`.5833`，超过冻结上限 `.40`。分歧主要不是两边走了完全不同的解法，而是一个 Complete 集合经常是另一个的严格子集。

所以 v8 不再让 AI 直接猜一个可能不唯一的“完整集合”。AI 标候选实际推理中 `哪一步依赖哪一步`、哪一步直接得出结论，以及错误链的第一个致命错误；程序用统一规则求闭包，得到最终 Key/Complete。

## 2. 本轮只做 60 条 smoke

本轮复用 v6 已经生成、完成 checker v3 和 exact-token unitizer v2 校验的 16,000 条候选，不需要重新运行 Phi，也不需要 GPU。它不是复用 v3 的已检查 Prior 行：v6 在生成前已经排除历史和 v2/v3/v4/v5 题目。

为避免和 Consistency 的新监督纠缠，本轮再排除 v6.1 已选 feature inventory 涉及的全部 612 个 query 和 602 个 template cluster。只从剩余 `train_acquisition` 中选择 60 个不同 query、60 个不同 cluster；同时机械核验整个 v6 池与 v7 H、v7 ranking 的 query/cluster 都是零重合。

固定四个 strata，每格 15 条：

| 来源 | checker 结果 | 数量 |
|---|---|---:|
| GSM8K train | numeric match | 15 |
| GSM8K train | numeric mismatch | 15 |
| MATH train | numeric match | 15 |
| MATH train | numeric mismatch | 15 |

checker 只用于冻结前分层，标注者看不到 source、checker、参考答案或 CLIR 分数。候选必须 normal stop、可监督、unitization=ok，且有 6–40 个 material claim。每题按固定 SHA-256 优先级最多选一条，不能在看 AI 输出后换题。

这 60 个 query 一旦用于 prompt smoke，永久烧成 prompt-development/train-side audit，不能再进入后续 Prior train、heldout、ranking 或 protected test。

## 3. AI 实际标什么

对 usable 候选，AI 输出：

- `path_status`：supported、flawed 或 uncertain；
- `conclusion_unit_indices`：直接得到候选最终答案的实质步骤，不含重复 final wrapper；
- `dependency_edges`：`[parent, child]`，表示 child 实际使用 parent 的新结果或主张；
- `first_flaw_unit_index`：仅 flawed 路径填写，且必须在通向结论的依赖闭包里；
- confidence 和一句理由。

程序随后确定性生成：

```text
Complete = 从 conclusion units 沿 parent 边向前得到的传递闭包
Key(supported) = conclusion units
Key(flawed)    = first flaw unit
```

这样保留的是候选实际走过的链，而不是另一位解题者能重算出的最短证明。题面复述、计划话术、没被结论使用的旁枝、重复等式、同义重复和 final wrapper 不连入结论闭包。

## 4. 双 AI 与盲包

- A：用户报告 GPT-5.6-sol，xhigh；
- B：用户报告 Claude Opus 5，high；
- 两者必须使用全新、彼此独立的上下文，也不能看到 H 标签；
- exact revision 和 temperature 若产品不暴露，继续记录为 unverified；
- 每边 60 条 natural +6 条隐藏控制；A 另有 12 条盲重复；
- 因此 A 只有一个 78 行 shard，B 只有一个 66 行 shard；
- `PRIVATE_package_index.jsonl` 永远不能发给标注模型。

无人类复核时，未来标签只能叫 `silver_dual_ai_dependency_prior_v8`，不能叫 Gold、verified 或人工近似标签。

## 5. 冻结的 raw 门

所有 60 条 natural 都留在分母。必须同时满足：

| 指标 | 门槛 |
|---|---:|
| eligibility agreement | ≥.95 |
| 双方 usable 支持 | ≥50/60 |
| usable path agreement | ≥.90 |
| derived Key macro F1 | ≥.90 |
| derived Complete macro F1 | ≥.90 |
| 非低置信 exact derived Key+Complete+path | ≥42/60 且 ≥.70 |
| 最低需裁决比例 | ≤.30 |
| 双方共同 flawed 支持 | ≥12 |
| first flaw exact | ≥.75 |
| hidden controls | A/B 各 6/6 |
| A self-repeat | ≥.95 |
| Complete=全部 material units | A/B 各 ≤.25 |

任一门失败就停在 `STOP_PRIOR_DEPENDENCY_SMOKE_V8_RAW_GATE_FAILURE`：不发第三模型、不改分母、不改闭包算法、不抽 feature、不训练。通过只说明依赖图目标在这套指南下可操作、较稳定，不证明标签事实准确，更不证明 Prior、gate 或 Best-of-N 有效。

## 6. 通过后也不能直接训练 Full

只有 smoke 全过，才另开并冻结 scale 协议；目标 train 400、query/cluster-disjoint heldout 150，允许范围分别为 300–500 和 100–200。scale 必须排除这 60 个 smoke query/cluster，最终训练标签默认只接收 A/B 非低置信 exact derived target 共识，不用第三模型救行。

训练顺序仍固定：

1. P0：先验证 direct Key/Complete 是否在新 heldout 上可学；
2. P1：再看 mutual distillation 是否有增量；
3. gate：只比较 gate-off 与固定 `gate_prior_weight=.25`，不重新扫权重；
4. gate 的排序效果必须用新的 query-disjoint ranking population；
5. smoke 通过不自动解锁 Full。

## 7. 当前权限边界

当前只授权只读 parent audit、确定性选样、建包和独立复算。AI 标注、裁决、feature extraction、训练都尚未由本协议预先授权。待两个公开包通过复算后，再由用户把 A/B 各自的单 shard 发给相应模型。
