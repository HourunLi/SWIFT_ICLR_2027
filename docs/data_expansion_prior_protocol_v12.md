# CLIR Prior v12：全新大池双标与预注册严格共识扩量

状态：`FROZEN_PREPARATION_ROLLOUT_NOT_STARTED`。本协议只先授权训练题源审计和
pre-rollout manifest 冻结；rollout、checker/unitizer、AI 标注、feature 和训练分别由后续
hash-bound 授权解锁。

## 1. 为什么另开 v12

v11 的两份 80 行标签已经完整写完并通过 schema、ID、盲包、控制题和自重复校验，但冻结
raw gate 以 `STOP_PRIOR_VERIFIED_SMOKE_V11_DEFINITION_FAILURE` 终止：60 条自然样本的
Key macro F1 为 `.8333 < .90`，Complete 正集合 IoU 为 `.7957 < .80`。这不是数量不足，
所以不能从 v11 事后挑 50 条共识行训练，也不能把门槛改到刚好通过。

只读诊断显示分歧集中在两类边界：错误链的“最早致命错误”与下游错误结果，以及长 MATH
解答中互相等价的文字/等式和重复枚举到底保留多少。简单限制到短题会产生查看结果后的容易题
偏差，因此 v12 不用该办法。

v12 是用户此前同意的另一条路线：**在全新 query 上先双标一个大池，再只保留满足事先写死
规则的严格共识行**。它不宣称整个 Key/Complete 定义已经稳定，也不复用 v8--v11 的任何标签。

## 2. 题源、去重和 split

- 只读 GSM8K train 与 MATH train，不读取或使用 GSM8K/MATH test；SVAMP 继续保护。
- GSM8K 继承 v7 的长链筛选；MATH 使用支持单一数值答案的 7 个学科、level 2--5、官方解答
  至少 25 词。Level 2 是公开记录的新题源扩展，不是根据 rollout/AI 结果挑题。
- 继承 v7 的精确文本、实体/数字模板、MinHash/Jaccard 聚类算法；任何 cluster 只要触及历史、
  v6、v7 ranking/H 或 Prior v8--v11 query，就整簇排除。
- 先在生成前冻结 2,000 个不同 query/cluster：GSM8K 1,200、MATH 800；train acquisition
  1,600、mechanism dev 400。train/dev query 与 cluster 均不重叠。
- 这 2,000 题永久烧成训练/机制开发侧，不能进入未来 ranking validation 或 protected test。

2026-08-31 的只读容量审计已通过：8,228 个筛选后 source candidate 与 608 个排除锚点重建为
8,387 个 cluster，得到 2,730 个 selectable rows、2,647 个 cluster representative，足以
冻结上述 2,000 题；与所有排除 query/cluster 的 overlap 均为 0，且没有读取 test 文件。

## 3. rollout 与 materialization

- 生成器固定为 `microsoft/Phi-3.5-mini-instruct` revision
  `2fe192450127e6a83f7441aef6e3ca586c338b77`。
- 每题一次 vLLM 请求生成 8 条候选：temperature `1.0`、top-p `.9`、最多 1,024 新 token；
  query seed 由冻结 namespace 和 query ID 决定。
- 共 40 个 shard，每 shard 50 query，合计 16,000 trajectories；先跑并核验一个 calibration
  shard，再允许最多 8 张单卡并行。完整 shard 复核后跳过，不完整文件停止且不覆盖。
- 保存的 `prompt_token_ids`/`output_token_ids` 是唯一 token 轴真值。
- rollout 全部完成后，另行授权用 `clir_numeric_multisource_v3` 与
  `clir_material_claim_unitizer_v2` 做 checker/unitizer；截断、解析失败、答案冲突、坏分区只
  留审计，不能进入标注池。

## 4. 冻结的 800 条自然标注池

每个 query 最多一条 trajectory，每个 cluster 最多一个 query；候选只按 checker、unitizer、
material unit 数 `6..40` 和冻结 SHA-256 priority 选择，不看 CLIR score 或 AI 标签。

| 题源 | checker | train | dev | 合计 |
|---|---|---:|---:|---:|
| GSM8K | numeric match | 128 | 32 | 160 |
| GSM8K | numeric mismatch | 80 | 20 | 100 |
| MATH | numeric match | 192 | 48 | 240 |
| MATH | numeric mismatch | 240 | 60 | 300 |
| 合计 |  | 640 | 160 | 800 |

若任一格凑不满，流程以 `FAIL-yield` 停止；不得临时换比例或再采一轮。

## 5. 双 AI 标注

- A：用户报告 GPT-5.6-sol / xhigh；B：用户报告 Claude Opus 5 / high。必须在互相独立的新
  上下文中完成，不能读取 PRIVATE、checker、参考答案、另一模型或历史标签。
- Key/Complete 定义和 v11 完全一致：先逐 unit 验算；错误实际主线取最早致命错误作为唯一
  Key，否则取首次完成答案的最后非包装 unit；Complete 按候选实际主线向后回溯。
- 800 条自然样本按每边 16 shard ×50 natural 分发；每边另有 16 条隐藏控制和 80 条盲重复。
- 不使用第三模型或裁决；所有 800 条自然行始终保留在 raw denominator 中。

## 6. 严格共识选择与最终规模

一行只有同时满足以下条件才进入 eligible pool：A/B 都判 usable、confidence 都非 low、唯一
Key 完全相同、Complete 交集非空。Key 共识 unit 为正、其他有效 token 为负；Complete 的
交集为正、并集外为负、对称差用显式 mask 跳过。attention 仍在完整有效 trajectory 上归一化，
不能只对 coverage 子集重新 softmax。

最终不按置信度或模型分数挑行，而是在下面每个预冻结格内按生成前已有的 selection priority
取前 N 条：

| 题源 | checker | train | dev | 合计 |
|---|---|---:|---:|---:|
| GSM8K | numeric match | 80 | 20 | 100 |
| GSM8K | numeric mismatch | 40 | 10 | 50 |
| MATH | numeric match | 120 | 30 | 150 |
| MATH | numeric mismatch | 160 | 40 | 200 |
| 合计 |  | 400 | 100 | 500 |

完整通过还要求：两边控制题各至少 `15/16`、自重复各至少 `.95`、所有最终格都凑满、最终 500
行的 Complete 平均正集合 IoU 至少 `.80`、平均 mask coverage 至少 `.90`、Complete=全部
material units 的比例每边不超过 `.25`。任一失败即终止，不追加 rollout、不重标、不降门。

## 7. 证据边界和后续顺序

最终标签只能称
`silver_dual_ai_strict_consensus_prior_v12_no_human_verification`。严格共识会偏向容易且定义清楚
的样本，必须同时报告 raw population 和入选前后题源、checker、长度分布；不能称 Gold、人工
验证、总体定义已稳定或事实准确。

数据门全过后，才可另行授权 selected-only feature extraction。训练顺序固定为：先做 matched
correctness-only vs P0 direct-prior learnability；P0 可学后再测 P1 mutual 增量；最后在新的
query-disjoint ranking population 上固定比较 gate off 与 `.25`，不得再调 gate 权重。任何数据
PASS 都不会自动解锁 Full，也不构成 Prior、gate 或 Best-of-N 效果证据。
