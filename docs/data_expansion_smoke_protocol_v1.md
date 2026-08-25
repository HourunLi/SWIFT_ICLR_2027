# CLIR 多题源数据扩充 smoke 协议 v1

冻结日期：2026-08-25 UTC

状态：`superseded_before_execution`（原状态 `frozen_pre_execution`）

证据等级：`pipeline smoke`

冻结时父代码：`clir-clean-integration@2e7250727cba2681797d9f02ac357b07230df247`

> **不要执行本版本。** 两份互盲外部审查均给出 block；v1 从未生成、标注、抽 feature 或训练任何数据。
> 当前协议为 [`data_expansion_smoke_protocol_v2.md`](data_expansion_smoke_protocol_v2.md)，审查裁决见
> [`data_expansion_smoke_review_resolution_20260825.md`](data_expansion_smoke_review_resolution_20260825.md)。

## 1. 这轮要回答什么

这轮只检验四件事：多题源取数是否可靠、Phi rollout 与数值 checker 是否兼容、双 AI
标注能否稳定落到 exact token 上，以及在正式扩量前预算与淘汰率是否可接受。

它不用于证明 Consistency、Hallucination、Key/Complete 或 gate 能提高 Best-of-N，也不用于
选择新的 loss 权重、gate 权重或 epoch。当前 `.25` prior-to-reward gate 继续作为
`dev-tuned engineering default` 保持开启；smoke 期间不调它。

如果本协议在执行前发生任何语义、数据源、模型、prompt、阈值或裁决规则变化，必须发布
`v2`，不得静默覆盖 v1。

## 2. 冻结的数据源与边界

| 数据源 | 本轮角色 | 固定来源 | 许可与限制 |
|---|---|---|---|
| GSM8K | 主要训练侧 smoke | `openai/gsm8k`，只用官方 train；数据 revision `740312add88f781978c0658806c59bc2815b9866` | MIT；官方 test 不得访问或进入训练/调参 |
| ASDiv-A | 多题源训练侧 smoke | `chaochun/nlu-asdiv-dataset` commit `883f90a9a65bf00304ba8f37423910fe743abc47`；`ASDiv.xml` 加 5 个 `asdiv-a` fold ID，共 1218 题 | CC BY-NC 4.0；本协议假定非商业论文研究 |
| SVAMP | 未来外部鲁棒性测试 | `arkilpatel/SVAMP` commit `689d7ccac74b9983a2ac7cc3b264f441b99e7c53` | MIT；完整 1000 题保持 protected，不进入本轮训练侧 smoke |

固定的 ASDiv 文件校验包括：

- `dataset/ASDiv.xml` SHA-256：`ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929`；
- `asdiv-a/fold0..4` 分别有 `238/238/238/238/266` 个 ID，总计 `1218`；
- 只接收能够解析为单个有限数值或有理数答案的 ASDiv-A 题目。

本轮不加入 MATH、AQuA 等会改变答案形态的题库。先保持“英文算术题 → 单一数值答案”的
共同任务，避免把模块效果与符号等价 checker、选择题 checker 混在一起。

## 3. 冻结的最终数量

下表中的数量都是**质检与裁决完成后的合格数量**，不是最初送标数量。

| 数据部分 | GSM8K | ASDiv-A | 最终数量 |
|---|---:|---:|---:|
| Outcome query | 30 | 20 | 50 queries |
| Outcome rollout | 240 | 160 | 400 trajectories，即每题 8 条 |
| Consistency group | 18 | 12 | 30 groups，每组 2 个 native views |
| 正 hallucination onset | 12 | 8 | 20 trajectories |
| explicit clean | 12 | 8 | 20 trajectories |
| Key/Complete | 复用正 onset 与 clean | 复用正 onset 与 clean | 同一批 40 trajectories |

约束如下：

- 50 个 outcome query 全部分配到训练侧 smoke 池，不作为 held-out dev/test；
- 30 个 Consistency group 来自 30 个不同 `query_id`，每个 query 最多一个 group；
- 40 个 H/P trajectory 来自 40 个不同 `query_id`，每个 query 最多一条；
- Consistency 与 H/P 可以在同一训练侧 query 上重叠，但这种重叠必须进入报告；
- 先从 400 条 rollout 中提议最多 40 个 Consistency pair 和最多 60 个 H/P trajectory，
  经过双标和裁决后分别留下 30 与 40；
- 若 400 条 rollout 无法在固定标准下凑足最终数量，smoke 判为未通过。不得放宽标签定义凑数；
  是否增加 query 或候选数必须另发协议修订。

## 4. query 选择、去重和 split

### 4.1 身份

统一使用带数据源命名空间的 ID：

```text
gsm8k:train:<source_index>
asdiv-a:<Problem-ID>
```

`query_id` 只表示原始问题及其候选池；`semantic_id` 只表示 Consistency 关系组，二者不得互相
替代。一个 query 的全部候选、视图和机制标签必须处于同一 split。

### 4.2 选择顺序

1. 先排除官方 test、SVAMP、旧 496-query outcome train、旧 500-query ranking population、
   旧机制标签 query，以及任何已经保留给新 dev/test 的 query。
2. 对题目正文做 Unicode/空白/大小写/标点规范化，规范化文本完全相同者硬删除。
3. 对剩余题目做跨数据源近重复筛查。高相似或只改数字/实体的模板题进入双 AI 重复判断；
   任一标注者认为是同一底题改写时，保守删除后出现者。
4. 使用 `sha256("clir-smoke-v1|" + query_id)` 排序；在各数据源合格池内按该顺序取题，
   不按模型表现或答案难度挑选。
5. 在生成任何候选前发布有序 query manifest、排除清单、数据 revision 与 SHA-256。

必须先按原始 query 完成选择和 split，再生成候选、挑 Consistency pair 或做机制标注。

## 5. 冻结的 rollout 协议

本轮继承已经实际跑通的 Phi 生成身份：

```text
model_id: microsoft/Phi-3.5-mini-instruct
model_revision: 2fe192450127e6a83f7441aef6e3ca586c338b77
tokenizer_revision: 2fe192450127e6a83f7441aef6e3ca586c338b77
generation_backend: vLLM
candidate_count: 8
temperature: 1.0
top_p: 0.9
max_new_tokens: 1024
max_model_length: 4096
seed: 42
terminal_token_policy: as_returned
```

统一 prompt 为：

```text
Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
<QUESTION>
```

必须保留原始 `prompt_token_ids`、`output_token_ids`、response、backend response、finish reason、
candidate index、模型/分词器 revision、chat-template hash、生成库版本与代码 commit。
`candidate_index` 固定为 `0..7`；Best-of-N 前缀不得重新排序。

保存的 token IDs 是唯一位置真相。不得从 response 文本重新 tokenize 后替代它们。

## 6. Outcome checker 协议

smoke 使用一个待实现并在执行前冻结 hash 的多题源数值 checker：
`clir_numeric_multisource_v1`。

- GSM8K 必须复用现有 `clir_gsm8k_numeric_v5` 的候选答案抽取与数值规范化语义；
- ASDiv-A reference 从 `<Answer>` 读取数值，单位另存为 provenance；
- 两个数据源都比较规范化后的单一有限整数、小数、分数或百分数；
- 缺失单位不影响本轮 numeric correctness；明确的单位信息作为诊断保留，不偷偷改变标签；
- reference 无法唯一解析的题目在生成前剔除；candidate 无法解析则记录明确 failure reason，
  按当前 numeric task 记为 incorrect；
- 保存 raw/reference answer、parsed answer、normalization path、status、checker version 与 hash；
- AI 不通过投票覆盖确定性 checker。若发现 checker 规则错误，先修 checker、升协议版本并全量重标。

train、未来 ranking validation 与正式 test 必须使用同一 checker 版本；不得再出现 v4/v5 混用。

## 7. Consistency group 怎么产生

每个 group 从同一 query 的 8 条 Phi native rollout 中选两条，不额外让改写模型生成文本：

- 两条候选的 numeric correctness 与规范化最终答案必须相同；
- 一条为相对 compact，一条为相对 expanded，token 长度比至少 `1.25`；
- 两条必须在措辞、步骤拆分或组织方式上有可见差异；
- 两条必须保持同一核心前提、解题方法、关键中间量、最终结论；
- 如果原推理有错误，两条必须保留同一个错误机制、语义位置和下游影响；
- 换了一条独立解法、修复旧错误、引入新错误或只是近乎复制，都必须 reject。

两条 view 共享 `query_id` 和新的 `semantic_id`，分别使用
`style_id=native_compact/native_expanded`。双 AI 只看问题和两条匿名 trajectory，不看
correctness、reference answer、长度排序规则或另一标注者输出。

## 8. H 与 Key/Complete 的同批标注

### 8.1 固定 reasoning unit

程序先依据保存的 exact output IDs 和原始 response 建立固定的 material-claim units。每个 unit 包含：

```json
{"unit_index":3,"text":"7 + 5 = 13.","token_start":42,"token_end":49}
```

`[token_start, token_end)` 直接指向保存的 `output_token_ids`。纯标题、过渡句和 final wrapper 要么
单独成 unit，要么标为 `non_claim`。若一个必要 claim 无法用固定 unit 表达，该 row 必须标为
`insufficient_unitization`，不能由 AI 自己拆句或重新分词。

### 8.2 Hallucination onset

- `clean`：所有 material claims 都可由题目、稳定常识或此前有效推理支持；
- `hallucinated`：至少一个 material claim 可被明确反驳，或缺少必要前提；
- onset：response 顺序中第一个明确 contradicted/unsupported material claim；
- `uncertain`：无法可靠判断最早错误，不得强行标 clean 或 positive；
- 最终答案正确不保证 clean；最终答案错误也不自动给出 onset；
- incorrect checker 结果只用于提议候选，标注 AI 看不到 checker 和 reference answer。

positive onset 最终 materialize 为所选 unit 的 `token_start`；explicit clean materialize 为
`hallucination_onset=-1`。缺失/淘汰标签保持字段缺失，绝不能用 `-1` 冒充未知。

### 8.3 Key/Complete

- `complete_unit_indices`：重建并审计 trajectory 实际推理链所需的最小非冗余 unit 集；
- `key_unit_indices`：complete 中最直接决定结论是否成立的最小 decisive unit 集；
- 错误 trajectory 仍然可标；key 通常是最早或最具因果决定性的错误步骤；
- 必须满足非空、升序、无重复且 `key ⊆ complete`；
- 只有答案、拒答或无法审计的 unitization 必须判 ineligible。

同一批 20 positive + 20 clean 都做 Key/Complete，但 H 与 prior 必须是两个独立 blind pass。
prior 标注者看不到 H 标签、correctness、reference answer或另一标注者输出，防止把“首错”机械复制成 key。

## 9. 双 AI 与盲裁

### 9.1 两个主标注者

- A、B 必须是两个独立调用；优先使用不同模型系列和固定 revision；
- exact model ID/revision、system/user prompt SHA-256、temperature、seed、API/provider 与原始响应
  必须在执行 manifest 中冻结；未填完不得开跑；
- 两者使用同一语义指南，但 item 顺序分别确定性打乱；
- 两者看不到对方输出、预期通过率、selection reason、checker label、reference answer、历史结果；
- 输出必须为严格 JSONL，并经 schema、item 顺序、unit index 和 exact-token range validator 检查。

如果只能使用同一模型，必须明确记录为 `same_model_independent_calls`，证据置信度下降；不得把两次调用
描述为两个独立模型。

### 9.2 分歧裁决

- A/B 完全一致且 confidence 非 low 时可直接接收；
- 任一 `uncertain/review/low`、C decision 分歧、H path/onset unit 分歧、prior eligibility 或 set 分歧，
  都进入盲裁；
- 裁决者只看到原 item 和匿名、随机顺序的方案一/方案二，不知道模型身份；
- 裁决只处理分歧行，不是第三套全量标注；优先使用第三模型。若只能复用 A/B 中较强模型，必须新建
  clean context 并如实记录；
- 裁决仍为 uncertain/low，或不能满足结构约束的 row 直接丢弃；不得多数票硬凑；
- 所有最终标签统一命名 `silver_dual_ai_v1`。无人类复核，因此绝不能写成 Gold。

## 10. 双标输出最小 schema

Consistency：

```json
{"schema_version":"clir-consistency-silver-v1","item_id":"COPY","decision":"accept","same_task_and_goal":true,"same_core_premises":true,"same_reasoning_method":true,"same_key_inferences":true,"same_intermediate_conclusions":true,"same_final_conclusion":true,"error_alignment":"both_clean","style_difference_satisfied":true,"confidence":"high","issues":[]}
```

Hallucination：

```json
{"schema_version":"clir-onset-silver-v1","item_id":"COPY","path_status":"hallucinated","onset_unit_index":3,"error_type":"arithmetic","confidence":"high","reason":"The first false claim is 7 + 5 = 13."}
```

Dual prior：

```json
{"schema_version":"clir-dual-prior-silver-v1","item_id":"COPY","eligibility":"usable","key_unit_indices":[3],"complete_unit_indices":[1,2,3],"confidence":"high","rationale":"Units 1-3 form the minimal audit chain and unit 3 is decisive."}
```

所有 schema 都必须 fail closed：缺字段、多字段、非法枚举、越界 index、错误 nesting 或非严格 JSON
都不能自动修成训练标签。

## 11. smoke 的预注册通过门

### 11.1 数据与生成硬门

- 最终恰好 `30 GSM8K + 20 ASDiv-A` query、每题 8 candidates，共 400 rows；
- candidate index 对每个 query 都为连续 `0..7`；
- 与旧 train/ranking/mechanism query、官方 test 和 protected SVAMP 的 overlap 为 0；
- normalized exact duplicate 为 0，所有 near-duplicate flag 均有处置记录；
- reference answer 100% 唯一可解析；所有 checker rows 有版本、status 和 failure reason；
- 同一 query 的 prompt IDs 完全一致；所有 output IDs 合法非空；
- unit text、字符 span、token span 与保存的 output IDs 100% 可复核。

### 11.2 标注质量门

| 部分 | v1 扩量门 |
|---|---|
| Consistency | 最终 30 groups；A/B 原始 decision agreement ≥90%；需盲裁行 ≤25% |
| H path | 最终 20 positive +20 clean；A/B path agreement ≥80%；需盲裁行 ≤50% |
| H onset | 在 A/B 都判 positive 的行上，exact onset-unit agreement ≥60% |
| Prior eligibility | A/B eligibility agreement ≥95% |
| Key | usable overlap 上 macro unit-set F1 ≥0.60 |
| Complete | usable overlap 上 macro unit-set F1 ≥0.80 |
| Prior structure | A、B、裁决和最终标签的 `key ⊆ complete` 均为 100%；需盲裁行 ≤75% |

这些门先看原始 A/B agreement，再做裁决；不能用裁决后的统一结果冒充两个标注者天然一致。
若任一门失败，先修改 unitization/指南/模型组合并发布新版本，不能直接正式扩量。

## 12. 产物与 provenance

每一步都必须保留：

- frozen source/query/split manifest 及 SHA-256；
- rollout shard、completion marker、candidate order 与完整生成 provenance；
- checker config、代码 commit、原始解析结果与版本 hash；
- unitization 文件、字符/token span、`output_token_ids` SHA-256；
- A/B 原始 item、prompt、响应、解析结果、模型 revision 与调用参数；
- agreement report、匿名裁决 package、裁决原始输出、最终 Silver 标签；
- 每一步的输入/输出 hash、失败/淘汰原因和可恢复 checkpoint；
- 最终 manifest 的 query/source/label coverage 报告。

大规模全层 BF16 feature 在文本、checker、unitization 和标注门通过后再抽取，写入全新目录，不就地
覆盖旧 feature。这样避免先花数十 GiB 甚至更多存储，再发现协议或标签不能用。

## 13. smoke 通过后的边界

smoke 通过只授权起草正式扩量执行协议。正式训练规模、GSM8K/ASDiv-A 最终比例、独立 dev/ranking
query 数量和 protected-test 开启条件仍需结合本轮各数据源的错误率、标注 yield、裁决成本和许可约束
再次冻结。

下一轮效果实验仍遵守现有裁决：

- C/H 主矩阵完整重跑 `C0/C1/H0/CH0`；
- gold negative tail、MIL 和 pseudo-onset tail 不进入下一轮 C/H 核心 cell；
- prior 先复核 direct Key/Complete，再在新数据上固定比较 gate off 与 `.25` on；
- `.25` 默认保持开启，但不得把它写成已经证明有效；
- 所有结论分别报告工程闭环、辅助目标可学性和 held-out Best-of-N 效果。
