# CLIR 多题源数据扩充 smoke 协议 v2

冻结日期：2026-08-25 UTC

状态：`review_integrated_pre_execution`

证据等级：`pipeline smoke`

父代码：`clir-clean-integration@2e7250727cba2681797d9f02ac357b07230df247`

v2 吸收了两份互盲外部 AI 审查。v1 尚未生成、标注或训练任何数据，现标记为
`superseded_before_execution`，不得继续执行，也不得原地修改成 v2。

## 1. 这轮只回答什么

这轮只检验下面五件事：

1. GSM8K train 与 ASDiv-A 能否按冻结规则取数、去重和生成；
2. Phi rollout 与统一数值 checker 能否稳定闭环；
3. Consistency、first-bad-unit 和 Key/Complete 的候选提议是否有足够 yield；
4. 两个不同模型系列的 AI 能否在互盲条件下稳定执行同一标注定义；
5. fixed unit 到保存的 exact output token IDs 能否 100% 无歧义物化。

本轮不训练 CLIR，不比较 Best-of-N，不选择 epoch、loss 权重或 gate 权重，也不证明三个模块有效。
`.25` prior-to-reward gate 只是现有 `dev-tuned engineering default`；本轮既不打开训练，也不再调它。

通过 smoke 只表示可以另发正式扩量协议。失败也必须区分：

- `FAIL_PIPELINE`：checker、unitizer、schema、盲标一致性或裁决规则有问题；修复后必须升版本；
- `FAIL_YIELD`：定义没有坏，但冻结题池产不出足量合格候选；保持定义不变，另发扩大 query 池的版本；
- `FAIL_DIVERSITY`：数量够但链长、onset 位置或来源分布退化；不得靠挑容易样本补门。

## 2. 数据源、SVAMP 和证据边界

| 数据源 | v2 角色 | 固定来源 | 数量/限制 |
|---|---|---|---|
| GSM8K | 训练侧 smoke 主来源 | `openai/gsm8k` revision `740312add88f781978c0658806c59bc2815b9866` | train only，60 queries |
| ASDiv-A | 多题源训练侧 smoke | `chaochun/nlu-asdiv-dataset` commit `883f90a9a65bf00304ba8f37423910fe743abc47` | 非商业研究，40 queries |
| SVAMP | 未来 protected contrast/challenge evaluation | `arkilpatel/SVAMP` commit `689d7ccac74b9983a2ac7cc3b264f441b99e7c53` | 本轮不读题、不训练、不调参 |

ASDiv-A 仍使用已核对的 `1218` 个 arithmetic fold IDs，`ASDiv.xml` SHA-256 为
`ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929`。只接受 reference answer
能够唯一解析为有限整数、小数、分数、混合数或百分数的题。

### 2.1 SVAMP 的准确说法

SVAMP [原论文](https://aclanthology.org/2021.naacl-main.168/)明确说明：其 1000 条题由从 ASDiv-A
选出的 100 个 seed examples 做变化得到。因此只要训练
使用 ASDiv-A，就不能把 SVAMP 叫作“与训练来源独立、完全未污染的外部测试集”。v2 采用以下保守表述：

- SVAMP 仍保持 protected，本轮不读取其题目、答案或模型表现；
- 若正式训练继续使用 ASDiv-A，SVAMP 只能报告为 **ASDiv-derived contrast/challenge set**，用于检查
  见过相关题型或 seed 来源后的变化鲁棒性；
- 它不能承担“独立来源泛化”结论；正式实验还需要另选真正独立的 holdout；
- 如果以后希望把 SVAMP 升格为更强的独立 holdout，必须先发布密封的 seed-family 去重协议，只向训练侧
  返回应排除的 ASDiv IDs/cluster IDs，不向选样和调参流程暴露 SVAMP 内容或性能。

本轮数据门只要求“没有直接使用任何 SVAMP row”，不再声称与 SVAMP 语义/模板 overlap 为 0。

### 2.2 公开基准和许可限制

- 生成器与标注模型可能在预训练中见过公开的 GSM8K/ASDiv；本轮只估计当前流水线的 yield，不能据此
  声称对新题泛化；
- GSM8K 继续按 MIT provenance 保存；ASDiv-A 每行保留来源和 CC BY-NC 4.0 标记，不发布成抹去来源的
  混合 blob；
- 外部标注 API 的 provider、数据保留和是否用于训练的设置必须记录；能使用 no-training/zero-retention
  时应启用。法律与服务条款是否满足由执行者另行确认，本协议不作法律结论。

## 3. 冻结规模：100 个 train-only query

v2 把 query 池从 50 扩到 100，增加的是 rollout GPU 时间，不增加自然标注行的上限。

| 数据部分 | GSM8K | ASDiv-A | 冻结数量 |
|---|---:|---:|---:|
| outcome queries | 60 | 40 | 100 train-only queries |
| generated rollouts | 480 | 320 | 800 rows，8/query |
| Consistency natural proposals | 40 | 0 | 40 pairs，40 distinct queries |
| Consistency final accepts | 30 | 0 | 30 groups |
| H/P natural proposals | 36 | 24 | 60 trajectories，60 distinct queries |
| final first-bad-unit positive | 每来源至少 5 | 每来源至少 5 | 20 trajectories |
| final explicit clean | 每来源至少 5 | 每来源至少 5 | 20 trajectories |
| Key/Complete | 与 H 同一 60 条 proposal 独立双标 |  | 最终复用同一批 40 |

约束：

- 100 个 query 永久记为 `train_only_smoke_v2`；今后可进入正式 train，但绝不能进入 mechanism dev、
  ranking validation、pilot test 或 final test；
- outcome、C、H、prior 共用一个全局 query 身份与永久排除表；
- C 与 H/P 允许 query 重叠，必须报告重叠 query 列表和比例；这不是 split 泄漏，但意味着多任务监督相关；
- 同一条 trajectory 不得同时承担 C view 和 H/P row；同 query 重叠时必须选择另一 candidate；
- 所有近重复/template cluster 在未来正式 split 时整体进入同一 split，不能只按 `query_id` 单行拆开。

## 4. query 选择、去重和永久排除

### 4.1 ID 与排序

```text
gsm8k:train:<source_index>
asdiv-a:<Problem-ID>
```

`query_id` 只表示原始问题与候选池；`semantic_id` 只表示经验证的 Consistency 关系。即使本轮每个
`semantic_id` 只对应一个 query，也不得用二者互相 fallback。

### 4.2 冻结步骤

1. 排除旧 outcome train、旧 ranking population、旧 mechanism queries、所有官方 test、已保留 dev/test
   和本轮永久排除表；
2. 对题目做 Unicode、空白、大小写和标点规范化，完全相同者聚为一簇；
3. 用数字/实体占位后的模板签名与冻结相似度检索器生成 near-duplicate candidate pairs；AI 只裁候选对，
   不自由浏览题库；
4. 一簇内保留 `sha256("clir-dedup-v2|" + query_id)` 最小者，其余删除；所有候选对、决定和 cluster ID
   落日志，不使用含糊的“删除后出现者”；
5. 合格池按 `sha256("clir-smoke-v2|" + query_id)` 排序，分别取 60/40；不按 Phi 正确率、错误率、
   标注难易或历史模块表现挑题；
6. rollout 前发布有序 query manifest、排除清单、cluster manifest、source revisions 和全部 SHA-256。

## 5. rollout 与截断规则

```text
model_id: microsoft/Phi-3.5-mini-instruct
model_revision: 2fe192450127e6a83f7441aef6e3ca586c338b77
tokenizer_revision: 2fe192450127e6a83f7441aef6e3ca586c338b77
backend: vLLM（执行前固定精确版本、TP、dtype、GPU 与 max_num_seqs）
candidate_count: 8（同一 query 的 n=8 放在一次请求中）
temperature: 1.0
top_p: 0.9
max_new_tokens: 1024
max_model_length: 4096
seed: 42
terminal_token_policy: as_returned
```

prompt：

```text
Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
<QUESTION>
```

必须保存 `prompt_token_ids`、`output_token_ids`、response、backend text、finish/stop reason、candidate index、
累计 logprob、chat-template hash、模型/分词器 revision、vLLM/Transformers/PyTorch/CUDA 版本、GPU 型号和代码
commit。保存 IDs 是唯一 token 轴；重跑不承诺逐 bit 复现，已保存 IDs/文本才是本轮事实。

`finish_reason=length`、空输出、非法 token ID 或明显未完成的 row：

- 仍保留在原始 800-row rollout 审计中；
- 不进入 correctness train label、C、H 或 prior proposal；
- 必须报告比例；截断比例超过 `2%` 时为 `FAIL_PIPELINE`，不得静默丢掉后继续报 800 条可训练数据。

## 6. checker：只承诺 numeric-value correctness

v2 checker 名为 `clir_numeric_multisource_v2`，必须在 rollout 前实现、测试并冻结代码 hash。

- GSM8K 继承 `clir_gsm8k_numeric_v5` 的 boxed/answer-cue、分数、小数、百分数、混合数和复合时长语义；
- ASDiv-A 从 `<Answer>` 读取 reference，并把 `<Unit>` 等来源字段单独保存；
- 公开主字段叫 `numeric_value_match`，训练兼容字段 `correctness` 只是它的 0/1 镜像，并必须附带
  `correctness_semantics=numeric_value_match_v2`；
- 缺单位不影响 numeric-value match；明确写错单位或实体可被 H 判为坏 claim，并记录
  `explicit_unit_status`，但在没有完整单位 ontology 时不假装 checker 已验证完整语义正确性；
- 因此任何报告只能说“数值 outcome 匹配”，不能说 checker 证明了完整语义答案正确；
- 多个 boxed answer 若规范化值相同，可取最后一个并记录计数；若互相冲突，状态为
  `ambiguous_multiple_answers`，该 row 不进入训练或机制 proposal；
- NaN/Inf、分母为 0、reference 多解或无法唯一解析的题在生成前剔除；candidate parse 失败记为
  numeric mismatch，并保存明确 failure reason；
- `checker_dispute` 只作审计 flag，不允许 AI 直接覆盖确定性标签；checker 规则变化必须升版本、重建
  proposal manifests，并重做所有依赖 checker 的标注。

最小回归集必须覆盖：整数、负数、逗号、小数、普通/LaTeX 分数、混合数、百分数、金额、复合时长、
无 box、空 box、占位 box、多个相同/冲突 box、显式单位、NaN/Inf 和截断。

## 7. `unitizer_v2` exact-token 契约

版本名：`clir_material_claim_unitizer_v2`。`panzhixin` 分支的行/句子 deterministic segmenter 可移植为
起点，但不能原样宣称满足 v2。

### 7.1 坐标和覆盖

每个 unit 至少包含：

```json
{
  "unit_index": 3,
  "kind": "material_claim",
  "text": "7 + 5 = 13.",
  "char_start": 41,
  "char_end": 54,
  "token_start": 42,
  "token_end": 49
}
```

- `token_start/token_end` 是相对于 `output_token_ids` 的零基、左闭右开 `[start,end)` 绝对位置；
- unit 按 token 顺序连续、无重叠，所有 unit 的 token ranges 必须恰好分割完整 `[0,T)`；
- 空白、标题、final wrapper 和不可见 terminal/control token 也必须由 `non_claim` unit 覆盖，不能消失；
- visible char ranges 必须无重叠并覆盖所有非空白可见字符；整段 decode 与保存 response 的关系必须记录；
- 允许用冻结 tokenizer 做 char-offset 验证，但只有当重新编码的 visible IDs 与保存 IDs 的可见前缀精确相同
  时才可映射；它不得替换保存 IDs。无法精确相等就整行 `insufficient_unitization`；
- 一个 `material_claim` unit 只能包含一个可独立判真假的实质 claim。若 token 边界迫使两个不同 claim
  融在一起，整行不适合 onset/prior，不能让 AI 猜一个更细位置；
- H/P natural proposal 每行至少 4 个 material-claim units，最终 40 条的 material-claim unit 中位数至少 5。

### 7.2 切分和回归测试

冻结实现需明确行边界、`Step N:`/`Answer:` 标题、列表编号、句末标点、LaTeX 公式和等式的处理，并对
`0.5`、`$1.50.`、`e.g.`、`Mr.`、多行等式、boxed answer、连续换行、Unicode 符号及 terminal control
token 建回归测试。所有 unit 文件、规则版本、代码 hash、输入/output-ID hash 和失败原因必须保存。

## 8. proposal manifests：先发布，再给 AI

自然 proposal 与隐藏 control items 分开计数。所有自然 proposal 的有序 manifest 和 SHA-256 必须在 A/B
看到任何 item 前发布；之后不能因为 A/B 不一致或 confidence 低而换候选。

### 8.1 Consistency：40 提议，30 入库

只从较长链的 GSM8K query 提议；ASDiv-A 仍参与 outcome/H/P，但本轮不强求它产生长短两种同方法推理。

对每个 query 的 8 条 candidate 枚举全部无序 pair，机械过滤：

- 两条均未截断、可精确 unitize，各至少 4 个 material claims；
- `numeric_value_match` 相同，规范化最终数值相同；
- output-token 长度比满足 `1.25 <= long/short <= 3.0`；
- candidate indices 不同，文本不是 exact/near copy。

每个 query 若有多个机械合格 pair，按
`sha256("clir-C-proposal-v2|<query_id>|<min_idx>|<max_idx>")` 取最小者；再按 query hash 取前 40 个不同
query。AI 决定它们是否真的保持同一前提、方法、关键中间量、错误机制与结论。所有 40 条进入原始 agreement
分母；按冻结顺序取前 30 个最终 `accept`。不足 30 为 `FAIL_YIELD`。

当前 clean 模型的 consistency 表示直接池化 generated-token features，不把 prompt tokens 拼进 pooled
trajectory；但这些 hidden states 和 condition fusion 仍携带 prompt 上下文。后续训练必须同时报告：

- accepted same-semantic/different-style pair 的表示与 score gap；
- different-semantic/same-style 负 pair 的分离；
- 同 query 但不同数值答案/不同方法的拒绝 controls。

“同 query 不同答案/不同方法”在本轮只是 verifier control，不自动改成新的训练 negative objective。当前
模型已有跨 `semantic_id`、同 `style_id` 的 negative consistency；若要增加 hard-negative loss，必须另发协议。

### 8.2 H/P：60 条共同 proposal，40 条联合入库

H 和 prior 必须对同一个 60-row natural manifest 各自做完整、互盲的 pass，不能先看 H 再只把容易行发给
prior。60 条来自 60 个不同 query，每 query/trajectory 最多 1 条，且不得复用 C 的两个 view。

冻结提议配额：

| source / numeric stratum | match=1 | match=0 | 合计 |
|---|---:|---:|---:|
| GSM8K | 18 | 18 | 36 |
| ASDiv-A | 12 | 12 | 24 |

每个 source/stratum 内，先对每个 query 取 candidate hash 最小的自动合格 row，再按 query hash 取冻结配额。
凑不满任何格即 `FAIL_YIELD`，不跨格偷换。checker stratum 只用于获得多样候选，A/B 看不到该字段。

双标和裁决完成后，从同时满足以下条件的交集中选最终 40 条：

- H 最终为 `hallucinated` 或 `clean`，不是 uncertain/ineligible；
- prior 最终为 `usable`；
- exact-token/unit/schema 全部合法；
- H 与 prior 的最终标签分别来自允许的 `auto_agree` 或 blind `adjudicated` 路径。

对 positive 和 clean 各自先按冻结 hash 取 GSM8K 5 条、ASDiv-A 5 条，再从两来源剩余合格行的统一 hash
顺序补到每类 20。不得按 confidence、F1、是否经裁决或 onset 位置挑行。不存在该子集即 `FAIL_YIELD`。

## 9. 三类标注的操作定义

### 9.1 Consistency

`accept` 需要两条 trajectory：任务和目标相同、核心前提相同、解题方法相同、关键中间量与结论相同，
只在措辞、步骤拆并、解释细度或组织上不同。若原路径有错，两条必须保留同一错误机制、最早语义位置和
下游影响。

以下一律 reject：不同方法、修复旧错、引入新错、改变关键中间量或结论、仅共享最终答案、近乎复制、
靠无意义尾部套话制造长度差。

### 9.2 Hallucination：first bad unit，不假装精确首错 token

- `clean`：每个 material claim 均由题目、已建立的有效推导或小学算术规则支持；
- `hallucinated`：至少一个 material claim 可明确反驳，或引入影响结论但题目/此前推导没有提供的事实或假设；
- 省略常规算术展开不等于“缺少前提”；只有实际声称了一个不受支持的桥梁或新事实才算；
- 明确写错单位、对象或实体是坏 claim，即使 numeric checker 的数值匹配；
- 多处错误取最小 `unit_index`；后来纠正也不抹掉第一次坏 claim；
- 若同一 unit 内融合了两个可独立判断的 claim，或最早坏 claim 不能由 fixed unit 表达，必须
  `insufficient_unitization`；
- 只有答案、拒答或没有可审计 material claim 的输出不是自动 clean，而是 ineligible/uncertain。

AI 输出的是 `first_bad_unit_index`。兼容训练字段
`hallucination_onset = units[first_bad_unit_index].token_start`，其准确名称是
`first_bad_unit_start_token`，不得写成 AI 找到了客观精确的“第一个坏 token”。explicit clean 才写 `-1`；
unknown/ineligible 保持字段缺失。

### 9.3 Key/Complete

- `complete`：从问题出发，复现并审计 trajectory **实际采用的** 推理路径所需的最小非冗余 unit 集；
- 删除 complete 中任何一个 unit 后，只凭剩余 unit 就不能完整复现或审计该路径；
- `key`：complete 内最直接决定这条 trajectory 所声称结论是否成立的最小子集；
- 对正确链选决定性答案推导；对错误链选最早或因果上最决定性的错误/无依据步骤，而不是机械选最后答案；
- 标题、计划话术、未变换的题面复述、重复等式和重复 final wrapper 默认不选；
- 多个同样合法的最小集合时，先最小化集合大小，再选择排序后 unit-index 数组字典序最小者；重复或等价
  unit 选择最早的可用者；
- usable 必须非空、升序、无重复且 `key ⊆ complete`。答案-only、拒答、不可审计或 unitization 不足为
  ineligible，两个数组均为空。

`key ⊆ complete=100%` 是 schema/结构门，不是标注质量证据。另报 `|Complete|/|material units|`、
`Complete=全部 material units` 比例和 key/onset 位置分布，防止平凡全集标签。

## 10. 双 AI、第三模型和重试

### 10.1 主标注者 A/B

- 正式 smoke 的 A/B **必须来自不同模型系列**，且都不得是 Phi-3.5 或与 rollout/frozen backbone 相同的
  模型系列；同模型两次新对话只可做 pipeline debug，不能通过正式 annotation-quality gate；
- exact provider、model ID、revision/date alias、temperature=`0`、seed（若可设）、prompt hash、调用时间、
  数据保留设置和原始响应都要记录；
- A/B 使用同一语义指南和同一 natural/control items，但在两个全新隔离上下文中以不同冻结顺序呈现；
- 看不到对方输出、checker/reference、selection stratum、历史结果、门槛或预期类比例；
- H 与 prior 即使复用同一 A/B，也必须是独立 clean-context pass，并在报告中承认跨任务错误可能相关。

### 10.2 schema 重试

- 首次响应始终保留；
- 最多允许 2 次格式修复，只提供原 item、原响应和具体 schema 错误，不提供正确答案或另一标注者输出；
- 不得借格式重试重新讨论语义；所有 attempts 均保存；
- 超限仍非法时，该行按 annotator failure 计入 proposal 分母并进入失败/裁决流程，不能静默删除后重算一致率。

### 10.3 自动接收、裁决和来源字段

- 只有语义 target 完全一致且双方 confidence 都不是 low 的行可标 `label_source=auto_agree`；
- 任一分歧、uncertain/review/low 或合法性冲突进入 blind adjudication；
- 裁决者应为第三个、不同于 Phi/A/B 的模型系列；如果没有合格第三模型，分歧行直接 unresolved/drop，
  不复用 A/B 投多数票；
- 裁决者先在看不到 A/B proposal 的 clean context 独立判断，再看到匿名、逐行随机顺序的 A/B 方案，
  可 adopt A、adopt B、synthesize 或 unresolved；随机种子和角色映射只放 private lineage；
- 报告 adopt-A/adopt-B/synthesize/unresolved 比例；裁决不能挽救已经失败的原始 agreement gate；
- 所有最终自然标签叫 `silver_dual_ai_v2`，并保留 `label_source in {auto_agree, adjudicated}`。

## 11. 没有人类复核时的额外控制

### 11.1 隐藏 protocol controls

每类任务另加约 10% 的合成、答案已知 control，不计入自然 proposal 数、不进入训练：

- C：原样复制/仅空白变化必须 reject；不同最终答案必须 reject；明显不同方法但同答案必须 reject；
- H：在简单、可机械验证的链中于指定 unit 植入明确假等式/假前提，必须定位到该 unit；全正确合成链必须 clean；
- prior：使用短小、唯一依赖链，预先固定可机械验证的 key/complete set。

controls 只能估计模型是否遵守本协议中的明显规则，不代表自然数据真实准确率下界。任何报告不得把 control
accuracy 冒充自然标签 accuracy。

### 11.2 自一致性与 auto-agree 抽审

- 对 A 的每类 natural items 按 hash 盲重跑 20%，新请求不得继承第一次输出；报告 self-agreement；
- 对 A/B `auto_agree` 行按 hash 抽 15%，由裁决模型先独立作答，再与共识比较；
- 该抽审只估计三模型稳定性，仍不是真实准确率，因为没有人类或外部 ground truth；
- 所有指标按 `auto_agree/adjudicated`、source、numeric stratum、unit-count bin 分层报告。

## 12. 预注册通过门

所有比例同时报告分子/分母；自然 proposal 是固定分母。bootstrap/Wilson 区间必须报告，但当前小样本不把
区间下界硬当成唯一开关。

### 12.1 数据、生成和 unitization

- query manifest 恰好 60 GSM8K +40 ASDiv-A，800 raw rollouts，candidate index 每题连续 `0..7`；
- 所有 100 queries 在永久 train-only 排除表中；旧 train/ranking/mechanism 与官方 test 的 source-ID overlap=0；
- normalized exact duplicate=0，near-duplicate candidates/决定/cluster IDs 完整；
- reference 100% 唯一可解析；每个 checker row 有 status/version/hash/failure reason；
- truncation/empty/illegal-ID 比例 `<=2%`，且这些行 0 条进入训练或机制 proposal；
- C manifest 恰好 40 pairs/40 queries；H/P manifest 恰好 60 rows/60 queries；两个 manifest 在标注前有 hash；
- 所有 proposal 的 unit/token contract 100% 通过；H/P 每行 material units>=4，最终 40 条中位数>=5；
- `key ⊆ complete`、索引范围、排序、无重复和 exact-token materialization 均 100%。

### 12.2 标注与反退化门

| 部分 | v2 硬门 |
|---|---|
| hidden controls | A、B 各任务均 100% 正确；自然数据 accuracy 不作推断 |
| schema | 最多 2 次格式修复后 100% 有明确 valid/failure 状态；初次失败率另报 |
| Consistency | 40-row raw decision agreement >=.90；裁决需求 <=.20；最终 accept >=30 |
| C prevalence | raw accept/reject 数、Cohen κ 与 class-specific agreement 必报；若任一类每位均至少 5 条，则 κ>=.60；否则 κ 只作描述，controls 防恒定 accept |
| H path | 60-row raw agreement >=.85；positive-specific 与 clean-specific agreement 各 >=.80；A/B positive rate 相差 <=.10；有足够双类支持时 κ>=.60 |
| H onset | common-positive 至少 15 条；material-unit>=5 子集 exact unit agreement >=.70，±1 unit agreement >=.85；同时报告逐行随机基线与 onset 位置直方图 |
| H 裁决 | 需裁决自然行 <=.35 |
| Prior eligibility | raw agreement >=.95；双方 usable overlap >=45 |
| Key | usable overlap 上 macro unit-set F1 >=.65 |
| Complete | usable overlap 上 macro unit-set F1 >=.82 |
| Prior 反退化 | 双方各自 `Complete=全部 material units` 比例 <=.50；选择比例均值/中位数必报 |
| Prior 裁决 | 需裁决自然行 <=.40 |
| self-agreement | A 的 C decision、H path、prior eligibility 各 >=.90；onset/set 自一致性另报 |
| 最终联合 yield | 固定规则能得到 30 C accepts、20 positive+20 clean 且每类每来源>=5，同 40 条 prior usable |

首/末 material unit 的 onset 合计比例超过 `.70` 时记为 `FAIL_DIVERSITY`；不得改标签，但正式扩量前必须
增加链长/位置分层。裁决后的 100% 统一率永远不能报告为 inter-annotator agreement。

## 13. 产物、存储与执行顺序

必须依次发布并 hash：

1. source/query/exclusion/cluster manifests；
2. rollout shards、completion markers 和 800-row audit；
3. checker code/config/tests 与逐行结果；
4. unitizer code/config/tests、unit files 和 exact-token audit；
5. C proposal、H/P proposal、control manifests；
6. A/B prompts、model roster、raw attempts、validated labels；
7. raw agreement/self-agreement/control report；
8. independent third-model audit 与 blind adjudication artifacts；
9. final `silver_dual_ai_v2` labels、selection lineage 和分层 coverage/yield report。

只有 1–9 全过门，才可在新目录抽取全层 BF16 feature。`33*3072` 个 BF16 数每 token 约 198 KiB；
执行前必须用实际 output/prompt token 长度计算 trajectory 与 canonical condition 的预计总存储，不能照搬
“约 30 GB”的粗估。feature extraction 仍必须保存 exact model revision、33×3072、BF16、路径、尺寸、
checksum 与 query-sharded completion；不就地覆盖任何历史 payload。

## 14. 允许与禁止的结论

通过后最多可以说：

- 这套冻结的多题源生成、numeric checker、dual-AI Silver、盲裁和 exact-token materialization 流水线闭环；
- 在具名模型、prompt 和 unitizer 下，各任务的原始 agreement、κ/F1、自一致性、三模型稳定性、yield、
  裁决率和成本是多少；
- 哪些概念在当前指南下可操作、哪些仍不稳定；
- 最终标签是 `dual-AI Silver, no human verification`。

绝不能说：

- Gold、人工/专家验证、真实准确率高、接近人工或客观唯一标签；
- AI 找到精确首错 token、唯一 Key 或唯一 Complete；
- numeric checker 证明完整语义正确；
- 三个模块、tail、mutual 或 `.25` gate 提高/降低 Best-of-N；
- Consistency×H 负交互已经消失或三个模块安全组合；
- SVAMP 是与 ASDiv 训练来源独立的未见测试；
- 结果可外推到 MATH/AQuA、长链、非 Phi 生成器或未标注领域。

## 15. smoke 后的下一道门

若 v2 全过，另发正式扩量协议：

- C：约 300–500 train groups +100–200 query/template-disjoint held-out groups；
- H：至少 200 positive +200 clean train，另有 100+100 query-disjoint dev；
- prior：约 300–500 train +100–200 query-disjoint dev；
- ranking：统一 checker 的约 1500–2000 independent queries ×16 candidates；
- 完整重跑 `C0/C1/H0/CH0`，共同估计两个单模块增量和 `CH0-C1-H0+C0`；
- current gold-tail、MIL、pseudo-tail 不进入下一轮核心 C/H 矩阵；H 先过 boundary/calibration；
- prior 先复核 direct target，再在新数据和新 ranking population 上固定比较 gate off 与 `.25` on；不再
  消费旧 16-row/500-query dev 调权重。

工程闭环、auxiliary target 可学性和 held-out Best-of-N 效果必须继续分开报告。
