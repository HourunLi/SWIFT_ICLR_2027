# CLIR 多题源数据扩充 smoke 协议 v3

冻结日期：2026-08-25 UTC

当前执行状态：`STOP_RAW_GATE_FAILURE`

机器契约：[`../configs/data_expansion_smoke_v3/protocol.json`](../configs/data_expansion_smoke_v3/protocol.json)，
冻结 SHA-256 为 `4c6dad263aa45232b04e40a7a58257f1f9b75130b050951994222206ef24b306`。

这是一轮数据与标注流水线 smoke，不是训练实验。当前没有抽 hidden state、没有训练 CLIR、没有选择
epoch/loss/gate，也不能用本轮数据声称模块改善 Best-of-N。prior→reward gate 的 `.25` 仍只是现有
`dev-tuned engineering default`，本轮不调它。

## 1. 为什么必须升到 v3

v2 已完成 100 题、800 条 Phi rollout 和 C/H/P 双标，但按预注册规则停止：旧 checker 把
`\boxed{38 cents}` 一类“数字加单位/说明文字”的正确答案误当成 mismatch，导致 H 的错误富集失真；
同时 Complete 的文字定义让一位标注者选择候选实际依赖链，另一位压缩成可重算的最短证明。

v3 做四个有边界的改动：

1. 保留 v2 原始 800 条 token-ID 轨迹，但全部用 `clir_numeric_multisource_v3` 重算，v2 标签不复用；
2. 新增较难、较长且答案仍能用同一数值 checker 处理的 MATH train 子集；
3. parse failure 不再冒充“答案明确错误”，H 的 mismatch 富集只认 `checker_status=numeric_mismatch`；
4. Complete 固定为“候选实际走过并被后续使用的全部唯一非冗余中间步骤”，不能压成另一条更短解法。

## 2. 数据源与永久 train-only 边界

| 来源 | 用途 | 固定范围 |
|---|---|---|
| GSM8K | 复用 v2 的 60 道 incumbent | `openai/gsm8k` pinned revision，train only |
| ASDiv-A | 复用 v2 的 40 道 incumbent，主要供 clean/prior | pinned repository commit，train only，CC-BY-NC-4.0 |
| MATH | 增加真实错解和长链 | `EleutherAI/hendrycks_math` pinned revision，train only |
| SVAMP | protected ASDiv-derived challenge set | 不训练、不调参；不能叫独立来源 holdout |
| MATH test | protected unused split | production loader 不请求/不读，不选样、不调参 |

MATH 原论文给出 7,500 train / 5,000 test、七个学科和难度 1–5 的结构；v3 只从 split-preserving mirror
直接下载四个明确命名的 train parquet，不调用会顺带物化 test 的整库 loader。冻结前曾有一次开发性
`datasets.load_dataset` 预检把 test 文件写入本机 cache，但没有查看 test row、答案或模型表现，也没有让
它进入选样/去重/调参；因此本轮能声称“执行路径只使用 train”，不能声称这个工作区从未下载过 test。
来源见
[MATH 论文](https://arxiv.org/abs/2103.03874)、[官方仓库](https://github.com/hendrycks/math)和
[固定镜像](https://huggingface.co/datasets/EleutherAI/hendrycks_math)。

MATH admission 规则在看到 rollout 前冻结：

- 只取 algebra、counting and probability、number theory、prealgebra；
- 只取 level 3/4/5，每个“学科×难度”单独分层；
- 官方 solution 至少 45 个词，排除 Asymptote；
- 只接受最后一个 boxed answer 可唯一化为整数、小数、分数、混合数或百分数的题；元组、区间、根式、
  含 π/变量/多答案的题在 source admission 阶段直接排除，而不是强行扩 checker。

实际 source export 为 7,473 GSM8K +1,218 ASDiv-A +1,852 strict-numeric MATH =10,543 行，
有序 hash `baf525a5…19ac`。全部选中 query 永久标为 `train_only_smoke_v3`，以后不得进入 mechanism dev、
ranking validation 或 test。

## 3. 冻结选样与 reserve 规则

v3 必须完整保留 v2 的 100 个 incumbent。MATH 在 12 个“4 学科×3 难度”strata 中各预选 9 题：

- 每层前 5 题是 primary，共 60；
- 每层后 4 题是 reserve，共 48；
- reserve 在生成前已经选定，只有 primary readiness 失败时才能整批启用，不能看模型输出后挑题。

因此完整冻结 manifest 是 208 题，hash `bc57c065…9481`；primary 是 160 题，hash
`c4dabbfe…4fe4`；reserve 48 题 hash `31873821…0ef`。本轮 primary 已通过，所以 reserve 没有生成。

去重沿用 27 条 v2 双 AI 决定；新增 scoped 检索只出现 1 对 MATH/MATH 近重复，按预注册保守规则合并。
28 条决定 hash 为 `f01ad242…d22`，没有未决 cross-source pair。incumbent 若与新增题同簇，incumbent
优先保留，避免后见数据替换旧冻结题。

## 4. 生成、checker 与 exact-token 结果

新增 60 道 MATH 使用与 v2 完全相同的 generator：固定 Phi-3.5-mini revision、每题 8 条、temperature
1.0、top-p .9、最多 1,024 新 token、query-derived seed、`vllm==0.5.3.post1`、单张 L20Z、BF16、TP=1。
新增 480 条均成功产生；raw hash `8ed41fad…185`。与绑定 hash 的旧 800 条合并后为 160 题、1,280 条，
combined raw hash `ab28c8ab…637`。

物化结果：

| 项目 | 数量 |
|---|---:|
| numeric match | 995 |
| 明确、可解析的 numeric mismatch | 240 |
| 多个冲突答案 | 10 |
| parse failed | 14 |
| 长度截断 | 21 |
| exact-token partition 通过 | 1,280 / 1,280 |

21 条截断占 1.64%，低于冻结的 2% generation-quality 上限；它们不进入训练或机制标签。冲突答案与
parse failure 保留审计，但不充当 H 的错误富集。两条输出存在“同一可见文本有多种合法 tokenization”，
unitizer 通过逐前缀解码把字符映射回原始 frozen IDs；没有重写 ID，且 terminal invisible token 仍留在
完整 token 轴上。

readiness 要求至少 30 个 query-distinct、可 exact-unitize 的明确 mismatch query；primary 实得 57 个。
40 个 C proposals 与 60 个 H/P proposals 的全部冻结 strata 也能一次生成。因此状态是
`READY_FOR_FROZEN_V3_PROPOSAL_AND_ANNOTATION`，不启用 reserve。

## 5. 冻结 proposal 分母

Consistency 固定为 40 个 GSM8K natural pairs，每 query 最多一组，两个 view 都至少 4 个 material units，
token 长度比在 `[1.25,3.0]`，数值结果相同，同一 trajectory 不再兼任 H/P。proposal hash 为
`1cd93bd…9a75`。

H 与 Prior 对同一批 60 条 natural trajectory 分开、互盲地标，固定 strata 为：

| stratum | 行数 |
|---|---:|
| GSM8K numeric match / mismatch | 10 / 8 |
| ASDiv-A numeric match | 12 |
| MATH numeric match / mismatch | 8 / 22 |

每 query 最多一条，parse failure 不可进入。H/P proposal hash 为 `cd5bd369…f3d`。所有原始 agreement、
F1 和裁决率都以这两个 natural manifest 为分母；格式失败、low、review/uncertain 或 ineligible 不能静默
从分母删除。

最终目标仍是 30 个 C accepts，以及 H/P joint-usable 交集中的 20 positive +20 explicit clean。positive
至少来自 5 GSM8K +5 MATH；clean 至少来自 5 GSM8K +5 ASDiv-A。最终排序不能看 confidence、F1、
裁决来源或 onset 位置。

## 6. 三类标签的大白话定义

### Consistency

两条推理只有在“同一道题、同一种实质解法、同一组关键中间量”时才 accept。换方法、修错、引入新错、
只碰巧同答案、错误机制不同、近乎逐字复制或靠废话拉长都 reject。两条都错仍可 accept，但必须同源地错。

### First-bad-unit

H 找的是第一个明确坏掉的 **unit**，不是精确首错 token。写错数字、等式、对象、显式单位，或凭空引入
会影响结论的新事实，都算 bad claim；省略普通算术展开不算缺前提。clean 的兼容 onset 值是 `-1`；
hallucinated 最终物化为被选 unit 的 `token_start`；unknown 不能伪装成 clean。

### Key / Complete

题面始终可见，不必选择原样题面复述。Complete 是候选实际依赖链里所有唯一、不重复、确实被后续使用的
中间变换和中间结果；排除计划话术、没参与结论的旁枝、重复等式和重复 final wrapper。不能因为另一位
解题者能一步重算，就删掉候选真实使用的前置中间量。

例如候选先写 `2+3=5`，再写 `5-1=4`，最后重复“答案 4”：Complete 是前两条计算，Key 通常是直接
得到 4 的第二条。错误链也照候选声称的结论标，Key 选最早或因果上最决定性的错误/无依据步骤。

## 7. 双 AI、隐藏控制与停止规则

A/B 必须来自两个不同、非 Phi 的模型系列，六个任务用六个全新上下文：A/C、A/H、A/P、B/C、B/H、
B/P。H 与 Prior 不能互相看标签。两边都混入约 10% hidden controls，A 另有约 20% self-repeat；自然项、
control 和 repeat 对标注者不可见地区分。当前盲包行数为：

| 任务 | A | B | natural 分母 |
|---|---:|---:|---:|
| C | 52 | 44 | 40 |
| H | 78 | 66 | 60 |
| Prior | 78 | 66 | 60 |

A/B 完成后，先机械验证 population/schema/index/control/self-repeat，再冻结 raw reliability report。第三个
不同模型系列先独立完成所有争议项与 15% auto-agree audit，落盘后才能查看匿名随机顺序 Option 1/2。
裁决不能把已经失败的 raw gate 改写成“通过”。无人类复核的最终标签只能叫
`silver_dual_ai_v3`，不能叫 Gold、verified 或近似人工。

关键门仍按 machine protocol 原样执行：C raw agreement、H path/类别/onset support、Prior eligibility/
Key/Complete F1、隐藏控制、自一致性、裁决率、onset 位置退化、Complete 全选退化和 joint final yield。
任一硬门失败就停止，不抽 hidden state、不训练。

## 8. 2026-08-25 raw triage 结果

六份互盲标签全部落盘，并通过 population、schema、item ID、枚举和 unit-index 校验。每个任务的 A/B
hidden controls 都是 100%，A-only self-repeat 也都是 100%。这证明两套输出稳定遵循格式与简单控制题，
不证明自然标签事实准确。

| raw 项目 | 实得 | 冻结门 | 结果 |
|---|---:|---:|---|
| C decision agreement | 26/40 = .6500 | ≥ .90 | FAIL |
| C 最低需裁决比例 | 14/40 = .3500 | ≤ .20 | FAIL |
| H path agreement | 59/60 = .9833 | ≥ .85 | PASS |
| H common positive | 30 | ≥ 15 | PASS |
| H 5+ unit exact onset | 26/30 = .8667 | ≥ .70 | PASS |
| H 5+ unit ±1 onset | 26/30 = .8667 | ≥ .85 | PASS |
| H 最低需裁决比例 | 5/60 = .0833 | ≤ .35 | PASS |
| Prior eligibility | 60/60 = 1.0 | ≥ .95 | PASS |
| Key macro F1 | .9167 | ≥ .65 | PASS |
| Complete macro F1 | .9267 | ≥ .82 | PASS |
| Prior 最低需裁决比例 | 35/60 = .5833 | ≤ .40 | FAIL |

其余 raw 门，包括 H 类别特异一致率/κ、Prior joint usable 与 Complete 全选退化门，也全部通过。C 的
A/B accept 数为 25/39，主要分歧是“近乎照抄”与关键中间量边界。Prior 的 Key exact 为55/60、Complete
exact 为26/60；Complete 分歧多数是一个集合为另一个的严格子集，说明链条接近但可选 unit 仍不唯一。

冻结规则规定 raw failure 不能由第三模型或裁决救活，因此本轮终态为 `STOP_RAW_GATE_FAILURE`，
`third_model_send_allowed=false`。旧 triage 已机械生成的第三模型文件不得发送；不运行 adjudication、
finalize、hidden-state extraction 或训练。H 的 raw 通过只能称为标注可操作性证据，不能称为标签准确或
模块有效。

下一版必须使用新自然样本并在标注前重冻：C 要把 near-copy/diversity 边界机械化并加入真实 hard
controls；Prior 要通过显式 dependency edge→确定性闭包或预注册等价组容错来唯一化 Complete。H 若继续，
也应在新 query 上独立确认，不能把已经看过结果的本批标签改名后当确认实验。

所有 source rows、rollouts、盲包和标签留在 `run_artifacts/`，不推 Git。远端只保留执行代码、机器协议、
标注语义、README、handoff 和本 canonical 文档。
