# CLIR 数据扩容主协议 v6：先扩 Consistency，再分别确认 H 与 Prior

状态：`FROZEN_PREPARATION_ROLLOUT_NOT_STARTED`

执行覆盖状态（2026-08-28）：`AUTHORIZED_ROLLOUT_ONLY`。基础协议与六份清单保持原 hash；用户在看过清单、
规模和预算汇总后明确回复“开始”，授权记录见
`configs/data_expansion_scale_v6/rollout_authorization.json`。该覆盖只允许 40 个 frozen shard 的生成、复核和
合并，不允许 checker/unitizer materialization、AI 标注、抽特征或训练。

执行结果状态（2026-08-28）：`PASS_ALL_16000_RAW_ROLLOUTS_VERIFIED_V6`。基础状态是生成前冻结时的不可变
记录，执行覆盖和结果是后加的 hash-bound 审计层；三者不是矛盾，也不能用 rollout PASS 解锁后续阶段。

标注前执行覆盖状态（2026-08-28）：`AUTHORIZED_PRE_ANNOTATION_ONLY`。用户明确要求“接着做，到要标数据再
叫我”，授权记录见 `configs/data_expansion_scale_v6/pre_annotation_authorization.json`。它只允许 CPU
checker/unitizer、原样机械筛选、候选冻结和隔离盲标包构造；真正的 AI 标注、标签 triage/finalize、feature、
训练、raw 重生成和阈值变更仍为 false。

标注前执行结果（2026-08-28）：`PASS_PRE_ANNOTATION_PACKAGES_VERIFIED_V6`。16,000 条材料化、708 对机械
候选和 15 个 A/B shard 均已由独立 verifier 重算；`annotation_started=false`，当前必须停下并通知用户。
首个 A/B `shard-000` 已经产生后，用户澄清 A 实际使用的是 GPT-5.6-sol/xhigh，不是基础索引误记的 5.5。
`configs/data_expansion_scale_v6/annotation_model_amendment_v1.json` 作为公开、可哈希的模型身份更正记录（SHA-256
`fadb3351…5a47`）：它绑定了更正时两个 `shard-000` 的 hash，证据等级明确为 user-reported，精确 provider
revision/temperature 仍未验证；B、包内容/顺序、controls、repeats、分母与 raw gate 均不变。

冻结日期：2026-08-26

机器契约：`configs/data_expansion_scale_v6/protocol.json`

## 1. 这版协议解决什么问题

v5 已在全新 MATH-train 样本上证明“固定机械筛选 + 两个不同模型系列做事实审计”这条 Consistency 数据流水线可以运行：12/12 自然 pair 的 A/B 判断一致，两个含明确错误的隐藏控制也都被正确拒绝。但 v5 是 smoke，12 对自然样本不能训练，也不能证明 Consistency 会提高 Best-of-N。

v6 把下一步固定为一个中等规模、可承受的 Consistency 扩容：

```text
2,000 个全新 query
  × 每题 8 个 Phi 候选
  = 约 16,000 条 raw rollout
      ↓ 先用 v5 原样机械筛选
      ↓ 再由两个不同、非 Phi 的 AI 只检查实质错误
      ├─ 400 个 query-distinct 训练正关系
      └─ 150 个 query/template-cluster-disjoint held-out 正关系
           + 150 个确定性 hard-negative held-out 关系
```

这里“冻结”指预算、来源、拆分、筛选、双标、失败门和最终入选顺序已经写死。协议冻结提交本身没有生成
数据；随后用户通过独立授权只解锁了 rollout。该阶段现已完成，但仍未标注、未抽 hidden state、未训练，
也尚未产生任何可训练 Consistency relation。

## 2. 为什么先扩 Consistency

三个模块当前不在同一个准备程度：

| 模块 | 最新数据门 | 当前裁决 |
|---|---|---|
| Consistency | v5 新鲜机械筛选审计全门通过 | 可以单独进入正式扩容准备 |
| Hallucination onset | v3 的双 AI 操作一致性通过，但样本已看过且不是独立确认 | 先做全新 H-only `30 positive + 30 clean` 确认 |
| Key/Complete Prior | v3 的集合 F1 高，但 35/60 需要裁决 | 先改成“标依赖边，再由程序算 Complete 闭包”的新 smoke |

因此 v6 不是把三个模块一起放大。先把已经通过数据生产门的 Consistency 做到可训练、可独立验证；H 和 Prior 仍要分别过自己的新鲜数据门。这样如果后续组合再下降，才有机会区分是模块本身、标签质量还是共享表示交互造成的。

## 3. 数据来源与数量

### 3.1 MATH train：1,400 题

- 固定镜像与 revision：`EleutherAI/hendrycks_math@21a563...`；
- 只读 train；test 不用于选题、调参或协议开发；
- 只取 algebra、counting/probability、number theory、prealgebra；
- level 3/4/5；
- 官方解至少 45 个英文词，最终答案必须是 checker 能唯一解析的单个数值；
- 排除 Asymptote 图形题。

其中 1,050 题属于 train-acquisition，350 题属于 heldout-acquisition。

### 3.2 长链 GSM8K train：600 题

- 固定 `openai/gsm8k@740312...`；
- 只读 train；test 完全保护；
- 官方推理至少 45 个词、至少 2 个显式计算标记、至少 3 个不同的中间数值；
- 目的不是凑简单加减法，而是补入表达更自然、链条仍足够长的小学应用题。

其中 450 题属于 train-acquisition，150 题属于 heldout-acquisition。

### 3.3 为什么本轮不放 ASDiv-A

ASDiv-A 很多题只有一两步。在这种短链上，“同一个方法的简写版和展开版”常常没有足够空间形成真实风格差异，容易让 Complete 退化成全集，也会让 Consistency pair 只剩表面改写。本轮不把它用于 C 扩容。SVAMP 继续只作为 ASDiv-derived challenge set 保护，不能称为独立来源 holdout。

## 4. 先按题和模板拆分，再生成

所有历史 outcome/mechanism/ranking query，以及 v2/v3/v4/v5 看过、生成过或标过的 query，都进入永久排除表。不能因为题面改了一个数字就重新当新题。

剩余题先做规范化和模板聚类：小写、Unicode NFKC、数字/实体占位、空白合并；exact duplicate 按固定 SHA 顺序保留一个；模板签名和 token trigram MinHash 负责召回近重复。无法确定的跨来源高相似题保守放进同一 cluster。

split 单位不是单行，而是整个 query/template cluster：

```text
train-acquisition:   1,500 queries = 1,050 MATH + 450 GSM8K
heldout-acquisition:   500 queries =   350 MATH + 150 GSM8K
```

分配键固定为 `sha256(clir-C-v6-split|cluster_id)`。同一 cluster 不得跨两边；候选、视图和后续标签始终跟随原 query。正式 rollout 前必须发布以下有序 manifest 及 SHA-256：source inventory、永久排除表、template cluster、两边 query 清单和 rollout shard 清单。

## 5. 生成与机械筛选

生成参数完全沿用 v5：

- `microsoft/Phi-3.5-mini-instruct@2fe192...`；
- vLLM `0.5.3.post1`、BF16、TP=1；
- 每题 8 条，temperature=1、top-p=.9、max_new_tokens=1024、seed=42；
- 保存的 `prompt_token_ids/output_token_ids` 是唯一 token 坐标真相。

2,000 题按每 shard 50 题拆成 40 个原子 rollout shard。每个 shard 独立写 manifest、行数和 hash；只有完整校验通过的 shard 才能合并。

checker 固定 `clir_numeric_multisource_v3`，只声称 `numeric_value_match`；unitizer 固定 `clir_material_claim_unitizer_v2`。截断、解析失败、多重冲突答案和 token partition 失败均留在 raw audit，但不能进入 C pair。

机械规则一项不改地复制 v5：

1. 两条都 numeric match，规范化最终答案完全相同；
2. 每条至少 4 个 material claims；
3. token 长度比 `[1.15,3.0]`；
4. 数学 trace 每边至少 6 项，相似度至少 `.60`；
5. 数字 trace 每边至少 4 项，相似度至少 `.75`；
6. 去除数学/数字/套话后的 word-bigram 每边至少 8 个，Jaccard 在 `[.10,.40]`；
7. 每题最多一对，多对通过时只按冻结 SHA 优先级取一对。

v5 的 48 题得到 16 个机械通过 query，观测 yield 是 `33.3%`。把它外推到 2,000 题大约是 667 对，足以覆盖 550 对目标并留约 17% 的事实审计损耗；这只是预算估算，不是保证。如果最终不足，状态是 `STOP_YIELD`，只能申请新增 query 预算，不能放宽阈值补数。

## 6. 双 AI 怎么标，怎样避免再手工搬六大包

机械程序已经判断“同路径”和“文字差异够大”。AI 只回答：任一视图里是否存在实质性的算术、代数、单位、实体、数量或内部矛盾错误；不得重新审理近抄、风格或方法等价。

- A、B 固定为两个不同、非 Phi 的模型系列。基础稿曾记为 GPT-5.5-sol/xhigh + Claude Opus 5/high；用户在首个 shard 后澄清实际 A 为 GPT-5.6-sol/xhigh，公开更正记录将有效 roster 记为 GPT-5.6-sol/xhigh + Claude Opus 5/high，但 exact revision/temperature 均不得冒充已验证。若之后还要换模型，必须再修订协议版本后才能继续；
- 每个模型一次最多处理 50 条自然 pair；脚本直接给它对应 JSONL 路径，不再让人逐条复制；
- 每 shard 混入 4 个隐藏控制：2 个事实正确必须 accept，2 个有明确错误必须 reject；
- 每个模型约 10% 的自然 item 在后续 shard 盲重复，测自身稳定性；
- `PRIVATE_manifest.json` 永远不发给标注者；
- 产品若不暴露精确 revision/temperature，就明确记为 `unverified`，不能假装完全可复现。

最终只有 A/B 都以非 low confidence 判 `accept` 的自然 pair 可入选。任何 reject、review、格式失败或 A/B 分歧都直接排除，不允许第三模型把 raw gate “救活”。它们仍留在所有 agreement/yield 分母中。

扩量门：

- 自然 decision agreement 至少 95%；
- 每边 review 不超过 2%；
- 每个模型隐藏控制 100%；
- 每个模型自重复一致率至少 95%；
- train common accepts 至少 400，heldout common accepts 至少 150。

任一门失败，不发布训练 manifest，也不抽特征。最终 400/150 不是按“理由更漂亮”挑，而是从 common accepts 中按标注前已冻结的 hash 顺序取前 N 个。

因为没有人工复核，这批标签统一叫 `silver_dual_ai_consistency_v6`。可以说“双 AI 按协议共同接受”，不能说 Gold、专家验证或客观准确。

## 7. held-out 为什么还要 150 个 hard negatives

只测试“同一道题的两种写法是否接近”会有一个偷懒解：模型把所有答案表示都压得很像。为识别这种塌缩，heldout 另外构造 150 个不同语义但外观相近的负关系：

- 只使用 heldout query；
- 两端必须来自不同 query 和不同模板 cluster；
- 最终规范化答案必须不同；
- 尽量匹配来源/学科/长度层；
- 长度比 `[.8,1.25]`，文字 bigram Jaccard `[.10,.40]`；
- 用来源层、长度距离和 SHA 做确定性 greedy matching。

这些负关系只用于机制评价，不作为新增自然 Gold。后续 Consistency 模型至少要同时报告：正关系的表示/score 差距、负关系的分离、worst-view correctness 和 score variance。不能只报训练 loss 下降。

## 8. 存储和算力预算

当前全层 BF16 每个 token 是：

```text
33 layers × 3072 values × 2 bytes = 202,752 bytes ≈ 198 KiB/token
```

v3 MATH 输出平均约 460 token，GSM8K 平均约 289 token；按 70/30 混合估计约 409 token，协议用 420 token 作保守预算。

| 阶段 | 规模 | 估算 |
|---|---:|---:|
| raw rollout | 16,000 条，约 672 万 output tokens | 主要是生成时间和 JSONL/token IDs，远小于全层 feature |
| 如果错误地给 16,000 条全抽全层 feature | 420 output token +每题约 150 prompt token | 约 `1.42 TB`，禁止这样做 |
| 正确做法：只抽最终 550 对 | 1,100 trajectories +550 个共享 prompt | 约 `105.5 GB`，实际值在入选后重算 |

因此流水线顺序必须是：先 rollout → checker/unitizer → 机械筛选 → 双 AI → 最终选 550 对 → **最后**才抽 1,100 条视图。不得为了省流程先把 16,000 条全部物化成 `33×3072` feature。

独立 1,500–2,000 题×16 的 ranking pool 是另一笔 TB 级预算，必须单独冻结协议与存储计划，不能混在本次 C 数据预算里。

## 9. H、Prior 和训练的后续顺序

Consistency v6 通过也只解锁 C 数据，不自动解锁其他模块或 Full。

### H：先做新鲜 H-only confirmation

目标是 30 条 positive onset +30 条 explicit clean，全部来自没用于 v3 标签的新 query。双 AI 要报告 path、exact/±1 unit onset、首尾位置分布和绝对位置基线。只有通过才扩为 train `200+200`、heldout `100+100`。第一轮 `C0/C1/H0/CH0` 只放 onset BCE；当前 negative tail、MIL、pseudo-tail 都不进核心矩阵。

### Prior：先改标注对象

新 smoke 先用 40–60 条新轨迹。AI 不直接猜一个可能不唯一的 Complete 集合，而是标“哪一步依赖哪一步”的显式边；程序对通向最终结论或首个致命错误的依赖图做确定性传递闭包得到 Complete，再在闭包内选 Key。只有 exact-set 分歧和裁决率明显下降，才扩到 train 300–500、heldout 100–200。

`gate_prior_weight=.25` 继续作为用户要求的 main-style 默认路径固定保留，但不再用旧 500-query dev 调参。等新的 prior 与独立 ranking pool 就绪后，只做 gate-off 对固定 `.25` 的新数据复测。

### 训练和最终选择

数据门通过后，先完整重跑 `C0/C1/H0/CH0`，至少 3 个、优先 5 个 seed，报告 paired query bootstrap。Full 只能在 C、H、Prior 各自数据与机制门通过后运行。下降的辅助 loss、通过的 smoke 或单个 seed 都不能替代 held-out Best-of-N。

## 10. 当前完成状态与唯一下一步

2026-08-27 的生成前澄清已经在机器契约中补齐并冻结：长链 GSM 的统计口径、
数字/实体模板算法、64-permutation/16-band MinHash、token/trigram Jaccard 阈值、历史排除项向整个 cluster
传播、MATH 学科×难度与 GSM 四分位分层、3,072-token prompt 上限，以及每个 shard 固定
`35 MATH +15 GSM8K`。这些都发生在任何 v6 rollout 之前，不是看过 yield 后调规则。

独立入口是：

```bash
python prepare_clir_scale.py freeze
python prepare_clir_scale.py verify
CUDA_VISIBLE_DEVICES=0 python prepare_clir_scale.py rollout --shard-id train-000
python prepare_clir_scale.py verify-rollouts --shard-id train-000 --require-complete
python prepare_clir_scale.py merge-rollouts
```

`freeze` 会拒绝 dirty worktree 和覆盖已有冻结目录。授权后的 `rollout` 每次只接受一个已冻结 shard，要求
进程只看见一张 GPU，逐题保留独立 seed、原始 prompt/output token IDs、模型/环境/commit/hash provenance，
完成后写 hash sidecar 和 completion marker。已有完整 shard 只复核并跳过；发现无 completion marker 的残留
文件则停止，不自动覆盖。`merge-rollouts` 只有在全部 40 个 shard 逐一重新通过时才发布 16,000 行总表。
入口仍刻意没有标注、抽特征或训练子命令。

生成前用该入口发布并核对的项目为：

1. 两个来源的完整 inventory 和许可字段；
2. 历史 query 永久排除表；
3. exact/near-duplicate 模板 cluster；
4. 1,500/500 query split；
5. 40 个原子 rollout shard；
6. 根据实际选中题目的 token 长度重算 GPU/磁盘预算；
7. 在干净 commit 上发布上述 manifest hash。

六份清单已经通过。用户只授权 rollout 后，入口在 clean commit
`078eb6b1c3d1aa8c1c950030e4aeff496ea1f342` 上先完成 `train-000` 校准，再完成全部 40 个 shard。独立
`verify-rollouts --require-complete` 与 `merge-rollouts` 均通过，结果如下：

- 2,000 个 query、16,000 条 row、每题 candidate 0–7，ID 全部唯一；
- 合并文件 `run_artifacts/data_expansion_scale_v6/rollouts/combined_raw.jsonl` 的 SHA-256 为
  `f538373b3d99791001cbe2119466b0bd52e23a8337223053920371cda7e75139`；
- 15,393 条 `finish_reason=stop`，607 条 `finish_reason=length`，截断率 3.79375%；
- output token 共 7,187,789，长度 min/median/mean/max 为 `87/392/449.2368/1024`；
- train-acquisition 为 12,000 行、453 条截断（3.775%）；heldout-acquisition 为 4,000 行、154 条截断
  （3.85%）；
- MATH 为 11,200 行、599 条截断（5.3482%）；GSM8K 为 4,800 行、8 条截断（0.1667%）；
- 1,999/2,000 个 query 至少有两条未截断候选；唯一不足者按冻结规则排除，不放松筛选阈值；
- vLLM 的 `backend_response_text` 对全部 16,000 行统一比 token-ID decode 多一个前导空格；strip 后 16,000
  行全部相等，因此这是展示层差异，不是 token 漂移。后续只认保存的 token IDs。

raw artifact 与完成报告均在 Git 忽略目录，远端只保存执行代码、授权、协议和交接文档。原先按 420 output
token 估计的 105.23 GB 选中-view feature 预算，因 raw 实测均值为 449.24，必须等最终 1,100 个 views
确定后按实际长度重算；全部 16,000 条抽 feature 的禁令不变。

rollout 授权在 merge 报告处终止。607 条截断输出必须在下一阶段排除；checker、
`clir_material_claim_unitizer_v2`、机械候选筛选、双 AI 审计、feature extraction 和训练均未启动。因此现在
只能写“raw acquisition 完成”，不能写“训练数据扩容完成”或“已有 400 对训练数据”。

新的标注前授权现已单独冻结。入口新增 `materialize → verify-materialization → propose → verify-proposals →
package → verify-pre-annotation`，只能在 clean commit 上执行，所有 artifact 原子写入且不得覆盖。材料化保留
16,000 行审计总体，但截断、parse failure、冲突答案或 unitization failure 一律不可进入 pair；`propose`
逐字复用 v5 阈值并把所有 query-distinct admitted pair 留在标注分母。只有机械池至少覆盖 train 400、heldout
150 才能打包。每包最多 50 个自然 pair，混 4 个平衡控制；A/B 各 10% 的盲重复只进入后续 shard，每个新
上下文只准处理一个 shard。`PASS_PRE_ANNOTATION_PACKAGES_VERIFIED_V6` 出现后必须停止并通知用户，不能由
脚本自行调用两个 AI。

该阶段已经按上述顺序完成，执行代码 commit 为 `5ab1fb743185a806f70269d66230768b7a4ad38d`：

- 16,000/16,000 unitization 成功；15,927 条使用 canonical fast offsets，73 条使用保留 frozen IDs 的
  prefix-decode fallback；
- checker 为 8,877 numeric match、5,907 numeric mismatch、607 truncated、350 parse failed、259
  ambiguous multiple answers；最终 14,784 行可监督，1,216 行只留审计；
- materialized 文件 SHA-256 为 `ca37d027…ea4fd`；
- 不改 v5 阈值后得到 708 个 query-distinct pair：train/heldout=`526/182`，MATH/GSM8K=`348/360`；proposal
  SHA-256 为 `795a1d47…33b73`；
- 15 个盲标 shard 中，每边为 708 natural +60 hidden controls +71 later-shard repeats =839 行；A/B 公共
  index SHA-256 分别为 `160d8c7f…429a` / `e895cd29…b1c3`；
- 两边合计 1,678 个 package item 已复核，标签目录未创建，provider call、feature 和训练均未开始。

正式标注前的余量是 train 126 对、heldout 32 对，因此 common-accept 最低比例分别为 76.0% 和82.4%。A/B
现已分别返回 15 个 shard、839 行，待 deterministic post-annotation evaluator 固化结果。标签前已冻结的 raw
gate 仍是 decision agreement≥95%、review≤2%、每边 60/60 controls、自重复≥95%、common accepts
train≥400/heldout≥150，且任何失败均不准第三模型救活。

用户报告两边全部 shard 完成后，独立的
`configs/data_expansion_scale_v6/post_annotation_authorization.json`（SHA-256 `7dcd096a…ab34`）只解锁
全量 label/schema 验证、冻结 raw gate、条件式 400/150 选择和原协议 150 hard-negative feasibility。它明确禁止
provider/第三模型、裁决或修标签、改阈值/分母、feature extraction 和训练；任何后置门失败都只写审计报告，不发布
relation manifest。
