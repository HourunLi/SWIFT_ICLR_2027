# Consistency 机械筛选 smoke v5

状态：`FROZEN_BEFORE_FRESH_ROLLOUT`。

v4 只改提示词后，两个模型在 14 条旧争议上仍只同意 7 条，因此 v5 不再让 AI 同时判断“数学路径一样”和“文字差异够大”。这两个条件先由固定程序处理；AI 只检查两条解答是否夹带实质错误。

## 1. 全新数据

使用 v3 在任何 rollout 前就选好的 48 个 MATH-train reserve query。它们没有进入 v2/v3 的任何 C/H/P 标注，query overlap 已机械验证为 0。v5 明确授权把整批 48 题用于 C-only pipeline smoke；所有题继续永久属于 train-only，不得进入 mechanism dev、ranking validation 或 test。

生成器、revision、prompt、n=8、temperature、top-p、token 上限、checker v3 和 exact-token unitizer v2 全部沿用 v3。v5 不复用旧 C pair，也不使用 v4 的 14 条作为自然验证数据。

## 2. 固定机械规则

候选首先必须同时满足：

- 两条都可 exact-token unitize，且各有至少 4 个 material-claim units；
- 两条都是 `numeric_value_match=1`，规范化最终答案完全相同；
- output-token 长度比在 `[1.15, 3.0]`；
- 从 TeX/等式抽取的有序数学 token 每边至少 6 个，SequenceMatcher ratio 至少 `.60`；
- 有序数字 literal 每边至少 4 个，ratio 至少 `.75`；
- 去掉数学、数字与固定套话后的英文 word bigram 每边至少 8 个，Jaccard 在 `[.10, .40]`。

每个 query 最多留一对；同一 query 有多对通过时只按固定 SHA-256 priority 取一对，不看 AI、置信度或人工判断。48 题若不足 12 对，终态为 `STOP_MECHANICAL_YIELD_FAILURE`，只能扩新 query，不能改阈值补数。

这些阈值在已检查的 14 条 v4 争议上做过开发回归：只放行 3 条，三条恰好都是 A/B 共同 accept。这个数字只说明实现符合预期，不是准确率或新可靠性证据。

## 3. 双 AI 只做事实审计

机械通过的 12 对分别交给 GPT-5.5-sol/xhigh 与 Claude Opus 5/high 的两个全新隔离上下文。AI 不得重新判断同路径、风格差异或近抄，只判断两条解答各自是否存在实质性算术、代数、单位、实体、数量或内部矛盾错误。

每边额外有 4 个隐藏控制：两个事实正确但故意近抄/换分组，必须 accept；两个含明确实质错误，必须 reject。A 另有 3 个盲重复，用来检查自身稳定性。

raw gate：

- 12 条自然样本至少 11 条 A/B decision 一致；
- 每边自然 `review <= 1`；
- 两边隐藏控制各 4/4；
- A 自重复 3/3；
- 至少 8 条自然样本为双方共同 accept；
- 所有理由前缀合法。

任何 raw gate 失败都直接停止，不由第三模型裁决救活。即使全部通过，也只允许另写正式扩量协议；v5 本身不发布训练 manifest、不抽 hidden state、不训练，也不产生 Consistency 模块效果结论。

## 4. 执行顺序

```bash
P=/prodcpfs/user/panzhixin/miniconda3/envs/SWIFT/bin/python

$P prepare_clir_smoke.py --protocol configs/data_expansion_smoke_v5/protocol.json rollout \
  --queries run_artifacts/data_expansion_smoke_v3/frozen/reserve_queries.jsonl \
  --output run_artifacts/data_expansion_smoke_v5/rollouts/reserve_raw.jsonl \
  --tensor-parallel-size 1

$P prepare_clir_smoke.py --protocol configs/data_expansion_smoke_v5/protocol.json materialize \
  --rollouts run_artifacts/data_expansion_smoke_v5/rollouts/reserve_raw.jsonl \
  --output run_artifacts/data_expansion_smoke_v5/materialized/reserve_rows.jsonl

$P prepare_clir_smoke.py consistency-v5-propose \
  --processed run_artifacts/data_expansion_smoke_v5/materialized/reserve_rows.jsonl \
  --output-dir run_artifacts/data_expansion_smoke_v5/proposals

$P prepare_clir_smoke.py consistency-v5-package \
  --items run_artifacts/data_expansion_smoke_v5/proposals/annotation_consistency_natural.jsonl \
  --output-dir run_artifacts/data_expansion_smoke_v5/packages
```

双标完成后运行 `consistency-v5-check`。所有原始数据、rollout、包、标签和报告继续只放 `run_artifacts/`；远端只保留代码、核心协议、README 和 handoff。
