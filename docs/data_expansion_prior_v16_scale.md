# Prior v16 正式扩量协议

v15 已在 48 条全新样本上通过：两个 AI 不再画依赖边，也不直接选 Key/Complete，只判断程序切好的
推理块是否属于最终答案的主链。v16 将这一目标扩大到 600 条候选，门全过后按冻结配额发布 500 条
Silver 数据（400 train、100 dev）。

这批数据不需要重新调用 Phi，也暂时不需要 GPU。它从 v12 已完成的 2000 题、16000 条采样中选择，
永久排除 v12、v13、v14、v15 已使用过的题和模板簇。容量审计后仍有 1054 个不同题/模板簇可选，
所以 600 条候选有真实余量。

两位 AI 各标 12 个 shard，每个 shard 是 50 条自然样本、1 条隐藏控制题和 5 条跨 shard 重复题，
共 672 行。最终 Key 只在两边都认同的最后计算块上生成；Complete 对双方都认为是主链的块标 1，
双方都认为不是主链的块标 0，一主一非的争议块直接遮住。这样既保留双 AI 的共同判断，也不让第三个
AI 或程序替双方裁决语义争议。

所有候选、配额、重复题、控制题、门槛和选择顺序都在标注前冻结。失败后不改提示词、不裁决、不从
失败批次另挑“看起来好”的子集。即使通过，也只能说明这套双 AI Silver 流程能稳定产生可训练目标，
不能说明标签等同人工 Gold，更不能直接说明 Prior、Gate 或 Best-of-N 一定有效。

## 冻结执行结果（2026-09-02）

GPT-5.6-sol/max 与升级后的 Claude Opus/max 各完成 12 个 shard。24 个输出文件的 JSON、行数、
唯一 ID、输入 ID 集合、七字段 schema、block 覆盖和顺序全部合法。冻结 evaluator 随后只运行一次，
返回 `STOP_PRIOR_V16_ROLE_ONLY_SCALE`；报告位于
`run_artifacts/data_expansion_prior_v16/pre_annotation/evaluation/raw_gate_report.json`，SHA-256 为
`eb4e82ef61de2275dd40446a3094b63ebc3b51ec09e281e7d5233b6ab7d27b4e`。

关键结果：controls A/B=`8/12,11/12`，self-repeat=`60/60,53/60`；600 条自然样本的 final-block
exact=`.8067`、role agreement=`.7670`、Complete IoU/coverage=`.6796/.7975`。冻结配额只能选出
473/500 条，选中部分的 IoU/coverage 也只有 `.7159/.8281`。因此
`trainable_labels_published=false`，没有 Silver manifest、feature 或训练。

失败的主要原因是主链边界在规模化样本上不稳定。A 在 10017 个 block 中标了 5887 个 main-step，
B 标了 3855 个；最大分歧来自标题/计划、题面复述和纯公式是否算主链。B 的 7 个 repeat 漂移表明
这也不是只把某一位标注者换成真值就能解决的问题。按本协议，v16 到此终止：不得修改 prompt、
门槛或标签，不得裁决、重标、混合尝试或选容易子集，也不得 materialize、抽 feature 或训练。
