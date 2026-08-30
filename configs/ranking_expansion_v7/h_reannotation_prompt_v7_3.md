# H0 v7.3 reserve 全量重标提示词

这不是换数据或放宽门槛。两位标注者都必须在新会话中，把原来同一批 800 个
公开条目完整重标一次。第一轮输出已经作为失败记录封存，禁止读取或复用。

## 标注者 A：GPT-5.6-sol xhigh

```text
你是 CLIR H0 reserve_attempt_2_v7_3 的独立标注者 A。请按文件名顺序逐个处理：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/reannotation_v7_3/packages/annotator_a/shard-000.jsonl
一直到 shard-015.jsonl。

每个输入 shard 的 50 行结果分别写到：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/reannotation_v7_3/annotation/annotator_a/shard-XXX.labels.jsonl

不得读取 annotator_b、PRIVATE_package_index.jsonl、第一次 reserve 标注、smoke 标签、checker/reference answer、其他模型输出或历史标签。不要上网搜索题目。

你必须自己解题并逐条审查 trajectory，找“第一个明确坏掉的 material_claim unit”：
- clean：每个 material claim 都能由题面、此前仍成立的推导或普通数学规则支持。
- hallucinated：至少一条 material claim 明确错误，或凭空引入会影响结论的新事实、数字、对象、单位、等式或推理关系。
- 省略常规算术展开不算缺前提；明确算错、抄错、换对象、换单位或使用无依据关系算坏 claim。
- 多处错误取最早的 material_claim；后面改正也不抹掉第一次错误。
- unit 切分真的无法稳定承载首错时才标 insufficient_unitization；只有答案、拒答或没有可审计推理时才标 no_auditable_reasoning。
- uncertain 只能用于你已经完整审查、但仍能给出该条目专属具体理由的真实两可情形，不能作为未审完或节省时间的占位符。

每行严格输出：
{"item_id":"...","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":整数或null,"confidence":"high|medium|low","rationale":"该条目专属的简短具体理由"}

禁止给大量无关条目复制同一句笼统 rationale。若你无法完成全部 shard，就保留尚未完成的输出文件不存在；绝不能用 uncertain 或模板理由补齐。每完成一个 shard，立即检查：恰好 50 行、逐行 json.loads、item_id 与该输入完全相同且无重复、hallucinated 的 unit_index 确实存在且是 material_claim。全部 16 个 shard 都完成后再结束。
```

## 标注者 B：Claude Opus 5 high

```text
你是 CLIR H0 reserve_attempt_2_v7_3 的独立标注者 B。请按文件名顺序逐个处理：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/reannotation_v7_3/packages/annotator_b/shard-000.jsonl
一直到 shard-015.jsonl。

每个输入 shard 的 50 行结果分别写到：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/reannotation_v7_3/annotation/annotator_b/shard-XXX.labels.jsonl

不得读取 annotator_a、PRIVATE_package_index.jsonl、第一次 reserve 标注、smoke 标签、checker/reference answer、其他模型输出或历史标签。不要上网搜索题目。

你必须自己解题并逐条审查 trajectory，找“第一个明确坏掉的 material_claim unit”：
- clean：每个 material claim 都能由题面、此前仍成立的推导或普通数学规则支持。
- hallucinated：至少一条 material claim 明确错误，或凭空引入会影响结论的新事实、数字、对象、单位、等式或推理关系。
- 省略常规算术展开不算缺前提；明确算错、抄错、换对象、换单位或使用无依据关系算坏 claim。
- 多处错误取最早的 material_claim；后面改正也不抹掉第一次错误。
- unit 切分真的无法稳定承载首错时才标 insufficient_unitization；只有答案、拒答或没有可审计推理时才标 no_auditable_reasoning。
- uncertain 只能用于你已经完整审查、但仍能给出该条目专属具体理由的真实两可情形，不能作为未审完或节省时间的占位符。

每行严格输出：
{"item_id":"...","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":整数或null,"confidence":"high|medium|low","rationale":"该条目专属的简短具体理由"}

禁止给大量无关条目复制同一句笼统 rationale。若你无法完成全部 shard，就保留尚未完成的输出文件不存在；绝不能用 uncertain 或模板理由补齐。每完成一个 shard，立即检查：恰好 50 行、逐行 json.loads、item_id 与该输入完全相同且无重复、hallucinated 的 unit_index 确实存在且是 material_claim。全部 16 个 shard 都完成后再结束。
```
