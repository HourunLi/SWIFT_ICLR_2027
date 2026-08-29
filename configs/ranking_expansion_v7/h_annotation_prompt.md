# H0 v7 双 AI 标注提示词

这份文件只规定标注行为。实际 smoke 包生成后，以 package report 里的绝对路径和行数为准。
A、B 必须在两个互相独立的会话中执行，不得读取对方的包、输出或
`PRIVATE_package_index.jsonl`。

## 给标注者 A（GPT-5.6-sol xhigh）

```text
你是 CLIR H0 数据的独立标注者 A。请直接读取：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/packages/smoke/annotator_a/hallucination.jsonl

把结果写到：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/annotation/smoke_a_gpt56sol.jsonl

不要读取 annotator_b、PRIVATE_package_index.jsonl、checker/reference answer、其他模型输出或历史标签。不要上网搜索题目。你需要自己解题并逐条审查 trajectory。

任务是找“第一个明确坏掉的 material_claim unit”，不是猜第一个坏 token：
- clean：每个 material claim 都能由题面、此前仍成立的推导或普通数学规则支持。
- hallucinated：至少一条 material claim 明确错误，或凭空引入会影响结论的新事实、数字、对象、单位、等式或推理关系。
- 省略常规算术展开不算缺前提；明确算错、抄错、换对象、换单位或使用无依据关系算坏 claim。
- 多处错误取最早的 material_claim；后面改正也不抹掉第一次错误。
- 如果最早坏点被 unit 切分粘连而无法稳定定位，标 insufficient_unitization。
- 只有答案、拒答、没有可审计推理时标 no_auditable_reasoning；确实无法稳定判断时标 uncertain。不要为了凑 clean/hallucinated 比例而猜。

每个输入 item 必须恰好输出一行 JSON，保持 item_id 原样，字段严格为：
{"item_id":"...","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":整数或null,"confidence":"high|medium|low","rationale":"简短具体理由"}

约束：
- clean/uncertain/insufficient_unitization/no_auditable_reasoning 的 first_bad_unit_index 必须是 null。
- hallucinated 必须填写输入里存在且 kind=material_claim 的 unit_index。
- 只写 JSONL，不加 Markdown、解释段、汇总行或空白模板。
- 覆盖输入每个 item_id 一次且仅一次；不要去重看起来相同的题，因为包中含有盲重复一致性检查。
- 完成后自行验证输出能逐行 json.loads、item_id 无重复、输出行数等于输入行数；若不满足就修复文件后再结束。
```

## 给标注者 B（Claude Opus 5 high）

```text
你是 CLIR H0 数据的独立标注者 B。请直接读取：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/packages/smoke/annotator_b/hallucination.jsonl

把结果写到：
/prodcpfs/user/panzhixin/ICLR_2027/run_artifacts/ranking_expansion_v7/pre_annotation/annotation/smoke_b_claude_opus5.jsonl

不要读取 annotator_a、PRIVATE_package_index.jsonl、checker/reference answer、其他模型输出或历史标签。不要上网搜索题目。你需要自己解题并逐条审查 trajectory。

任务是找“第一个明确坏掉的 material_claim unit”，不是猜第一个坏 token：
- clean：每个 material claim 都能由题面、此前仍成立的推导或普通数学规则支持。
- hallucinated：至少一条 material claim 明确错误，或凭空引入会影响结论的新事实、数字、对象、单位、等式或推理关系。
- 省略常规算术展开不算缺前提；明确算错、抄错、换对象、换单位或使用无依据关系算坏 claim。
- 多处错误取最早的 material_claim；后面改正也不抹掉第一次错误。
- 如果最早坏点被 unit 切分粘连而无法稳定定位，标 insufficient_unitization。
- 只有答案、拒答、没有可审计推理时标 no_auditable_reasoning；确实无法稳定判断时标 uncertain。不要为了凑 clean/hallucinated 比例而猜。

每个输入 item 必须恰好输出一行 JSON，保持 item_id 原样，字段严格为：
{"item_id":"...","status":"hallucinated|clean|uncertain|insufficient_unitization|no_auditable_reasoning","first_bad_unit_index":整数或null,"confidence":"high|medium|low","rationale":"简短具体理由"}

约束：
- clean/uncertain/insufficient_unitization/no_auditable_reasoning 的 first_bad_unit_index 必须是 null。
- hallucinated 必须填写输入里存在且 kind=material_claim 的 unit_index。
- 只写 JSONL，不加 Markdown、解释段、汇总行或空白模板。
- 覆盖输入每个 item_id 一次且仅一次；不要去重看起来相同的题，因为包中含有盲重复一致性检查。
- 完成后自行验证输出能逐行 json.loads、item_id 无重复、输出行数等于输入行数；若不满足就修复文件后再结束。
```
