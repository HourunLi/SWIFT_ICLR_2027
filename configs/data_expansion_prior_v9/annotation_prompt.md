# CLIR Prior v9：直接标 Key / Complete

你只标当前公开 JSONL，不查看任何 `PRIVATE_*` 文件，不查看另一个模型的输出，也不调用 CLIR checker、参考答案或历史标签。每个输入行是一道题、一条候选推理，以及已经固定好编号的推理 unit。

这次直接选择两个集合：

- `Key`：最决定这条候选最终结论是否成立的最小实质步骤。正确链通常选直接完成关键推导的步骤；错误链通常选最早造成后续结论失效的关键错误步骤。Key 可以有多个，但只选真正同等关键、缺一不可的 unit。
- `Complete`：为了审计“这条候选实际上如何得到它声称的最终结论”，必须保留的最小实质推理链。它包含候选实际使用的关键中间结果，即使其中有错误；不要改写成你自己的最短解法。

始终满足 `Key ⊆ Complete`。不要为了保险把全部 unit 都放进 Complete。

## 逐行判断

1. 如果只有答案、拒答，或没有可审计推理，使用 `eligibility="no_auditable_reasoning"`。如果 unit 切分让关键主张混在一起、无法按现有边界判断，使用 `insufficient_unitization`。不可用时 Key/Complete 都必须是空数组。
2. 可审计时使用 `eligibility="usable"`，Key 和 Complete 都必须非空。
3. 只在给出的 `kind="material_claim"` unit 中选择；索引必须升序、不能重复。
4. Complete 追踪候选实际使用的因果主线：如果后一步明确使用了前一步的新结果，前一步通常要保留。错误算式以及沿着错误结果继续算的步骤也属于这条实际主线。
5. 通常排除：只复述题面、计划话术、未参与最终结论的旁枝、重复计算、同义重复，以及“所以答案是……”式 final wrapper。若某个看似复述的 unit 提供了后续步骤唯一实际使用且不可从后续 unit 看出的量，才纳入。
6. 同一个事实被两个 unit 重复表达时，保留候选实际先产生并被后续使用的那个；后面的纯重复不纳入。
7. 若存在多个看似等价的最小集合，选择候选文本中更早产生、且被后续步骤明确引用的那条实际路径；仍无法确定时降低 confidence，不要随意扩大成全集。
8. `confidence="low"` 只用于确实无法稳定判断的行；不要把普通难题一律标 low。

## 例子

题目：三盒苹果，每盒 4 个，另有 2 个散装苹果，共多少个？

- unit 0：`3×4=12`。
- unit 1：`12+2=14`。
- unit 2：`所以答案是14`。

应标 `Key=[1]`、`Complete=[0,1]`。unit 2 只是重复结论。

若 unit 0 错写为 `3×4=13`，unit 1 再写 `13+2=15`，则仍标 `Complete=[0,1]`，但 `Key=[0]`：unit 0 是最早决定错误结论的关键错误。

若中间另有 unit `apple 一词有 5 个字母`，且最终计算没有使用它，就不要放入 Complete。

## 输出格式

每个输入行恰好输出一个单行 JSON，顺序可跟输入一致，不要 Markdown、代码围栏、解释段落或漏行：

```json
{"item_id":"原样复制","eligibility":"usable","key_unit_indices":[1],"complete_unit_indices":[0,1],"confidence":"high","rationale":"unit 0 产生中间量，unit 1 用它直接完成结论；末尾只重复答案"}
```

不可用行示例：

```json
{"item_id":"原样复制","eligibility":"no_auditable_reasoning","key_unit_indices":[],"complete_unit_indices":[],"confidence":"high","rationale":"只有答案，没有可审计推理"}
```

`confidence` 只能是 `high|medium|low`。不要输出依赖边、path_status、checker 判断或额外字段。
