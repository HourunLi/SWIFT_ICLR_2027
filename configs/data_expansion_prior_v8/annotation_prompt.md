# CLIR Prior v8：依赖图标注提示词

你只标当前 JSONL，不查看任何 `PRIVATE_*` 文件，不查看另一个模型的输出，也不调用 CLIR checker、参考答案或历史标签。每个输入行是一道题、一条候选推理，以及已经固定好编号的推理 unit。

这次不要直接猜 `Key` 和 `Complete` 集合。你只需要标“候选实际使用了哪些依赖边”，程序随后统一计算：

- `Complete`：从直接得到候选最终结论的 unit 开始，沿依赖边向前做传递闭包；
- 正确/有支撑的链：`Key` 是直接得到最终结论的 unit；
- 有错误的链：`Key` 是 Complete 链里第一个导致结论不成立的错误或无依据 unit。

## 逐行判断

1. 如果只有答案、拒答，或者没有可审计推理，`eligibility="no_auditable_reasoning"`。如果 unit 切分本身不足以判断，使用 `insufficient_unitization`。这两种情况下其余图字段必须为空。
2. 可审计时使用 `eligibility="usable"`，并判断 `path_status`：
   - `supported`：实际推理链足以支撑它声称的最终答案；
   - `flawed`：实际依赖链中存在会使结论失效的算术、代数、数量、实体、单位或无依据步骤；
   - `uncertain`：确实无法判断，只能同时使用 `confidence="low"`，该行不会用于训练。
3. `conclusion_unit_indices` 只选直接算出或推出候选最终答案的实质步骤。不要选“所以答案是……”这种只重复答案的 wrapper。
4. 每条 `dependency_edges` 写成 `[parent, child]`：只有 child 实际使用 parent 的新结果或主张时才连边。仅仅相邻、主题相关、按时间先后出现，都不算依赖。边必须从较小 unit index 指向较大 index，数组按 `[parent,child]` 升序排列且不重复。
5. 原样复述题面、计划话术、未参与最终结论的旁枝、重复等式、同义重复和 final wrapper 不要为了“看起来完整”强行连进图。
6. 不要把候选改写成你自己的最短解法。候选若先算 A，再用 A 算 B，就保留 A→B，即使你能一步重算。
7. `flawed` 时填写 `first_flaw_unit_index`：必须是通向 conclusion 的依赖闭包中，第一个因果上使后续结论失效的 unit。`supported/uncertain` 时必须为 `null`。

## 例子

候选 unit：

- 2：`2+3=5`
- 5：`5-1=4`
- 7：`所以答案是4`

正确输出的核心是 `conclusion_unit_indices=[5]`、`dependency_edges=[[2,5]]`；unit 7 只是 wrapper，不连边。程序会得到 `Complete=[2,5]`、`Key=[5]`。

若 unit 2 错写成 `2+3=6`，unit 5 又用 6 继续算，则 `path_status="flawed"`、`first_flaw_unit_index=2`；程序会得到 `Key=[2]`。

## 输出格式

每个输入行恰好输出一个单行 JSON，顺序可跟输入一致，不要 Markdown、代码围栏、解释段落或漏行：

```json
{"item_id":"原样复制","eligibility":"usable","path_status":"supported","conclusion_unit_indices":[5],"dependency_edges":[[2,5]],"first_flaw_unit_index":null,"confidence":"high","rationale":"一句简短中文理由"}
```

不可用行示例：

```json
{"item_id":"原样复制","eligibility":"no_auditable_reasoning","path_status":null,"conclusion_unit_indices":[],"dependency_edges":[],"first_flaw_unit_index":null,"confidence":"high","rationale":"只有答案，没有可审计推理"}
```

`confidence` 只能是 `high|medium|low`。不要自行输出 `Key` 或 `Complete`；它们由程序从图中计算。
