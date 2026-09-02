# CLIR Prior v16：只判断主链 block，Key/Complete 由程序生成

你只标当前公开 JSONL。机器已把候选推理切成带编号的 `block`。你不标依赖边，也不直接输出
Key 或 Complete：只判断每个 block 的角色、整条路径是否有致命错误，以及最后哪个 block 真正完成
候选所声称的答案。

禁止查看 `PRIVATE_*`、代码、测试、协议、checker、参考答案、历史标签、另一位标注者的包或输出。
`role_hint`、`proposed_final_block_id` 和 `final_block_candidates` 都只是机器提示，不是真相。

## 程序如何使用你的判断

- 两位标注者都判为 `main_step` 的 block，成为 Complete 正例；
- 两位都判为非主链的 block，成为 Complete 负例；
- 一位判主链、一位判非主链的 block 会被程序遮住，不强行定对错；
- 两位相同的 `final_block_id` 生成 Key；
- 路径即使算错，Key 仍是“最终完成候选答案的结构步骤”，不是最早错误；
- 最早错误完全属于 Hallucination 模块，本轮不要定位它。

所以你必须按候选实际写出的推理判断主链，不能把它改写成自己更短的解法。

## Block 角色

每个 block 必须按 `block_id` 顺序恰好判断一次：

- `main_step`：候选实际用来形成所声称答案的定义、设元、方程、变形、计算、选择或结论；算错也仍是主链。
- `premise_restatement`：只复述题面已经明确给出的事实，没有引入候选自己的变量、别名、方程或新关系。
- `plan_or_heading`：只说“接下来计算……”或只是标题，没有产生可用内容。
- `formula_only`：只写通用公式，没有代入本题对象、数值或约束。
- `duplicate`：完整重复前面已经表达的同一个结果。
- `answer_wrapper`：只包装或重复答案，没有新计算或选择。
- `unused_branch`：做了计算或推导，但候选最终答案没有使用它。

特别注意：`Let x=...`、`设 D=...`、把题意写成方程、连续代数改写，只要后面实际使用，都是
`main_step`，不能因为它像“定义”或“复述”就标成 `premise_restatement`。

## Path 与最终 block

- 有可审计推理：`eligibility="usable"`；只有答案或拒答用 `no_auditable_reasoning`；切分后仍无法辨认
  主线用 `insufficient_structure`。
- `path_status="supported"`：这些 main_step 足以支撑候选所声称答案；存在会让结论失效的算术、代数、
  对象、单位、逻辑或无依据步骤则为 `flawed`。
- `final_block_id` 是最后一个真正完成候选所声称答案的 `main_step`，不能选纯答案包装。
- 错误路径照样标出它实际使用的全部 main_step；不要在 rationale 里给出 first-error unit 编号。

## 严格输出

每个输入行输出一个单行 JSON，不要 Markdown，不要额外字段：

```json
{"item_id":"原样复制","eligibility":"usable","path_status":"supported","block_roles":[{"block_id":0,"role":"main_step"},{"block_id":1,"role":"answer_wrapper"}],"final_block_id":0,"confidence":"high","rationale":"block 0 完成候选答案，block 1 只包装答案"}
```

不可用行必须完全清空结构字段：

```json
{"item_id":"原样复制","eligibility":"no_auditable_reasoning","path_status":null,"block_roles":[],"final_block_id":null,"confidence":"high","rationale":"只有答案，没有可审核推理"}
```

只允许七个字段：

- `item_id`
- `eligibility`: `usable|no_auditable_reasoning|insufficient_structure`
- `path_status`: usable 时 `supported|flawed`，否则 `null`
- `block_roles`: usable 时覆盖全部 block 且按序，否则 `[]`
- `final_block_id`: usable 时为一个 role=`main_step` 的整数，否则 `null`
- `confidence`: `high|medium|low`
- `rationale`: 一句具体说明主链、错误和最终步骤

不要输出 `dependency_edges`、`missing_edges`、`key_unit_index`、`Key` 或 `Complete`。普通难题不要一律
标 low；只有按上述规则仍无法稳定判断时才用 low。
