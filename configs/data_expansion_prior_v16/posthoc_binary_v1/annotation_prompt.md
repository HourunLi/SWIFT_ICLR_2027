# CLIR Prior v16-posthoc：只判断“这块内容有没有被固定 Key 实际用到”

这是单独命名的 post-hoc 数据重标，不是重开原 v16/v17。你只标当前公开 JSONL。

程序已经：

1. 把回答切成编号 block；
2. 把最后一块明确产出候选数值答案的计算固定为 `fixed_key`；
3. 把明显的计划/标题、纯题面复述、没有代入的通用公式、重复句、答案包装和 Key 后总结固定为
   `fixed_non_main`。

你不审核程序固定的块，只判断 `residual_block_ids` 中每个 block 是 `used` 还是 `not_used`。

## 唯一标准：从 Key 向前做删除检查

题目 `question` 在阅读和核验时始终可见。从 `fixed_key` 向前追这份候选实际写出的路线：

- 删除某块后，无法理解或核验 Key 里的数值、变量、方程、代入、单位换算或分支选择怎样由其余内容
  得到，标 `used`。
- 删除后，仍能结合题目和其余 `used` 块完整理解、核验 Key，标 `not_used`。

请特别遵守：

- 只是重抄题目已经给出的数量或条件，一般是 `not_used`；但若这块新建了变量、方程或关系，而且后续
  确实沿它计算，则是 `used`。
- 题目没有给出的换算常数（如 1 小时=60 分钟），如果固定 Key 用到了它，是 `used`。
- 中间数、变量定义、代数变形、必要的换算和必要的分支选择，只要后续路线实际依赖，就是 `used`。
- 未采用的猜测、试算、旁支、验算或替代解是 `not_used`。
- 通用公式若后面的具体计算已自包含，删掉仍能核验，就是 `not_used`。
- 候选算错也不要修正；错误步骤若确实被候选后续采用，仍是 `used`。
- 不要把“看起来有帮助”“位于 Key 之前”当成 `used`。必须真的通过删除检查。

先在心里完整追一次路线，再一次性输出所有 residual block，不要逐句凭印象标。

## 严格输出

每个输入行恰好输出一个单行 JSON，不要 Markdown，不要额外字段：

```json
{"item_id":"原样复制","residual_decisions":[{"block_id":0,"decision":"used"},{"block_id":2,"decision":"not_used"}],"confidence":"high","rationale":"block 0 的中间量被 Key 使用；block 2 是未采用的旁支"}
```

只允许四个字段：

- `item_id`：原样复制；
- `residual_decisions`：按 `residual_block_ids` 原顺序覆盖每个残余 block，不能多、少或换序；
- `confidence`：`high|medium|low`；
- `rationale`：一句话概括真正使用的链和删掉的旁支。

不要输出 Key、Complete、路径对错、首错位置、role、依赖边或对程序固定块的判断。普通难题不要一律
标 low；只有做完删除检查仍无法稳定判断时才用 low。
