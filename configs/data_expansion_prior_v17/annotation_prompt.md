# CLIR Prior v17：只判断“这一步有没有被最终计算实际用到”

你只标当前公开 JSONL。程序已经做完三件事：

1. 把回答切成编号 block；
2. 把最后一条明确产出候选答案的计算固定为 `fixed_key`；
3. 把明显的计划句、题面复述、通用公式、重复句、答案包装和 Key 后面的总结固定为
   `fixed_non_main`。

你不审核程序固定的 block，只判断 `residual_block_ids` 中的 block。每个残余 block 只有两个选择：
`used` 或 `not_used`。

禁止查看 `PRIVATE_*`、代码、测试、协议、checker、参考答案、历史标签、另一位标注者的包或输出。

## 唯一判断标准

从 `fixed_key` 往前倒着追候选实际写出的路线：

- 如果删掉某个残余 block，会让人无法理解或核验 `fixed_key` 中的数值、变量、方程、分支选择或结论
  是怎样沿这份回答得到的，标 `used`。
- 如果删掉它，`fixed_key` 仍然能沿其余已使用 block 被理解和核验，标 `not_used`。

不要按“这一步在数学上有帮助”判断，也不要因为它出现在 Key 前面就自动标 `used`。只看候选这份回答
是否真的用了它。

常见例子：

- 后面使用的中间数、变量定义、方程、代数变形、必要的单位换算、必要的分支选择：`used`。
- 没被最终路线使用的试算、猜测、旁支、验算或替代解：`not_used`。
- 通用公式如果后面的具体计算已经自包含，删掉公式不影响核验：`not_used`。
- 候选算错了也不要修正；错误步骤若确实被后续采用，仍标 `used`。
- 题面本身始终可见。只是再次抄一遍题面、且没有建立新变量或新关系的句子，不应因为“提供了已知数”而
  标 `used`。

先在心里从 Key 做一次反向删除检查，再一次性输出全部残余 block；不要逐句凭印象判断。

## 严格输出

每个输入行恰好输出一个单行 JSON，不要 Markdown，不要额外字段：

```json
{"item_id":"原样复制","residual_decisions":[{"block_id":0,"decision":"used"},{"block_id":2,"decision":"not_used"}],"confidence":"high","rationale":"block 0 的中间量被 Key 使用；block 2 是未采用的旁支"}
```

只允许四个字段：

- `item_id`：原样复制；
- `residual_decisions`：必须按 `residual_block_ids` 的原顺序覆盖每一个残余 block，不能多、不能少；
- `confidence`：`high|medium|low`；
- `rationale`：一句话概括真正用到的链和被删掉的旁支。

不要输出 Key、Complete、路径对错、首错位置、七类 role、依赖边或任何程序固定 block 的决定。普通难题
不要一律标 low；只有按“删除后是否还可核验 fixed_key”仍无法稳定判断时才用 low。
