# CLIR 三模块阶段报告：大白话图解版

副标题：不先背术语，也能看懂“我们做了什么、数据怎么标、现在效果到哪一步”

- 日期：2026-08-24
- 分支：`clir-clean-integration`
- 方法状态基线：`049eb091113ad92b615bd4bd0dce6fb782ea115e`
- 当前默认 gate：`gate_prior_weight=.25`，默认开启
- 证据等级：小规模真实数据筛选，不是论文级最终结论

> 阅读建议：先看第 1、2、7、9 节，就能掌握主要结论；第 4–5 节专门用实际数据举例解释几种标注；
> 第 11 节再讲下一步怎么批量扩数据。

## 1. 先用一句话说清项目在干什么

同一道数学题，我们先让大模型生成多份候选解答，再训练一个“小评委”给每份解答打分，最后选择分数
最高的那份。

普通评委只学习“最后答案对不对”。CLIR 还想额外教它三件事：

1. **Consistency（语义一致性）**：同一种正确思路，换个说法，分数不应该大变。
2. **Hallucination（推理何时开始出错）**：不仅知道答案错了，还要尽量找到第一处错误。
3. **Dual Prior（关键点 + 完整链）**：既要看到最关键的一步，也要看到支撑结论的整条必要推理链。

可以把整个系统想成：

```text
一道题
  └─ 大模型生成 16 份候选解答
       └─ CLIR 小评委逐份打分
            ├─ 最后答案对不对？
            ├─ 换个写法是否还应得到相近分数？
            ├─ 从哪一步开始不可信？
            └─ 哪些步骤最关键、哪些步骤构成完整依据？
                 └─ 选择分数最高的一份
```

这里的最终目标不是“把辅助标签预测得漂亮”，而是：**在同一道题的多份候选解答中，更常把正确的那份
排到第一名。**

## 2. 当前结论，不绕弯子

### 2.1 已经确认的

- 三个模块都已接入真实全层 hidden state 数据，能够训练、保存、恢复、打分和做机制诊断。
- Consistency、Hallucination、Dual Prior 都不是只有代码空壳；当前训练集中确实有对应标签。
- Consistency 和只做“首错位置分类”的 H0 有正向点估计，值得扩数据复测。
- Direct Key/Complete 在很小的机制验证集上确实学到了一些标注结构。
- Prior→gate 路径确实会改变最终候选选择，不是“接了但没起作用”。
- 用户要求保留 `main` 原版 shared-gradient coupling 后，gate 已固定为 `.25` 并默认开启。

### 2.2 还不能下结论的

- 不能说三个模块已经稳定提高最终 Best-of-16；现有置信区间普遍跨 0。
- 不能说 Consistency 和 H0 天生冲突；当前只看到小数据上的负交互信号。
- 不能说 Hallucination 已经能准确找到第一处错误；6 条正例上，`±5 token` 命中仍为 0。
- 不能说 Mutual Prior 有用；它没有比 Direct Prior 带来额外收益。
- 不能说 gate `.25` 已被独立验证有效；它是在同一开发集上选出来的工程默认值。
- 旧 Full 没赢 correctness-only；而加入新 `.25` gate 后的当前完整默认配置还没有重新做独立 Full 交互实验。

### 2.3 为什么现在不应该只加 epoch

`epoch` 就是“把现有训练数据再看一遍”。当前 Consistency 只有 27 对，H 只有 17 条正首错 + 31 条
明确无错，Prior 只有 48 条训练轨迹。多跑几轮只是在重复这批标签，不会凭空增加新题型、新错误类型或
更可靠的边界。之前已经看到部分训练 loss 继续下降，但小验证集反而变差。

因此下一步的主矛盾是扩充独立数据，而不是继续消费同一小验证集调参数。

## 3. 术语翻译：英文到底在说什么

| 术语 | 大白话解释 | 在当前项目中的具体含义 |
|---|---|---|
| Candidate / trajectory | 一份候选解答 / 一整条推理过程 | 同一道题会有 8 或 16 份候选 |
| Token | 模型眼里的小文字块 | 标签和 hidden state 都必须和这些位置一一对应 |
| Hidden state | 大模型读到某个 token 时内部形成的特征 | 当前保存 embedding + 32 个 block，共 33 层 |
| Reward / score | 小评委给整份解答的分数 | Best-of-N 最后按这个分数选第一名 |
| Gate | 给每个 token 分配“这一步在总分里有多大话语权”的权重 | gate 越大，该 token 的 value 对总分影响越大 |
| Value | 每个 token 对整份解答的局部加分或减分 | onset 后的 negative tail 会直接训练这个量变负 |
| Prior | 一张“应该重点看哪里”的教学提示图 | 这里不是玄学先验，而是 Key/Complete 标注产生的 token 重要性图 |
| Onset | 第一处错误开始的位置 | `-1` 表示明确检查过且整条 clean；字段缺失表示没标 |
| Tail | 从首错位置到解答结尾 | main-style H 会把这一段当作受污染尾部 |
| BCE | 教模型做 0/1 判断的一种常见损失 | 如这个 token 是否位于首错之后、答案是否正确 |
| BoN / Best-of-N | N 份候选里按分数挑一份 | BoN@16 就是每题看 16 份，挑最高分，统计挑对比例 |
| Pairwise accuracy | 同题里随机拿一对“正确/错误”解答，正确的分数更高的比例 | 它衡量一般排序，不完全等同于最顶部能否选对 |
| Seed | 一次训练使用的随机种子 | 不同 seed 会改变初始化和批次顺序；当前主要比较 42/43/44 |
| Confidence interval | 结果的不确定范围 | 如果“增益区间”跨过 0，就不能排除只是抽样波动 |
| Detach / stop-gradient | 把某个输出当作教学目标，不让这条 loss 反向改它 | fused prior 在 gate loss 中 detach，主要去训练 gate 和共享 encoder |
| Coupling | 把一个辅助模块接到最终打分路径上 | 例如 prior 通过 loss 去约束本来就参与 score 的 gate |

### Gold、Silver 到底是什么意思

这两个词说的是**标签可信度和生产方式**，不是模型效果等级。

- **Gold（金标）**：经过较严格的独立标注和分歧裁决，成本高、可信度相对高，但仍不等于绝对真理。
  当前 Key/Complete Prior 使用两份独立标注；完全一致的直接收下，有分歧的做 role-blind 裁决。
- **Silver（银标/工程标）**：通常由模型、规则或较轻量人工流程辅助，再进行一定检查，便宜但噪声更大。
  当前 H onset 数据明确标记为内部盲裁 pipeline pilot，不是论文级人工 Gold。
- **Pseudo label（伪标签）**：模型根据自己的预测生成标签。例如没有外部 onset 时，用 H head 第一次过阈值
  的位置当 pseudo onset。当前这条路默认关闭，因为模型还没学准就用自己教自己，容易把错误循环放大。

历史实验名里的 `gold tail` 容易误导。它主要想表达“tail 起点来自外部已标 onset，而不是模型自猜的
pseudo onset”。它**不表示当前 H onset 数据已经达到论文级 Gold**。本报告后文统一称为“外部标注起点后的
负分尾部”。

## 4. 目前到底有哪些标注

| 标注类型 | 标签长什么样 | 教模型什么 | 当前来源 |
|---|---|---|---|
| Correctness | 整条解答一个 `0/1` | 最终答案对不对 | 数值答案 checker v5 |
| Consistency | `semantic_id + style_id` | 同思路不同写法应该接近 | 两份盲审关系标注 + 分歧裁决 |
| H onset | 一个 token 下标；clean 用 `-1` | 第一处错误从哪里开始 | 内部双标 + 冲突盲裁，工程级 pilot |
| Path H | 整条解答一个 `0/1` | 这条推理里是否 somewhere 有错 | 与 onset 同源；MIL 当前关闭 |
| Key Prior | 每个 token 一个 `0/1` | 最关键的一步或最致命错误在哪里 | 双人独立标注 + role-blind 裁决，Gold 流程 |
| Complete Prior | 每个 token 一个 `0/1` | 哪些步骤合起来构成完整支撑链 | 与 Key 同源，覆盖更宽 |

几个很重要的规则：

1. `correctness=0` 只代表最后答案错，不自动告诉我们从哪一步开始错。
2. `hallucination_onset=-1` 表示明确检查过且 clean；没有这个字段表示没标，不能偷偷当 clean。
3. Key/Complete 必须和保存的 output token 一一对齐，不允许从文本重新分词后“差不多映射”。
4. 同一份数据可以只有部分标签；每种标签有独立 mask，没标的部分不会被当负样本。

## 5. 四种标注的具体例子

以下例子来自当前真实训练 row；为了让中文读者容易理解，题目和推理做了忠实中文改写，数字、逻辑和
标签位置保持原意。

### 5.1 Correctness：只判断最后选对没有

**题目：** Kantana 每个周六买 2 块巧克力给自己、1 块给妹妹。上个周六还额外买了 10 块送朋友。
一个月按 4 个周六算，她这个月共买多少块？

```text
正确思路：每周固定 3 块，共 3×4=12；生日礼物只买一次，再加 10。
正确答案：22  → correctness = 1

错误候选：把上周额外买的 10 块也当成每周都买，算成 (3+10)×4=52。
错误答案：52  → correctness = 0
```

Correctness 只告诉模型“52 这条最后错了”。它本身没有告诉模型错误恰好发生在“把一次性礼物重复四次”。

### 5.2 Consistency：同一个思路，简写和详写不应差太多

**实际题目：** 开胃菜 10 美元，4 份主菜每份 20 美元，再付 20% 小费，总共多少钱？

```text
简写版：4×20=80；80+10=90；小费 18；总计 108。

详写版：
1. 逐份解释 4 份主菜为什么是 80；
2. 解释 subtotal 为什么是 90；
3. 把 20% 写成 0.2，再算 0.2×90=18；
4. 90+18=108。
```

两条的表达长度和 style 不同，但核心推理、关键中间量和结论相同，所以：

```text
semantic_id 相同
style_id 分别是 native_compact / native_expanded
→ positive consistency pair
```

反过来，两道不同题即使都写成“Step 1 / Step 2 / Final answer”的格式，也不能因为表面 style 一样就被
模型当成同一语义。

### 5.3 Hallucination onset：从哪一句开始错

还是巧克力题。真实 row `gsm8k-train-00333-cand-003` 有 279 个 output token，外部标注 onset 是
第 196 个 token，对应句子开头：

```text
前面尚可接受：上个周六，她常规买 3 块，又额外买 10 块，所以那个周六买了 13 块。

                 ↓ onset = token 196，从 “Since” 开始
错误开始：Since there are 4 weeks in a month, ... 13 × 4 = 52。
          因为 10 块生日礼物只是一次性的，不能跟着每周重复。
```

模型得到两类训练信号：

- H0：onset 前标 0，onset 及以后标 1，让 H head 学“哪里开始不可信”。
- H1：除了 H0，再要求 onset 后的 token value 变成负值，试图让最终 score 直接降低。

当前结果是：H0 的最终选择有正向点估计，但定位边界仍不准；H1 反而把整条解答的 value 普遍推低，
没有形成理想的“只在首错之后明显下降”。

### 5.4 Key 和 Complete：一个是“最关键”，一个是“完整证据链”

**实际题目：** 周六 80 人；周一少 20 人；周三比周一多 50 人；周五等于周六和周一之和。全周实际
人数比预计 350 人多多少？

正确推理：

```text
[Complete] 周一：80 - 20 = 60
[Complete] 周三：60 + 50 = 110
[Complete] 周五：80 + 60 = 140
[Complete] 总人数：80 + 60 + 110 + 140 = 390
[Key + Complete] 最后差值：390 - 350 = 40
```

- **Key** 只抓最能决定答案的那一步：`390 - 350 = 40`。
- **Complete** 把得到 390 和最终得到 40 所需的整条链都覆盖。

错误解答也可以标 Prior：Key 不一定是“正确答案那一步”，也可以是“最致命的错误”。在巧克力错误候选中，
Key 正是 `Since there are 4 weeks ... 13×4=52` 这一段。

### 5.5 Clean、未标注和伪标签不是一回事

```text
hallucination_onset = -1    明确检查过，认为整条 clean
字段缺失                      没有人对这条做 onset 判断
pseudo onset                模型自己猜出的首错位置
```

把“字段缺失”当 clean，会制造大量假负例；把 pseudo onset 当 Gold，会让模型把自己的错误判断不断写回训练。
当前 reader 和 loss 都保留独立 mask，避免这两类偷换。

## 6. 最终分数到底怎么来的

每个 token 都有两个重要量：

- `gate_t`：这一步在整条解答总分里有多少话语权；
- `value_t`：这一步倾向于加分还是减分。

大白话版本的公式是：

```text
token 部分总分 = 所有 token 的“话语权 × 局部价值”的加权平均
最终分数       = token 部分总分 + 整条解答的总体补充分
```

可以把 gate 想成舞台上的聚光灯：灯照得越亮，这一步的 value 越能影响最后分数。Prior→gate coupling 做的
不是在演出结束时另外加一张评分表，而是在训练时教聚光灯“哪些步骤应该多看”。

## 7. 三个模块各自怎么影响最后选择

### 7.1 Consistency

有两条路：

1. 更新共享 encoder，让同语义不同 style 的内部表示更接近；
2. 直接约束两条同义解答的最终 scalar score 不要相差太大。

所以 Consistency 不只是“共享表示间接影响”，它确实有一项直接作用在最终 score 的训练约束。

### 7.2 Hallucination

- **H0（只有 onset BCE）**：H 概率本身不会在推理时直接从 score 里扣掉。它主要通过共享 encoder 间接
  改变 gate、value 和总体补充分。
- **H1（H0 + 外部起点后的 negative tail）**：额外直接训练 score 使用的 token value，让 onset 后变负。

因此 H0 和 H1 不能混为一谈：H0 是“教一个定位头”，H1 是“把已知首错后的局部价值直接往负方向推”。

### 7.3 Dual Prior

- Direct Key/Complete BCE：主要训练两个 prior head 和共享 encoder；如果 gate coupling 关闭，它对最终
  score 主要是间接影响。
- Mutual：让 Key 和 Complete 两张图互相靠近，但两边都用 stop-gradient 保护对方目标。
- 当前默认 gate coupling：把 `0.5 Key + 0.5 Complete` 融合图 detach 后，用 `.25` 权重约束 reward gate。
  gate 本来就直接参加最终 score，所以现在 Prior 获得了一条更直接的训练路径。

推理时不需要 Key/Complete 人工标签，也不会临时把 prior 分数加到 reward 上；模型使用训练后已经学到的
gate。只有训练 row 同时具有 Key 和 Complete coverage 时，gate alignment loss 才会计算。

## 8. 单模块效果：现在到底成了什么样

主要看 BoN@16：每题有 16 份候选，模型按分数选一份，统计选对率。以下都是 3 个 seed 的平均值。

| Cell | 大白话含义 | BoN@16 | 相对 correctness-only |
|---|---|---:|---:|
| C0 | 只学最后答案对错 | `91.73%` | 基线 |
| C1 | C0 + Consistency | `92.20%` | `+0.47` point |
| H0 | C0 + 首错位置 BCE | `92.67%` | `+0.93` point |
| H1 | H0 + 首错后 value 负分 | `91.87%` | `+0.13` point；相对 H0 `-0.80` |
| P0 | C0 + Direct Key/Complete | `91.80%` | `+0.07` point |
| P1 | P0 + Mutual | `91.80%` | `+0.07` point；Mutual 增量为 0 |
| CH0 | C1 + H0 | `91.53%` | `-0.20` point |
| 旧 Full | C1 + H1 + P1，gate 当时关闭 | `91.60%` | `-0.13` point |

如何翻译这些数字：

- C1 和 H0 的点估计向上，但区间跨 0，所以是“值得扩量复测”，不是“已经证明有效”。
- H0 是当前最好点估计；加上 negative tail 后 H1 三个 seed 都比 H0 低，说明当前 tail 实现有问题。
- P0 的 Key/Complete 在小机制集上能学，但没有转化成明显的最终选对提升。
- P1 没比 P0 更好，所以 Mutual 暂时没有增量价值。
- CH0 和 Full 都没有出现“模块收益相加”。

## 9. 组合为什么可能互相拖累

### 9.1 C1 + H0

CH0 是最干净的二因子组合：只把 Consistency 和 H0 放在一起，不混入 tail 或 prior。

```text
C0  = correctness only
C1  = C0 + Consistency
H0  = C0 + onset BCE
CH0 = C0 + Consistency + onset BCE
```

CH0 的 BoN@16 是 `91.53%`，比 H0 低 `1.13` points。二因子交互点估计为 `-1.60` points，三个 seed
都是负方向，但把训练 seed 也作为不确定性来源后，区间仍跨 0。

一种与现象相符、但还没证明的解释是：Consistency 倾向于把同义解答的分数拉平，而 H0 在当前小数据上
形成的顶部排序很脆弱；一拉平，最高分候选的顺序可能变化。也可能只是 27 对关系和 17 条正 onset 太少，
两个 loss 在共享 encoder 上争抢有限信号。

### 9.2 H0 + negative tail

H0 有一点正向排序信号，H1 却回退。机制诊断发现 H1 没学成“首错之前正常、首错之后明显变负”，而是把
整条 trajectory 的 value 大体都推到约 `-.62`。这叫 global shift（全局平移）：模型找到了一个满足 loss
的省事办法，却不是我们真正想要的局部惩罚。

### 9.3 Full

旧 Full 的名字容易让人误以为“全功能就应该最好”。实际上它同时启用 C1、H1、P1，而各模块使用同一个
共享 encoder；更多 loss 可能提供更多信息，也可能产生梯度冲突。旧 Full 为 `91.60%`，没有赢 C0。

注意：旧 Full 训练时 gate alignment 是 0。当前 `best_current` 新增 `.25` gate，但还没有在扩充后的独立
数据上重新验证完整组合，所以不能拿 isolated gate 调参结果替代 Full 交互实验。

## 10. Gate：为什么最后选 `.25`

### 10.1 它做的事情

当前保持 `origin/main` 原版实现方式：

```text
gate attention = 归一化后的 sigmoid(gate logits)
fused prior    = 归一化后的 0.5×Key + 0.5×Complete，并 detach
gate loss      = 两张分布之间的 squared-L2
```

没有改成 KL，没有只训练 gate head，没有在推理时额外融合 prior，也没有另造一个 gate。

### 10.2 六个权重点的结果

| gate weight | BoN@16 | gate↔prior L2，越低越贴近 | gate 有效覆盖比例 |
|---:|---:|---:|---:|
| `0` | `91.80%` | `.011953` | `.8149` |
| `.0625` | `91.67%` | `.013347` | `.7907` |
| **`.25`** | **`91.87%`** | `.012798` | `.7952` |
| `1` | `91.80%` | `.010270` | `.7722` |
| `4` | `91.73%` | `.005609` | `.7290` |
| `10` | `92.07%` | `.000934` | `.3645` |

`10` 的 BoN@16 点估计最高，但只比 `.25` 高 `0.20` point，恰好落在预先写好的 near-tie 边界内。规则规定
这种情况下选更小权重，所以固定 `.25`。

选小一点的理由也比较直观：`10` 虽然强行把 gate 拉得非常贴近 prior，却把 gate 的有效覆盖压到约
36%；`.25` 仍保持约 80% 的宽覆盖，更不容易把 prior 小数据中的偶然位置模式硬写进最终 gate。

### 10.3 一个必须说清的系数差异

`origin/main` 里外层 `prior_weight=.25`、内层 `gate_prior_weight=.25`，总有效系数是 `.0625`。clean 当前
外层 `prior_weight=1`，所以 `.25` 的总有效系数就是 `.25`，是原 main 绝对强度的 4 倍。

因此准确说法是：**公式、detach、mask 和 shared-gradient 路径保持 main 原版；强度经过当前开发集工程
调参后固定为 `.25`。**

### 10.4 为什么不能说 gate 已经证明有效

- `.25` 相对 gate-off P0 只有 `+0.07` point；两个配对区间都跨 0。
- 选参和汇报使用了同一个 500-query development population。
- `.25` 的 held-out gate L2 并没有比 P0 更低。
- 本轮只隔离了 correctness + direct prior，没有重新跑当前完整 Full。

所以它的标签是 `dev-tuned engineering default`，即“开发集调出的工程默认值”，不是“独立验证过的增益”。

## 11. 当前数据够不够，以及怎么批量扩充

### 11.1 当前规模

| 数据用途 | 当前量 | 能做什么 | 主要不足 |
|---|---:|---|---|
| Outcome correctness train | 496 题 ×8 = 3968 rows | 训练基础排序骨架 | 题量仍小；与 ranking checker 版本不同 |
| Consistency train | 27 个正关系对，54 rows | 验证 grouped batch 和 loss 能跑 | 太少；没有 held-out relation set |
| H train | 17 条正 onset +31 条 clean | 小规模机制筛选 | 边界噪声大、错误类型覆盖窄 |
| H mechanism dev | 6 正 +10 clean | 只能看很粗的点估计 | 任何 1 条都会明显改变比例 |
| Prior train | 48 条 trajectory | 看 Direct Key/Complete 是否能学 | 独立轨迹太少，位置/长度 shortcut 风险高 |
| Prior dev | 16 条 trajectory | 小门诊断 | 不足以选复杂 routing 或强度 |
| Ranking dev | 500 题 ×16 | 做 matched screening | 已被多轮比较和 gate 选参使用 |

结论是：**够做工程闭环和初筛，不够做正式机制归因或最终论文结论。**

### 11.2 建议扩到多少

| 模块 | 下一阶段建议 |
|---|---:|
| Consistency | 300–500 个 train semantic groups；100–200 个完全 held-out groups |
| Hallucination | 至少 200 条正 onset +200 条明确 clean train；dev 至少 100+100 |
| Dual Prior | 300–500 条独立 train trajectories；100–200 条 query-disjoint dev |
| Ranking validation | 1500–2000 个独立 queries ×16 candidates |
| Outcome train（预算允许） | 1500–2000 queries ×8 candidates |

### 11.3 批量扩充的推荐流水线

#### A. Outcome / correctness

1. 先按 query 切 train/dev/test，再生成候选，避免同题泄漏。
2. 每题固定生成 8 或 16 个候选，保存原始 prompt/output token IDs 和 candidate order。
3. 用统一的 checker v5 判断最终答案，保存解析答案、参考答案、失败原因和 checker 版本。
4. 抽查边界案例，如单位、分数、百分比、货币和多答案格式。

#### B. Consistency

1. 从同一题候选里找“方法、关键中间量和结论都相同，但长短/style 不同”的候选对。
2. 也可以受控生成 compact、expanded、formal、conversational 等 rewrite，但必须验证语义没有变。
3. 两份独立标注分别判断是否 reasoning-equivalent；一致则收下，分歧进入 blind adjudication。
4. 按原始 query 划分，保证 dev 的 semantic group 从未出现在 train。

#### C. Hallucination onset

1. 先把 trajectory 按 reasoning claim 切成稳定单元，而不是让标注者面对一长串 token。
2. 两位标注者逐步判断 supported / contradicted / unsupported / non-claim / uncertain。
3. 给出第一条 material error 的原文 quote 和理由；clean 必须显式确认。
4. 分歧进入盲裁；统计 path agreement、onset 距离和不同错误类型覆盖。
5. 最后才通过保存的 output token IDs 把 claim/unit 映射到精确 token onset。

#### D. Key / Complete Prior

1. 用确定性规则先把解答切成 reasoning units。
2. 两位标注者独立选：最关键/最致命的 Key units，以及足以支撑判断的 Complete units。
3. 完全一致直接接收；不一致时隐藏标注者身份，由第三方选择 A、选择 B 或综合新方案。
4. 同时保留正确和错误 trajectory；错误 trajectory 的 Key 可以是决定性错误。
5. 检查 `Key ⊆ Complete`、位置分布、长度分布，再映射为 exact-token `0/1` target。

#### E. 质量与发布

1. 每批先做 20–50 条 smoke，计算一致率和问题分布，再扩大。
2. 每次发布不可变 manifest，绑定标注协议、模型/checker 版本、query IDs 和 SHA-256。
3. 训练、机制 dev、ranking validation 必须 query-disjoint。
4. 选完参数后锁 validation，再使用一次 protected test；不能边看 test 边改权重。

## 12. 下一轮实验怎么设计

### 12.1 Consistency × H0

扩数据后完整重跑：

```text
C0   correctness only
C1   C0 + Consistency
H0   C0 + onset BCE
CH0  C0 + Consistency + onset BCE
```

四个 cell 使用相同候选、初始化策略、训练预算和 seeds。主看各单模块增量以及
`CH0 - C1 - H0 + C0` 交互。

### 12.2 Prior 与 gate

先复核新数据上的 P0 Direct Key/Complete learnability，再固定 `.25` 比较：

```text
gate off = 0
gate on  = .25
```

不再用当前 16-row / 500-query dev 继续扫权重。若研究 KL、head-only 或推理时 prior fusion，应明确当作
新方法另开实验，不能混成“原 gate 修复”。

### 12.3 Hallucination tail

先让 onset boundary 和 calibration 在独立数据上过门，再讨论新的局部 reward coupling。当前 H1 的全局
value shift 问题没有解决，因此不应该靠多加 epoch 继续顶。

## 13. 现在对外应该怎么表述

可以说：

- 三模块已在真实 exact-token 全层特征上完成工程接入和多 seed 小规模筛选。
- Consistency 和 H0 有正向点估计，值得扩大独立数据后复测。
- Direct Key/Complete 在小机制集上有 learnability；Mutual 没有增量。
- 当前 negative-tail 和旧 Full 没有通过效果门。
- main-style prior→gate 路径默认开启，`.25` 是开发集调出的工程默认值。

不可以说：

- “三个模块已经稳定提高 Best-of-N。”
- “H0 已经准确找到第一处错误。”
- “Gold tail 已经有效。”
- “Dual Prior 已经通过 gate 提高最终选择。”
- “Consistency 和 Hallucination 天生不兼容。”
- “Full 是当前效果最好的配置。”

## 14. 最后总结

现在最准确的状态不是“方法失败”，也不是“方法已经成功”，而是：

1. **工程闭环完成。** 三模块、数据 mask、训练、评分、诊断、配对比较都能真实运行。
2. **部分标签可学。** Direct Prior 和 H token/path 有小样本信号；Consistency 能影响训练和排序。
3. **最终收益尚未建立。** C1/H0 是候选，tail/mutual/旧 Full 未过门，组合出现负交互信号。
4. **Gate 默认保留。** 按方法身份要求和冻结规则选 `.25`，但只标记为工程默认值。
5. **下一步先扩独立数据。** 扩标、统一 checker、建立 held-out relation/mechanism set，再固定协议复测。

## 附录 A：例子与结果来源

- Consistency 餐厅账单 pair：`gsm8k-train-01458-cand-000/007`
- Hallucination 巧克力错误候选：`gsm8k-train-00333-cand-003`；279 tokens；onset `196`
- Key/Complete 足球观众候选：`gsm8k-train-04355-cand-006`
- 当前 joint train：3968 rows / 496 queries ×8
- 当前 mechanism dev：16 trajectories
- 当前 ranking dev：500 queries ×16
- Gate v2 paired summary SHA-256：
  `3aaf0d67378c2bb189bf1794e6d0f683822f1b5aca2836037dedd11084db277c`

例子中的中文为便于理解的忠实改写；精确训练标签仍绑定原始 Phi output token IDs，不使用中文文本重新映射。
