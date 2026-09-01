# CLIR：Consistency-Localized Intrinsic Rewards

CLIR 是一个自包含的 hidden-state reward model 研究实现。它参考 SWIFT 的 token reward / gate 聚合方式，在 frozen LLM 的生成 token hidden states 上加入三类监督：语义一致性、幻觉起点后的局部负奖励，以及 key/complete 双先验定位。仓库不依赖 SWIFT 源码。

当前分支的目标是提供一个小而可运行的研究主干：保留 `main` 的方法语义，吸收 `panzhixin` 分支中已经证明有工程价值的 exact-token 数据链路、全层特征压缩、严格 mask、可恢复训练和查询级评估；历史协议、标注流水账和失败实验实现不进入核心目录。

需要先明确证据边界：[`configs/best_current.json`](configs/best_current.json) 是当前唯一的**整合配置**，不是已经证明优于 correctness-only baseline 的“最优效果配置”。三模块联合训练的历史结果没有通过扩展门，详见 [`docs/handoff.md`](docs/handoff.md)。

## 当前标注账本

必须把“已经用于小规模训练的数据”和“后来只做流程审计的数据”分开：

| 数据层 | 现有规模 | 现在能否训练 | 准确含义 |
|---|---:|---|---|
| Outcome/correctness | 3,968 条候选轨迹 | 是 | 3,590 条数值答案匹配、378 条不匹配，训练基础 reward score |
| Consistency | 27 个 compact/expanded 正 pair（54 个 view）+702 个负 pair | 是，但只够筛选实验 | 同一道题、同一路径的一简一详应得到接近表示；没有独立 held-out relation set |
| Hallucination H | 历史 17 positive +31 clean；v7.4 另有 400 train +200 dev，均为正负各半且每条来自不同 query | v7.4 可作探索性 H0 训练，不能作确认性证据 | 双 AI 标出“从哪个推理单元开始出现无依据/错误主张”。v7 原门失败；v7.4 只从现存标注中保留严格多路共识子集，属于无人工复核的 post-hoc Silver |
| Dual Prior | 历史 48 条；另有 v12-posthoc 202 train +51 dev | 只允许已登记的探索性 R0/P0；原 v12/v13 仍不可训练 | v12 原 blocker 是 Complete 边界，v13 在 schema 门终止；后续明确授权的 post-hoc exact-consensus 子集能学会 Key/Complete，但未改善最终排序 |

失败批次不能直接混入可训练数据：v2 因 checker 与 H/P yield 失败，v3 因 Consistency 原始一致率和 Prior 裁决率失败，v4 提示词回放失败；v5 的 12 对新鲜 Consistency 只通过机械筛选流程审计，协议明确 `eligible_for_training=false`。H 的原始 v7 也仍是 `FAIL_H0_V7_RESERVE`；只有另行登记的 v7.4 严格共识子集获准做探索性训练，不能反过来宣称 v7 通过。Prior v8--v13 分别保留各自冻结失败状态；v12 是 `STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE`，v13 是 `FAIL_PRIOR_V13_SCHEMA`。后续用户另行授权并明确命名的 `v12-posthoc` 只是一条带 easy-sample bias 的探索路线，不修改原门、不重标、不降阈值，也不能写成 v12/v13 通过。

## 2026-08-23 clean integration 审计与训练试跑

本轮在 `clir-clean-integration` 的 `8b116c4` 上完成了代码、旧 artifact 兼容和真实训练审计，并修复了一个 CUDA 续训回归：full-state checkpoint 现在先加载到 CPU，再恢复 model/optimizer/RNG；若把整个 checkpoint 映射到 CUDA，CPU RNG state 会被搬到 GPU，`torch.set_rng_state` 会直接失败。修复后固定 SWIFT Python 环境的完整测试为 `37 passed`，CUDA 上的 interrupted→resume 回归也通过。

工程与数据门结果：

- toy generate→epoch 1→resume 到 epoch 2→score→evaluate 闭环完成；toy total loss 从 `4.5132` 降到 `4.2594`。这是随机数据的代码路径证据。
- clean reader 成功接入旧 3968-row train manifest（496 queries × 8 candidates）与 8000-row validation manifest（500 × 16），train/validation `query_id` 无交集；全部引用路径存在，代表性 trajectory/condition 为 BF16 `[221,101376]` / `[105,101376]`。
- 默认配置的真实模型有 `5,347,593` 个训练参数；全宽 consistency batch 和 hallucination/prior batch 的前向、反向与梯度均 finite，实测峰值显存约 `2.94 GiB`。
- seed 42、默认整合配置、3968-row train、query-disjoint 16-row mechanism dev 的 1 epoch 真实试跑完成：train total `.5374`，mechanism-dev total `3.6609`。checkpoint 位于 `run_artifacts/clean_integration_audit_20260823/real_full_seed42_epoch1.pt`，SHA-256 为 `e1dba08f91d6529213db1acadfc274a422a76c7c8d096a74da2576290c7c891f`。

同一 checkpoint 在 checker `clir_gsm8k_numeric_v4` 的 500×16 validation pool 上得到：

| K | reward BoN | random expected | oracle |
|---:|---:|---:|---:|
| 1 | `.884` | `.884` | `.884` |
| 2 | `.898` | `.892` | `.936` |
| 4 | `.910` | `.888` | `.956` |
| 8 | `.910` | `.89075` | `.970` |
| 16 | `.906` | `.8925` | `.976` |

query 内 correct-vs-wrong pairwise accuracy 为 `.6241`（5076 comparisons）。BoN@16 相对 random expected 的 paired query delta 为 `+.0135`，10,000 次 query bootstrap 95% 区间 `[-.0045,+.03175]`，跨 0。正确结论是：clean integration 已建立 `small-scale real pipeline pilot` 和弱排序信号；它只有 1 epoch/1 seed，且没有 matched clean correctness-only baseline，不能把结果归因给 consistency、hallucination 或 dual prior，也不能声称三模块联合有效。

## 2026-08-24 clean ablation v1

[`configs/clean_ablation_v1`](configs/clean_ablation_v1) 的 7-cell × 3-seed × 3-epoch
matched matrix 已全部完成：correctness-only、`+C`、H BCE only、H BCE+gold tail、
direct prior、direct+mutual prior 和 full 共 21 个 run。所有 run 使用同一 496-query
train、16-row mechanism dev 和 500×16 ranking population；checkpoint code/environment
provenance、候选 parity、checkpoint hash 与 scored-input hash 均通过。

BoN@16 三 seed 均值为：C0 `.9173`、C1 `.9220`、H0 `.9267`、H1 `.9187`、P0
`.9180`、P1 `.9180`、full `.9160`。相对 C0 的 paired delta 中，C1 为 `+.47`
points，H0 为 `+.93` points，P1 为 `+.07` points，full 为 `-.13` points；所有
query-bootstrap 区间都跨 0。H0→H1 的 gold-tail 增量在三个 seed 都回退，均值
`-.80` points；机制诊断同时显示 tail 没有形成 onset-localized value drop，而是把全局
token value 推到约 `-.62`。direct priors 在 16-row dev 上可学，但 mutual 没有额外
机制或 ranking 收益；consistency 因没有 held-out relation set 仍无法验证关系泛化。

3→5 epoch 的预注册扩展门未通过：多个 auxiliary cell 的 train loss 下降时 mechanism-dev
反而恶化，继续训练只会重复 27 个 consistency pair、17 个正 onset + 31 个 clean 和 48
条 prior trajectory。关键数字、置信区间和裁决已收敛在本 README 与
[`docs/handoff.md`](docs/handoff.md)；完整历史结果可从归档提交 `596a5e4` 恢复。这组结果仍是
`small-scale real screening`；不能声称三模块联合有效。

### CH0：consistency 与 onset BCE 的二因子交互补测

为避免把 H tail 和 dual prior 混入交互判断，本轮在查看新指标前冻结并运行了
`CH0 = correctness + consistency + onset BCE`，仍用同一 train/dev/ranking population、
3 epochs 和 seeds 42/43/44。CH0 的 BoN@16 为 `.9153 ± .0042`，逐 seed 为
`.920/.912/.914`；相对 C0、C1、H0 分别为 `-.20/-.67/-1.13` points。H0→CH0 三个
seed 都下降，fixed-seed query bootstrap 为 `[-2.20,-.13]` points；但只有三个训练 seed，
seed+query hierarchical interval 仍为 `[-2.73,+.47]` points，不能升级为正式稳定负效应。

二因子交互 `CH0 - C1 - H0 + C0` 为 `-1.60` points，三个 seed 都为负；当前数据支持的
筛选结论是 consistency 和 onset BCE **没有叠加，且有负交互信号**。这不等于二者天然
不兼容：consistency 只有 27 个正 pair、H 只有 17 个正 onset + 31 个 clean，机制 dev
只有 6+10 条，ranking 只有 500 queries。下一轮应扩数据后按同一 `C0/C1/H0/CH0` 2×2
矩阵复测，而不是直接把本轮写成最终论文结论；`best_current` 也不因这次筛选而更换。

三个模块的实现路径、最终分数耦合、历史数据生产流程、单模块效果和组合现象已合并到本
README 与 [`docs/handoff.md`](docs/handoff.md)。为保持远端分支精简，阶段报告、图解版和 PDF
不再放在当前分支顶端；需要追溯时可从提交 `596a5e4` 读取原文件。Gold/Silver/Pseudo、
clean/未标/pseudo onset、Key/Complete 和 gate `.25` 的含义与证据边界仍由 handoff 保留。

### Prior→reward gate 独立消融

[`configs/clean_gate_ablation_v1`](configs/clean_gate_ablation_v1) 已完成 P0 direct-prior 与
PG0 direct-prior+gate 的 2-cell × 3-seed × 3-epoch 对比。PG0 使用 `.0625`，等于
`origin/main` 的总有效系数 `.25×.25`；没有加入 mutual、C、H 或 Full 混杂项。

工程路径和 prior protection 均通过，但机制与 ranking 未通过：gate↔fused-prior
squared-L2 从 `.01195` 恶化到 `.01335`，只有 1/3 seed 改善；BoN@16 从 `.9180`
变为 `.9167`，paired delta `-.13` points，fixed-seed query interval
`[-.87,+.60]` points，seed+query interval `[-1.20,+1.00]` points。gate 虽使
`53%–76%` 的 query 更换最终候选，但三 seed 合计净少选对 2/1500 次。这仍是 `.0625`
这个单点的有效反证；关键历史结果保留在 [`docs/handoff.md`](docs/handoff.md)，完整旧文档可从
提交 `596a5e4` 恢复。

用户随后明确将“保留 `main` 原始 shared-gradient coupling 且默认开启”定为方法身份约束，
并允许在当前开发 population 上选择一个保守强度。为此在查看新结果前冻结
[`configs/clean_gate_tuning_v2`](configs/clean_gate_tuning_v2) 的 `.25/1/4/10` 网格，并
复用 `0/.0625` anchors，完成 6-weight × 3-seed × 3-epoch 严格配对评测。所有正权重通过
机制健康门；`10` 的 BoN@16 点估计最高 `.9207`，`.25` 为 `.9187`，差值恰好是冻结的
near-tie 边界 `.002`，因此按“近似时选更小权重”规则固定 `.25`。

`best_current` 和 `RewardConfig` 现已默认启用 `.25`。它保持 `origin/main` 的内部系数、
MSE、detach、mask 与 shared-gradient 路径，但 clean 的外层 `prior_weight=1`，所以总 loss
中的绝对 coupling 系数是 `.25`。`.25−P0` 的 BoN@16 只有 `+.07` point，两个配对区间都
跨 0；这是使用同一 dev 选出的 **dev-tuned engineering default**，不是 gate efficacy
结论。选择规则、结果摘要和证据边界保留在对应训练配置与
[`docs/handoff.md`](docs/handoff.md)。扩大数据后将
固定 `.25` 重做独立 `off/on` 诊断，而不在当前 dev 上继续调参。

## 2026-08-25 多题源扩量 smoke v2：双标完成，按预注册规则停止

v1 收到两份互盲外部 AI 审查后被共同判为 block，且从未执行，状态是
`superseded_before_execution`。v1 原文、审查提示词和逐项裁决已从当前分支顶端移出；最后一份
完整快照是提交 `596a5e4`。当前执行只认 v2，不应再从 v1 衍生实现。

修订后的可执行规格是
[`docs/data_expansion_smoke_protocol_v2.md`](docs/data_expansion_smoke_protocol_v2.md)，机器可读配置为
[`configs/data_expansion_smoke_v2/protocol.json`](configs/data_expansion_smoke_v2/protocol.json)。流水线、checker
v2、unitizer v2、C/H/P proposal、隐藏控制项、自一致性、第三模型先独立后匿名裁决和 fail-closed
finalizer 已实现；8-query/64-row 确定性 fixture 已通过。

真实 acquisition 与自然 proposal 冻结已经完成。本地按固定 revision 导出 `7473` 条 GSM8K train 和
`1218` 条 ASDiv-A，共 `8691` 条；从旧 outcome/ranking/mechanism population 汇总并规范化出 `1108` 个
不可复用的 GSM8K query ID。冻结检索器找到 29 对 near-duplicate candidate，其中 2 对的两端都已在历史
排除表内；实际送标的 27 对由用户报告的 OpenAI `gpt-5.5-sol`（xhigh）与 Anthropic
`claude-opus-5`（high）互盲判断。两者是不同、非 Phi 模型系列，target `27/27` 一致：`7` 对
duplicate、`20` 对 distinct，Cohen κ=`1.0`，无需第三模型。产品界面没有暴露精确 revision 与
temperature，因此本地 roster 明确记录为 unknown/unverified；这不能冒充预注册的 temperature=`0`。

去重决定随后冻结出 100 个永久 train-only query（60 GSM8K +40 ASDiv-A），与 1108 个历史排除项交集为
0；选中 query ID 集合 SHA-256 为 `08fc850d…df78b`。在一张 L20Z 上用固定 Phi-3.5 revision、
`vllm==0.5.3.post1`、TP=1、BF16、`max_num_seqs=32` 完成 100×8=800 条 rollout：800 条均为正常 stop，
没有空输出或长度截断，原始有序行 hash 为 `3b47bd39…101f`。

冻结的 v2 checker/unitizer 当时得到 680 条 numeric match、39 条明确 numeric mismatch、78 条
candidate-not-numeric mismatch 和 3 条 conflicting-boxed-answer ineligible；799/800 条通过 exact-token
unitization。由此冻结了 40 个 Consistency pairs（hash `6a97b2bf…9145`）与 60 条 H/P rows（hash
`36a2b380…80cb`）。A=`gpt-5.5-sol`/xhigh、B=`claude-opus-5`/high 已分别完成 C/H/P 六个隔离上下文，
所有文件通过 population、schema、index 与 A-self-repeat 校验。原始结果是：

| 任务 | 原始结果 | 预注册裁决 |
|---|---|---|
| Consistency | decision 36/40 一致，A/B accept 为 34/38，κ=.459；隐藏控制各 4/4，A 自重复 8/8 | 90% agreement 与 10% 分歧率达到工程门；因 B 仅 2 个 reject，κ 门按协议不适用 |
| Hallucination | path 60/60 一致；54 clean、仅 6 个共同 positive，onset exact 与 ±1 均为 5/6；隐藏控制各 6/6，A 自重复 12/12 | positive 支持数 6<15、±1=.833<.85，且不可能凑出最终 20 positive，失败 |
| Key/Complete | eligibility 60/60；Key F1=.928，Complete F1=.784；exact set 仅 27/60，33/60 需裁决；隐藏控制各 3/6，A 自重复 12/12 | control<100%、Complete F1<.82、裁决率 .55>.40，三项失败 |

H 的异常不是“Phi 几乎没有错误”这么简单。30 条冻结 numeric-mismatch proposal 中，24 条实际推理和
答案都正确，只是写成 `\boxed{38 cents}`、`\boxed{9 glasses}` 等“数字+单位/文字”；v2 checker 把整个
boxed 内容当纯数字，解析失败后误记为 0。两位盲标者都把这 24 条判为 clean，因而准确暴露了 checker
系统性假阴性。按协议，checker 改动必须升版、重建 proposal 并重做所有依赖标注，所以 v2 在此正式记为
`FAIL_PIPELINE`：不送第三模型粉饰原始门、不 finalize、不抽 hidden state，也不训练。

代码已加入协议钉住的 backward-compatible checker：v2 仍可复现旧结果；新的
`clir_numeric_multisource_v3` 会从 boxed prose 提取受支配数值，并覆盖单位短语、答案句、等式、金额和
复合时长回归。用 v3 对同一批 800 rollout 只读重算为 754 numeric match、42 parsed mismatch、1 个
non-numeric refusal 和 3 个冲突 box；可用于机制 proposal 的真实 mismatch 只覆盖 13 个 GSM8K query 和
1 个 ASDiv-A query。现有池因此仍不足以支持 20 个 query-distinct positive onset，下一版必须先扩大/加难
acquisition，而不是再次标同一份包。Prior 另暴露了定义边界：55/60 条 Key 完全一致；Complete 的 33 条
分歧中有 29 条是 B 取 A 的严格超集，说明“实际依赖链”与“可压缩的最短证明”需要在新版指南和控制题中
明确区分。

v2 把 train-only query 池扩为 `60 GSM8K +40 ASDiv-A`，每题 8 条 Phi rollout，共 800 raw rows；
自然标注预算仍只送 40 个 Consistency proposals 和同一批 60 个 H/P proposals。所有 proposal 的机械
过滤、每 query 上限、source/numeric strata、hash tie-break、固定分母与 joint H/P 入选顺序都必须在
A/B 看标签前发布。最终目标仍是 30 个 C accepts，以及同一批 prior-usable 的
`20 first-bad-unit positive +20 explicit clean`。

两个主标注者现在必须来自不同模型系列，且都不得与 Phi generator/backbone 同系列；分歧只能交给第三个
不同系列模型先独立判断再匿名盲裁，没有合格第三模型就丢弃分歧行。unitizer v2 必须在保存的
`output_token_ids` 上给出连续、无重叠、完整覆盖的半开 token ranges；H 的兼容 onset 字段只表示
`first-bad-unit start token`，不再冒充精确首错 token。所有自然比例都以预先 hash 的 proposal manifest
为分母，格式失败、low 或 uncertain 不能静默移除。

SVAMP 原论文由 100 个 ASDiv-A seed 生成变化题，因此 v2 不再把它称为与 ASDiv 训练来源独立的外部
holdout。它继续保持不训练、不调参的 protected 状态，但准确角色是 **ASDiv-derived contrast/challenge
set**；独立来源泛化还需另选 holdout，或以后做密封 seed-family 排除。

无人类复核的最终标签统一叫 `silver_dual_ai_v2`，不能叫 Gold。smoke 通过只证明多题源生成、
numeric checker、dual-AI Silver、盲裁和 exact-token materialization 流水线达到预注册门，不证明三模块
提高 Best-of-N。当前 `.25` gate 继续作为默认方法身份保留，但本轮不训练、不调参。

离线验收入口：

```bash
python prepare_clir_smoke.py fixture \
  --output-dir run_artifacts/smoke_v2_fixture
```

正式执行的完整逐步命令及 A/B/第三模型分发顺序在
[`docs/data_expansion_smoke_protocol_v2.md`](docs/data_expansion_smoke_protocol_v2.md#16-2026-08-25-实现状态与唯一执行入口)；
可复制提示词在
[`configs/data_expansion_smoke_v2/annotation_prompts.md`](configs/data_expansion_smoke_v2/annotation_prompts.md)。
v2 不再继续裁决或训练；它只保留为失败审计与 backward-compatible checker 回归。v3 acquisition 已按
下面的新协议完成，旧标签不跨 manifest 搬运。

## 2026-08-25 多题源扩量 smoke v3：H 通过，C/Prior raw gate 停止

canonical 规格是
[`docs/data_expansion_smoke_protocol_v3.md`](docs/data_expansion_smoke_protocol_v3.md)，机器契约与标注语义在
[`configs/data_expansion_smoke_v3`](configs/data_expansion_smoke_v3)。v3 完整保留 v2 的 60 GSM8K +40
ASDiv-A train-only incumbents，并从 MATH train 的 algebra、counting/probability、number theory、
prealgebra 中按 level 3/4/5 分层预选 60 个 primary +48 个 reserve。只纳入 final boxed answer 能由同一
checker 唯一解析为标量数值、官方解至少 45 词且不含 Asymptote 的题。production loader 只请求 train
parquet，MATH test 不用于读取、选样或调参；冻结前一次开发性 loader 预检曾把 test 写入本机 cache，
但没有查看 test row/答案/表现，因此不能声称该工作区从未下载过 test。

primary 新增 60 题×8=480 条 Phi rollout，与绑定 hash 的旧 800 条合并为 160 题、1,280 条。v3 checker
得到 995 match、240 个明确 parsed mismatch、10 个冲突多答案、14 个 parse failure、21 个长度截断；
1,280/1,280 exact-token partition 通过。截断占 1.64%，低于冻结 2% 上限且全部排除。可用于机制提议的
query-distinct parsed mismatch 有 57 个，超过预注册最低 30；40 个 C 与 60 个 H/P 固定 strata 也都能
一次凑齐，因此状态为 `READY_FOR_FROZEN_V3_PROPOSAL_AND_ANNOTATION`，48 题 reserve 没有生成。

H/P 的 60 条固定为 GSM8K match/mismatch `10/8`、ASDiv-A match `12`、MATH match/mismatch `8/22`；
parse failure 不得冒充错误富集。Complete 现明确指候选实际依赖链里所有被后续使用的唯一非冗余中间
步骤，不得压缩成另一条更短证明。盲包已生成：A=`52/78/78`，B=`44/66/66`（C/H/P，含隐藏控制和
A-only self-repeat）。六份 A/B 标签已全部通过 population/schema/index 校验；每个任务两边的 hidden
controls 都是 100%，A 的 self-repeat 也都是 100%，说明输出不是随机或格式性失败。

raw triage 随后按冻结门停止。C 只有 `26/40=.65` 决策一致，低于 `.90`；至少 `14/40=.35` 必须裁决，
高于 `.20`。A/B accept 数为 `25/39`，且 B 只有 1 个 reject，表明两者对“近乎照抄”和“关键中间量是否
改变”的边界并不一致。H 是这一轮唯一完整通过所有 raw 门的任务：path agreement=`59/60=.9833`，
共同 positive=30，五个以上 material units 上 exact/±1 onset 都是 `26/30=.8667`，最低需裁决
`5/60=.0833`。Prior 的 eligibility=`60/60`、Key F1=`.9167`、Complete F1=`.9267` 都通过，且没有
Complete=全部 material units 的退化；但 exact target 仅 `25/60`，最低需裁决 `35/60=.5833`，超过 `.40`。

因此终态是 `STOP_RAW_GATE_FAILURE`，`third_model_send_allowed=false`。第三模型包虽由旧 triage 路径机械
生成在本地，但不得发送；本轮不 finalize、不抽 hidden state、不训练，也不产生模块 efficacy 或 Best-of-N
结论。代码现会在 triage 时显式执行这个停止规则，避免把裁决误当成能救活 raw failure。

## Consistency 提示词修复 v4：14 条开发回放未通过

v2/v3 有 36 个完全相同的 C pair；B 的 36 个 decision 全部复现，A 却把其中 9 个从 accept 改为 reject，
8 个理由是“近乎照抄”。因此先按
[`docs/data_expansion_smoke_protocol_v4.md`](docs/data_expansion_smoke_protocol_v4.md) 做一个不改 schema、
不加相似度算法的提示词修复：先判断数学路径，再判断表达差异，并明确相同公式、数字和必然顺序不能单独
作为照抄证据。

两个新上下文已经完成回放，冻结检查器得到 `STOP_REPLAY_FAILURE`：A/B 只在 `7/14=.50` 条上同意，
低于 `13/14` 门，decision kappa 为 `-.0426`。两边都没有使用 review，A 的 accept/reject 为 `8/6`，
B 为 `9/5`，理由前缀与反塌缩门全部通过；因此失败不是格式问题或全部选同一类，而是判尺仍不一致。
七个分歧中，四个是“真实展开还是近似照抄”的边界，两个是单位表示/乘法分组是否仍属同一路径，
一个是是否应因夹带的实质性错误而拒绝。30 条全新确认不再允许，提示词-only 路线按预注册规则停止；
这些回放仍不训练、不算可靠性证据，也不能由第三模型裁决救活。

## Consistency 机械筛选 v5：新鲜双盲审计通过

v4 说明自然语言提示词无法稳定同时处理“数学路径相同”和“表达差异足够大”两个边界。v5 因此按
[`docs/data_expansion_smoke_protocol_v5.md`](docs/data_expansion_smoke_protocol_v5.md) 把工作拆开：固定程序先检查
最终数值、数学/数字序列、长度和非数学文字重合区间；两位 AI 只审查两条解答有没有实质性的算术、代数、
单位、实体或内部矛盾错误。14 条已检查的 v4 争议只用于开发回归：程序仅放行 3 条，且三条都是 v4 的
A/B 共同 accept；它们不进入 v5 自然样本，也不算准确率证据。

规则先在干净提交 `a60b2cb` 冻结，再使用 v3 从未 rollout、从未标注的 48 个 MATH-train reserve queries。
Phi-3.5-mini-instruct 生成 `48×8=384` 条候选，provenance 记录 `code_dirty=false`；384/384 通过 exact-token
unitization。checker 得到 201 numeric matches、154 parsed mismatches、10 parse failures、16 truncations 和
3 ambiguous-multiple-answer rows，后 29 条不会被冒充正确候选。机械规则在 38 个具有正确候选的 query 中
找到 16 个 query-distinct pairs，超过冻结目标 12；程序只按预先固定的 SHA-256 顺序选择前 12 个，没有
人工看答案挑题或事后改阈值。

GPT-5.5-sol/xhigh 与 Claude Opus 5/high 在两个隔离上下文中完成了 A/B 包。冻结检查器终态为
`PASS_FRESH_MECHANICAL_AUDIT`，没有失败门：12/12 natural decisions 一致且均为 accept，两边 review=0、
hidden controls 各 4/4，A self-repeats=3/3，理由前缀和 schema 全部合法。A/B label SHA-256 分别为
`7004f129…b149` / `185f0b15…ca1`，审计报告 SHA-256 为 `e88d389b…196f`。

自然集本来就是机械预筛后的正 pair，所以 12/12 全 accept 符合任务结构；它也意味着自然集只有一个类别，
报告中的 κ=1 不应被单独解释为强统计证据。两个明确错误的 hidden controls 均被正确 reject，说明两边并非
无条件 accept。v5 因此只证明这条“机械筛选 + 双 AI 事实审计”流水线在新鲜样本上可操作，并按预注册规则
允许另写正式 Consistency 扩量协议。v5 自身仍 `eligible_for_training=false`、禁止第三模型补救，不抽 hidden
state、不训练，也不产生 Consistency efficacy 或 Best-of-N 结论。

## 数据扩容主协议 v6/v6.1：关系与 selected feature 已通过核验

用户已确认先完成正式扩容准备，不直接继续旧 Full 或在旧小数据上增加 epoch。新的
[`docs/data_expansion_scale_protocol_v6.md`](docs/data_expansion_scale_protocol_v6.md) 与
[`configs/data_expansion_scale_v6/protocol.json`](configs/data_expansion_scale_v6/protocol.json) 固定了中档
Consistency 扩量方案：2,000 个全新 train-only query（1,400 MATH train +600 长链 GSM8K train），每题
8 个 Phi 候选，共约 16,000 条 raw rollout；query/template cluster 在生成前拆成 1,500 个
train-acquisition 和 500 个 heldout-acquisition。目标是 400 个训练正关系、150 个 query/cluster-disjoint
held-out 正关系，再配 150 个确定性 hard negatives 检查表示塌缩。

v6 原样保留 v5 的 numeric/path/surface 机械阈值，不允许看完新 rollout 后调松；两个不同、非 Phi 的 AI
只审查实质错误，最多 50 条自然 pair 一包，并混入隐藏控制和跨包自重复。只有 A/B 都 accept 的 pair 才能
按预先冻结的 hash 顺序入选；自然 agreement<95%、任一控制失败、自重复<95% 或 common accepts 不足
400/150 都直接停止，不由第三模型补救。没有人工复核，因此未来标签只叫
`silver_dual_ai_consistency_v6`，不能叫 Gold。

存储顺序也已固定：16,000 条只先保存文本和 token IDs；原 v6 正关系部分是550 对、1,100 条 view，v6.1
再按 hard-negative 实际端点做 query 级去重并发布精确 inventory。原先按420 output token 估计约105.23 GB，
不能继续把它当成实际占用。若先给全部 rollout 抽
`33×3072` BF16 feature，预计约 1.42 TB，协议明确禁止。H 仍需全新 `30 positive +30 clean` 的 H-only
确认；当时冻结的 Prior 建议是先标 dependency edges，但该路线后来在 v8 raw gate 失败，当前改为 v9
direct-set 局部共识 smoke；独立 1,500–2,000×16 ranking pool 另行冻结预算。

基础协议仍准确写作 `FROZEN_PREPARATION_ROLLOUT_NOT_STARTED`：它本身不授权任何生成、标注、hidden
state 或训练。2026-08-27 又在生成前冻结了长链 GSM 统计、实体/数字模板、
MinHash/Jaccard 阈值、历史排除传播、分层配额、prompt 上限与每 shard `35 MATH +15 GSM8K` 等执行细节。
来源 inventory、永久排除表、近重复 cluster、1,500/500 query split、40 个原子 shard 和实际预算 hash 已
全部通过。2026-08-28 用户明确回复“开始”，单独的
[`rollout_authorization.json`](configs/data_expansion_scale_v6/rollout_authorization.json) 因此只解锁 rollout；
checker/unitizer materialization、标注、抽特征和训练仍关闭。入口
[`prepare_clir_scale.py`](prepare_clir_scale.py) 现支持逐 shard 原子生成、完成标记、复核和全量合并；先跑
`train-000` 校准，机械运行门通过后最多 8 张 L20Z 并行完成其余 shard。任何已有完整 shard 只复核跳过，
不完整 artifact 默认停止且不覆盖。

该授权现已执行完毕：40/40 shard、2,000/2,000 query、16,000/16,000 raw rows 通过独立复核与全量合并，
终态为 `PASS_ALL_16000_RAW_ROLLOUTS_VERIFIED_V6`。合并文件 SHA-256 是
`f538373b…5139`，有 15,393 条自然停止，607 条因达到 1,024-token 上限而以 `length` 结束（3.79%）；后者
必须在材料化阶段排除。1,999/2,000 个 query 至少有两条未截断输出，剩余 1 个 query 不能组成正 pair，按
冻结规则排除，不能据此放松阈值。MATH 截断率为 5.35%，GSM8K 为 0.17%；train/heldout acquisition
分别为 3.78%/3.85%，没有异常 split 偏移。vLLM 返回的展示文本统一比 token-ID 解码多一个前导空格，
16,000 条去掉首尾空白后全部相等；保存的 `output_token_ids` 仍是唯一权威，没有 token 漂移。

raw 文件和完成报告只保存在被 Git 忽略的 `run_artifacts/data_expansion_scale_v6/rollouts/`，不会推到远端。
这只表示生成阶段完成，不表示已经得到 400 个可训练 Consistency pair：checker、
`clir_material_claim_unitizer_v2`、机械筛选、双 AI 审计、feature 和训练均未启动。

用户随后明确要求“接着做，到要标数据再叫我”。新的
[`pre_annotation_authorization.json`](configs/data_expansion_scale_v6/pre_annotation_authorization.json) 只解锁
CPU checker/unitizer 材料化、原样 v5 机械筛选、自然候选冻结和 A/B 盲标包构造，并在真正调用 AI 标注前强制
停止；它仍禁止 provider call、标签裁决/发布、抽 feature、训练、重跑 raw rollout 或修改阈值。每个标注
shard 最多 50 个自然 pair，另混 2 accept +2 reject 控制；A/B 各约 10% 自重复只放到后续 shard。启动词
要求一个全新上下文只处理一个 shard，避免同一上下文记住重复项。执行前先把实现、启动词和 raw gate
评估器冻结在干净提交上，然后才运行下述标注前阶段。

标注前阶段现已在 clean commit `5ab1fb7` 上执行并经独立重算通过，终态为
`PASS_PRE_ANNOTATION_PACKAGES_VERIFIED_V6`：16,000/16,000 exact-token unitization 成功，其中 15,927 条走
canonical fast offsets，73 条走不改写 frozen IDs 的 prefix fallback。checker 得到 8,877 numeric match、
5,907 numeric mismatch、607 truncated、350 parse failed 和259 conflicting answers；后 1,216 条 audit-only
row 全部不可监督。原样 v5 机械阈值最终留下 708 个 query-distinct pair：train 526、heldout 182，MATH
348、GSM8K 360，超过标注前最低 400/150，未调阈值。

708 对现已冻结为 15 个 shard。每边总计 839 行 =708 natural +60 controls +71 later-shard repeats；两个公共
索引和全部 1,678 个 A/B package item 已复核；在该 pre-annotation 报告的冻结时点，标签目录尚不存在且
`annotation_started=false`。要最终通过，
两个 AI 共同接受率至少需达到 train `400/526=76.0%`、heldout `150/182=82.4%`，同时自然 decision agreement
≥95%、每边 review≤2%、60/60 控制正确、自重复≥95%。这些 raw gate 的纯机械评估器也在零标签状态下实现，
不会由第三模型补救。首个 A/B shard 已经存在后，用户澄清 A 的实际模型是 GPT-5.6-sol/xhigh，而不是
基础索引文字中的 5.5；公开的
[`annotation_model_amendment_v1.json`](configs/data_expansion_scale_v6/annotation_model_amendment_v1.json) 如实绑定了当时两个
`shard-000` 的 hash，并把模型证据标为 user-reported、精确 revision/temperature 未验证。708 对、包顺序、
控制/重复项和 raw gate 均未改变。A/B 现已分别完成 15 个 shard、839 行；使用的启动词是
[`launch_prompt_a_5_6.txt`](configs/data_expansion_scale_v6/launch_prompt_a_5_6.txt) /
[`launch_prompt_b.txt`](configs/data_expansion_scale_v6/launch_prompt_b.txt)。全量 schema、agreement、controls、
self-repeat 和 common-accept gate 已由 commit `657d471` 上的冻结 evaluator 执行；raw report SHA-256 是
`4c80626e…a598`，终态 `PASS_SCALE_V6_RAW_ANNOTATION_GATES`：自然 decision agreement 为
`676/708=95.48%`，A/B controls 都是 `60/60`，A self-repeat `71/71`、B `69/71=97.18%`，A/B review
分别为 `7/708=.99%` 与 `0`。双方非 low 共同 accept 为 train `474`、heldout `167`，均超过 `400/150`。

按冻结顺序取前 N 个时，400 个 train 正关系和 150 个 heldout 正关系都能形成，query 跨 split overlap=0、
template-cluster 跨 split overlap=0；对应有序行 hash 为 `079d014b…f9d2` / `704a8317…73b6`。但原协议还要求
用这 300 个 heldout 正视图配出 150 个 different-answer/similar-surface hard negatives。保持 response-surface
Jaccard `[.10,.40]`、长度 `[.8,1.25]` 和所有分离条件不变，只得到 10 条 eligible edge，greedy 不复用视图
只能选 8/150。因此 post plan 终态是 `STOP_SCALE_V6_POST_ANNOTATION_PLAN`，报告 SHA-256
`42052c40…bb0`；没有 relation manifest 被发布，feature extraction 与训练仍为 false。

随后用户明确同意一个只改 hard-negative 来源池、匹配器和存储重算的
[`hard_negative_amendment_v6_1.json`](configs/data_expansion_scale_v6/hard_negative_amendment_v6_1.json)
（SHA-256 `8cbcf62b…20b3`）。它如实标为看过 v6 失败和可行性诊断后的工程修订，不是盲预注册；A/B 标签、raw
gate/分母、400/150 正关系顺序、Jaccard `[.10,.40]`、长度 `[.8,1.25]` 和 query/cluster/answer 分离条件都没改。
唯一数据改动是从所有现有 heldout `numeric_match + eligible_for_supervision` 视图取负例端点，唯一算法改动是
固定 NetworkX 3.6.1 的 maximum-cardinality/preference matching。

正式 plan 在 clean commit `bb99261` 上通过：端点池2,178 条，shared-bigram pair 331,122 条，同阈值 eligible
edge 605 条，maximum matching 167，最终不复用300 个端点选满150 条 hard negatives。400 train positives、
150 heldout positives 和150 heldout negatives 均已发布到 Git 忽略的 `post_annotation_v6_1/`；plan report
SHA-256 为 `71a470e6…bb75`。第二次从父级标签、proposal 和 materialized rows 独立重算逐行一致，终态
`PASS_SCALE_V6_1_INDEPENDENT_RECOMPUTATION`，verification report SHA-256 为 `48d69371…2756`。

精确 inventory 是1,357 个 trajectory、612 个 query prompt，共520,103 个 feature token，按 `101376×BF16`
为98.210 GiB（105.452 GB）。其中 hard negatives 与正关系重叠43 个 view，真正新增257 个 trajectory 和62 个
prompt；早先“101 个新 prompt”的只读估算没有扣掉44 个已随正关系保存的 query prompt，现已纠正。这个 PASS
只说明 Consistency 扩量关系构造闭环，不证明模块提高 Best-of-N。关系报告在其冻结时点正确保持
`feature_extraction_allowed=false`、`training_allowed=false`；后续特征抽取必须另行授权且只能消费 inventory，
不能抽16,000 条全池。

用户随后授权只推进这份 inventory 的 exact feature extraction。机器授权
[`feature_extraction_authorization_v6_1.json`](configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json)
与人读协议 [`feature_extraction_protocol_v6_1.md`](docs/feature_extraction_protocol_v6_1.md) 把范围固定为：1,357 条
trajectory、612 份 condition、Phi 固定 revision、33×3072 全层 BF16、8 个 query-balanced GPU worker，再由8个
CPU verifier 逐文件复核 shape/dtype/finiteness/SHA。新入口 `extract_clir_scale_features.py` 支持原子 tensor、
query completion marker 和安全续跑；完整16,000条抽取、重生成、改标签/关系/阈值和训练仍禁止。

该授权现已完整执行。正式 run3 绑定 code commit `64470c8`，plan、最长样本全宽 preflight、8个 GPU writer、
8个 CPU independent verifier 和 finalize 全部通过，终态
`PASS_SELECTED_FEATURE_EXTRACTION_AND_VERIFICATION_V6_1`。1,357个 trajectory 与612个 condition 共1,969个
BF16 payload 全部逐文件重读，shape/contiguous/finiteness/SHA 与 writer digest 一致；raw/serialized bytes 分别为
`105451923456` / `105455351485`。最终报告 hash 为 `a1ce2d9b…8daa`，extracted manifest hash 为
`ac1f35ff…7a8b`，训练读取器也已只读加载普通样本和980-token最长样本。大 artifact 继续 Git ignore；远端只保存
代码、授权、完成摘要和关键文档。这个 PASS 仍只是 exact feature pipeline 证据，`training_allowed=false`；
下一门是单独冻结并授权 C-only 训练与 heldout 正/负关系机制评估。

该下一门随后已由用户单独授权并完成。C0/C1 使用同一 4,768-row 训练清单、seeds
42/43/44 和3 epochs，只有 `consistency_weight=0/1` 不同；150 个正关系与150 个
hard negative 始终只用于 held-out 评估。冻结主指标“正对平均 cosine−负对平均
cosine”的 C1−C0 增量为 `+0.1760`，三个 seed 都为正，relation bootstrap 95%
区间 `[+0.1389,+0.2155]`。但正对 cosine 从 `.99991` 降到 `.98335`，cosine
AUROC 的 seed 方向混合，说明主要学到的是 hard-negative 分离和去塌缩，不是干净的正对
不变性。完整协议与结果见
[`docs/consistency_training_protocol_v6_1.md`](docs/consistency_training_protocol_v6_1.md)，
机器摘要见
[`completion.json`](configs/data_expansion_scale_v6/consistency_training_v6_1/completion.json)。
本轮没有新的 ranking population，不能据此声称 Best-of-N 提升。

[`post_annotation_authorization.json`](configs/data_expansion_scale_v6/post_annotation_authorization.json)（SHA-256
`7dcd096a…ab34`）在其阶段只解锁 deterministic audit、条件式正关系选择和 frozen hard-negative feasibility；
它没有授权 provider、第三模型、改标签/阈值、feature extraction 或训练。后来的 selected feature 完成状态只来自
上面的独立 feature authorization；再后来的 C0/C1 训练只来自独立的 C-only authorization，不能反向扩大
post-annotation 授权范围。

## 目录

```text
configs/best_current.json             唯一默认模型与训练配置
configs/clean_ablation_v1/            可复现的 C/H/P/full 消融训练配置
configs/clean_gate_ablation_v1/       P0→PG0 prior-to-gate 冻结消融
configs/clean_gate_tuning_v2/         gate 工程默认值选择训练配置
configs/data_expansion_smoke_v2/      双审查后多题源扩量 smoke 的机器可读协议
configs/data_expansion_smoke_v3/      当前 MATH 扩量 smoke 的机器协议与标注语义
configs/data_expansion_smoke_v4/      Consistency 提示词修复、14-ID 回放清单与机器门
configs/data_expansion_smoke_v5/      Consistency 机械筛选与双 AI 事实审计协议/启动提示词
configs/data_expansion_scale_v6/      正式扩容：v6/v6.1 授权、修订与 selected-feature 完成摘要
  consistency_training_v6_1/          C0/C1 配置、独立训练授权与可提交结果摘要
prepare_clir_smoke.py                  v2/v3 数据管线与 v4/v5 Consistency gate 入口
prepare_clir_scale.py                  v6 rollout/材料化/标注审计及 v6.1 关系规划与独立核验
src/clir_smoke.py                      checker/unitizer/proposal/label 核心契约
src/clir_scale.py                      v6 来源过滤、历史排除、模板分簇、split/shard/预算契约
src/clir_scale_post_annotation.py      v6/v6.1 raw gate 后的正负关系与精确 inventory 契约
src/clir_scale_pre_annotation.py       v6 exact-token 材料化、机械 pair 和隔离盲标包契约
src/clir_features.py                  identity/layer-axis 特征编码器
src/clir_data.py                      JSONL 数据、严格 token 对齐、collate、sampler
src/consistency_localized_reward.py   reward model、三模块和 loss
extract_hidden_states.py              exact-ID teacher-forced 全层特征抽取
extract_clir_scale_features.py        v6.1 inventory-only 分片抽取、续跑与独立逐文件核验
prepare_clir_consistency_training.py  v6.1 C0/C1 训练视图、独立重算与全宽预检
train_clir.py                         训练、验证、原子 checkpoint、精确续训
score_clir.py                         打分、定位诊断、Best-of-N 选择
evaluate_clir.py                      query-level Best-of-N 与 pairwise 评估
evaluate_clir_mechanisms.py           H/onset/value 与 dual-prior 机制诊断
evaluate_clir_consistency.py          冻结正负 relation 的 representation/score-gap 评估
summarize_clir_ablation.py            多 seed 候选 parity 与 paired contrast
summarize_clir_consistency.py         C0/C1 多 seed relation 配对与分层 bootstrap
examples/create_toy_clir_data.py      仅供管线 smoke test 的合成数据
tests/                                模型、数据、续训与评估测试
docs/proposal.md                      与当前实现一致的方法说明
docs/handoff.md                       迁移裁决、历史证据和下一步
docs/data_expansion_smoke_protocol_v2.md  双审查整合后的 100-query smoke 协议
docs/data_expansion_smoke_protocol_v3.md  当前 160-query primary acquisition 与双标协议
docs/data_expansion_smoke_protocol_v4.md  Consistency 提示词修复与新样本确认门
docs/data_expansion_smoke_protocol_v5.md  Consistency 机械筛选与新鲜双盲事实审计
docs/data_expansion_scale_protocol_v6.md  当前数据扩容主协议；v6.1 关系与 selected feature 均完成
docs/feature_extraction_protocol_v6_1.md  v6.1 selected inventory 精确特征抽取协议
docs/consistency_training_protocol_v6_1.md v6.1 C0/C1 训练、结果与证据边界
```

当前分支顶端只保留训练/打分/评测代码、测试、可运行配置、README、handoff、核心方法说明和
当前关键扩量协议。历史报告与被 supersede 的协议仍可从 Git 历史恢复，但不再占用当前
远端目录视图。

## 环境

```bash
pip install -r requirements.txt
```

`torch`、`numpy` 和 `pytest` 是训练与测试依赖；`transformers`、`huggingface_hub` 与 `pyarrow` 用于
exact-ID materialization、MATH train parquet 和抽取。v2/v3/v5 source export 固定 `datasets==3.6.0`，rollout 固定
`vllm==0.5.3.post1` 与 `numpy==1.26.4`；训练和打分本身仍不会导入 task LLM 或 vLLM。

## 唯一默认配置

`train_clir.py` 默认读取 `configs/best_current.json`。loss 权重只在这个 JSON 中定义，CLI 仅允许覆盖 epoch、batch size、学习率、设备等运行参数，避免形成第二套隐含方法。

当前真实数据配置为：

- 输入是 Phi-3.5-mini 风格的 embedding + 32 blocks，共 33 层；每层宽 3072，原始 token 特征宽度为 `33 × 3072 = 101376`。
- layer-axis encoder 使用 2 层 Transformer、宽度 256、8 heads 和 4 个 learned pooling queries，把每个 token 压到 `model_dim=768`。
- condition attention 再通过独立的 256 维瓶颈计算，避免在原始全层宽度上建立二次参数层。
- 训练默认 5 epochs、batch size 4、learning rate `1e-4`、BF16、seed 42、gradient clipping 1.0。
- dual-prior 使用 `joint` phase；semantic grouping sampler 默认开启。

`--hidden_dim` 是 toy/开发用覆盖项：指定后会把 encoder 切为 identity。真实全层训练应直接使用默认配置，不要传这个参数。

## 三模块的默认开关

模型始终输出各诊断 head，但只有同时满足“权重非零”和“batch 中存在对应监督”时才产生训练 loss。缺失标签通过 mask 跳过，不会被填成负样本。

| 模块/分量 | 默认权重 | 当前行为 |
|---|---:|---|
| final correctness BCE | `1.0` | 训练 scalar reward |
| consistency | `1.0` | 同 semantic、异 style 拉近；异 semantic、同 style 分离；score consistency 为 `0.1` |
| hallucination onset BCE | `1.0` | 使用 `main` 的 onset→tail 二值目标 |
| token reward target | `0.5` | onset 后 target 改为 `-0.5` |
| negative-tail margin | `0.5` | onset 后要求 `token_values <= -0.5` |
| path-level MIL | `0.0` | 保留稳定 log-space 实现，默认关闭 |
| pseudo-onset tail | `0.0` | 保留实现，默认关闭 |
| progress loss | `0.0` | head 仍输出；且 `progress_score_weight=0.0`，不进入 scalar score |
| key / complete direct prior | `1.0 / 1.0` | 有对应 token target 时启用 |
| bidirectional prior distillation | `0.25` | 双向 stop-gradient mutual MSE |
| gate-prior alignment | `0.25` | 默认开启；保持 `main` 原始 shared-gradient 公式，当前是 dev-tuned 工程值 |
| complete reconstruction | `0.0` | 只接受外部 target，默认关闭 |

当前 hallucination 默认不是 `panzhixin` 的 sparse-span diagnostic：它回到 `main` 的定义——若 `hallucination_onset=k`，则第 `k` 个生成 token 起都属于受污染 tail，H head 做 token BCE，reward value path 同时受到负 tail 约束。这是方法身份选择，不是已经建立的效果结论。

## 数据格式

每个 JSONL row 表示同一个 query 下的一条候选 trajectory。建议所有真实数据都保留 exact token IDs，即使训练阶段实际读取的是预抽取 feature。

### 抽取前的最小 row

```json
{
  "id": "q000-cand00",
  "query_id": "q000",
  "candidate_index": 0,
  "prompt_token_ids": [1, 2, 3],
  "output_token_ids": [4, 5, 6],
  "correctness": 1
}
```

- `id`：trajectory 唯一标识。
- `query_id`：必填；Best-of-N 分组和 train/validation query-disjoint 检查使用，不要拿 `semantic_id` 代替。
- `candidate_index`：可选；评估时决定冻结的候选前缀顺序。也支持 `completion_index` 或 `vllm_completion_output_index`。
- `prompt_token_ids`、`output_token_ids`：生成时保存的原始 token IDs。两个序列都必须非空、非负，且元素必须是真正的整数；字符串、浮点数和布尔值即使可强转也会被拒绝。抽取脚本不从 response 文本重新 tokenize。
- `correctness`：建议为数值 `0/1`。允许部分 row 缺失，此时 final BCE 对该 row 跳过。

### 抽取后增加的字段

`extract_hidden_states.py` 会原样保留输入 row，并增加：

```json
{
  "hidden_states_path": "features/00000000.hidden.pt",
  "condition_states_path": "features/condition-<prompt-sha>.pt",
  "hidden_states_sha256": "...",
  "condition_states_sha256": "...",
  "feature_dim": 101376,
  "num_feature_layers": 33,
  "per_layer_dim": 3072,
  "feature_model": "your/model-or-local-path",
  "feature_revision": "pinned-or-resolved-commit",
  "feature_dtype": "bfloat16"
}
```

路径相对于输出 JSONL 所在目录。也可以用 inline `hidden_states` / `condition_states`，或 `.pt`、`.pth`、`.npy`、`.json` feature 文件。数据/续训兼容层同时支持 `panzhixin` 旧 manifest 的嵌套 `feature_metadata` 维度字段，以及 `feature_sha256` / `condition_sha256` checksum 别名；新旧 metadata 同时存在时必须一致。

### 可选监督

| 字段 | 形状/语义 |
|---|---|
| `semantic_id` | consistency 语义组 |
| `style_id` | consistency style/domain 属性 |
| `hallucination_onset` | 生成 token 的首错索引；`-1` 表示已知 clean；字段缺失表示未标注 |
| `path_hallucinated` | path-level `0/1`，只供默认关闭的 MIL/pseudo-tail 使用 |
| `token_advantage` | 长度严格等于输出 token 数；token value 的外部 target |
| `progress_targets` | 等长 token target；当前默认关闭 |
| `key_prior_target` | 等长的 key evidence `0/1` target |
| `complete_prior_target` | 等长的 complete support `0/1` target |
| `key_prior_mask` | 可选的等长 `0/1` coverage；`0` 表示该 token 的 Key 标签未知，不计 loss |
| `complete_prior_mask` | 可选的等长 `0/1` coverage；`0` 表示该 token 的 Complete 标签有分歧，不计 loss |
| `complete_reconstruction_target` | 外部生成的固定宽度向量；当前默认关闭 |

`token_advantage`、`progress_targets`、Prior target 及其显式 mask 必须与 trajectory feature 的 token 长度完全一致；不一致直接报错，不做截断或补零。Prior mask 不能脱离对应 target 单独出现；未提供显式 Prior mask 时保持历史行为，即该 target 的全部有效 token 都有 coverage。`correctness`、onset 和各 auxiliary target 都有独立 mask，因此一份 manifest 可以混合不同监督覆盖的 row。Key/Complete attention 仍在完整有效 trajectory 上归一化，coverage 只控制 loss，不对子集重新 softmax。

## Exact-ID 全层特征抽取

输入必须已经包含 `query_id`、`prompt_token_ids` 和 `output_token_ids`。脚本对后两者拼接后做一次无 padding、teacher-forced causal forward，读取模型返回的全部 hidden-state layers；随后按保存的 prompt 长度精确切分：

```text
condition_states = all_layer_states[:len(prompt_token_ids)]
hidden_states    = all_layer_states[len(prompt_token_ids):]
```

整个过程不 decode response，也不调用 tokenizer 重建 output IDs，因此 token target、onset 和 hidden-state 位置使用同一身份坐标。

```bash
python extract_hidden_states.py \
  --input_jsonl data/rollouts.jsonl \
  --output_jsonl data/extracted.jsonl \
  --feature_dir data/features \
  --model your/model-or-local-path \
  --revision pinned-model-commit \
  --dtype bfloat16 \
  --expected_num_feature_layers 33 \
  --expected_per_layer_dim 3072
```

同一 `query_id` 的所有候选必须使用完全相同的 prompt IDs；相同 prompt 序列只保存一份 condition tensor，各候选复用同一相对路径和 checksum。`--revision` 会传给模型加载器，manifest 记录 `feature_model`、解析到的 commit/所给 revision、dtype 和两个 feature 文件的 SHA-256。

每个 tensor 和最终 manifest 都原子发布；脚本还会拒绝不等长 token target、非法 onset 和跨 row 不一致的 layer contract。若输出 manifest 或 feature 已存在，默认直接失败；`--overwrite` 只应用于明确可整体替换的重跑。正式数据优先写入新目录，避免中途失败后在旧 manifest 下留下部分已替换的 feature。脚本不负责 rollout 生成、correctness 判断或标注。

兼容性 smoke 已使用现有 3968-row 旧 manifest 完成 schema 解析和首条真实 feature 读取：trajectory 为 `[221,101376]` BF16，condition 为 `[105,101376]` BF16，`33×3072` metadata 一致。这只是 reader/schema smoke，不是新 extractor 的全量复制，也不是联合训练效果。

## 训练

推荐使用显式、query-disjoint 的 train/validation manifest：

```bash
python train_clir.py \
  --train_jsonl data/train_extracted.jsonl \
  --val_jsonl data/validation_extracted.jsonl \
  --config configs/best_current.json \
  --output_model outputs/best_current.pt
```

若只有一个 manifest，可按 `query_id` 而不是按 row 切分：

```bash
python train_clir.py \
  --train_jsonl data/extracted.jsonl \
  --val_fraction 0.1 \
  --output_model outputs/best_current.pt
```

训练前会检查 feature width；显式 validation 若与 train 有重复 `query_id` 会直接失败。每个 epoch 都会检查 total loss 和梯度是否 finite，执行 gradient clipping，然后原子发布 full-state checkpoint 和 `<checkpoint>.metrics.jsonl`。checkpoint 包含 model、optimizer、完成 epoch、RNG、配置、数据 hash 和 metrics。

### 精确续训

`epochs` 表示续训后的总目标 epoch 数。下例从已完成 epoch 5 的 checkpoint 继续到 epoch 10：

```bash
python train_clir.py \
  --train_jsonl data/train_extracted.jsonl \
  --val_jsonl data/validation_extracted.jsonl \
  --config configs/best_current.json \
  --output_model outputs/best_current.pt \
  --resume_from outputs/best_current.pt \
  --epochs 10
```

除 `epochs` 外，模型配置、训练设置和数据必须与 checkpoint 一致，否则拒绝续训。sampler 使用显式 `(seed, epoch)` 顺序，optimizer 和 RNG 也会恢复；CPU 测试覆盖了 interrupted 与 uninterrupted 训练最终状态完全一致。

## 打分与 Best-of-N 标记

```bash
python score_clir.py \
  --input_jsonl data/validation_extracted.jsonl \
  --model outputs/best_current.pt \
  --output_jsonl outputs/validation_scored.jsonl
```

打分默认 `batch_size=2` 和 BF16 autocast，是针对 101376 维全层 feature 的保守设置；需要 FP32 时显式传 `--amp_dtype none`。大规模 ranking 若只需要最终打分，可显式加 `--scalar_only`；它仍保留 checkpoint SHA、`clir_score` 和每题 Best-of-N 标记，但不会复制逐 token 诊断数组。

输出保留原 row，并增加：

- `clir_score` 和 `clir_checkpoint_sha256`；
- `clir_path_hallucination_prob`、`clir_path_no_hallucination_log_prob` 和 `clir_pseudo_onset`；
- 逐 token `clir_hallucination_prob`、`clir_token_reward` 和 `clir_token_value`；
- `clir_mean_gate`、逐 token `clir_gate_attention` 和 `clir_condition_relevance`；
- 归一化的 `clir_key_prior` / `clir_complete_prior`，独立 sigmoid membership `clir_key_prior_membership` / `clir_complete_prior_membership`，以及 overlap 诊断 `clir_prior_gate_alignment` 和与训练目标同定义的 `clir_prior_gate_squared_l2`；
- `clir_selected_best_of_n`，每个 `query_id` 恰有一个最高分候选被标记。

`clir_pseudo_onset` 始终作为诊断输出；默认配置中的 pseudo-tail 训练仍为关闭状态。输出 JSONL 原子写入，且不得覆盖输入 manifest 或 checkpoint；已存在的其他输出需显式 `--overwrite`。

## 查询级评估

```bash
python evaluate_clir.py \
  --input_jsonl outputs/validation_scored.jsonl \
  --output_json outputs/validation_metrics.json \
  --k 1,2,4,8,16 \
  --bootstrap_replicates 2000
```

评估器按 `query_id` 分组，按冻结的 candidate index 排序，对每个 `k` 只使用前 `k` 个候选，报告：

- reward Best-of-N accuracy 和 query bootstrap 95% 区间；
- random expected accuracy；
- oracle accuracy；
- query 内 correct-vs-wrong pairwise accuracy 和 tie 数。

score tie 使用最早 candidate，保证结果稳定。默认要求每个 query 都至少有 `max(k)` 个候选，因此所有 K 使用同一 query population；任一 query 不足都会直接失败。只有探索性报告才应传 `--allow_incomplete_queries`，此时改为每个 K 单独过滤候选不足的 query，各 K 的 population 可能不同。

评估前会检查 score 全部 finite、correctness 严格为 finite `0/1`，candidate index 不重复且显式 index 从 0 连续。报告原子写入，记录 `input_jsonl_sha256` 以绑定输入 scored manifest；已存在的输出需显式 `--overwrite`。

机制标签存在时，可把 H 排序、onset 阈值、pre/tail value shift 和 key/complete prior
learnability 与 task ranking 分开评估：

```bash
python evaluate_clir_mechanisms.py \
  --input_jsonl outputs/mechanism_dev_scored.jsonl \
  --output_json outputs/mechanism_metrics.json
```

matched 多 seed 比较使用 `summarize_clir_ablation.py`。它要求目录为
`seed_<seed>/<cell>/validation_{scored,metrics}.*`，逐行核对所有 run 的候选身份、顺序、
correctness、scored-input hash 和 checkpoint hash，再对同 query outcome 做 paired
bootstrap；预声明 cells 见 `configs/clean_ablation_v1/`，已完成结果和结论见
[`docs/handoff.md`](docs/handoff.md)。

### Consistency v6.1 扩量复测

扩充后的 400 个训练正对、150 个 held-out 正对、150 个 held-out hard negative
和 1,357 条 exact-ID 全层 feature 已完成独立核验。C-only 的 C0/C1 matched
复测也已完成；冻结设计、执行结果和证据边界见
[`docs/consistency_training_protocol_v6_1.md`](docs/consistency_training_protocol_v6_1.md)。

本轮确定性构造了一份 C0/C1 共用的 4,768-row 训练清单：3,968 条历史
correctness 行加 400 个新 relation 的 800 个 endpoint。C0/C1 只差
`consistency_weight=0/1`，固定 seeds 42/43/44、3 epochs；150+150 关系只做
机制留出评估。新增入口分别是 `prepare_clir_consistency_training.py`、
`evaluate_clir_consistency.py` 和 `summarize_clir_consistency.py`。这轮不训练 H、
Dual Prior 或 Full，也没有新的 ranking population，因此不能给 Best-of-N 效果结论。

正式结果中，C0/C1 的主 cosine separation 为 `.00016/.17614`，配对增量
`+.17597`，冻结 relation bootstrap 95% 区间 `[+.13889,+.21553]`；score-gap
separation 也从 `.18622` 增至 `.42970`。同时，正对 cosine 下降、cosine AUROC
仅从三 seed 均值 `.6875` 到 `.7209` 且方向不一致，hard negatives 大多仍未低于
`.2` margin。结论是 Consistency 已有“分开 hard negative、打破表示塌缩”的部分机制
证据，尚未建立完整正对不变性或最终选答案增益。

### 排序评测与 H0 扩量 v7（原协议失败，v7.4 另做探索性子集）

冻结协议见
[`docs/ranking_expansion_protocol_v7.md`](docs/ranking_expansion_protocol_v7.md) 和
[`configs/ranking_expansion_v7/protocol.json`](configs/ranking_expansion_v7/protocol.json)。
现已生成并校验 1,500-query 新排序池的 24,000 条候选，以及 H0 采集、补样和最终冻结的
800 条 proposal。80 条 smoke 通过后开放 720 条 reserve；reserve 首轮失败，v7.3 按
一次性修正案让 GPT-5.6-sol xhigh 和 Claude Opus 5 high 对原封不动的 800 条公开包全部
重标。第二轮 path 一致率为 `698/720=96.94%`，共同 positive 的首错 unit 精确一致率为
`310/403=76.92%`，控制项两边均为 `8/8`；但 A/B 的盲重复自一致性分别只有
`65/72=90.28%` 和 `64/72=88.89%`，没有达到预先冻结的 95%。因此最终状态为
`FAIL_H0_V7_RESERVE`。原协议没有发布 H0 Silver train/dev，也不允许第三次重标或事后
降低门槛。终止报告 SHA-256 为
`93260683…2c01`，原始标注和完整报告只保留在本地 `run_artifacts/`。

用户随后明确授权“看看这些 H 数据里有没有能用的，挑子集使用”。这不是追认 v7 通过，
而是一个单独登记的 post-hoc exploratory 路线。v7.4 只接收 smoke 的 A/B 精确非低置信
共识，以及 reserve 中 retry A、retry B、原始 B 三者对 `(clean/hallucinated, onset unit)`
完全一致的自然行；任何 retry 自重复失败的自然题都排除，attempt-1 A 完全不用。按原冻结
优先级而非模型效果取满后得到 600 条、600 个不同 query：train 为 200 positive + 200
clean，dev 为 100 + 100，train/dev 无 query 重叠。标签名是
`silver_posthoc_triple_consensus_h0_v7_4`，只能称“双/三路 AI 共识的探索性 Silver”，不能称
Gold、人工验证或 confirmatory 数据。原始 `FAIL_H0_V7_RESERVE` 保持不变。

新实验协议见
[`configs/ranking_expansion_v7/h0_experiment_v7_4/protocol.json`](configs/ranking_expansion_v7/h0_experiment_v7_4/protocol.json)。
四格 `C0/C1/H0/CH0` 共用同一个 5,168-row 训练 manifest：原 C0/C1 的 4,768 行加
400 条 H train；只有 `consistency_weight` 与 `hallucination_weight` 开关不同。H0 仅开启
onset-tail token BCE，H1 negative-tail reward、Path MIL、pseudo-tail、Dual Prior 和 Full
全部关闭。H dev 的 200 条不进训练。

24,000 条原始排序候选全部保留。为满足 Best-of-N evaluator 对每题 16 个二值 correctness
标签的硬契约，v7.4 在抽特征前机械保留“16 条都能被 checker 明确判成 numeric match 或
mismatch”的 892 题、14,272 条；不看任何 CLIR 分数，也不按正负比例挑题。其中 347 题
同时含正确和错误候选，另作配对区分力副报告。选定 H + ranking 共抽取 5,247,658 个
全宽 token，BF16 payload 约 990.9 GiB；8 个 worker 和逐文件 SHA/shape/finite 复核均通过。

四格、三 seed、三 epoch 共 12 个训练和两类评测均已完成。BoN@16 如下；括号内是相对
C0 的配对均值，区间为 seed+query hierarchical bootstrap 95% interval：

| Cell | BoN@16 | seed 42/43/44 | 相对 C0 |
|---|---:|---:|---:|
| C0 correctness | `84.08%` | `86.43/86.32/79.48%` | — |
| C1 consistency | `85.35%` | `85.76/85.43/84.87%` | `+1.27` points `[-1.68,+5.23]` |
| H0 onset BCE | **`86.06%`** | `85.76/86.55/85.87%` | `+1.98` points `[-1.12,+6.13]` |
| CH0 C1 + H0 | `85.58%` | `85.20/85.54/85.99%` | `+1.49` points `[-1.94,+6.39]` |

H0 的点估计最高，而且 13,028 个 query 内 correct-vs-wrong 比较中，平均区分率也是四格
最高的 `.6697`（C0/C1/CH0 为 `.6232/.6612/.6642`）。但三项增益区间都跨 0，C0 的
seed 44 又明显塌到 `79.48%`，所以只能说“严格子集有可用排序信号”，不能说 H0 已稳定
提高 Best-of-N。`CH0-C1-H0+C0` 交互为 `-1.76` points，区间
`[-5.38,+1.05]`：组合仍没有胜过 H0，但也不足以确认稳定负交互。

200 条独立 H dev 更清楚地界定了这些标签到底能用来学什么：H0 的 token AUROC/AP 为
`.878/.848`，path AUROC 为 `.841`，正路径检出率 `.827`、clean 拒报率 `.787`；说明它
能学到“从某个区域开始，后面的推理不可信”。但固定 `.5` 阈值下，first-bad start exact
为 `0%`、±5 token 仅 `4%`。因此这 400 条训练子集适合继续用作 **tail/path 风险监督**，
目前不适合宣称“精确首错 token 定位”。CH0 的 H 指标略低于 H0，也没有带来组合收益。

最终汇总状态是 `COMPLETE_H0_V7_4_POSTHOC_EXPLORATORY_EVALUATION`，本地 summary
SHA-256 为 `d80fff82…291e`。这是扩大后的探索性 Silver 复测，不是原 v7 翻盘，也不是
Gold、人工验证、protected-test 或 H1/Prior/Full 的证据。

### Dual Prior v8--v12 均按冻结门停止

v8 的 60 条依赖图标注已经完成并按冻结门评估。结果不是“待标”：eligibility=`60/60`、
path agreement=`.95`，但 Key/Complete F1 只有 `.7667/.8040`，非低置信完整训练共识仅
`8/60`，控制题 A/B 为 `4/6`、`5/6`。最终状态固定为
`STOP_PRIOR_DEPENDENCY_SMOKE_V8_RAW_GATE_FAILURE`，不裁决、不抽 feature、不训练。

诊断表明依赖图闭包把任务变复杂了。历史 v3 direct-set 数据中，Key 有 55/60 行完全一致；
Complete 虽只有 26/60 行完全一致，但 unit decision agreement 仍为 `.9341`，真正有分歧的
unit 只有 `.0659`。因此新 v9 回到让两个 AI 直接标 Key/Complete，同时把“整行必须完全一样”
改成局部共识：Key 仍要求非低置信 exact set；Complete 的交集为正、并集外为负、对称差位置
用显式 `complete_prior_mask=0` 跳过。attention 仍在完整 trajectory 上归一化，Prior 网络、
mutual 和 main 固定 `.25` gate coupling 均未改变。

协议见 [`docs/data_expansion_prior_protocol_v9.md`](docs/data_expansion_prior_protocol_v9.md)，
入口是 `prepare_clir_prior_v9.py`。它在代码提交 `3331edf` 上从 v6 的 16,000-row 池重新选择
60 个全新 query/cluster，排除了 v6.1 Consistency 和 v8 的全部 query/cluster，并核验与 v7
H/ranking 均零重合。GSM8K/MATH × numeric match/mismatch 仍各 15 条。

公开 A/B 盲包分别为 78/66 行（每边 60 natural +6 hidden controls，A 另有 12 blind
repeats），发布和独立复算均通过：

- natural ordered SHA-256：`ba0133d0…5852`；
- A/B package ordered SHA-256：`e816f353…71d1` / `fa131ce2…7847`；
- 状态：`PASS_PRIOR_PARTIAL_SMOKE_V9_PACKAGES_READY`；
- 独立复算：`PASS_PRIOR_PARTIAL_SMOKE_V9_RECOMPUTATION`。

GPT-5.6-sol xhigh 与 Claude Opus 5 high 随后完成全部标签，schema、population 和 ID 契约均
通过，但 raw 数据门失败。eligibility 为 60/60，双方非低置信 usable 为 53；Key/Complete
macro F1 只有 `.7778/.7280`，非低置信 exact Key 为 39/60，能够同时训练 Key+Complete 的也
只有 39/60。Complete unit agreement=`.7891`、分歧比例=`.2109`、正集合交并比=`.5665`、
平均 coverage=`.7999`，均未达到 `.90/.10/.80/.90` 的冻结门；controls A/B=`4/6,5/6`，
A self-repeat=`12/12`。

这不是少量随机边界噪声：在 53 条双方非低置信行里，B 的 Complete 是 A 的严格子集 47 条，
两边平均 Complete 大小约为 `10.87` 对 `6.03`。因此局部 mask 会屏蔽约五分之一 unit，不能
把剩余 39 行包装成扩量成功。最终状态为
`STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE`，raw report SHA-256=`30695a1a…1f3`；
不裁决、不挑子集、不扩 400/150、不抽 feature、不训练。Prior 网络和固定 `.25` gate 路径仍
保留在代码中，但独立 Prior 扩量的最早 blocker 已回到“Complete 到底包含多宽”的标签定义。
不含原始标签的机器摘要见
[`configs/data_expansion_prior_v9/completion.json`](configs/data_expansion_prior_v9/completion.json)。

用户随后批准先优化提示词；若统一口径仍只差数量，再采用“先扩量、按事先固定规则取严格
共识子集”。v10 不重标 v9，也不复用 v9 的 39 行：它在提交 `61aeab6` 上把 usable Key
强制为单个锚点（错误链取最早致命错误，否则取首次完成答案的最后非包装步骤），并把 Complete
固定为从最终实质结论向前回溯候选实际计算链；拆开的“代入算式+结果”都保留，自包含计算不收
重复结果，题面复述/计划/未用旁枝/包装排除。该口径保持 `origin/main` 的 Key 窄、Complete 宽。

v10 从 v6 池另选 60 个全新 query/cluster，四格各 15，并在选择前排除 v6.1 C、v7 H、
v7 ranking、v8、v9 的全部 query/cluster。A/B 各一个 80 行包（60 natural +8 controls +12
self-repeats），已发布并独立复算通过：natural ordered hash=`37ae27fa…ecd9`，A/B package
ordered hash=`01642639…4f3` / `d6c7a5e5…2ee5`，状态分别为
`PASS_PRIOR_CANONICAL_SMOKE_V10_PACKAGES_READY` 与
`PASS_PRIOR_CANONICAL_SMOKE_V10_RECOMPUTATION`。协议见
[`docs/data_expansion_prior_protocol_v10.md`](docs/data_expansion_prior_protocol_v10.md)，入口为
`prepare_clir_prior_v10.py`。

两份 v10 标签随后完整通过 schema、ID、顺序和盲包绑定。自然 60 条的 Key/Complete F1 为
`.9000/.9253`，exact non-low Key=`54/60`，Complete unit agreement=`.9291`、positive
IoU=`.8462`，A/B self-repeat 都是 `12/12`；所有自然、数量、coverage、repeat 和反退化门均
通过。但 A 漏看隐藏题里的 `7+5=13`，把下游步骤选成 Key，controls=`7/8`；B 为 `8/8`。
冻结门要求两边都 `8/8`，所以最终状态是
`STOP_PRIOR_CANONICAL_SMOKE_V10_DEFINITION_FAILURE`，report SHA-256=`2ecc2e80…3025`。
这不能被解释成 yield-only，也不能重标控制题、降低门槛或挑 54 条共识行训练。

用户随后批准再做一次 v11。它不改 singleton Key、canonical Complete、partial mask、模型、
loss 或固定 `.25` gate，只在提示词前增加“按 unit 顺序重新验算算术/代数/单位/对象/所求量，
再选最早致命错误”，并要求错误 rationale 写出具体校验。v11 从同一已 materialize 池确定性
选择 60 个新的 query/cluster，排除 v6.1 C、v7 H/ranking、v8、v9、v10；8 个控制题也全部
换新，A/B 仍各 60 natural +8 controls +12 repeats。协议见
[`docs/data_expansion_prior_protocol_v11.md`](docs/data_expansion_prior_protocol_v11.md)，入口为
`prepare_clir_prior_v11.py`。代码/协议 commit `1b0e35d` 上的正式准备与独立复算已经通过：
状态为 `PASS_PRIOR_VERIFIED_SMOKE_V11_PACKAGES_READY` /
`PASS_PRIOR_VERIFIED_SMOKE_V11_RECOMPUTATION`；natural ordered hash=`26aec3c9…28e4`，
A/B package ordered hash=`6b826261…80fc` / `e042b8e6…468c`。两个公开包各 80 行、ID 唯一，
没有 source/checker/reference/control-answer 字段。

两份 v11 标签后来均完整写入并通过 schema、ID、盲包、控制题和 self-repeat 校验；纯 evaluator
两次得到完全相同的 report SHA-256 `0729d982…fa8d`。但 60 条自然样本的 Key macro F1
只有 `.8333 < .90`，Complete positive IoU 为 `.7957 < .80`，因此最终状态固定为
`STOP_PRIOR_VERIFIED_SMOKE_V11_DEFINITION_FAILURE`。A/B controls 都是 `8/8`、self-repeat
都是 `12/12`，说明失败不在文件或标注稳定性，而在 Key/Complete 边界仍未达到预注册一致度；
不能把其中 50 条 exact-Key 行或任何“容易题子集”拿去训练。

在不回收 v11 的前提下，v12 冻结为一条全新的严格共识扩量路线。协议见
[`docs/data_expansion_prior_protocol_v12.md`](docs/data_expansion_prior_protocol_v12.md)，入口为
`prepare_clir_prior_scale_v12.py`：先从与历史/v6/v7/Prior v8--v11 query/cluster 全部隔离的
GSM8K/MATH train 中冻结 2,000 题，每题生成 8 条；再按 checker × 题源 × train/dev 预冻结格
选 800 条交给 GPT-5.6-sol xhigh 与 Claude Opus 5 high 独立双标，最后只按事前写死的 exact
singleton-Key 与 partial-Complete 共识规则取 500 条。只读审计确认 2,647 个可用新 cluster，
足以组成 2,000 题且与排除集零重合；40 个 rollout shard 已全部完成并逐项复验，共 16,000 条、
其中 15,822 条正常 stop、178 条截断。combined raw SHA-256 为 `ce18c0f7…e7552`。CPU
materialization 通过 16,000/16,000 unitization，得到 15,520 条 supervision-eligible；冻结的
800 条 proposal 文件 SHA-256 为 `334c7dbf…65bdc`，八个题源/checker/split 格均精确满足配额。
提交 `a669e9a` 随后构造并独立复算了 32 个公开盲包：A/B 各 16 shard，每 shard 50 natural +1
control +5 repeat，共每边 896 行；package report/verification SHA-256 为 `ffe02fb1…f452` /
`37837dc0…c27e`，且冻结时没有 label。

两边后来各完成 16 个 shard、896 行标签。提交 `f7e3a9e` 在首次读取标签前冻结了 exact
schema/ID/package evaluator、16-control/80-repeat 门、八格固定配额和选择后 Complete 质量门；
完整测试为 `175 passed`。首次评估与独立复算逐字节一致，raw report/verification SHA-256 为
`0dce22ba…d0a1` / `cf68ba49…f20a`，最终状态
`STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE`。

数量并不是失败原因：800 条自然样本全部双方 usable/non-low，687 条达到 exact singleton-Key
与非空 Complete 交集，八个预冻结格均足以按原 priority 凑满 400 train +100 dev。失败的是
稳定性与 Complete 质量：A controls=`11/16<15/16`，B=`16/16`；A/B self-repeat 都是
`51/80=.6375<.95`；固定 500 条的 Complete IoU=`.7065<.80`，mask coverage=`.8709<.90`。
只读诊断中，Key repeat 实为 A=`76/80`、B=`78/80`，而 Complete repeat 两边都只有 `51/80`；
说明主要 blocker 是长链 Complete 边界仍不稳定，而不是候选规模不足。协议要求终止：不发布
这 500 条、不事后挑 687 条或其他容易子集、不重标、不抽 feature、不训练。

### Dual Prior v13：机械局部审核烟测因 schema 终止

v12 的只读回放先验证了一个重要负结果：只把 unit 碎片合成较大的 block，A 的 Complete
重复一致仍为 `51/80`，B 也只从 `51/80` 变成 `53/80`，所以问题不是简单的切分偏一格。
v13 因此不再让 AI 自由列整个 Complete 集，而是让程序先合并安全碎片、给出非强制角色提示，
并为每个步骤最多提出两个直接父步骤；AI 只审核 block 角色、局部依赖、最终步骤和 singleton
Key，Complete 最后由程序沿保留依赖边向前回溯。原始 unit index 始终保留为训练轴。

协议见 [`docs/data_expansion_prior_protocol_v13.md`](docs/data_expansion_prior_protocol_v13.md)，
入口为 `prepare_clir_prior_mechanical_v13.py`。提交 `83775b7` 从 v12 已冻结但从未送标的 rollout
里选择 48 条 train-side 新轨迹，排除了 v12 800 条 proposal 的所有 query/cluster，并精确覆盖
GSM8K/MATH × numeric match/mismatch × medium/long 八格各 6 条。A/B 各有 4 个 shard，每个
固定 12 natural +2 hidden controls +4 跨 shard repeats=`18` 行；每边总计 72 行。

盲包状态为 `PASS_PRIOR_V13_FRESH_BLIND_PACKAGES_READY`，独立复算状态为
`PASS_PRIOR_V13_PACKAGE_INDEPENDENT_RECOMPUTE`；package report/verification SHA-256 分别为
`8de2a666…1422` / `abfaad27…fdd`。GPT-5.6-sol xhigh 与 Claude Opus 5 high 随后各完成
4 个 shard：A/B 各 72 行，八个文件均为合法 JSON，行数、唯一 ID 和对应 package 全部匹配。

冻结 evaluator 仍在 schema 门终止为 `FAIL_PRIOR_V13_SCHEMA`。唯一错误是 B 的隐藏控制项
`prior-v13-control-b-07`：它正确给出 `eligibility=no_auditable_reasoning`，但仍输出了一个
`answer_wrapper` block role；不可用行按冻结契约必须把所有结构字段清空。因为 evaluator 在
schema 失败后不会计算 controls、repeats 或 48 条自然样本的机制指标，本轮没有可报告的
Complete 稳定性结果。报告 SHA-256 为 `179d1006…577f`，且 `trainable_labels_published=false`。
协议预先禁止失败后的格式/标签修补、重标和自适应救援，因此不能删掉该 role 后重跑，也不能
启动 v14、抽 feature 或训练 v13。若仍要使用 v12 严格共识子集，只能另立明确的 post-hoc
探索版本，并保留 v12/v13 的失败结论。

### Dual Prior v12-posthoc：可学，但没有改善最终排序

用户随后明确选择“V12吧”，因此新建了独立的
[`posthoc_v1`](configs/data_expansion_prior_v12/posthoc_v1) 路线；它不修改 v12/v13 的失败报告。
机械规则只保留 A/B 都 usable、非 low、singleton Key 完全相同、非空 Complete 集合完全相同的
行；若某个 natural parent 被任一侧抽到 self-repeat 且 target 漂移，也整行排除。800 条原始
proposal 中有 266 条满足双边 exact Key+Complete，再排除 13 条已观察到 repeat 不稳定的行，
最终得到 253 条：202 train、51 dev。标签名明确包含
`posthoc_dual_ai_exact`，没有人工复核，也明显偏向两个 AI 都容易判断的样本。

253 条 selected-only exact-token 全层 BF16 feature 已完成逐 tensor 验证，原始 feature 约
`19.75 GiB`。第一阶段只比较 matched R0/P0，均为 3 epochs、seeds 42/43/44：两边共享
4,170 条训练行；R0 只用 correctness，P0 额外使用旧 48 + 新 202 =250 条 direct
Key/Complete 监督。Consistency、H0/H1、mutual、gate、MIL、pseudo-tail 和 Full 全关。
六个 checkpoint 均可加载且全部 finite。

51 条独立 Prior dev 表明 direct target 确实学会了，而不只是 loss 能下降：

| 指标（三 seed 均值） | R0 | P0 |
|---|---:|---:|
| Key AUROC / AP / BCE | `.488 / .063 / .616` | **`.904 / .596 / .145`** |
| Complete AUROC / AP / BCE | `.527 / .381 / .683` | **`.961 / .935 / .270`** |
| correctness AUROC / BCE | `.898 / .491` | `.883 / .500` |

但同一批 892 query ×16 candidate 的冻结 v7.4 排序复用集没有出现最终收益：

| K | R0 BoN | P0 BoN | P0−R0 |
|---:|---:|---:|---:|
| 1 | `82.74%` | `82.74%` | `0.00` point |
| 2 | `84.87%` | `85.13%` | `+0.26` point |
| 4 | `85.80%` | `85.76%` | `-0.04` point |
| 8 | **`86.17%`** | `84.60%` | **`-1.57` points** |
| 16 | **`85.72%`** | `85.54%` | `-0.19` point |

主指标 BoN@16 的逐 seed 差为 `-.67/+.22/-.11` points；fixed-seed query interval
`[-1.31,+.90]`、exploratory seed+query interval `[-1.57,+1.12]` points，均跨 0。
BoN@8 则三个 seed 都下降，两个区间分别为 `[-2.65,-.49]` 和 `[-2.95,-.22]` points。
题内 correct-vs-wrong pairwise 也从 R0 `.6682` 降到 P0 `.6594`。事后描述性检查显示，
P0 在 K=16 每个 seed 改了约 `77%--83%` 的候选选择，但三 seed 合计只有 78 次错→对、
83 次对→错，其余 1,982 次换候选不改变 correctness；说明共享表示路径确实会传导到 score，
只是当前传导没有形成净收益。

准确裁决是：**扩量后的 direct Key/Complete 已建立很强的 held-out 可学习性，但 gate-off 的
间接共享表示路径没有改善最终答案选择，K=8 还有稳定回退。** 排序集是复用的探索性 population，
不是新 protected test；原 v12/v13 仍失败。mutual、固定 `.25` gate 和 Full 仍未因本轮自动
解锁，若继续必须另冻下一阶段，不能在这 51 条 dev 或 892 题上再挑 epoch、权重或子集。

### Dual Prior v12-posthoc：固定 `.25` Gate 学到对齐，但排名门失败

随后按独立冻结的单因素协议比较已有 P0（Gate 关）与 PG0（只把
`gate_prior_weight` 从 `0` 改为 `.25`）。PG0 使用相同的 4,170 条训练行、250 条 direct
Prior 监督、三个 seed 和三个 epoch；没有开启 Consistency、H0/H1、mutual 或 Full。

在 51 条 Prior dev 上，Gate 到 fused Key/Complete 分布的平方 L2 从 `.03114` 降到
`.02584`，2/3 seed 改善；Key/Complete AP 下降分别只有 `.01071/.00018`，Gate 的归一化
熵 `.87191`、有效 token 比例 `.41419`。因此“Gate 学会跟随双先验”、Prior 保护和防塌缩
三道机制门都通过。

但复用的 892×16 探索性排名没有通过主门：BoN@8 从 `84.60%` 到 `85.05%`
（`+0.45` point，三 seed 同正但区间跨 0），BoN@16 从 `85.54%` 到 `85.01%`
（`-0.52` point，三 seed 全负；fixed-seed query interval `[-1.20,+.11]` points）。
K=16 时 PG0 每 seed 改变 `30%--57%` 的候选，说明 Gate 确实直接影响最终 score；只是净方向
略差。冻结裁决是：**机制对齐成立，固定 `.25` Gate 的 standalone ranking benefit 不成立，
并按预注册规则在当前探索性 screen 被拒绝。** 用户要求三模块阶段仍保留 main-style `.25`
路径，所以后续只把它作为固定方法身份测试交互，不在这批 51/892 数据上重新调权重。

### 三模块扩量组合 v1：24 组已完成，Full 结果不确定

[`three_module_expansion_v1`](configs/three_module_expansion_v1) 已完成统一数据、八格
`2×2×2`、三 seed、三 epoch 的全部 24 次训练。5,370-row 训练清单、198-row H dev、
49-row Prior dev、所有 checkpoint/optimizer、机制评分和 892×16 排序评分均通过完整性检查。
完整数字见 [`docs/three_module_expansion_v1.md`](docs/three_module_expansion_v1.md)，冻结结果见
[`completion.json`](configs/three_module_expansion_v1/completion.json)。

机制层面，三个目标都学到了：C 能把同义不同写法与 hard negative 拉开；H0 的 token AP
从 U0 `.465` 提到 H `.838`，但 Full 在首错位置 ±5 token 内只约 `3.3%`，所以它更像“坏尾部
风险检测”而不是精确首错定位；P 的 Key/Complete AP 从 `.072/.370` 提到
`.595/.928`。固定 `.25` Gate 在全部 P-on cell/seed 中都比 uniform attention 更接近同一个
learned prior（`12/12`），但这是 scale-aware 的事后机制诊断，不等于排序收益。

复用 892-query 探索性 ranking 的 BoN@16 为：U0 `84.16%`、C `85.80%`、H `85.54%`、
P `85.72%`、CH **`86.14%`**、CP `85.20%`、HP `84.04%`、Full `84.68%`。Full−U0
为 `+0.52` point，但 fixed-seed query 95% interval=`[-0.71,+1.76]` points，未通过收益门，
也未触发整体伤害门。最清楚的冲突是 H×P：`-1.96` points，三 seed 全负，fixed 与
hierarchical interval 都低于 0；Full 也稳定低于 CH `1.46` points。准确结论是：模块本身
都能学到各自 Silver target，单模块 point estimate 都高于 U0，但当前 P/Gate 与 H 的组合
会抵消收益；Full 尚未证明有效。不能继续在这批 dev/ranking 上调参数，下一步需要新的
query/template-cluster-disjoint ranking population 做确认。

### Prior/Gate 新题归因与权重确认 v1：已完成，Full 相对 CH 触发伤害规则

前瞻协议、锁定选择与最终裁决分别见
[`protocol.json`](configs/prior_gate_tuning_v1/protocol.json)、
[`confirmation_lock.json`](configs/prior_gate_tuning_v1/confirmation_lock.json) 和
[`completion.json`](configs/prior_gate_tuning_v1/completion.json)。本轮没有复用看过结果的 892 题：
调参与确认各使用 800 个 query、每题 16 个候选，两边 query 与近模板簇重合均为 0。确认集在
权重和 checkpoint 写入 Git 锁文件前没有计算 CLIR score，锁定后只打开一次。

Stage A 把 Prior 拆成两条影响。关闭 Gate 时，direct Key/Complete 监督相对 CH 平均
`-0.50` point；在 direct 监督之上恢复固定 `.25` Gate 平均补回 `+0.50` point。因此负项来自
direct 监督，而不是 Gate 本身，按冻结规则只开放 direct 权重 `{.25,.5,1}`，Gate 始终为 `.25`。
开放调参集的 BoN@16 分别为 `95.250% / 95.375% / 95.958%`；减弱 direct 监督反而更差，最终锁回
原始 `direct=1、Gate=.25`。该候选与 CH 在调参集同为 `95.958%`，没有任何 Prior 候选超过 CH。

一次性确认集上，锁定 Full 为 `93.917%`，CH 为 `94.458%`，U0 为 `92.708%`。主对比
Full−CH=`-0.542` point，逐 seed 为 `+0.375/-1.125/-0.875`，fixed-seed query 95% interval
为 `[-1.375,+0.250]` points。区间仍跨 0，不能夸大效应大小；但均值为负且两个 seed 为负，按
事先写死的规则裁决为 `CONFIRMATION_HARM`。Full−U0=`+1.208` points，三 seed 全正且 fixed interval
为 `[+0.083,+2.333]` points，说明三模块整体仍强于无辅助任务基线，只是把 Prior 加到更强的 CH
上会抵消一部分收益。

当前训练数据下的排名推荐因此是 **CH**。`.25` Gate 继续作为 main-style 工程默认路径保留，但这不
等于 Full 有效性结论；不能再用这 1,600 题改权重。下一步若继续 Prior，应扩充或改善 Prior Silver
监督并单独诊断 H0×Prior 交互，再使用新的预注册训练/排名 population，而不是继续扫 `.25/.5/1`。

## Toy smoke test

Toy 数据只验证代码路径，不能证明方法有效：

```bash
python examples/create_toy_clir_data.py \
  --output_jsonl examples/toy_clir.jsonl \
  --feature_dir examples/features \
  --hidden_dim 8

python train_clir.py \
  --train_jsonl examples/toy_clir.jsonl \
  --feature_root . \
  --output_model outputs/clir_toy.pt \
  --hidden_dim 8 \
  --epochs 2 \
  --amp_dtype none \
  --num_workers 0

python score_clir.py \
  --input_jsonl examples/toy_clir.jsonl \
  --feature_root . \
  --model outputs/clir_toy.pt \
  --output_jsonl outputs/clir_toy_scored.jsonl

python evaluate_clir.py \
  --input_jsonl outputs/clir_toy_scored.jsonl \
  --output_json outputs/clir_toy_metrics.json \
  --k 1,2
```

完整测试：

```bash
pytest -q
```

## 当前限制

- 仓库已有 v2 rollout、numeric checker、first-bad-unit/dual-prior Silver 标注包与盲裁管线，但没有人工
  标注或人工复核；它不生成 Gold。
- 多题源管线已完成 2,000-query/16,000-row Phi rollout；v6.1 的 400/150/150
  Consistency 关系和 1,357-view feature 已发布并独立复核。它们仍是双 AI Silver，
  不能称为 Gold 或人工验证。
- 默认仍使用预抽取全层 feature，真实数据的磁盘开销很大；没有集成 batch-local online extraction。
- 当前 objective 是 pointwise correctness BCE 加可用 auxiliary supervision，尚无 pairwise/listwise reward objective。
- clean integration 已完成历史小数据矩阵、扩充后的三模块完整 `2×2×2` 三-seed 矩阵，
  以及 C/H/P/Gate 的机制复测；仍没有新的 protected ranking test。扩量 Full 在复用
  892-query ranking 上只比 U0 高 `0.52` point 且区间跨 0，不能称为有效提升。
- Consistency 已有400个训练正对和150+150 held-out 正负关系的三 seed C0/C1
  复测；均值分离与 score-gap 结构改善，但正对 cosine 下降、cosine AUROC seed 方向
  混合。Hallucination/Prior 使用的是无人工复核的事后 Silver 子集；H0 的 tail/path
  判别明显改善，但精确 onset 仍差。Prior v12 原门和 v13 schema 门仍保持失败；
  另立的 post-hoc exact 子集新增 202 train +51 dev，证明 direct Key/Complete 可学，但复用
  ranking 上 gate-off BoN@16 无增益、BoN@8 三 seed 一致回退；固定 `.25` Gate 虽学到
  Prior 对齐，却在 BoN@16 三 seed 全负，也不能替代新的确认性数据。
- clean checkpoint 已记录配置、数据/split hash、feature reference、optimizer/RNG、metrics、code commit/branch/dirty state、完整命令与运行环境；这不替代缺失的数据 provenance 上游与 protected-test protocol。
- clean 已有 frozen-prefix evaluator、机制诊断和 parity-checked multi-seed paired
  summarizer；v6.1 新增了单独的 held-out consistency relation evaluator，但尚未重建
  strict/encoded SWIFT 等预算 baseline。

研究假设、已有证据与未验证部分见 [`docs/proposal.md`](docs/proposal.md)；迁移依据和历史负结果见 [`docs/handoff.md`](docs/handoff.md)。
