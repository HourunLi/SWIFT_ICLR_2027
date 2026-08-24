# CLIR clean integration 交接说明

本分支从 `main` 的十余文件结构重新出发，只整合 `panzhixin` 分支中可复用的工程进展和通过相应门的模块部分。它不是对 `panzhixin` 的压缩复制，也没有继承其大量冻结协议、标注流水线、版本化 runner、历史结果文件或失败诊断类。从 tracked file 数量看，origin/main 约 10 个，`panzhixin` 为 429 个（configs 201 / scripts 104 / tests 46 / docs 45 / src 24），clean integration 约 18 个；结构仍是 `main` 的扁平十余文件骨架。

当前唯一运行配置是 `configs/best_current.json`。这里的 “best current” 指当前最清晰、最可维护的**整合方案**；历史上最高的单次联合矩阵 BoN@16 是 correctness-only J0 `.920`，不是三模块联合成功。

## 2026-08-23 复核状态

在 commit `8b116c4` 的 clean integration 上完成了完整测试、toy resume、旧真实 manifest 接入、全宽梯度 smoke、1-epoch 真实训练和 500×16 排名评估。审计中发现 CUDA 续训会失败：`torch.load(..., map_location=cuda)` 把 CPU RNG state 搬到 GPU，随后 `torch.set_rng_state` 拒绝该 tensor。当前工作树已改为 checkpoint 先在 CPU 加载，再由 `load_state_dict` 恢复 model/optimizer；针对性测试与完整测试均通过，最终为 `37 passed`。

真实数据门与试跑：

- train 为 3968 rows / 496 queries × 8，validation 为 8000 rows / 500 queries × 16，query overlap 为 0；代表性 BF16 feature shape 为 trajectory `[221,101376]`、condition `[105,101376]`。
- train 中有 3968 correctness rows、27 个 consistency positive pairs、702 个 negative pairs、48 个 onset rows，以及 48 行/14,307 token 的 paired key/complete supervision，覆盖当前所有 active objective。
- 默认模型实测 `5,347,593` trainable parameters；全宽两个模块 batch 的 loss/gradient finite，峰值显存不超过约 `2.94 GiB`。
- seed 42、1 epoch 的 train/mechanism-dev total 为 `.5374/3.6609`。checkpoint SHA-256 是 `e1dba08f91d6529213db1acadfc274a422a76c7c8d096a74da2576290c7c891f`。
- checker v4 的 ranking validation 上，BoN@16/random/oracle 为 `.906/.8925/.976`，within-query pairwise 为 `.6241`；selected-minus-random paired delta `+.0135`，10,000-query-bootstrap interval `[-.0045,+.03175]`。

证据等级是 `small-scale real pipeline pilot`。它证明 clean 代码可以消费旧全层 BF16 artifact、联合 loss 可以真实更新并产出完整 checkpoint/score；1 epoch/1 seed、区间跨 0、没有 matched correctness-only clean baseline，所以不能证明任何单模块或 full integration 增益。

`configs/clean_ablation_v1` 冻结了 7-cell 三 epoch 筛选矩阵：C0 correctness-only、C1 consistency、H0 onset BCE、H1 onset BCE+negative tail、P0 direct priors、P1 direct+mutual priors、full integration。所有 cell 的架构、初始化 RNG、sampler、优化器和预算相同，仅 loss-family 权重变化。预注册 staging 是先完整跑 seed 42；若数据/hash/finite gate 正常，则不按 seed-42 排名挑选，而是全矩阵扩到 seeds 43/44；只有所有 cell 的三 epoch 曲线均未饱和且 mechanism dev 未明显恶化时，才统一续到五 epoch。该协议现已执行完毕，结果见下一节。

数据承载边界也已量化：train correctness 为 3590 正/378 负；consistency 只有 27 个训练正 pair 且没有 held-out relation；H train 为 17 个正 onset + 31 个 clean，dev 为 6 + 10；prior 虽有每个 head 14,307 个 token 标签，但只来自 48 条相关 trajectory，dev 只有 16 条。故这套数据足够做 matched engineering/screening ablation，不足以做正式机制归因；更多 epoch 只会重复相同标签，不能替代扩标和独立 held-out mechanism set。

## 2026-08-24 三随机种子消融状态

`configs/clean_ablation_v1` 的 7 个 cell 已完整运行 seeds 42/43/44、每个 3 epochs，
没有按 seed-42 结果挑选扩跑对象。21 个 checkpoint 都绑定
`da913318dae92ed8d436564729b92afb4c93f44c`、`dirty=false`、配置/数据/环境/命令；
统一 500×16 ranking scoring 和严格 candidate parity 也已完成。新增
`summarize_clir_ablation.py` 做跨 cell/seed 的候选、标签、input/checkpoint hash 校验及
paired-query uncertainty，`evaluate_clir_mechanisms.py` 分开评估 H、onset、value shift
和 dual-prior target。

BoN@16 mean ± sample SD：C0 `.9173 ± .0061`、C1 `.9220 ± .0040`、H0
`.9267 ± .0110`、H1 `.9187 ± .0058`、P0/P1 均 `.9180 ± .0072`、full
`.9160 ± .0080`。相对 C0，C1 为 `+.47` points、H0 为 `+.93`、P1 为
`+.07`、full 为 `-.13`；所有 paired query bootstrap 区间跨 0。H0→H1 的 gold-tail
增量是 `-.80` points，三个 seed 都回退。H1 的 `post-onset − pre-onset` token value
约为 0，但所有 token value 整体约为 `-.62`，再次呈现 global shift 而非 locality。

direct priors 在 16-row dev 上可学（P0 key/complete AUROC `.663/.869`），但 mutual
没有增量机制或 ranking 收益；H0/H1 有弱 token/path 排序信号，onset ±5 仍为 0；
consistency 没有 held-out relation set。多个 auxiliary cell 的 mechanism-dev 随 epoch
恶化，故统一 3→5 epoch 门失败，没有继续训练。完整数字、区间与裁决见
[`clean_ablation_v1_results.md`](clean_ablation_v1_results.md)。正确状态是：工程矩阵闭环
通过，部分 auxiliary target 在小 dev 上可学，任何模块的 held-out ranking efficacy 和
full integration 增益都尚未建立。

### CH0 交互补测

用户提出的缺失组合已按查看结果前冻结的协议完成：`CH0 = correctness + consistency +
onset BCE`，不含 gold negative tail、direct/mutual prior 或其他辅助 loss。它在 commit
`582da9af65da622608d796f68b035f523b13009d`、`dirty=false` 上使用与原矩阵完全相同的
train、16-row mechanism dev、500×16 ranking population、3 epochs 和 seeds 42/43/44。
三个 checkpoint、scored input、candidate identity/order/labels 和配置/数据 hash 均通过。

| Cell | BoN@16 mean ± SD | seed 42/43/44 | pairwise mean ± SD |
|---|---:|---:|---:|
| C0 correctness | `.9173 ± .0061` | `.916/.912/.924` | `.6860 ± .0200` |
| C1 consistency | `.9220 ± .0040` | `.918/.922/.926` | `.6937 ± .0088` |
| H0 onset BCE | `.9267 ± .0110` | `.932/.914/.934` | `.6753 ± .0174` |
| CH0 C1 + H0 | `.9153 ± .0042` | `.920/.912/.914` | `.6942 ± .0181` |

CH0 相对 C0/C1/H0 的 BoN@16 均值分别为 `-.20/-.67/-1.13` points；其中 H0→CH0
逐 seed 为 `-1.2/-.2/-2.0` points，fixed-seed query bootstrap 95% interval 为
`[-2.20,-.13]` points，seed+query hierarchical interval 为 `[-2.73,+.47]` points。
二因子交互 `CH0 - C1 - H0 + C0` 为 `-1.60` points，逐 seed
`-1.4/-1.2/-2.2`，fixed-seed query interval `[-3.07,-.20]`，hierarchical interval
`[-3.40,+.13]` points。准确裁决是“小数据筛选中没有加和，并出现一致的负交互信号”；
三个 seed 和稀疏辅助监督不足以声称两个模块天然不兼容。

CH0 的 pairwise `.6942` 与 C1 基本相当且高于 H0，但 BoN@16 低于三者，说明它并非让
所有排序全面变坏，更像是改变了最顶部候选的极值排序。机制 dev 上 CH0 的 H token
AP/AUROC 为 `.450/.749`、path AUROC `.872`，优于 H0 的小样本点估计；但阈值 `.5`
完全不报正 onset，onset ±5 仍为 `0/6`。因此辅助 H 排序诊断变好不能替代最终选择效果，
也不能证明 onset boundary 已学好。

这里的命名关系必须保持清楚：`H0 = correctness + onset BCE`；`H1 = H0 + gold tail`；
`CH0 = C1 + H0`；旧 `full = C1 + H1 + P1`。所以旧 full 既不是 H0 也不是单独 H1，
它同时包含 consistency、onset BCE、gold-tail、direct priors 和 mutual priors。CH0 才是
检验 C 与 H0 是否能组合的干净二因子 cell。

### Prior→reward gate 补测

`configs/clean_gate_ablation_v1` 已在查看新指标前冻结并完成：P0 是 correctness + direct
key/complete prior，PG0 只增加 `gate_prior_weight=.0625`。该值匹配 `origin/main` 的总
有效强度 `.25×.25`；mutual、C、H、tail、progress 和 reconstruction 全部关闭。两 cell
使用同一 3968-row train、16-row mechanism dev、500×16 ranking pool、3 epochs 与
seeds 42/43/44，checkpoint 绑定 commit
`649747f3605e820430d4c93d788e368676ff37ea`、`dirty=false`。重新训练的 P0 state dict
与旧 clean P0 逐 tensor bit-exact。

PG0 的 held-out gate→fused-prior squared-L2 从 P0 `.01195` 变为 `.01335`，只有 seed42
改善，seeds43/44 恶化；key AP `.2969→.2926`、complete AP `.9210→.9251`，prior
protection 与 gate-collapse guard 通过。BoN@16 从 `.9180→.9167`，逐 seed delta
`+.2/-.8/+.2` points，fixed-seed query interval `[-.87,+.60]`、seed+query interval
`[-1.20,+1.00]` points。pairwise 点估计约 `+.35` points，但没有转化为主指标增益。

gate 确实影响了 score：各 seed 有 `53.2%/75.6%/57.6%` query 更换最终候选；其中绝大
多数 correctness 不变，错→对与对→错合计为 `18 vs 20`，净少 2/1500。准确裁决是：
工程 direct coupling 闭环，但 main-scale objective 没建立 alignment learnability 或
ranking efficacy。该实验当时支持保持默认 0；它作为 `.0625` 单点的历史结果保留，完整
证据见 [`clean_gate_ablation_v1_results.md`](clean_gate_ablation_v1_results.md)。

### Prior→reward gate 工程默认值选择 v2

用户随后把“保留 `main` 原始 shared-gradient coupling，并默认开启”定为方法身份约束，
同时授权在当前开发 population 上选一个相对保守的固定强度。新协议在查看结果前提交
`.25/1/4/10` 四点配置，复用已完成的 `0/.0625` anchors；12 个新 run 使用相同 3968-row
train、16-row mechanism dev、500×16 ranking development、3 epochs 和 seeds 42/43/44。
mutual、C、H/tail、progress 与 reconstruction 全关，只改变 `gate_prior_weight`。

所有正权重均通过 finite、key/complete AP protection、normalized entropy 与 effective
support 门。BoN@16 三-seed mean 为：`0=.9180`、`.0625=.9167`、`.25=.9187`、
`1=.9180`、`4=.9173`、`10=.9207`。`10` 是 raw best，但 `.25` 与它恰好相差冻结的
near-tie 阈值 `.002`，所以按“近似时选更小权重”规则固定 `.25`。`.25−P0` 仅 `+.07`
point，fixed-seed interval `[-.80,+.87]` points、hierarchical interval
`[-1.20,+1.53]` points，均跨 0。

因此 `RewardConfig` 和 `configs/best_current.json` 现在都默认 `.25`。它保持
`origin/main` 的 gate normalization、detached 50/50 fused prior、full-trajectory
squared-L2、shared mask 与 shared-encoder gradient 路径；没有引入新的推理分数项。
clean 外层 `prior_weight=1`，故绝对 coupling 系数 `.25`，是原 main 总有效系数 `.0625`
的 4 倍。这个值的证据标签只能是 `dev-tuned engineering default`：同一 500-query dev
参与了选择，而且选中点没有改善 held-out gate L2。扩大独立 prior/ranking 数据后应固定
`.25` 做 off/on 复测，不再消费当前 dev 调参。完整结果见
[`clean_gate_tuning_v2_results.md`](clean_gate_tuning_v2_results.md)。

## 与 `origin/main` / 历史 artifact 的兼容性

| 维度 | 结论 | 边界 |
|---|---|---|
| 最新 `origin/main` model API | 兼容 | identity 模式严格加载最新 main state dict；21 个共同 forward 输出在运行时对照中 bit-exact，clean 只额外输出 `layer_attention` |
| 旧 main checkpoint | 部分兼容 | condition bottleneck 修复之后的 identity checkpoint 可加载；修复前、缺少 `condition_hidden_proj/condition_delta_out` 的旧 checkpoint shape 不兼容 |
| clean real checkpoint | 不与 main raw-width 权重互换 | clean 新增 layer-axis encoder，真实参数 shape 与 main 直接吃 raw width 的模型不同 |
| 打分 checkpoint schema | 向后读取 | `score_clir.py` 接受 clean 的 `model_config` 或 main 的 `config`；是否能 strict-load 仍取决于具体架构版本 |
| 训练续训 schema | 不兼容 main | clean 需要 optimizer/epoch/RNG/data_state 的 full-state checkpoint；main 的 weights-only checkpoint 只能打分，不能精确 resume |
| CLI/config | 有意 breaking | main 的 loss-heavy CLI 改为一个 JSON 方法配置加少量运行覆盖项，旧命令不能原样复用 |
| `panzhixin` manifest | 已验证兼容 | nested `feature_metadata`、`feature_sha256`/`condition_sha256` 和全层 BF16 路径已在 3968/8000 manifests 上通过 |
| `panzhixin` 研究协议 | 不自动兼容 | sparse H、strict/encoded variants、versioned runner、标注与多 seed summarizer 没有迁入；不能把可读 manifest 等同于可复现旧协议 |
| evaluator | 工程兼容，已补多 seed 配对 | frozen prefix、stable tie、common population、random/oracle/pairwise aggregate、机制诊断及 parity-checked multi-seed paired summary 已有；formal 仍缺 protected test、held-out consistency evaluator 和完整 baseline 矩阵 |

## 迁移裁决

| 内容 | 当前处理 | 来源与理由 |
|---|---|---|
| SWIFT-style token reward/gate、trajectory residual | 保留 | `main` 的核心 score 语义 |
| 通用 semantic/style consistency | 保留 | `main` 接口不绑定某一种 rewrite 路线 |
| Hallucination onset BCE 与负 tail reward | 实现与整合默认保留，但分开裁决 | onset BCE 有弱诊断/排名点信号；2026-08-24 gold-tail 的 locality/ranking 门失败，不能因默认 active 就声称有效 |
| condition attention 256 维瓶颈 | 保留 | `main` 修复了 raw hidden width 上的二次参数爆炸 |
| exact prompt/output token ID 对齐 | 移植 | `panzhixin` 最重要的数据正确性经验 |
| embedding + 全部 block hidden states | 移植 | 真实 Phi 数据链已经跑通 |
| feature provenance/checksum 与旧 manifest schema | 精简后移植 | 新产物记录 model/revision/dtype/SHA-256，reader 解析嵌套 metadata，resume 识别旧 checksum 别名 |
| layer-axis Transformer encoder | 移植 | 把 `33×3072` 压到 768，真实配置少于一千万训练参数 |
| BF16 feature 原 dtype 读取 | 移植 | 避免 CPU 侧无意义扩成 FP32 |
| token target 严格等长、缺失标签 mask | 移植 | 禁止静默截断、补零和把未标注 row 当负样本 |
| semantic group sampler | 移植 | 确保 consistency pair 进入同一 batch |
| query-disjoint split | 移植 | 避免同 query 候选跨 train/validation |
| finite loss/gradient、grad clip | 移植 | 基础训练可靠性 |
| 原子 full-state checkpoint 和精确 resume | 精简后移植 | 保留 model/optimizer/RNG/data contract，不搬复杂 run-record 系统 |
| query-level Best-of-N evaluator | 精简后移植 | 固定 candidate prefix、tie 和 bootstrap 语义 |
| Dual-prior direct key/complete supervision | 默认启用 | standalone 与 clean 小 dev 均显示 target learnability；clean ranking 增益未建立 |
| 双向 stop-gradient mutual distillation | 默认启用，权重 `.25` | 历史保护门通过，但 clean P0→P1 没有机制或 ranking 增量，不应再写成 efficacy 证据 |
| shared-gradient gate-prior alignment | 公式保留，默认权重 `.25` | 用户要求保留 main 方法路径；v2 按冻结近优规则选出 `.25` 作为 dev-tuned 工程默认值。排名 efficacy 仍未建立，扩大数据后固定 off/on 复测 |
| sparse-span hallucination | 不迁移到当前核心 | 点估计小门通过，但 onset、blind transfer 和联合门失败；且用户指定回 main |
| online batch-local extraction | 暂不迁移 | 只有小样本等价性，没有大规模吞吐结论；会显著扩大 trainer |
| Strict / Encoded baseline model variants | 不迁移 | 保持 clean 主干单一模型；后续 matched ablation 在独立分支或最小 baseline 中重建，不把多 variant 类塞回核心 |
| annotation/adjudication/protocol/versioned scripts | 不迁移 | 研究档案留在旧分支，不应成为核心运行依赖 |

不要整文件复制 `panzhixin` 的模型文件：那会覆盖 `main` 后续增加的 condition bottleneck。当前正确组合是：

```text
exact all-layer features
  → layer-axis encoder (101376 → 768)
  → main-style condition attention bottleneck (768 → 256 → 768)
  → reward / consistency / hallucination / dual-prior heads
```

## `panzhixin` 的真实数据规模

旧分支的磁盘规模很大，但统计监督规模很小，二者不能混为一谈。

| 数据部分 | 历史真实规模 |
|---|---:|
| outcome train | 496 queries × 8 candidates = 3968 trajectories |
| 每 epoch 生成 token | 1,116,541 |
| 5 epochs 每个 cell 的 token exposure | 5,582,705 |
| correctness-only rows | 3866，约 97.43% |
| consistency rows | 54，即 27 对，约 1.36% |
| hallucination + prior rows | 48，约 1.21% |
| 任意 auxiliary rows | 102/3968，约 2.57% |
| H train sparse supervised tokens | 6681：922 positive / 5759 negative |
| prior supervised token units | 14,307 |
| mechanism dev | 16 trajectories；H sparse tokens 2451 |
| ranking validation | 500 queries × 16 candidates = 8000 trajectories |
| ranking validation tokens | 2,315,243 |

全层 BF16 feature 被多轮重复保存后，旧分支本地 artifact 约 833 GB。这个数字主要反映 `[T,101376]` payload 的重复物化，不代表有海量独立训练样本。clean integration 不跟踪这些 feature 或 checkpoint，也不在配置中写机器绝对路径。

已对现有 3968-row manifest 做 clean reader 兼容性 smoke：schema 完整解析，首条真实 trajectory `[221,101376]` 和 condition `[105,101376]` 均以 BF16 读取，`33×3072` 约束一致。这证明旧 artifact 可作为数据接入起点，不证明新 extractor 全量跑通或新配置的训练效果。

## 联合训练的真实负结果

旧分支完成过 seed 42、5 epochs 的 J0/JP/JALL single-stream 矩阵。loss、监督计数、梯度和 checkpoint 工程闭环均成功，但冻结效果门失败。

| Cell | BoN@16 | H span AP | H claim AP | Key AP | Complete AP | C cosine gap |
|---|---:|---:|---:|---:|---:|---:|
| J0 correctness | `.920` | `.169` | `.421` | `.098` | `.277` | 约 0 |
| JP prior | `.918` | `.192` | `.172` | `.432` | `.946` | `.023` |
| JALL 三模块 | `.912` | `.272` | `.289` | `.314` | `.931` | `.789` |

关键解释：

- JALL 相对 J0 的 BoN@16 为 `-.008`，query-paired 95% bootstrap 区间为 `[-.026,+.010]`。没有建立正增益，区间跨 0 也不足以宣称稳定负效应。
- H span `.272` 低于冻结位置基线 `.393`，claim `.289` 低于 `.422`，两门均失败。
- JALL key AP 相对 JP 从 `.432` 降到 `.314`，下降 `.118`，超过允许的 `.05`。
- complete prior 和 consistency 的训练内 geometry 通过，但 consistency gap 只来自训练关系，不能当作 held-out 泛化证据。
- 因扩展门失败，seeds 43/44 没有继续跑。
- 后续 drop-one、supervision packing、condition gradient routing、frozen probe、temporal smoother和 H-v3/v3a 都没有修复核心问题。

因此新分支不能写成“三模块联合有效”。2026-08-24 的 clean 三 seed matched matrix 也已
完成，但 full 相对 C0 为 `-.13` points 且区间跨 0；准确状态是：三模块接口已整合，
部分 auxiliary target 有小样本 learnability，联合或单模块 ranking efficacy 未建立。

## 为什么 Hallucination 回到 `main`

`panzhixin` 的 S1 sparse reviewed-span BCE 在 48/16 小数据上得到 span AP `.416`、claim AP `.464`，点估计略高于位置基线 `.393/.422`；但 bootstrap 区间跨 0，exact onset `±5` 为 `0/6`。后续 mixed-domain blind validation 和 position-control 路线继续失败。

更重要的是，旧联合方案只让 sparse BCE 更新独立 H head 和共享表示：MIL、token reward、absolute/relative/pseudo tail、progress 都为 0，`hallucination_logits` 不进入 scalar score。因此它实现的是“幻觉诊断 head”，不是项目最初的“首错之后降低 reward”。

当前分支按用户裁决恢复 `main` 语义：

```text
hallucination_onset = k
  → H target 在 k 前为 0、从 k 起为 1
  → token_values 从 k 起被监督到负 margin
  → gate-weighted scalar score 的 value path 因此受到影响
```

需要同时保留反面证据：旧分支对 absolute-margin tail 做过多 fold / seed 复核，tail-specific locality 0/3 seeds 通过，存在全局 value shift；relative 和 clean-matched repair 也失败。2026-08-24 的 clean gold-tail 消融再次出现全局 value shift，且 H0→H1 的 BoN@16 三 seed 都回退。因此“回 main”只解释方法身份和为何保留实现；当前 objective 已未通过这轮 locality/ranking 筛选门，不应继续在同一 16-row dev 上调权重。

## 可选与关闭支线的当前裁决

| 支线 | 当前代码状态 | 默认 | 重新开启条件 |
|---|---|---:|---|
| path-level MIL | 保留稳定 log-space noisy-or | `0` | 更大、定义稳定的 path labels；单独 matched ablation |
| pseudo-onset tail | 保留 | `0` | H boundary 在独立数据通过后再启用，避免循环自训练 |
| progress | head 与 loss 保留 | loss `0`，score weight `0` | 有独立于 token advantage 的 target，并明确 reward/progress 分工 |
| gate-prior alignment | 原 shared-gradient 公式保留 | `.25`，开启 | 用户方法身份约束下的 dev-tuned 工程值；扩大 prior/ranking 数据后固定权重做 off/on 独立复测，不再在当前 dev 调参 |
| complete reconstruction | 仅外部 target 接口保留 | `0` | 获得独立 evidence/answer embedding；禁止同 trajectory 自重构 |
| sparse-span H | 未迁移 | 不适用 | 若重开需新实现、独立标签和与 onset-tail 的明确语义比较 |
| relative tail | 未迁移 | 不适用 | 新方法、新 validation，不继续消费旧 16-row dev |
| clean-matched tail | 未迁移 | 不适用 | 先解决优化隔离与 comparator 定义，再重新预注册 |
| H-v3/v3a、probe、smoother | 未迁移 | 不适用 | 不作为当前主线 |
| supervision packing、condition routing | 未迁移 | 不适用 | 只有新证据表明 schedule/routing 是主要瓶颈时才考虑 |

“保留开关”不代表推荐启用，也不代表历史负结果被抹去；“未迁移”也不等于永久否证整个研究假设。

## 核心代码入口

### `configs/best_current.json`

唯一默认配置。真实输入为 `33×3072`，layer encoder 输出 768，condition bottleneck 256。默认 active loss 是 final + consistency + main hallucination onset/tail + direct/mutual dual prior + `.25` main-style gate alignment；MIL、pseudo、progress 和 reconstruction 关闭。

### `src/clir_features.py`

- `IdentityFeatureEncoder`：toy 或已压缩 feature。
- `LayerAxisFeatureEncoder`：当前真实默认；reshape 为 `[B*T,L,D]`，共享投影、layer Transformer、learned-query pooling。
- `build_feature_encoder`：由 `RewardConfig.encoder_type` 选择。

### `src/clir_data.py`

- `CLIRTrajectoryDataset`：读取 inline/path feature，核对 exact output/prompt token 长度和 layer metadata，保留 BF16；兼容旧的嵌套 `feature_metadata` schema。
- `clir_collate`：padding 和每类监督的独立 mask。
- `SemanticGroupBatchSampler`：把同 semantic 的多 style row 放入同一 batch。
- `EpochRandomSampler`：以 `(seed,epoch)` 固定普通 shuffle，供精确 resume。

`query_id` 只用于候选分组与 split；`semantic_id/style_id` 只用于 consistency。不要互相 fallback。

### `src/consistency_localized_reward.py`

- `RewardConfig`：模型结构与 loss 权重。
- `ConsistencyLocalizedReward.forward`：encoder、condition fusion、score 和诊断 heads。
- `ConsistencyLocalizedReward.loss`：按标签存在性和权重路由三模块 loss。
- `prism_style_consistency_loss`：正/负 pair 和 score consistency。
- `hallucination_localization_losses`：当前 main onset-tail 实现。
- `path_level_hallucination_mil` / `pseudo_onset_tail_loss`：保留但默认关闭。
- `dual_prior_losses`：direct、mutual、gate、external reconstruction。

### 顶层命令

- `extract_hidden_states.py`：必填 `query_id` + strict exact integer IDs → all-layer token features；复用同 prompt condition，记录 revision/dtype/checksum，支持受控 `--overwrite`。
- `train_clir.py`：唯一 JSON config、query split、finite checks、原子 full-state checkpoint/resume；resume feature contract 兼容 `feature_sha256` / `condition_sha256` 旧别名。
- `score_clir.py`：默认 batch 2 + BF16，输出 checkpoint SHA-256、scalar/path-clean log/逐 token H-reward-value/prior membership/condition 诊断和每 query Best-of-N 标记；原子写入且默认不覆盖。
- `evaluate_clir.py`：candidate-prefix Best-of-N、bootstrap、pairwise accuracy；默认要求全部 query 满足 max K，仅 `--allow_incomplete_queries` 启用逐 K 过滤，报告记录输入 SHA-256。

## 已知限制

- 没有 rollout generator、answer checker、rewrite verifier 或机制标注系统；extractor 假设 exact IDs 已由上游可靠保存。
- extractor 现在会把 `--revision` 传给模型加载器，并写 model/revision/dtype 与 feature checksum；但本地模型若无 config commit 且未显式传 revision，`feature_revision` 仍可为 null。resume data contract 会使用 checksum 字段，不会每次重新 hash 巨大 feature 来验证 manifest 中的声明。
- extraction 虽原子发布单个 tensor 和最终 manifest，`--overwrite` 中途失败仍可在旧 manifest 下留下部分新 feature；正式运行应使用新目录而不是就地覆盖。
- 只支持预抽取 feature 训练，全层 payload 存储昂贵；online extraction 尚未进入 clean trainer。
- 当前模型用 pointwise correctness BCE，没有 pairwise/listwise ranking objective。
- clean 已在历史 3968-row 数据上完成 7-cell 主矩阵和 CH0 二因子补测、三 seed matched evaluation；它没有扩大独立机制样本，也没有使 `best_current` 成为 efficacy winner。
- consistency 证据只有 27 对且没有 held-out relations。
- H 证据来自很少的 Silver trajectory，首错边界一致性弱；clean onset ±5 仍为 0，恢复的 main gold-tail 再次未通过 locality/ranking 门。
- dual prior 只有很少的 adjudicated Gold trajectory；clean direct target 可学，但 mutual 增量与 ranking improvement 都未建立。
- gate-prior 现在默认 `.25`，只在 row 同时具有 key/complete coverage 时计算；progress、reconstruction 等权重为 0 时，对应 loss/value 路由会直接跳过，不通过 `0×NaN` 污染 score 或 total。
- score 中始终输出 pseudo onset 和 path probability；这不表示 MIL/pseudo-tail 训练已经打开。
- resume 的相同设备 CPU 测试为 bit-exact；不要假设跨设备、跨 PyTorch/CUDA 版本也逐 bit 相同。
- checkpoint 已写 code commit/branch/dirty-worktree、完整命令和 Python/PyTorch/CUDA/device；上游标签/checker一致性、protected test 和 baseline completeness 仍不满足正式论文级协议。
- trainer 的 feature reference 会绑定 path、size、mtime 与 manifest 内 checksum 声明，但不会在每次训练前重 hash 数百 GiB payload；必须先确认 durable exhaustive mirror-verification report。
- clean evaluator 与新增 summarizer 已能完成跨 variants/seeds 的 parity 检查和 paired contrast；本地大 scored/checkpoint artifact 默认不进 Git，正式发布仍需独立 artifact manifest/存储。

## 下一步

1. 保持完整测试、toy resume 和真实 feature contract gate；如重新抽取，在新目录做 extraction smoke 并绑定 model revision/checksum，不就地覆盖历史 payload。
2. 先扩大并 query-disjoint 划分 consistency relations 与可靠 H onset labels：C train 至少 300–500 个语义组、held-out 100–200 组；H train 至少 200 个正 onset + 200 个显式 clean，H dev 至少 100+100。当前 16-row mechanism dev 已用于筛选，不再用它选择 threshold、weight 或 epoch。
3. 用新数据预注册并完整重跑 `C0/C1/H0/CH0` 2×2；主检验仍是各单模块增量和 `CH0-C1-H0+C0` 交互。当前负交互只用于决定复测设计，不能直接写成普遍结论。
4. 统一 train/ranking checker 为 v5，并把 ranking validation 扩到约 1500–2000 个独立 query ×16；若预算允许，outcome train 扩到约 1500–2000 queries ×8。锁验证结论后再开 protected test。
5. H 先单独通过 token/path calibration 与 onset boundary，再设计新的 localized reward coupling；当前 gold-tail、MIL 和 pseudo-tail 都不进入下一轮 full。
6. prior 先扩到约 300–500 条独立 train trajectory 与 100–200 条 query-disjoint dev，复核 direct target 的跨样本/跨域 learnability；在新 ranking population 上固定 `.25` 比较 gate off/on，不再调权重。若提出 KL/gradient-balanced/head-only/runtime-fusion 等新 coupling，必须作为新假设预注册。若要归因 encoder，最小重建 strict/encoded SWIFT 等预算 baseline。

任何后续结果都应把三件事分开报告：工程闭环是否运行、auxiliary target 是否可学、是否真正改善 held-out Best-of-N。三者不能互相替代。
