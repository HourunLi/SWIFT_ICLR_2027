# Clean ablation v1：三随机种子筛选结果

状态：2026-08-24 完成 7-cell 主矩阵及预注册 CH0 交互补测。证据等级是
`small-scale real screening`，不是正式 efficacy result。

## 冻结协议与数据承载量

训练使用 commit `da913318dae92ed8d436564729b92afb4c93f44c`，branch 为
`clir-clean-integration`，所有 checkpoint 均记录 `dirty=false`、完整命令、配置
hash、数据/split hash、Python/PyTorch/CUDA/device 和 RNG/optimizer 状态。7 个 cell
只改变预声明的 loss-family 权重；架构、初始化 RNG 流、sampler、batch size、学习率、
BF16 和 3-epoch 预算一致。每个 cell 完整运行 seeds 42/43/44，共 21 个训练 run、63
个 cell-epoch；全部 finite，没有 OOM、静默失败或候选人口不一致。

随后在查看 CH0 指标前冻结 `CH0 = C1 + H0`，按同一预算完成另外 3 个 run、9 个
cell-epoch；累计为 24 个 run、72 个 cell-epoch。CH0 checkpoint 绑定 commit
`582da9af65da622608d796f68b035f523b13009d`、`dirty=false` 和配置 SHA-256
`d5f67aca2696c08a6331b26beb0840b4e197c146a4dbec57841290a06c741545`。结果更新后使用
固定 SWIFT Python 环境运行完整仓库测试，`42 passed`；CLIR skill validator 也通过。

| 数据 | 独立单位与标签量 | SHA-256 | 结论 |
|---|---|---|---|
| train | 496 queries × 8 = 3968 rows；correctness 3590 正/378 负 | `dc868b60494c0417902f4f4679d24771a75cd5c9f51107104bb365579a0f4d48` | 足够做小规模排序筛选 |
| consistency train | 54 rows、27 个正 relation pair；sampler preflight 产生 702 个负 pair | 同上 | 没有 held-out relation set，不足以证明关系泛化 |
| H train | 17 个正 onset + 31 个显式 clean trajectory | 同上 | token 多不等于独立 onset 多，规模很小 |
| prior train | 48 trajectories、每个 head 14,307 token labels | 同上 | token 强相关，独立 trajectory 仍只有 48 |
| mechanism dev | 16 query-disjoint trajectories；H 为 6 正/10 clean；prior 为 5008 token | `22b4dadc0c75cfe9422bc9805697028dd31e19fe129ac100ab5a10b69a16678d` | 只适合诊断，不适合选很多超参或做正式置信结论 |
| ranking validation | 500 queries × 16 = 8000 frozen candidates | `42d83ab29bcbf7070f5480b6f209f522aa402d841d850c6fea3c9835c999d801` | 约 1 point 的差异仍难稳定分辨 |

train correctness 使用 checker v5，ranking validation 使用 checker v4。该差异对 21 个
run 完全冻结，适合 matched screening，但正式协议应统一 checker 版本。候选 parity
摘要 SHA-256 为
`93f5d1bafcc81c109c8f2d7f8672e8e233baa45ff57eb88a888d0a7090e5a040`。

## 为什么停在 3 epochs

预注册规则是：只有所有 cell 的 3-epoch train/dev 曲线都没有饱和或明显 dev 恶化，才
把**全部** cell 一起续到 5 epochs，不能按结果挑选部分 cell。实际 train loss 普遍继续
下降，但 16-row mechanism dev 已出现明显不稳定，例如：

- seed-42 `H1` dev total：`3.2264 → 3.5930 → 3.9667`；
- seed-42 full：`3.6609 → 3.7147 → 4.2087`；
- seed-42 `P0`：`2.9610 → 3.2185 → 3.4368`；
- seed-42 `P1`：`2.9590 → 3.4739 → 3.7207`；
- 其他 seed 有时改善、有时反向，说明小机制集上的 seed 方差已经主导。

因此 5-epoch 扩展门失败。增加 epoch 只会重复 17/31 个 H 标签、27 个 consistency pair
和 48 条 prior trajectory，不会增加独立信息；本轮没有续到 5 epochs。

## Ranking 结果

下表是 seeds 42/43/44 的 mean ± sample SD。K=1 对所有 cell 都是 `.884`，因为只看
冻结前缀中的第一个候选，reward model 没有选择自由度。

| Cell | BoN@2 | BoN@4 | BoN@8 | BoN@16 | query 内 pairwise |
|---|---:|---:|---:|---:|---:|
| C0 correctness only | `.8993 ± .0023` | `.9100 ± .0035` | `.9120 ± .0053` | `.9173 ± .0061` | `.6860 ± .0200` |
| C1 + consistency | `.9047 ± .0012` | `.9087 ± .0023` | `.9180 ± .0020` | `.9220 ± .0040` | `.6937 ± .0088` |
| H0 + onset BCE | `.9000 ± .0087` | `.9133 ± .0012` | `.9200 ± .0069` | `.9267 ± .0110` | `.6753 ± .0174` |
| H1 + onset BCE + gold tail | `.9007 ± .0064` | `.9100 ± .0053` | `.9133 ± .0050` | `.9187 ± .0058` | `.6778 ± .0240` |
| P0 + direct priors | `.9027 ± .0023` | `.9087 ± .0031` | `.9107 ± .0050` | `.9180 ± .0072` | `.6891 ± .0113` |
| P1 + direct + mutual priors | `.9000 ± .0000` | `.9120 ± .0040` | `.9140 ± .0040` | `.9180 ± .0072` | `.6950 ± .0162` |
| Full integration | `.9020 ± .0020` | `.9100 ± .0060` | `.9173 ± .0090` | `.9160 ± .0080` | `.6869 ± .0101` |

主指标是同 seed、同 query 的 BoN@16 paired delta。区间对每个 query 先平均三个固定 seed
的 paired outcome，再做 10,000 次 query bootstrap；另外生成的 seed+query hierarchical
bootstrap 区间也全部跨 0，且只有 3 seeds，因此只作探索性检查。

| 对比（右减左） | seed 42 / 43 / 44 | mean delta | fixed-seed query 95% CI | 裁决 |
|---|---:|---:|---:|---|
| C0 → C1 consistency | `+.2 / +1.0 / +.2` pt | `+.47` pt | `[-.47,+1.40]` pt | 三 seed 同向，但幅度小且区间跨 0；保留候选，先补 held-out relations |
| C0 → H0 onset BCE | `+1.6 / +.2 / +1.0` pt | `+.93` pt | `[-.13,+2.07]` pt | 本轮最佳点估计，仍未达到稳定 efficacy 门 |
| H0 → H1 gold negative tail | `-1.0 / -.2 / -1.2` pt | `-.80` pt | `[-1.87,+.27]` pt | 三 seed 都回退；当前 tail coupling 不接受 |
| C0 → H1 combined H | `+.6 / 0 / -.2` pt | `+.13` pt | `[-.93,+1.27]` pt | onset BCE 的点增益基本被 tail 抹掉 |
| C0 → P0 direct priors | `-.6 / +1.2 / -.4` pt | `+.07` pt | `[-1.00,+1.13]` pt | seed 不稳定，没有 ranking 增益证据 |
| P0 → P1 mutual prior | `+.6 / +.2 / -.8` pt | `.00` pt | `[-.87,+.80]` pt | mutual 没有增量 ranking 证据 |
| C0 → P1 dual prior combined | `0 / +1.4 / -1.2` pt | `+.07` pt | `[-.93,+1.07]` pt | seed 不稳定 |
| C0 → full | `0 / +1.2 / -1.6` pt | `-.13` pt | `[-1.07,+.80]` pt | full 没有优于 correctness-only |

`H0` 的 BoN@16 点估计较好，但其 pairwise accuracy 反而比 C0 低约 `1.07` points，说明
这个结果依赖 top-1-of-prefix 行为，不是整体正确/错误排序全面改善。所有区间跨 0，不能
把上述点估计写成显著增益或稳定负效应。

## CH0：Consistency × onset BCE 交互补测

原 7-cell 矩阵缺少一个干净的二因子组合：full 同时含 consistency、onset BCE、gold
negative tail、direct priors 和 mutual priors，无法回答 C1 与 H0 本身能否相加。为此新增
`ch0_consistency_onset_bce`：final correctness、consistency、onset BCE 权重均为 1，
tail/MIL/pseudo/progress/direct prior/mutual prior/gate/reconstruction 全为 0。训练、候选、
初始化策略和评估均与原矩阵 matched。

### Ranking

| Cell | BoN@2 | BoN@4 | BoN@8 | BoN@16 | query 内 pairwise |
|---|---:|---:|---:|---:|---:|
| C0 correctness only | `.8993 ± .0023` | `.9100 ± .0035` | `.9120 ± .0053` | `.9173 ± .0061` | `.6860 ± .0200` |
| C1 + consistency | `.9047 ± .0012` | `.9087 ± .0023` | `.9180 ± .0020` | `.9220 ± .0040` | `.6937 ± .0088` |
| H0 + onset BCE | `.9000 ± .0087` | `.9133 ± .0012` | `.9200 ± .0069` | `.9267 ± .0110` | `.6753 ± .0174` |
| CH0 + consistency + onset BCE | `.9007 ± .0012` | `.9100 ± .0000` | `.9140 ± .0020` | `.9153 ± .0042` | `.6942 ± .0181` |

CH0 的 BoN@16 逐 seed 为 `.920/.912/.914`。严格相同候选上的主对比如下；fixed-seed
区间先对每个 query 的三个 seed outcome 求均值再 bootstrap，hierarchical 区间同时重采
seed 与 query，因此后者更诚实地反映只有三个 seed 的不确定性。

| 对比（右减左） | seed 42/43/44 | mean delta | fixed-seed query 95% CI | seed+query 95% CI |
|---|---:|---:|---:|---:|
| C0 → CH0 | `+.4/.0/-1.0` pt | `-.20` pt | `[-1.07,+.67]` pt | `[-1.60,+1.07]` pt |
| C1 → CH0 | `+.2/-1.0/-1.2` pt | `-.67` pt | `[-1.53,+.13]` pt | `[-2.00,+.60]` pt |
| H0 → CH0 | `-1.2/-.2/-2.0` pt | `-1.13` pt | `[-2.20,-.13]` pt | `[-2.73,+.47]` pt |
| interaction `CH0-C1-H0+C0` | `-1.4/-1.2/-2.2` pt | `-1.60` pt | `[-3.07,-.20]` pt | `[-3.40,+.13]` pt |

固定三个 seed 的 query-bootstrap 中，H0→CH0 与 interaction 都排除 0；但把训练 seed
也视为需要泛化的随机维度后区间仍跨 0。正确措辞是：**在当前冻结筛选集上，C1 与 H0
没有叠加，并出现三个 seed 同向的负交互信号**；不能据此声称二者天然或普遍不兼容。

CH0 pairwise `.6942` 明显高于 H0 `.6753`，且与 C1 `.6937` 几乎相同，但 BoN@16
反而更低。这说明 consistency 恢复了平均 correct-vs-wrong pair ordering，却没有保住
H0 在每组最高分候选上的优势。一个合理但尚未被因果验证的解释是：consistency 约束让
score 更平滑或改变了高分极值，因而损失 top-1-of-16 的选择收益；现有 evaluator 只能把
它标成待验证机制，不能当成已证明原因。

### CH0 的 H 机制诊断

| 指标 | H0 | CH0 |
|---|---:|---:|
| H token AP | `.357 ± .103` | `.450 ± .061` |
| H token AUROC | `.667 ± .127` | `.749 ± .042` |
| path AP | 未作为主裁决报告 | `.801 ± .016` |
| path AUROC | `.806 ± .140` | `.872 ± .063` |
| 阈值 `.5` 正 path 找到 onset | `.389 ± .419` | `.000` |
| onset ±5 | `.000` | `.000` |
| clean specificity | `.967 ± .058` | `1.000` |
| post-onset value − pre-onset value | `-.090 ± .114` | `-.446 ± .188` |
| all-token value | `2.859 ± .557` | `2.229 ± 1.259` |

CH0 在 tiny dev 上的 H 排序型 AP/AUROC 点估计更高，但阈值变得极保守：三个 seed 对
6 个正 path 都不报 onset，且 `±5` 仍为 0。因此它没有建立 boundary localization，机制
指标变好也没有转化成 BoN@16。由于没有 held-out consistency relation evaluator，本轮
仍不能判断 C head 是否真正泛化了语义/风格关系。

### CH0 裁决与扩量门

- 不把 CH0 提升为默认配置，也不因它否定 C1 或 H0 单模块；旧 full 更不能用来代表 H0。
- 当前数据足够做“负交互筛选”并决定下一轮实验矩阵，不足以做最终 efficacy 或模块天然
  不兼容结论。
- 下一轮冻结 `C0/C1/H0/CH0` 2×2：C train 目标 300–500 个语义组、held-out
  100–200 组；H train 至少 200 个可靠正 onset + 200 个显式 clean，H dev 至少
  100+100；ranking validation 目标 1500–2000 个独立 query ×16，并统一 checker v5。
- 新数据先统一跑 3 epochs/3 seeds；只有更大的 mechanism dev 曲线健康，才对**全部四个
  cell** 统一续到 5 epochs。增加当前数据的 epoch 不会提高独立样本量。

## 机制诊断

以下指标都来自同一 16-row mechanism dev，报告三 seed mean ± sample SD。token 数量
不能当作独立样本数；没有对这个小集合继续调 threshold。

### Hallucination localization 与 reward coupling

| Cell | H token AP | H token AUROC | path AUROC | 正 path 在阈值 .5 找到 onset | onset ±5 | clean 不报 onset | post-onset value − pre-onset value | all-token value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | `.285 ± .094` | `.590 ± .138` | `.694 ± .077` | `1.000` | `.000` | `.000` | `-.276 ± .285` | `-1.286 ± 2.762` |
| H0 onset BCE | `.357 ± .103` | `.667 ± .127` | `.806 ± .140` | `.389 ± .419` | `.000` | `.967 ± .058` | `-.090 ± .114` | `2.859 ± .557` |
| H1 onset + tail | `.409 ± .056` | `.719 ± .073` | `.828 ± .113` | `.056 ± .096` | `.000` | `1.000` | `.000 ± .014` | `-.620 ± .112` |
| Full | `.351 ± .134` | `.588 ± .169` | `.789 ± .193` | `.000` | `.000` | `1.000` | `.032 ± .040` | `-.564 ± .230` |

H token 的绝对位置 AP baseline 是 `.241`。H0/H1 有一些 token/path 排序信号，但
`±5` onset 命中仍是 `0/6`，阈值 `.5` 的 boundary calibration 不可用。noisy-or path
probability 的 Brier 约 `.62`，也显示 path probability 接近饱和，path AUROC 只能解释为
排序而不是良好概率校准。

更关键的是，H1 并没有产生局部的 post-onset 下降：`post − pre ≈ 0`，而所有 token
value 整体移到约 `-.62`。这与 ranking 上 H0→H1 三 seed 全回退一致，属于 global value
shift，而不是成功的 onset-localized negative tail。

### Dual priors

| Cell | key AP / AUROC | complete AP / AUROC | key position AP | complete position AP | key↔complete map squared-L2 |
|---|---:|---:|---:|---:|---:|
| C0 | `.174 / .508` | `.665 / .517` | `.242` | `.586` | `.0003` |
| P0 direct | `.297 ± .038 / .663 ± .033` | `.921 ± .016 / .869 ± .036` | `.242` | `.586` | `.0017 ± .0008` |
| P1 direct + mutual | `.292 ± .051 / .662 ± .043` | `.924 ± .017 / .870 ± .038` | `.242` | `.586` | `.0042 ± .0040` |
| Full | `.258 ± .028 / .639 ± .013` | `.895 ± .020 / .828 ± .040` | `.242` | `.586` | `.0011 ± .0005` |

direct prior 明显学到了这个小 dev 上的 membership target，尤其 complete prior；但 P0 的
BoN@16 没有稳定提升。P1 相对 P0 没有进一步提高 AP/AUROC、没有降低 held-out map
discrepancy，也没有 ranking 增益。因此当前证据只支持“direct target 在小样本上可学”，
不支持 mutual collaboration 或 reward efficacy。

consistency 没有独立 held-out relation set，所以不能报告其目标指标（same-semantic
different-style gap、negative separation、worst-view accuracy）。C1 的 BoN@16 三 seed
同向 `+.47` points 是值得扩数据后复核的 screening 信号，不是 consistency 已学会关系
泛化的证据。

## 四类 H 目标的区别与当前裁决

| 目标 | 使用的标签 | 训练什么 | 是否直接改变 scalar reward | 当前状态与理由 |
|---|---|---|---|---|
| Hallucination onset BCE | 显式 gold onset `k`；clean 为 `-1` | `k` 前 H target=0，`k` 起=1，训练 token H logits | 只通过共享表示间接影响；H logits 本身不进入 score | active/保留为诊断候选；有弱 AP/AUROC 信号，但 boundary 未通过 |
| Onset→negative tail reward | 同一个 gold onset `k` | 从 `k` 起把 token value 拉到负 margin，并加 tail hinge | 是，value 经 gate 聚合进 score | `best_current` 为保留 main 方法身份仍有实现；本轮 locality/ranking gate 失败，不应宣称有效或继续同数据调权重 |
| Path MIL | 只需整条 path 是否含幻觉 | 用 log-space noisy-or 让正 path 至少一个 token 为正、clean path 全负 | 默认只训练 H head，不直接改 value | 实现保留、权重 0；弱标签无法确定 onset，长序列易饱和，需更大独立 path 数据单独消融 |
| Pseudo-onset tail | path 为正但没有 gold onset；boundary 来自当前 H head 首次越阈值 | 从预测 boundary 起惩罚 token value | 是 | 实现保留、权重 0；会把 H 自身错误循环写入 reward。当前 onset ±5=0，更不能开启 |

“实现保留”表示代码/API 和数值稳定测试仍在，不表示推荐默认开启；“active”也只表示
方法配置，不等于通过了效果门。

## 总裁决与下一步

1. **工程闭环通过。** clean 分支已完成真实全宽、多 seed、全 cell 训练、机制打分、
   500×16 ranking 和严格候选 parity/paired uncertainty；checkpoint code provenance 已补齐。
2. **Consistency：候选保留，证据不足。** 排名点估计小幅、三 seed 同向；先增加并冻结
   held-out semantic/style relations，再谈模块有效。
3. **Hallucination：H BCE 只保留为诊断候选；当前 gold-tail objective 拒绝。** 不再用
   这 16-row dev 调 threshold/weight，也不打开 MIL/pseudo-tail。下一轮要先获得更大、
   query-disjoint 的可靠 onset set，并把 boundary/calibration 与 reward coupling 分开验收。
4. **Dual prior：direct target learnability 通过小样本诊断，ranking efficacy 与 mutual
   增量未通过。** 扩充独立 trajectory 后先复核 direct，再决定是否重开 mutual/gate。
5. **Full integration 未通过。** BoN@16 与 C0 基本相同且 seed 不稳定；不要直接扩跑 full。
6. **CH0 未通过组合门。** 它低于 C0/C1/H0，二因子交互为 `-1.60` points；把它作为扩量后
   完整 2×2 复测的预警信号，不写成两个模块天然冲突。
7. **暂不增加 epoch。** 优先增加独立监督和 ranking query 数，并统一 checker；如果要归因
   encoder，还需补 strict/encoded SWIFT 等预算 baseline。只有新数据下预注册的 dev 曲线
   支持时，才重新决定 3→5 epoch。

复现排名汇总使用 `summarize_clir_ablation.py`，它会逐行验证 21 份 scored JSONL 的
candidate identity/order/label、metric input hash 和 checkpoint hash，并输出固定 seed 的
paired-query bootstrap 及探索性 seed+query hierarchical bootstrap。机制指标使用
`evaluate_clir_mechanisms.py`；完整本地产物位于
`run_artifacts/clean_ablation_v1/`，其中 `paired_summary.json` 是原 21-run 报告，
`paired_summary_ch0.json` 与 `ch0_interaction.json` 是 CH0 对比和二因子交互报告。
