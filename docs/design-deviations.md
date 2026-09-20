# 与权威设计文档的偏差记录

> 权威设计文档：`state-sequence-model-design.md`（v2.0，已迁入本文档同目录）。本文件只记录实现时经文档作者确认的偏差，权威文档原文不动。

## 1. §10.1 验收 #3 的断言口径（2026-09-14，作者确认：方案 A）

- **原文**：整序列 vs 逐步递推的 logits 差 < 1e-4（fp32）。
- **实测**：官方 `mamba_ssm` kernel 并行 scan 与单步 decode 两条数值路径存在固有浮点噪声（每层 ~1e-5，12 层 + policy 头放大后 raw logits 差 ~2e-4，且 T=4 即达 1.1e-4，非递推累积、非实现错误——逐步路径重跑 diff=0.0，同输入喂两条 R 路径 diff~3e-4）。
- **现口径**：policy 断 **softmax 概率差 < 1e-4**（实测 ~1e-5，推理实际消费的是 masked softmax 概率）；WDL / moves-left 维持 logits < 1e-4；policy raw logits 差（~2e-4）作诊断指标随冒烟持续记录。真实现法 bug 的差在 2e0 量级（曾抓出 color 条件 bug，diff=2.22），任何 1e-3 门槛均可捕获。

## 2. 实现层记录（非文档偏差）

- `aux.py` 更名为 `model_d.py` / `model_g.py`（`aux` 是 Windows 保留设备名，任何扩展名均不可创建）。
- §5.6 g 侧枝"MLP 1024→1024→512，残差"按同维首层残差实现：`h = z + GELU(W₁z)`（1024→1024），再 `W₂`（1024→512）；predictor 为 512→512 两层。
- 参数实测 26.40M（预算表 ~27M）：E 1.14M / R 19.25M / f 1.52M / D 1.39M / g+predictor 3.09M（g 超预算 1.6M 主因 emb_a 1936×512≈0.99M，总量仍在 D4 的 ~28M 内）。
- §3.2 特征实际 785 维（文档称"约 790"）：12×64 棋子 + 走子方 1 + 易位 4 + 过路兵 8 + 半回合 1 + 全回合 1 + 重复 2 = 785。

## 3. 监控指标记录口径问题（2026-09-15 代码核查确认，训练器下轮运行修复，不影响已训模型）

- **train metrics 只记录 accum 窗口最后一个 microbatch**（`stage_a.py:174` 循环内覆盖 `pending_metrics`），不是 512 局窗口均值 → 单点波动大，绘图标题取末点曾出现 value CE 0.552 的误导读数（平滑/验证口径 ~0.8/0.85）。修复：改为窗口内汇总（sum/count）。
- **`dyn_rel_err` 指标分子/分母未加 pos_mask**（`losses.py:120-121`），padding 位计入（delta≈0 主要污染分母）→ 记录的 rel 系统性偏低（@27000 记录 0.48，正确掩码口径 0.26）。dyn loss 本身是 masked 的，仅诊断指标口径有瑕。
- **grad norm 记录于裁剪之后**（clip 在前 `stage_a.py:175`、gn 计算在后 :184），clip 前总范数与触发率未记录 → 若裁剪频繁触发，gn 曲线被压平丢失信号。修复：记 clip 前范数 + 触发比例。
- **`val_loss_total` 中 w_r 恒为 1.0**（验证时 step=0 传入退火函数），验证 total 不反映当期退火权重；各任务分量不受影响。

## 4. 数据事实核查（2026-09-15，排除标签 bug）

- value 诊断发现训练/验证和棋率仅 5.9%/3.9%，一度怀疑 PGN Result 解析错误；实测原始 PGN 前 10 万局标签和棋率 3.7%，v2 分片全量 result 分布（白胜 49.67% / 和 3.85% / 黑胜 46.47%）与之一致 → 低和棋率为 Lichess 超快棋池的真实分布，标签可信。
- >200 ply 的局重放截断不影响"距真实终局步数"离线计算（分片 meta `n_plies` 为未截断总局数），但 moves_left 训练标签对这类局在 200 处饱和（已知设计，重放侧截断）。

## 5. Arena 逐局诊断输出（2026-09-18，review.txt §2 响应）

- **审查意见**：64/64 和棋未经终止原因分析，A/A 对称性不能发现"全记和"bug，权重加载未验证。
- **实现变更**：
  - `tools/ssm_gumbel_arena.py`：输出 games.jsonl（逐局 PGN/ply/termination/is_truncated/board_result/anomaly）+ model_ids.json（参数哈希 + 前向输出比较）+ `--test-scoring` 模式
  - `tools/eval_dual_checkpoint.py`：新增双检查点统一验证集评估工具
  - `tools/test_multi_gen_control_flow.py`：多代替代控制流单测
  - `tools/trace_pipol.py`：Q→σ→π′ 数值轨迹审计工具
- **更新后的 Arena 流程**：每局含诊断 → 逐行 JSON → 聚合统计写 arena.json → 联邦哈希验证 → 记分正向测试
- **不涉及超参/规格改动，无需设计文档变更。**

## 6. Review 2026-09-18/19 代码审查修复（commit 8d14a58）

### 6.1 审查范围与方法
对 `6c8bc62..HEAD`（4 个 commit，10 文件，862 行变更）进行完整源码审计，对照 `docs/state-sequence-model-design.md` D1–D10 及 `docs/stage-b-implementation.md` §2–§3 锁定规格。

### 6.2 发现与修复

| # | 严重度 | 文件 | 问题 | 修复 |
|---|---|---|---|---|
| 1 | CRITICAL | `ssm_gumbel_selfplay.py:211` | `occ_root` 未定义 NameError，生成器完全不可运行 | 改为 `occ` |
| 2 | CRITICAL | `ssm_gumbel_arena.py:295-314` | 换色局胜者归属双重映射，32 局 A/B 结论 12/20 颠倒 | 直接用 `arena_result` 计数 |
| 3 | WARNING | `ssm_gumbel_selfplay.py:204-206,250-252` | occurrence 编码时机根节点/展开路径不一致 | 展开路径改为 encode-before-increment |
| 4 | WARNING | `ssm_gumbel_selfplay.py:205` | `features = encode(...)` 死代码 | 删除 |
| 5 | WARNING | `ssm_gumbel_arena.py:247` | `if False else` 死代码 | 删除 |
| 6 | WARNING | `adapter.py:22-23` | `ELO_MEAN/ELO_STD` 重复定义，未从 `conditions` 导入 | 移除未使用导入，补充注释 |
| 7 | WARNING | `model.py:111` | dyn 索引语义变更未标注 Stage A checkpoint 兼容性 | 补充注释 |

### 6.3 数据正确性核查

- **v3 pipol 格式**：与规格 §2.5 完全一致（`u16 count + count×(u16 action_id + f16 prob)`）
- **v3 meta 字段**：generator 写入完整；`dataset_selfplay.py` 正确读取 `elo_mean`→`elo_std`；`mlh_valid` 正确剔除截断局
- **一条格式偏离**：v3 meta 实际存储为 `.meta.npz`（56B/局 numpy structured array），规格描述为 `.meta.bin`（32B 原始二进制）；当前无 C++ 读取器，不构成互操作 bug，但需在 C++ 读取器实现时对齐

### 6.4 验证结果（修复后重跑）

| 测试 | 修复前 | 修复后 |
|---|---|---|
| A/B 32 局（Stage A vs Round 2） | **12/20**（不可信，计分 bug） | **28/4**（Stage A 87.5%，可信） |
| A/A 4 局 | 2/2 | 2/2（50%，对称） |
| 计分正向测试 | 4/4 PASS | 4/4 PASS |
| 单测 48 项 | 48/48 PASS | 48/48 PASS |

**不涉及模型架构或超参变动。**

## 7. Review 2026-09-19 第二轮：终局裁决口径 bug（commit 784dc64）

### 7.1 审查范围与方法
对 `gumbel.py` / `ssm_gumbel_selfplay.py` / `ssm_gumbel_arena.py` / `adapter.py` /
`gshards.py` / `losses.py` / `dataset*.py` / `model*.py` / `train/stage_b2.py` 做规格–代码
逐条比对，并在远端 GPU 上做两项实测验证（分片重放审计、σ 归一化在线探针）。

### 7.2 发现与修复

| # | 严重度 | 位置 | 问题 | 修复 |
|---|---|---|---|---|
| A | CRITICAL | `ssm_gumbel_selfplay.py::GameState.result` | 分类用**严格**判定 `is_repetition(3)`/`is_fifty_moves()`，而对局循环用 `is_game_over(claim_draw=True)` 退出——后者含"下一着可申和"，早一 ply。规则申和局全部掉进兜底分支被记成"300 ply 封顶截断" | 新增 `adapter.classify_final_board` 为唯一裁决入口，生成器 + arena 共用 |
| B | CRITICAL | `ssm_gumbel_arena.py::play_one_game` | 开局库写成 `push_san → _advance`，初始局面 B₀ 从未入 R，最后一个开局局面被重复步进两次且 occurrence 多记一次 | 改为 encode-before-move（`_advance → push_san`） |
| C | WARNING | `ssm_gumbel_selfplay.py::_play_ply` | `_resolve_move` 返回 None 时静默 `return False`，会把编解码损坏伪装成正常终局 | 改为 `raise RuntimeError` |

### 7.3 实测验证（bug A 的影响面）

重放全部分片动作序列、按 `claim_draw=True` 口径重算：

| 数据集 | 局数 | 封顶率（修复前记录） | 封顶率（实际） | reason 修正 | **result 修正** |
|---|---|---|---|---|---|
| `stage_b_gen2k` | 2,000 | 54.2% | **15.9%** | 766 | **0** |
| `stage_b_gen_round2` | 2,500 | 53.0% | **16.8%** | 906 | **0** |
| `stage_b_val64` | 256 | 50.8% | **11.3%** | 101 | **0** |

关键结论：**`result`（z 标签）修正数为 0**——申和局本就记和，因此**不需要重跑 9.3 小时的
自对弈生成**，只需就地重算 meta 三字段（`tools/repair_v3_meta.py`，已对以上三个目录执行）。

> **⚠️ 边界更正（§8.3）**：原文此处写"动作序列 / π′ / z 全部正确"是**过度推论**。
> `result 修正数 = 0` 只证明 **z 在本次重新裁决前后相同**，不证明生成当时的**搜索目标**正确。
> π′ 的可用性取决于**生成时的搜索版本**，而非本次 meta 修复——经核查，全部现有分片均
> 生成于 adapter 修复之前，π′ 一律定级为 `legacy_teacher`，详见 §8.3。

修复后真实终局分布（gen2k）：checkmate 33.2% / threefold 38.0% / truncated 15.9% /
stalemate 7.2% / insufficient_material 5.4% / fifty_move 0.3%。
**三次重复是仅次于将杀的第二大终局原因**，而非此前记录的"0 局"。

### 7.4 连带失效的既有结论

- `docs/stage-b-experiment.md` §4.1「记录 vs 实际终止不匹配 0 局」「实际五十步和棋 0 / 实际三次重复 0」——**错误**。
- 同文 §8「封顶问题专题」：其假设表以"五十步=0、三次重复=0"为由否决了「重复/五十步绕行导致封顶」，
  该否决的前提不成立，整节分析作废。真实封顶率仅 ~16%。
- `mlh_valid_mask` 此前错误剔除 ~54% 的对局；修复后 gen2k 有效局从 46% 升至 **84.1%**，
  round2 升至 **83.2%**。moves-left 头的有效训练样本翻倍。
- Arena A/A 16/16 与 A/B 28/4 因 bug B 失效，已重跑（见 §7.6）。

### 7.5 记录在案但未改动的偏离

1. ~~**q̂ 归一化的视角混用**~~ → **已于本轮修复，见 §8.1**。原记录的放行理由（"不发散、
   越界仅 4.6%"）**不成立**：判据应是它是否改变 ℓ 与 Q 的相对权重，而不是有无 NaN。
2. **来源权重未归一**：自对弈 0.85 + 人类 0.10 = 0.95（谜题 0.05 未接线）。
   ~~等效缩放了学习率 5%~~ ——**该解释错误**，见 §8.2。
3. **π′ 熵与文档不符**：`docs/stage-b-experiment.md` §4.3 记 H(π′)=0.0605、75% 硬目标，
   §11.3 记 1.94。当前 gen2k 实测 **均值 1.2644 / 中位 1.4447 / 零熵 8.1% / max_prob>0.999 占 8.5%**，
   §4.3 的数字不描述当前数据，已在实验文档更正。

### 7.6 回归与重跑

- 新增 `tests/test_termination_classify.py`（6 项）：可申和三次重复 → `threefold` 且非截断、
  未终局 → `truncated`、将杀 result 白视角映射、逼和/子力不足、生成器 `result()` 同口径、
  arena 开局每个局面恰好入 R 一次。
- 全量单测 **70/70 PASS**（A 组门禁满足）。
- Arena A/A 与 A/B 按修复后代码重跑（`runs/arena_aa_round4` / `runs/arena_ab_round4`）：

| 对局 | 修复前 | **修复后** |
|---|---|---|
| A/A（Stage A vs 自身，4 局） | 2/2，50%，4 checkmate | **2/2，50%，4 checkmate，0 anomaly** |
| A/B（Stage A vs Round 2，32 局） | 28/0/4，**87.5%**，218s | **22/2/8，71.9%，1,762s，30 checkmate + 2 子力不足** |

Stage A 强于 Round 2 的方向不变（Round 2 不应晋级），但优势幅度从 87.5% 回落到 71.9%
——此前被错误的开局历史放大。用时 8× 增长与非将杀终局的出现，是"历史正确后棋力表现
更合理"的旁证。32 局置信区间仍宽（±~16pp），正式换代须按 §2.8 C 组跑 400 局。

**不涉及模型架构或超参变动。**

## 8. Review 2026-09-19 第三轮：Q 归一化收口与数据可用性定级（commit 154d673 起）

本轮按审查意见执行，**不因元数据错误重生成棋谱**；Stage A 继续担任 champion。

### 8.1 【规格变更】Q 归一化改为逐节点 completed-Q 量程

原规格（`stage-b-implementation.md` §2.2）写"q 按本树本次搜索的 min-max 统计归一"，
实现为一个全树 `qbox`，且该 qbox 以父视角 `root.q` 起算、用子视角 `child.q` 扩展。

**放行判据修正**：此前以"不发散、越界仅 4.6%"为由搁置是错的。关键在于量程出现在打分的分母上：

```
log(π′(a)/π′(b)) = ℓ(a) − ℓ(b) + α·(Q(a) − Q(b)) / span,   α = (c_visit + maxN)·c_scale
```

span 被其他节点撑大，会**压缩**价值差异在打分中的相对权重；"span 变大"不是安全信号而是风险信号。
纯数值反例（非实测）：Q=(0.05, 0.25)、ℓ=(0, −7)、α=60 时，
span=1.83 给出 −7+60×0.2/1.83 ≈ **−0.44**（选 a₀），span=0.20（本节点自身量程）给出
−7+60×0.2/0.20 = **+53**（选 a₁）——**全部归一值都在 [0,1] 内，选择却相反**。

**新规格**：在每个待选择节点，对该节点行棋方视角的全部合法动作 completed Q 取局部 min/max
归一。新增 `gumbel.qtransform_completed` 为**唯一** Q→打分变换，
**根评分（顺序减半淘汰）/ 非根选择 / π′ 导出三处共用**，保证同一节点上"选择依据"与
"监督目标"一致。对齐 mctx `qtransform_completed_by_mix_value`。

保留的约束：**原始价值尺度与零和回传不变**（`Node.q`、边统计、终局真值照旧，归一值不回传）；
**本轮不动 `c_scale`**，避免把归一化定义变更与缩放调整混在一起。
`order_halving` 的 `qmin/qmax` 降级为诊断输出，不再参与打分。

验证：新增 4 项回归测试（本节点量程、外部量程不泄漏、选择与目标共用同一变换、
零量程下 π′=π 不除零），全量单测 **74/74 PASS**。未跑 A/B——按审查意见，本项用固定局面
数值对照与既有算法测试收口即可。

### 8.2 来源权重 0.95：保留，但纠正解释

`L = 0.85·L_sp + 0.10·L_human` 是整体损失缩放。**它不等价于学习率乘 0.95**：
AdamW 的自适应分母会抵消梯度的统一缩放——若 g→cg，则 m→cm、v→c²v，
而 m/√v 不变。实际还受梯度裁剪、ε 与解耦 weight decay 干扰，更无法简单折算。

**决定**：本轮保留 0.85/0.10，不补偿 LR，不为凑满 1 强行接入谜题。训练器改为显式打印
`w_selfplay / w_human / w_puzzle / active_source_weight_sum=0.95`，并删除"等效学习率缩 5%"的错误说明。

### 8.3 【重要】数据可用性定级：全部现有分片的 π′ 均为 legacy_teacher

`result 修正数 = 0` 只证明 **z 在重新裁决前后相同**，**不证明搜索目标正确**。按生成时间与
git log 比对（分片无 commit 字段，属既有疏漏，已补 `manifest["provenance"]`）：

| 数据集 | 生成完成 | 生成时 HEAD | 教师状态 |
|---|---|---|---|
| `stage_b_smoke` | 2026-09-17 19:26 | pre-f1146ac | legacy_teacher（另含 504/1000 动作解析错误，已隔离） |
| `stage_b_val64` | 2026-09-17 21:23 | pre-f1146ac | legacy_teacher |
| `stage_b_gen2k` | 2026-09-18 02:59 | pre-f1146ac | legacy_teacher |
| `stage_b_gen_round2` | 2026-09-18 22:28 | pre-f1146ac | legacy_teacher |

**四个数据集全部生成于 `f1146ac`（2026-09-19 00:04，adapter 修复）之前**，当时的搜索路径：

1. `q = wdl[0] − wdl[2]` 直接作用于**原始 logits**（未过 softmax），量纲不是 [−1, 1]；
2. 条件输入使用**未标准化**的原始 Elo；
3. 终局叶子由**网络前向估值**，而非规则真值 `get_terminal_q`。

三项都进入 completedQ → σ → π′。**修 meta 不会重算当时的搜索目标。**

**处置**：文件全部保留，不重生成。`actions` / `result z` / 已修复的 termination meta **可用**；
`pi_prime` 标为**不可用作"修复后搜索的正确目标"**。若日后需要修正，可沿原完整历史
重新搜索、只重标策略目标（无需重下对局），但成本需实测，且重标不会把旧行为下得到的 z
自动变成新策略下的结果。

### 8.4 meta 修复的可追溯性（含一处疏漏）

- **疏漏**：修复时 `.meta.npz` 被原子替换，**未保留原始副本**。缓解：旧值可确定性重建——
  旧分类 = 严格判定链（checkmate→stalemate→fifty→repetition(3)→insufficient→否则 truncated），
  对同一动作序列重放即可复现（`tools/verify_meta_repair_smoke.py::legacy_is_truncated`）。
- **影响范围已核验**：`*.actions.bin` / `*.pipol.bin` / `*.pipol.offsets.bin` 的 mtime 保持在生成日
  （09-17/09-18），仅 `*.meta.npz` 为修复日（09-19）；game ID 与 π′ 未变。

### 8.5 修复后训练冒烟（4/4 PASS，`tools/verify_meta_repair_smoke.py`）

目的不是再证明程序能跑，而是证明 **metadata 修复只改变了它应该改变的监督**。
固定种子、`dropout=0`、无 autocast。

| 检查 | 结果 |
|---|---|
| 1 相同权重同 batch，旧/新 meta 前向 | policy / value / recon / dyn **逐位一致**（4.52785015、2.37362790、0.04448702、0.12660037）；仅 mlh 变化 |
| 2 mlh_valid 全 False 的 batch | `loss_mlh = 0.000e+00`，finite，无除零 |
| 3 新恢复的申和样本 | 5/5 的 `n_plies` 分别为 115/138/145/206/203，均 <300，moves-left 指向真实终止 |
| 4 4 次真实优化器更新 | loss 12.23→8.14，裁剪前梯度范数 36.4→12.2，全部有限；checkpoint 保存/恢复 OK |

取样 8 局中 5 局被恢复，mlh 有效位 **423 → 1,230**。注意：
**`loss_mlh` 从 50.35 升到 57.00**——监督覆盖扩大不等于梯度按局数比例放大，
按有效位置平均时分母同时改变。**本轮维持 `w_mlh=0.1` 不动**，先观察真实贡献。

**边界**：修复 meta **不会修复已经完成的历史更新**。Round 1 / Round 2 仍是在旧有效位下
训练出来的检查点，可保留、可继续评测，但**不得重新标注为"使用修复后监督训练的模型"**。

### 8.6 截断率：局级与位置级并记

| 数据集 | 局级截断 | **位置级截断 ply** |
|---|---|---|
| `stage_b_gen2k` | 318/2,000 = 15.9% | 95,400/403,094 = **23.7%** |
| `stage_b_gen_round2` | 420/2,500 = 16.8% | 126,000/500,021 = **25.2%** |
| `stage_b_val64` | 29/256 = 11.3% | 8,700/49,423 = **17.6%** |

均未越过 §1.2 的 20% 调查线（局级），维持 300 ply 与现行裁决约定，**不加和棋惩罚、不提高 cap**。

### 8.7 保留为假设、未采纳为结论的两项

1. **"重复和多 ⇒ 终局兑现能力弱"仅是假设。** 三次重复只描述结束方式，不说明进入重复前
   孰优孰劣；均势主动求和、劣势成功逼和、优势错失胜机都会产生重复。
   **不据此修改和棋奖励**，`stage-b-experiment.md` §8.5 第 2 点相应降级为待验证项。
2. **Round 2 的退化幅度未定。** 32 局 22胜/2和/8负 → Stage A 71.875%、Round 2 28.125%。
   准确表述是"在这批开局和修复后的搜索设置下 Round 2 明显落后；具体幅度与泛化范围尚未确定"。
   **不晋级、保留为诊断检查点，暂不追加 400 局**——门禁 400 局要求的是**晋级证据**，
   不是给每个明显落后的候选都消耗同等评测预算。按本次 1,762s/32 局线性外推，
   400 局约需 **6.1 小时**，该预算更值得投入修复版短训与新候选筛查。
   正式 400 局时应对应 **200 个不同开局前缀 × 交换颜色各一局**，并按开局对分析；
   不可把少数开局重复跑到 400 局。±16pp 不是固定模板。

## 9. c_scale 1.0 vs 0.1 对照（2026-09-20，commit 8331930 起）：改用 0.1

§8.1 归一化收口时保留 `c_scale=1.0` 不动，本轮补做尺度本身的对照实验。

### 9.1 128 固定局面机制对照（`tools/compare_c_scale.py`，无对弈，只重跑搜索）

取 fix500 真实对局的 126 个前缀（opening/midgame/late 各 42），逐 ply 精确推进 R 到同一局面，
对每个位置分别用 `c_scale∈{1.0, 0.1}` × 2 个 Gumbel 噪声种子各跑一次完整搜索（`runs/c_scale_compare/`）：

| 指标 | c_scale=1.0 | c_scale=0.1 |
|---|---|---|
| 目标熵 mean / median | 0.238 / **0.0001** | 1.255 / 1.326 |
| 零熵占比 / max_prob>0.999 占比 | **60.3%** / 60.3% | 9.5% / 9.1% |
| KL(π′‖π) mean | **2.195**（> 原 policy 熵 1.811） | 0.747 |
| 打分差 mean/median | **18.2 / 11.2** | 1.91 / 1.19 |
| 原始 Q 差 mean（[−1,1] 量程） | 0.032 | 0.025 |
| 预算合规 | 100% | 100% |

[跨尺度] 同一局面两个尺度的选着一致率仅 **52.0%**（65/125，接近随机）；
[跨噪声] c_scale=1.0 下两个独立噪声种子的选着一致率 77.6%（97/125）。

**机制判断**：`α=(c_visit+maxN)·c_scale`，c_scale=1.0 时 α 常在 50~80 量级，把原始仅 ~0.03
的 Q 差（[−1,1] 量程里很小的价值分辨力）放大到 18 个 logit 的打分差，**KL(π′‖π) 反超原始
policy 熵本身**——即目标几乎丢弃了策略先验，60% 的局面退化为近乎确定选择。这正是权威文档
警告的"用不可靠的微小价值差异伪造确定性"失效模式的实测证据，不是理论担忧。
c_scale=0.1 下 α 降到 ~5~8，打分差与 ℓ 差同量级，policy 先验仍参与决策。

**熵高本身不作为判据**（避免把"更分散"直接等价于"更好"）；判据是价值分辨力与打分幅度是否匹配。

### 9.2 32 局同权重搜索对抗验证（`tools/ssm_gumbel_arena.py --c_scale_a 1.0 --c_scale_b 0.1`）

因 9.1 显示"确有明显决策差异"（跨尺度选着一致率仅 52%），按既定判据补做同权重（均为
`stage_a_20260915/best.pt`）、仅搜索尺度不同的对抗：16 组开局×交换颜色＝32 局，
`n_sims=64, m0=16`，`runs/arena_scale_cmp/`：

| | c_scale=1.0（A） | c_scale=0.1（B） |
|---|---|---|
| 胜局 | 11 | **21** |
| 胜率 | 34.4% | **65.6%** |

0 和棋、0 异常、0 截断，32 局全部将杀终局。**c_scale=0.1 明显更强**，与 9.1 的机制分析一致
（1.0 把噪声级的价值差异误判为决定性优势，实战中体现为更差的落子质量）。

### 9.3 决定

- **`c_scale` 由 1.0 改为 0.1**（`stage-b-implementation.md` §3 锁定表随附变更）。
- 此前用 `c_scale=1.0` 生成的 `stage_b_gen_fix500`（`current_teacher`，adapter 修复后但
  scale 未改）**判定不可用于短训**：其 π′ 目标处于 60% 近乎确定的退化状态，与新尺度下的
  搜索目标系统性不同，继承训练无意义。
- **`runs/stage_b_gen_fix500` 降级为诊断数据集，不进入短训**；用 `c_scale=0.1` 重新生成
  500 局教师数据（复用相同种子与开局分布，命名 `stage_b_gen_fix500_cs01`），完成后作为
  fix500 短训唯一数据源，`provenance.teacher_status` 标 `current_teacher`。
- **不涉及 D1–D10 架构冻结项**，`c_scale` 属 §3 超参锁定表条目，变更已按 §5 流程记入
  `stage-b-implementation.md` 并附本节证据。
