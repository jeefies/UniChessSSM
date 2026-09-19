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

关键结论：**`result`（z 标签）修正数为 0** ——申和局本就记和，动作序列 / π′ / z 全部正确，
因此**不需要重跑 9.3 小时的自对弈生成**，只需就地重算 meta 三字段
（`tools/repair_v3_meta.py`，已对以上三个目录执行）。

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

1. **q̂ 归一化的视角混用**（`gumbel.py`）：`qbox` 以 `root.q`（父视角）起算、用 `child.q`
   （子视角）扩展，再去归一化父视角的 `completed_q`。论文为每节点独立归一。在线探针
   28,540 次调用实测：span 中位 1.83（混用是**放大**而非压缩量程），q̂ 越界 [0,1] 仅 4.6%
   （范围 −1.68…1.86），σ 展幅 p50 1.2 / p90 15.8 / max 114。不发散，暂记为偏离与风险项，
   不触发重跑；后续可改为逐节点归一并做 A/B。
2. **来源权重未归一**：自对弈 0.85 + 人类 0.10 = 0.95（谜题 0.05 未接线），等效缩放了学习率 5%。
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
