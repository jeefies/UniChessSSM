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
