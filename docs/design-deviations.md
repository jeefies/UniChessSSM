# 设计偏差与改进审计日志 (Audited Changelog)

> **状态**：已完全审核并合并 (Merged into `docs/state-sequence-model-design.md` v3.0)  
> **更新日期**：2026-09-21  
> **定位说明**：本文档作为系统设计演进过程中的**历史审查记录与变更审计索引**保留。文档中记录的所有经确认的技术偏离、漏洞修复与离线增强，均已作为真实架构与实现规范**合入 `docs/state-sequence-model-design.md`（UniChess SSM 系统设计与框架说明书 v3.0）**。

---

## 1. 变更审计与系统说明书映射表

下表列出系统自 v2.0 原案演进至当前 v3.0 真实代码实现过程中确认的所有核心设计调整及其在系统设计说明书中的对应章节：

| 变更项 | 涉及文件与模块 | 原始设计或历史状态 | 最终实现规范与裁定口径 | 系统设计说明书对应节 |
|---|---|---|---|---|
| **785 维精确无损特征** | `stateseq/features.py` | 早期文档描述为"约 790 维" | 精确定义为 785 维（12×64 棋子 + 17 全局位），白方绝对坐标 | §2.1 |
| **零分配高速位棋盘提取** | `stateseq/features.py` | Python 字典动态遍历与堆分配 | `encode_board_fast` 位棋盘直接解包，延迟 $49.9\mu\text{s} \to 8.9\mu\text{s}$ | §2.1 |
| **条件嵌入逐步对齐** | `stateseq/conditions.py` | 曾存在首步条件静态绑定错误 | `color` 与行棋方按步动态绑定，整序列 vs 递推严格对齐 | §2.3 |
| **模块命名与拓扑** | `model_d.py` / `model_g.py` | 早期设计名为 `aux.py` | 避开 Windows 保留名，明确独立模块命名与参数预算 | §3 |
| **整序列 vs 逐步递推界** | `stateseq/model_r.py` | 早期假定 raw logits 差 $<1\text{e-}4$ | Mamba-2 并行 Scan 硬件浮点噪声，口径改为 Softmax 概率差 $<1\text{e-}4$ | §3.2 |
| **非法着法数值掩码** | `stateseq/heads.py` | 理论公式写 $-\infty$ | 严格采用有限大负数 $-3\times 10^4$，防 $0 \times (-\infty) = \mathrm{NaN}$ | §3.3 |
| **动力学双侧 Stop-Gradient** | `stateseq/losses.py` | 仅单侧截断存在坍缩风险 | 目标增量项双侧双重截断，答案侧绝对保护 | §3.4 |
| **Moves-Left Log-Huber 扩展** | `stateseq/losses.py` | 原始绝对步数 Huber 损失 | 增加可选 `--mlh-log` 变换，有效缓解开局残差对主干表征的劫持 | §4.1 |
| **逐节点 Completed-Q 归一化** | `stateseq/gumbel.py` | 早期使用全树全局 qbox | 改为当前节点合法动作局部 min-max 归一化，三处共用唯一入口 | §5.2 |
| **价值缩放常数 $c_{\text{scale}}=0.1$** | `stateseq/gumbel.py` | 早期默认沿用 $c_{\text{scale}}=1.0$ | 实测与机制对照证明 1.0 导致伪确定性坍缩，锁定改为 0.1 | §5.1, §5.2 |
| **终局裁决唯一权威入口** | `stateseq/adapter.py` | 严格判定与申和判定差一 ply | 统一收口至 `adapter.classify_final_board`，准确区分截断与正常申和 | §7.2 |
| **开局序列注入支持** | `tools/ssm_gumbel_selfplay.py` | 纯噪声冷启动 | 支持 `--openings` 预设特级大师局面注入，提升兵形信息覆盖 | §8 |
| **换代 Arena Wald SPRT 早停** | `tools/ssm_gumbel_arena.py` | 必须固定跑满 400 局 | 支持 `--sprt` 快速淘汰明显落后候选（Fail-Fast），节省无效算力 | §8 |
| **数据可用性定级** | `data/` 早期分片 | 早期分片视为完全可用 | `pi_prime` 降级为 `legacy_teacher`，仅 `actions` 与 $z$ 标签可用 | §9.2 |
| **对局/生成管线迁移到 UniChessKit（P3）** | `stateseq/kit_adapter.py`、`tools/ssm_gumbel_{arena,selfplay}.py` | 两个工具各自实现对局循环、跨局拼批、多进程 | 对局循环/拼批/多进程/续跑/裁决由 kit 负责，S 只提供 Player（ReplayStore）与 `V3Sink`；切换前并发 1 下逐位对照一致（§2.4） | §8 |
| **换代 Arena SPRT 改为五项式 GSPRT** | `tools/ssm_gumbel_arena.py` | 逐局伯努利 LLR（p 0.50 vs 0.55），只在拒绝 H1 时停 | kit 逐对五项式 GSPRT，H0 Elo 0 vs H1 Elo 35（≈p 0.55），接受或拒绝都停（§2.4） | §8 |
| **自对弈多进程按全局局号分段** | `tools/ssm_gumbel_selfplay.py` | 每 worker 局号从 0 起、种子由 `SeedSequence(seed).spawn(workers)` 派生 | 各 worker 取不相交全局局号区间（`--first-game`）、同一 seed；修复多 worker 下 `game_key` 重复（§2.4） | §8 |

---

## 2. 详细审查历史归档（保留关键实验证据与上下文）

### 2.1 §10.1 验收断言口径与浮点噪声界定
- **实测**：官方 `mamba_ssm` kernel 并行 scan 与单步 decode 两条数值路径存在固有浮点噪声（每层 ~1e-5，12 层 + policy 头放大后 raw logits 差 ~2e-4，且 T=4 即达 1.1e-4，非递推累积、非实现错误——逐步路径重跑 diff=0.0，同输入喂两条 R 路径 diff~3e-4）。
- **现行权威口径**：policy 断 **softmax 概率差 < 1e-4**（实测 ~1e-5，推理实际消费的是 masked softmax 概率）；WDL / moves-left 维持 logits < 1e-4；policy raw logits 差作诊断指标随冒烟持续记录。

### 2.2 终局裁决口径修复与真实数据分布（对应修复 commit `dd620d5`，历史代号 `784dc64`）
- **根因**：生成器早期使用严格判定 `is_repetition(3)`，与对局循环中 `is_game_over(claim_draw=True)` 差 1 ply。导致大量规则申和局被误记为 300 ply 封顶截断。
- **重放审计结果**：
  - `stage_b_gen2k` 真实封顶率实测仅 **15.9%**（原记录 54.2%）；
  - 终局因果修复前后，所有局的终局价值 $z$ 修正数为 0（和棋本质不变）；
  - 经 `tools/repair_v3_meta.py` 修复元数据后，`mlh_valid` 训练有效局从 46% 恢复至 84.1%。

### 2.3 Completed-Q 局部归一化与 $c_{\text{scale}}=0.1$ 对照
- **归一化量程缺陷**：若使用全树全局 qbox，其他深层节点的极端 Q 值会撑大分母 span，严重压缩局部细微价值差在策略打分中的权重。改为逐节点 completed-Q 归一化后，彻底消除了量程外泄。
- **尺度实验证据**：128 局面机制对照显示，在 $c_{\text{scale}}=1.0$ 下 $\text{KL}(\pi' \parallel \pi)$ 高达 2.195，60.3% 局面目标退化为近乎确定选择；而在 $c_{\text{scale}}=0.1$ 下目标熵健康恢复至 1.255。32 局同权重对抗中，0.1 尺度以 21:11（胜率 65.6%）大幅战胜 1.0 尺度。

### 2.4 P3：对局与自对弈管线迁移到 UniChessKit（2026-09-23）
- **结构**：`tools/ssm_gumbel_arena.py` / `tools/ssm_gumbel_selfplay.py` 改为 kit `run_match` / `run_selfplay` 的薄封装，CLI 参数与输出文件（`arena.json`/`games.jsonl`/`model_ids.json`、v3 分片 + manifest）保持兼容。S 侧只剩 `stateseq/kit_adapter.py`：`SsmEvaluator`（批量 `SeqModel.step`）、`SsmExpander`（ReplayStore：叶子从根 cache 重放路径）、`SsmPlayer`（懒追赶 + Gumbel，arena g=0）、`SsmSelfPlayer` + `V3Sink`。
- **切换判据（均已满足）**：并发 1 下与原实现逐位对照——arena 真实权重 gen2 vs gen3 逐局 `_pgn_fingerprint` 一致（8/8、16/16）；自对弈真实权重 gen3 8 局 598 ply 的 v3 分片逐字节一致，book π′ 缓存命中、预算违例、扩展深度直方图、节点/模拟/深度计数全部相同，用时持平。证据脚本在提交 `93f1ca8`（切换后删除，git 历史保留）。
- **只在并发 1 下逐位**：`SeqModel.step` 随批大小有 ~1e-5 浮点差，拼批组成不同时 argmax 可能翻转；并发 > 1 只验证结构与统计口径，这与原实现相同。
- **前向数略少**：懒追赶省掉最后一步非行棋方的 1 次步进；kit 的 Gumbel 在调 Expander 之前判终局，终局叶子不再重放路径（原实现先重放 d−1 步才发现终局）。着法与 π′ 不受影响。
- **ply 上限口径**：arena 的 `--max_plies` 仍只计开局之后（kit `MatchConfig.max_plies_after_opening=True`）；自对弈的 `--max_plies` 计整局（含 book ply），与原生成器相同。
- **SPRT 方法变更**：原实现是逐局伯努利 LLR（p₀=0.50 vs p₁=0.55）、只在拒绝 H1 时早停；现用 kit 逐对五项式 GSPRT（H0 Elo 0 vs H1 Elo 35，α=β=0.05，`--sprt-min-games` 折算成对数），接受或拒绝都会停，并且考虑了配对开局的相关性。换代门槛本身（400 局 ≥55%）不变。
- **自对弈多进程**：原实现每个 worker 局号从 0 起、种子由 `SeedSequence(seed).spawn(workers)` 派生，多 worker 时 `game_key` 重复（只影响 train/val 划分）。现各 worker 取不相交的全局局号区间、同一 seed，每局的 rng / book 分配 / `game_key` 只取决于全局局号，与 worker 数无关（`tests/test_kit_selfplay.py` 拆段 == 整段）。**同 seed 的多 worker 生成结果因此与旧版不同**；单 worker 结果不变。
- **开局文件**：由 kit `OpeningBook` 解析（SAN/UCI 均可、去重、不裁切）；非法行改为直接报错，不再静默跳过（静默跳过会悄悄缩小开局库）。
- **arena 断点续跑**：逐局结果写 `kit_results.jsonl`，同一命令重跑会核对配置哈希后续跑；逐局扩展深度直方图另存 `expand_hist.jsonl` 以便续跑读回。出错整批停止（kit 语义），`anomaly` 字段保留恒为 None。
