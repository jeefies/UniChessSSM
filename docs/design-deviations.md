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
