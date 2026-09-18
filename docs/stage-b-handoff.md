# Stage B 阶段① 交接指令（给实现 agent）

> 你的任务：**只做阶段① —— 实现与算法单测**。不要生成正式训练数据，不要提前进入阶段②。
> 唯一规格来源：`docs/stage-b-implementation.md`（下称"规格"）。本文件只做导航，与规格冲突时以规格为准。

## 0. 环境

- 项目根：`C:\Users\jeefy\Documents\ChatGPT\UniChessSSM`（纯 Windows，**无 WSL 依赖**，所有命令在 Windows 侧执行）。
- 训练/推理入口、模型定义、Stage A checkpoint 位置见规格与 `docs/stage-a-experiment.md`。

## 1. 背景 30 秒版

- Stage A（人类棋谱监督预训练）已完成：28M 状态序列模型（Recurrent-Transformer 局面编码器 E + Mamba 骨干 R + 解码头 g/ori/dec），终值 policy CE 1.864、Top-1 44.5%，实测棋力约 Elo 400，相对 champion −880。
- B0 审计（`docs/stage-a-experiment.md` §10）结论：链路无 bug，弱在网络本身 → 进入自对弈强化。
- 路线（规格 §0.2）：**阶段① 实现+单测 → ② 首轮 2k–5k 局 Gumbel 闭环 → ③ 多代主循环**；B1 是条件触发的后备，**默认不启用**。

## 2. 必读文档（按序）

1. `docs/stage-sequence-model-design.md` —— 架构定义，**D1–D10【锁定】不可改**。
2. `docs/stage-b-implementation.md` —— 全文，重点 §2.2（Gumbel 精确规格）、§2.3（树与 R cache）、§2.4（生成器）、§2.5（v3 分片）、§2.6（训练器）、§2.8（门禁）。

## 3. 交付物

| # | 交付 | 规格依据 |
|---|---|---|
| 1 | `tools/ssm_gumbel_selfplay.py` 原生 Gumbel 树生成器（含 learner/champion 分离、根部快照 + 独占工作 cache） | §2.1–§2.4 |
| 2 | v3 分片读写（变长目标 + 终局元数据） | §2.5 |
| 3 | `train/stage_b2.py`（改自 `train/stage_a.py`：遍历预算、软 CE −3e4 数值安全、全 300 ply） | §2.6 |
| 4 | A 组 7 项算法单测全部通过（对拍脚手架复用 `tools/ssm_path_audit.py`） | §2.8 A 组 |
| 5 | B 组冒烟：1,000 局生成 + ≤3 遍训练，管道健康 | §2.8 B 组 |
| 6 | 规格 §4 实测回填清单逐项填回文档 | §4 |

## 4. 硬约束（违反即返工）

- **A 组单测全过之前，禁止生成任何正式训练数据。**
- 旧 MCTS-400 链路**禁止**用于数据生成（O(T²)，实测 0.63s/步），仅作 B0 对齐基线。
- 条件输入锁定（规格 §0.2）：自对弈 Elo=2567.5 + RAPID 桶；人类混批保留原始条件；谜题固定 2567.5 + unknown 桶。
- 超参以规格 §3 锁定表为准；任何偏离先停下来报告，不得自行调整后继续，确认的变更记入规格 §5。
- 架构 D1–D10 冻结；不做消融实验。

## 5. 完成定义（DoD）

A 组 7 项 + B 组冒烟全绿，§4 回填完毕，代码与单测可复现（固定随机种子），向作者汇报：单测结果、吞吐实测、遇到的规格歧义清单。
