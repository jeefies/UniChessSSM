# UniChessSSM

UniChess 新架构：状态序列模型（格子级 Transformer E → 12 层 Mamba R → policy/WDL/moves-left 三头），
训练期辅助 D（重建）/g（残差动力学）。Lichess 人类棋谱行为克隆预热，明确不用 Stockfish 蒸馏。

- 权威设计文档：`docs/state-sequence-model-design.md`（D1–D10 已锁定）
- 开发约定与双设备路由：见 [AGENTS.md](AGENTS.md)
- 当前阶段：**Stage A · 人类棋谱 BC 预训练进行中**（run `runs/stage_a_20260915`，1 epoch = 37,758 步）
- 实验报告（审查用，含管线/公式/流程图/曲线）：[docs/stage-a-experiment.md](docs/stage-a-experiment.md)

## 快速开始（远端 5070 Ti）

```bash
cd /home/jeefy/UniChess/SSM
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
```
