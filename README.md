# UniChessSSM

UniChess 新架构：状态序列模型（格子级 Transformer E → 12 层 Mamba R → policy/WDL/moves-left 三头），
训练期辅助 D（重建）/g（残差动力学）。Lichess 人类棋谱行为克隆预热，明确不用 Stockfish 蒸馏。

- 权威设计文档：`../UniChess/docs/state-sequence-model-design.md`（D1–D10 已锁定）
- 开发约定与双设备路由：见 [AGENTS.md](AGENTS.md)
- 当前阶段：**阶段 0 · 接口验证**（七项验收清单见设计文档 §10.1）

## 快速开始（远端 5070 Ti）

```bash
cd /home/jeefy/UniChessSSM
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v
```
