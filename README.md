# UniChessSSM

UniChess 新架构：状态序列模型（格子级 Transformer E → 12 层 Mamba R → policy/WDL/moves-left 三头），
训练期辅助 D（重建）/ g（残差动力学）。Lichess 人类棋谱行为克隆预热，明确不用 Stockfish 蒸馏。

**扁平布局（2026-09-25）**：仓库根目录即包（`import SSM`），import 根是 `~/UniChess`；
训练、搜索、自对弈、arena 都在 `Kit/`，本仓库只剩网络 + 前端 + kit 接入 + Server 插件。
旧 `stateseq/` 包、`tools/`、`train/` 已删除，git 历史可查。

- 权威系统设计与框架说明书：`docs/state-sequence-model-design.md`（v3.0，D1–D10 已锁定）
- 开发约定、对拍纪律与双设备路由：见 [AGENTS.md](AGENTS.md)
- 训练配置：`configs/stage_a.json`（人类棋谱预热）、`configs/stage_b2.json`（人类 + 自对弈混合）

## 快速开始（远端 5070 Ti）

```bash
cd ~/UniChess/SSM
export UNICHESS_IMPORT_ROOT=~/UniChess
# 单测（123 项，5 项按环境跳过）
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests
# 训练
cd ~/UniChess && python -m Kit train SSM/configs/stage_a.json
```

对局 / 观战走 kit 原生 Player：

```bash
cd ~/UniChess
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

实验报告（审查用，含管线/公式/流程图/曲线）：`docs/stage-a-experiment.md`。
