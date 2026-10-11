# UniChessSSM

UniChess 新架构：状态序列模型（格子级 Transformer E → 12 层 Mamba R → policy/WDL/moves-left 三头），
训练期辅助 D（重建）/ g（残差动力学）。Lichess 人类棋谱行为克隆预热，明确不用 Stockfish 蒸馏。

**扁平布局（2026-09-25）**：仓库根目录即包（`import SSM`），import 根是 `~/UniChess`；
训练、搜索、自对弈、arena 都在 `Kit/`，本仓库只剩网络 + 前端 + kit 接入 + Server 插件。
旧 `stateseq/` 包、`tools/`、`train/` 已删除，git 历史可查。

- 权威系统设计与框架说明书：`docs/state-sequence-model-design.md`（v3.0，D1–D10 已锁定）
- 开发约定、对拍纪律与设备/连接路由（单远端，域名 `ulab.jeefy.top` 优先）：见 [AGENTS.md](AGENTS.md)
- 训练配置：`configs/stage_a.json`（人类棋谱预热）、`configs/stage_b2.json`（人类 + 自对弈混合）
- 实验记录与战报：`docs/stage-a-experiment.md`（Stage A 预热）、`docs/stage-b-experiment.md`（Stage B 自对弈换代大循环与三元联盟）、`docs/elo-degradation-analysis.md`（诊断报告）

## 当前 Champion 霸主与最新进展
- **现役 Champion**：`runs/champion.pt`（**Champion gen4**，Round 4 胜出，对前任霸主 gen3 在 508 局成对换先对决中取得 55.51% 胜率，五项式 GSPRT $H_1$ 早停胜出，将杀率超 90.9%）；
- **强化学习循环（Gen 10+）**：引入 4 代滑动窗口 Replay Buffer（12,000 局）、三元多模型博弈联盟生成器（50% 自博弈 + 25% 历史最强 + 25% 联盟轮换）与基于负对数似然自信息的惊讶度重要性采样（Surprise-based PER），在 RTX 5070 Ti 自动化持续迭代中。

## 快速开始（远端 5070 Ti）

```bash
cd ~/UniChess/SSM
export UNICHESS_IMPORT_ROOT=~/UniChess
# 单测验证
/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests
# 运行自动化强化大循环
bash tools/start_loop.sh
```

对局 / 观战走 kit 原生 Player：

```bash
cd ~/UniChess
python -m Kit match <config.json> --out runs/<name>/results.jsonl
```

