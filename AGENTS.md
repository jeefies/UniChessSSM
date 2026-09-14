# UniChessSSM AGENTS.md

> 本文件面向 AI 编码 agent。最后更新：2026-09-14。

## 1. 项目概述

UniChessSSM 是 UniChess 的**新架构独立项目**：状态序列模型（state-sequence model）——
每步局面显式编码为向量序列（白方绝对坐标，785 维无损特征），格子级 Transformer E（权重共享走两遍）
→ 基础 Mamba R（12 层）→ policy/WDL/moves-left 三头 f；训练期辅助模块 D（MLP 重建）与
g（残差动力学）不参与推理。预热用 Lichess 人类棋谱行为克隆（**不用 Stockfish 蒸馏**）。

**权威设计文档**：`../UniChess/docs/state-sequence-model-design.md`（v2.0）。
决策 D1–D10 已锁定，实现时不得偏离；不确定处回到文档作者（用户）确认。

与原项目 UniChess（ResNet 46M + MCTS + autoloop）**完全隔离**：本目录独立开发、独立数据、
独立 git 仓库；不得修改 `/home/jeefy/UniChess` 的任何文件或服务配置。

## 2. 双设备路由

| 设备 | 路径 | 用途 |
|---|---|---|
| Windows 本机 | `C:\Users\jeefy\Documents\ChatGPT\UniChessSSM` | 代码编辑、文档、小规模 CPU 验证；**不在本机训练** |
| 远端 5070 Ti 主机（`jeefy@172.16.2.12`，SSH 免密） | `/home/jeefy/UniChessSSM`（工作 clone）+ `/home/jeefy/UniChessSSM.git`（bare 中枢） | GPU 训练、评测、阶段 0 验收测试的实际运行 |

- git 同步：本地 commit → push 到远端 bare → 远端工作目录 `git pull`。
- 远端 conda 环境（与原项目共用，**复用不改动**）：
  `/home/jeefy/miniconda3/envs/unichess/bin/python`（Python 3.12.14，torch 2.11.0+cu128）。
- mamba-ssm / causal-conv1d 已安装进该 conda env（2026-09-14 源码编译，CUDA 12.9）。

## 3. 目录结构

```
UniChessSSM/
├── AGENTS.md  README.md  pyproject.toml
├── stateseq/            # Python 包（model.py 为总装）
│   ├── actions.py       # 1936 动作空间常量表 + 双射
│   ├── features.py      # 785 维局面特征编码/解码（设计文档称"约790"）
│   ├── conditions.py    # time_control/elo/color 条件
│   ├── model_e.py       # 格子级 Transformer E
│   ├── model_r.py       # Mamba 主干 R（官方 mamba_ssm）
│   ├── model_d.py       # D 重建解码器（仅训练；原拟名 aux.py 系 Windows 保留名）
│   ├── model_g.py       # g 残差动力学侧枝（仅训练）
│   ├── heads.py         # f: policy/WDL/moves-left
│   ├── losses.py        # 五损失（统一归约口径）
│   └── data/            # PGN 下载/序列构建/分片/Elo 加权
├── tools/               # 冒烟与一次性脚本
├── tests/               # 阶段 0 七项验收单测（unittest）
├── data/  runs/         # 运行时产物（gitignore；磁盘水位见 §5）
└── docs/
```

## 4. 构建与测试

- 无构建步骤；远端运行用 conda python 直接执行脚本/测试。
- 阶段 0 验收：`python -m unittest discover -s tests -v`（在任一 clone 根目录）。
- 阶段 0 全部通过才允许进 Stage A（全量 Lichess 数据 + 正式训练）。

## 5. 开发约定与安全事项

- **远端是生产主机**：网站、隧道、autoloop（4 个 actor 进程批量 GPU 推理）常驻。
  跑任何 GPU 任务前 `nvidia-smi` 确认占用；训练先小步验证吞吐再放大。
- **不碰原项目**：`/home/jeefy/UniChess` 下的一切（源码、数据、systemd 服务、检查点）只读引用。
- **磁盘**：本目录数据/模型总量水位暂定 60 GiB（原始 Lichess 文件解析后删除中间产物）。
  检查点原子保存（写 `.tmp` 再 `replace`）。
- **检查点纪律**：正式权重不容许被冒烟测试覆盖（用隔离 runs 子目录）。
- **实验纪律（D8）**：不做消融矩阵；只保留最小冒烟门禁。
- **判定权威**：走法合法性、重复/终局判定一律以规则引擎（python-chess）为准，网络特征不作判定依据。
- 代码注释与文档用中文；git 提交信息用中文。
- git 提交节奏：每个里程碑完成后 commit + push（用户已授权此节奏）。
