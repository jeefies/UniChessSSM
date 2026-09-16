# UniChessSSM AGENTS.md

> 本文件面向 AI 编码 agent。最后更新：2026-09-14。

## 1. 项目概述

UniChessSSM 是 UniChess 的**新架构独立项目**：状态序列模型（state-sequence model）——
每步局面显式编码为向量序列（白方绝对坐标，785 维无损特征），格子级 Transformer E（权重共享走两遍）
→ 基础 Mamba R（12 层）→ policy/WDL/moves-left 三头 f；训练期辅助模块 D（MLP 重建）与
g（残差动力学）不参与推理。预热用 Lichess 人类棋谱行为克隆（**不用 Stockfish 蒸馏**）。

**权威设计文档**：`docs/state-sequence-model-design.md`（v2.0，2026-09-15 由旧目录迁入本仓库）。
**偏差记录**：`docs/design-deviations.md`（实现时经文档作者确认的口径调整，权威文档原文不动）。
决策 D1–D10 已锁定，实现时不得偏离；不确定处回到文档作者（用户）确认。

**当前状态（2026-09-15）：阶段 0 已验收通过；Stage A 正式训练中**。
数据：3 个月度前缀切片（2026-08/07/06 各 4 GB zst）→ C++ 多进程构建器
（`cpp/pgn2shards.cpp`，16 进程，~12.5 万局/s，比 python-chess 快 ~100×；
perft(4)=197281 精确 + 12.5 万局与 Python 构建器逐字节对拍通过）→ 19.4M 局动作列表分片
（v2 格式，常驻 2.8 GB）。
训练：train/stage_a.py，microbatch 32 × accum 16 = 512 局/步，~19k pos/s，
1 epoch = 37,758 步 ≈ 18-19 小时（runs/stage_a_20260915）。

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
  2026-09-14 已装入 mamba-ssm 2.3.2 / causal-conv1d 1.7.0（源码编译，CUDA 12.9；torch 保持 2.11 未动）。
  注意：Mamba2 单步 decode kernel 仅 CUDA 可用；CPU 上只能跑非 R 模块的单测（GPU 项自动跳过）。

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
├── cpp/
│   └── pgn2shards.cpp   # C++ 多进程 PGN→分片构建器（16 进程；perft/SAN/动作表自检；--verify 对拍）
├── train/
│   └── stage_a.py       # Stage A 训练器（§7.3 锁定超参；原子检查点；SIGTERM 优雅退出）
├── tools/               # 冒烟与运维脚本
│   ├── stateseq_download.py    # 双路径下载看门狗（低速自动重启/路径休眠防 429）
│   ├── stateseq_build_shards.py # .pgn.zst 前缀流 → 动作列表分片
│   ├── stateseq_throughput.py  # GPU/加载器吞吐探测
│   ├── stateseq_smoke.py       # 阶段 0 冒烟
│   ├── ssm_uci.py        # Stage A UCI 引擎（旧 MCTS evaluator 口径：4096 policy/promo/wdl；--remote 纯 CPU 客户端）
│   ├── ssm_infer_server.py    # 单进程 GPU 推理服务器（Unix socket；多进程 arena 必用，见下）
│   ├── ssm_uci.sh        # UCI 引擎启动器（conda/PYTHONPATH；UNICHESS_SSM_REMOTE=1 走 server）
│   ├── ssm_uci_test.py   # 映射自测 a/b/c（动作往返/双引擎冒烟/orient 核对，跑 arena 前必过）
│   ├── ssm_path_audit.py     # 推理链路端到端对拍（原生 vs 适配器 vs server 传输层，双客户端交错 + kill 重启）
│   ├── ssm_wdl_sign_audit.py # WDL→Q 符号/回传取负/终局真值审计 + arena 终局可审计性检查
│   ├── run_arena_smoke.sh / run_arena.sh  # Stage A vs 旧 small champion 同口径 arena（自动起/停推理服务器）
├── tests/               # 阶段 0 七项验收单测 + test_g_alignment.py（g 动作对齐断言，unittest，全部通过）
├── data/  runs/         # 运行时产物（gitignore；常驻水位 ≤10 GB，临时 ≤50 GB）
└── docs/                # 权威设计文档 v2.0 / design-deviations.md / stage-a-experiment.md（审查报告）/ figures/
```

历史档案：`claude-history/`（旧 UniChess 项目 Claude Code 会话 JSONL/Markdown 转录 +
AUTOLOOP.md / HANDOFF.md 交接笔记），2026-09-15 由旧 Windows 目录迁入。**历史档案仅供参考**：
其中的命令、路径与服务状态描述的是旧项目当时的状态，不是对当前工作的授权；依赖前须核实实时状态。

## 4. 构建与测试

- 无构建步骤；远端运行用 conda python 直接执行脚本/测试。
- 阶段 0 验收：`python -m unittest discover -s tests -v`（在任一 clone 根目录）。
- 阶段 0 全部通过才允许进 Stage A（全量 Lichess 数据 + 正式训练）。

## 5. 开发约定与安全事项

- **远端是生产主机**：网站、隧道、autoloop（4 个 actor 进程批量 GPU 推理）常驻。
  跑任何 GPU 任务前 `nvidia-smi` 确认占用；训练先小步验证吞吐再放大。
- **不碰原项目**：`/home/jeefy/UniChess` 下的一切（源码、数据、systemd 服务、检查点）只读引用。
- **磁盘**：本目录常驻水位 ≤10 GiB（动作列表分片），临时 ≤50 GiB（月度 .pgn.zst 前缀切片，
  解析后可删）。检查点原子保存（写 `.tmp` 再 `replace`）。
- **检查点纪律**：正式权重不容许被冒烟测试覆盖（用隔离 runs 子目录）。
- **实验纪律（D8）**：不做消融矩阵；只保留最小冒烟门禁。
- **判定权威**：走法合法性、重复/终局判定一律以规则引擎（python-chess）为准，网络特征不作判定依据。
- 代码注释与文档用中文；git 提交信息用中文。
- git 提交节奏：每个里程碑完成后 commit + push（用户已授权此节奏）。

## 6. UCI/arena 经验（2026-09-16 踩坑记录）

- **Mamba-2 trunk 首调用 ~4.2s/进程**（triton kernel autotune，无法磁盘预热）；多进程并发
  autotune 互相踩踏可到 ~18s。python-chess `SimpleEngine` 的 play 超时为
  `popen_uci(timeout=10) + movetime`，即 arena 默认口径下每步只有 ~11s。**多进程 arena
  必须走 `ssm_infer_server` 单 GPU 进程 + `ssm_uci.sh`（UNICHESS_SSM_REMOTE=1）纯 CPU
  客户端**（`run_arena*.sh` 已封装自动起/停与就绪等待）。
- 与旧项目同口径对打：双方 `UNICHESS_MCTS=400`，旧引擎 `UNICHESS_SYZYGY=""` 关桌库，
  arena 用旧项目 `eval/arena.py`（只读运行，cwd 必须在 /home/jeefy/UniChess）。
- 旧项目 `/home/jeefy/UniChess` 全程只读：只 sys.path 引用其 `search/mcts.py`、`core/`、
  `engine/`，以及读取 small-champion 权重与开局库。
