# UniChessSSM AGENTS.md

> 本文件面向 AI 编码 agent。最后更新：2026-09-20。

## 1. 项目概述

UniChessSSM 是 UniChess 的**新架构独立项目**：状态序列模型（state-sequence model）——
每步局面显式编码为向量序列（白方绝对坐标，785 维无损特征），格子级 Transformer E（权重共享走两遍）
→ 基础 Mamba R（12 层）→ policy/WDL/moves-left 三头 f；训练期辅助模块 D（MLP 重建）与
g（残差动力学）不参与推理。预热用 Lichess 人类棋谱行为克隆（**不用 Stockfish 蒸馏**）。

**权威系统设计与框架说明书**：`docs/state-sequence-model-design.md`（v3.0，唯一权威架构与代码手册，D1–D10【锁定】不可改）。
**设计偏差与审计日志**：`docs/design-deviations.md`（历史审查与变更审计索引，内容已合入设计说明书 v3.0）。
**Stage B 实施规格**：`docs/stage-b-implementation.md`（唯一规格来源，与 handoff 冲突时以规格为准）。
**阶段交接**：`docs/stage-b-handoff.md`（阶段① 实现+算法单测；A 组全过前禁止生成正式训练数据）。

**当前状态（2026-09-20）**：Stage A 已验收通过（policy CE 1.864、value CE 0.767、held-out value_gain 0.100）；
B0 链路审计全 PASS（对拍 fp32 逐位一致、WDL 符号正确、80 局 0 超时负）；
归因明确：−880 Elo 来自网络本身棋力不足，非链路 bug。

**当前阶段**：Stage B 阶段① 代码已落地（生成器 / v3 分片 / `train/stage_b2.py` / arena 全部存在，
单测 14 个文件 74/74 PASS），正在做首轮小规模教师数据闭环，**尚未进入阶段② 2k–5k 局闭环**。

已跑闭环：round1 小规模；round2（2,500 局 → 14 步训练 → 64 局 arena，28.1% 未达换代门槛，
保留为诊断检查点，不为落后候选消耗 400 局预算）。

**四轮审查修复（均为真 bug，勿回退）**：`6cde89b` 生成器 `occ_root` NameError + arena 换色胜者归属；
`dd620d5` 终局分类与 `claim_draw=True` 差一 ply（真实封顶率 15.9% 而非 54.2%），统一到
`adapter.classify_final_board` 唯一裁决入口；`b423ede` Q 归一化改为**逐节点** completed-Q 量程
（全树 qbox 会被其他节点撑大而压缩价值差权重）；`b274360` `c_scale` 配置贯通至 π′ 导出与 arena 两侧。

**历史哈希已失效（2026-09-20 核实）**：早期文档广泛引用的 `8d14a58`/`784dc64`/`154d673`/`f1146ac`/
`2c12930`/`8331930` 均为 rebase 前的**悬空提交**，在两个 clone 里都**不可从 HEAD 到达**
（远端 objects 仍在故 `git cat-file` 可解析，易误判为"存在"；判据用
`git merge-base --is-ancestor <h> HEAD`）。对应的真实提交：`8d14a58`→`6cde89b`、
`784dc64`→`dd620d5`、`154d673`→`b423ede`、`f1146ac`→`4835b79`、`2c12930`→`45fc1b6`；
`8331930` 实为 `ee0801b`（只新增 `tools/compare_c_scale.py`，**未改任何默认值**）。
修复本身都在代码里，只有哈希失效。`docs/` 下仍有约 24 处旧哈希引用未改（见 §13）。

**数据可用性定级（重要）**：`stage_b_smoke`/`val64`/`gen2k`/`gen_round2` 四个分片的 π′ 均为
`legacy_teacher`（生成于 `4835b79` adapter 修复前），`actions`/`z`/修复后 meta 可用，
**`pi_prime` 不可用作搜索目标**；`stage_b_gen_fix500` 虽在修复后生成，但用了 c_scale=1.0，
**降级为诊断数据集**。方法论教训：`result 修正数 = 0` 只证明 z 在重新裁决前后相同，
**不证明搜索目标正确**。

**当前待办（阶段① 收尾）**：用 c_scale=0.1 重生成 500 局教师数据（`stage_b_gen_fix500_cs01`，
复用相同种子与开局分布），作为 fix500 短训唯一数据源，`provenance.teacher_status` 标
`current_teacher`。2026-09-20 13:15 首次尝试因 CUDA OOM 全部 worker 失败（原因见 §4 并发陷阱），
产出目录只剩 worker.log，需重跑。

**Stage A 产物**：`runs/stage_a_20260915/` 含 `best.pt`（step 37758）与 `latest.pt`（step 37000）。

与原项目 UniChess（ResNet 46M + MCTS + autoloop）**完全隔离**：本目录独立开发、独立数据、
独立 git 仓库；不得修改 `/home/jeefy/UniChess` 的任何文件或服务配置。

## 2. 双设备路由

| 设备 | 路径 | 用途 |
|---|---|---|
| Windows 本机 | `C:\Users\jeefy\Documents\UniChess\SSM` | 代码编辑、文档、CPU 单测；**不在本机训练** |
| 远端 5070 Ti 主机（`jeefy@172.16.2.12`，SSH 免密） | `/home/jeefy/UniChess/SSM`（工作 clone）+ `/home/jeefy/UniChess/SSM.git`（bare 中枢 `origin`） | GPU 训练、评测、单测实际运行 |

**路径迁移（2026-09-20，已核实）**：远端工作 clone 从 `/home/jeefy/UniChessSSM` 迁到
`/home/jeefy/UniChess/SSM`，**旧路径已不存在**。根目录 `run_round2_*.sh`、`tools/*.sh`、
`tools/stage_b_verify.sh` 等脚本硬编码旧路径已统一修复完成。迁移只动目录位置，未改动"与原项目隔离"约束：
`/home/jeefy/UniChess` 下除 `SSM/` 与 `SSM.git/` 外一切仍是只读。
GitHub 远端为 `gh` → `git@github.com:jeefies/UniChessSSM.git`。

- git 同步（2026-09-23 起）：本地 commit → push 到 GitHub（`origin`/`gh`）→ 远端工作目录
  `git pull --ff-only origin main`。远端 bare `SSM.git` 已不再使用；**勿向 `legacy-unichess` 推送**。
- **UniChessKit 依赖（P3 起）**：arena 与自对弈需要兄弟仓库 `../Kit`（或环境变量
  `UNICHESS_KIT_ROOT`）；口径变化见 `docs/design-deviations.md` §2.4。Kit 只能**追加**到 sys.path 末尾；
  且本仓库 `tests/` 无 `__init__.py`（命名空间包），Kit 的常规 `tests` 包在 sys.path 任何位置都会胜出，
  测试间互相导入须带 `from test_xxx import` 回退（见 `tests/test_cache_isolation.py`）。
- 远端 conda 环境（与原项目共用，**复用不改动**）：
  `/home/jeefy/miniconda3/envs/unichess/bin/python`（Py 3.12.14，torch 2.11.0+cu128，
  mamba-ssm 2.3.2，causal-conv1d 1.7.0，CUDA 12.9）。
- **Windows 陷阱**：用户名 `jeefy`（j-e-e-f-y），曾多次拼成 `jeffy` 致路径错误。
- **权限陷阱**：Windows 新建文件在远端丢 +x 执行权限（需 `chmod +x`）。
- 远端无外网（装不了 pytest，测试用标准库 unittest runner）。

## 3. 目录结构

```
UniChessSSM/
├── AGENTS.md  README.md  pyproject.toml
├── stateseq/            # Python 包（model.py 为总装）
│   ├── actions.py       # 1936 动作空间常量表 + 双射
│   ├── features.py      # 785 维局面特征编码/解码
│   ├── conditions.py    # time_control/elo/color 条件
│   ├── model_e.py       # 格子级 Transformer E
│   ├── model_r.py       # Mamba 主干 R（官方 mamba_ssm）
│   ├── model_d.py       # D 重建解码器（仅训练）
│   ├── model_g.py       # g 残差动力学侧枝（仅训练）
│   ├── heads.py         # f: policy/WDL/moves-left
│   ├── losses.py        # 五损失（统一归约口径）
│   ├── gumbel.py        # Gumbel 顺序减半搜索核心（纯 numpy，无 GPU 可测）
│   ├── kit_adapter.py   # 接入 UniChessKit：SsmPlayer / SsmExpander（ReplayStore）/ SsmSelfPlayer / V3Sink
│   └── data/            # 分片读写
│       ├── shards.py    # v2 分片（动作列表）
│       └── gshards.py   # v3 分片（变长合法目标 + 终局元数据）
├── cpp/
│   └── pgn2shards.cpp   # C++ 多进程 PGN→分片构建器（16 进程）
├── train/
│   ├── stage_a.py       # Stage A 训练器（§7.3 锁定超参；原子检查点；SIGTERM）
│   └── stage_b2.py      # Stage B 训练器（改自 stage_a.py；已落地）
├── tools/               # 冒烟与运维脚本
│   ├── stateseq_download.py    # 双路径下载看门狗
│   ├── stateseq_build_shards.py # .pgn.zst → 动作列表分片
│   ├── stateseq_throughput.py  # GPU/加载器吞吐探测
│   ├── stateseq_smoke.py       # 阶段 0 冒烟
│   ├── ssm_uci.py        # UCI 引擎适配器（--remote 纯 CPU 客户端）
│   ├── ssm_infer_server.py    # 单 GPU 推理服务器（Unix socket）
│   ├── ssm_uci.sh        # UCI 引擎启动器
│   ├── ssm_uci_test.py   # 映射自测 a/b/c
│   ├── ssm_path_audit.py     # 推理链路端到端对拍（fp32 逐位一致）
│   ├── ssm_wdl_sign_audit.py # WDL→Q 符号审计
│   ├── ssm_gumbel_selfplay.py # Stage B 自对弈生成器（kit run_selfplay 薄封装，P3）
│   ├── ssm_gumbel_arena.py    # Stage B 换代 arena（kit run_match 薄封装，g=0，P3）
│   ├── repair_v3_meta.py      # v3 meta 终局语义修复（只改 meta 不重跑生成）
│   ├── align_pipol.py / trace_pipol.py # π′ 对齐与 σ/Q 追踪
│   └── run_arena*.sh     # arena 启动器（自动起/停推理服务器）
├── tests/               # unittest 单测
│   ├── test_actions.py
│   ├── test_features.py
│   ├── test_legal_mask.py
│   ├── test_consistency.py
│   ├── test_overfit.py
│   ├── test_g_alignment.py # g 动作对齐断言（5/5 通过）
│   ├── test_cache_isolation.py # 分支缓存隔离（需 CUDA）
│   ├── test_value_sign.py # WDL→Q 符号单测
│   ├── test_gumbel.py / test_gshards_v3.py # 搜索算法 + v3 分片往返
│   ├── test_termination_classify.py # 终局裁决唯一入口
│   ├── test_kit_replay.py / test_kit_adapter.py / test_kit_selfplay.py / test_arena_kit.py # kit 接入
│   └── test_soft_policy_integration.py / test_board_key_consistency.py
├── data/  runs/         # 运行时产物（gitignore；常驻 ≤10 GB，临时 ≤50 GB）
└── docs/                # 权威设计文档 / design-deviations.md / stage-a-experiment.md / stage-b-*
```

历史档案：`claude-history/`（旧 UniChess 项目转录 + 交接笔记），**仅供参考**；
其中的命令、路径与服务状态描述的是旧项目当时的状态，依赖前须核实实时状态。

## 4. 构建与测试

- 无构建步骤；远端运行用 conda python 直接执行脚本/测试。
- **本机（Windows）**：无 torch/chess/mamba-ssm，只能跑纯 Python 单测或语法检查。
  `python -m unittest discover -s tests -v` 会因缺依赖而 ERROR，这是预期行为。
- **远端**：`/home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests -v`
  跑全量单测；阶段 0 全部通过才允许进 Stage B。
- **Stage B A 组单测**：必须在生成正式训练数据前全过；对拍脚手架复用 `tools/ssm_path_audit.py`。
  `stateseq/gumbel.py` 是纯 numpy 实现，可在无 GPU 环境下做算法单测。

**并发陷阱（2026-09-20 实测，OOM 根因）**：`ssm_gumbel_selfplay.py` 的 `--concurrency`
是**每 worker** 的并发局数，`run_workers` 原样透传给每个子进程，
且每进程各建独立 CUDA context。`--workers 3 --concurrency 128`（默认 concurrency）会让 3 个进程
各占约 5 GiB，合计 ~15 GiB 撞满 15.51 GiB 显存 → 全部 worker `torch.OutOfMemoryError` 退出码 1。
已验证可行档位：round2 用 `--concurrency 24 --workers 4`（GPU 利用率 97%）；成功的 fix500 用
`workers=4`，0.427 games/s（500 局约 20 分钟）。**显存预算按 workers × concurrency 算，不是 concurrency**。
失败时生成器**故意不合并**已产出的部分分片，避免未验证批次混入正式数据。
**P4（2026-09-24）后上述陷阱只适用于 `--engine fast/reference`**：默认 `--engine server` 下 worker 纯 CPU、
不建 CUDA context，显存只有一个 GPU 服务进程（权重 + 全体状态槽，按空闲显存自动限槽）。
推荐 `--workers 16 --concurrency 24`（arena 55 plies/s，P3 为 3.83）；再大会报“状态池耗尽”。
结果与 workers / concurrency 无关、逐位可复现。细节见 `docs/design-deviations.md` §2.5。
P5（2026-09-24）：服务端两拍流水线 + 搜索侧热点优化，fp32 arena 71 plies/s、自对弈 94 plies/s
（192 局 × 40 ply 口径，结果与 P4 逐局相同）；`--server-precision tf32` 再快约 22%（有小偏差，默认 fp32）。见 §2.6。

## 5. Stage B 硬约束（阶段① 实现+单测）

**违反即返工，不得自行调整后继续，确认的变更须记入 `docs/stage-b-implementation.md` §5 变更表。**

- **A 组 7 项算法单测全过之前，禁止生成任何正式训练数据。**
- **旧 MCTS-400 链路禁止用于数据生成**（O(T²)，实测 0.63s/步），仅作 B0 对齐基线。
- 条件输入锁定（规格 §0.2）：自对弈 Elo=2567.5 + RAPID 桶；人类混批保留原始条件；谜题固定 2567.5 + unknown 桶。
- 架构 D1–D10 冻结；不做消融实验。
- 超参以 `docs/stage-b-implementation.md` §3 锁定表为准。
- **换代门槛【锁定】**：challenger vs champion **400 局 ≥55%**（和计 0.5），Gumbel `g=0`、`n=64`，
  配对开局 + 交换颜色，记录完整 W/D/L 并按开局对看波动；**连续 3 代失败即暂停，不放宽门槛**。
  未达门槛的候选不消耗 400 局预算（round2 用 32/64 局对照即判定落后并留作诊断）。
- **learner 连续性**：权重与优化器状态跨代持续，**不随换代成败回滚**；champion 仅在晋级时替换。
- 阶段推进：阶段② 首轮闭环 2k–5k 局 → 阶段③ 主循环 25k 局/代 + 最近 10 代 replay buffer
  （按代均匀采样）。单卡**分时**：生成窗口与训练窗口交替，不并行争显存。

## 6. Stage B 交付物（阶段①）

| # | 交付 | 规格依据 |
|---|---|---|
| 1 | `tools/ssm_gumbel_selfplay.py` 原生 Gumbel 树生成器 | §2.1–§2.4 |
| 2 | v3 分片读写（变长合法目标 + 终局元数据） | §2.5 |
| 3 | `train/stage_b2.py`（改自 `train/stage_a.py`） | §2.6 |
| 4 | A 组 7 项算法单测全部通过 | §2.8 A 组 |
| 5 | B 组冒烟：1,000 局生成 + ≤3 遍训练，管道健康 | §2.8 B 组 |
| 6 | 规格 §4 实测回填清单逐项填回文档 | §4 |

## 7. 开发约定与安全事项

- **远端是生产主机**：网站、隧道、autoloop（4 个 actor 进程批量 GPU 推理）常驻。
  跑任何 GPU 任务前 `nvidia-smi` 确认占用；训练先小步验证吞吐再放大。
- **不碰原项目**：`/home/jeefy/UniChess` 下的一切（源码、数据、systemd 服务、检查点）只读引用。
- **磁盘**：本目录常驻水位 ≤10 GiB（动作列表分片），临时 ≤50 GiB（月度 .pgn.zst 前缀切片，解析后可删）。
- **检查点纪律**：正式权重不容许被冒烟测试覆盖（用隔离 runs 子目录）。
- **实验纪律（D8）**：不做消融矩阵；只保留最小冒烟门禁。
- **判定权威**：走法合法性、重复/终局判定一律以规则引擎（python-chess）为准，网络特征不作判定依据。
- 代码注释与文档用中文；git 提交信息用中文。
- git 提交节奏：每个里程碑完成后 commit + push（用户已授权此节奏）。

## 8. Gumbel 搜索关键事实（避免踩坑）

- **价值口径【锁定】**：节点 q 一律存**该行棋方视角** `q = pW − pL ∈ [−1,1]`，和棋=0；跨边取负（零和）。
  与 WDL 训练标签（行棋方归一）一致，且与 B0 核对的 `Q=wdl[0]−wdl[2]` 合约同构。
- **根节点选择**：Gumbel-Top-k 取 top-m₀=16 候选，顺序减半分配 n=64 模拟（4 轮 16→8→4→2→1）。
  **2026-09-22 变更：n_sims 默认值 64→256**（`stateseq/gumbel.py:29` 及两个工具 CLI，
  用户确认，依据 `docs/ssm_vs_transformer_analysis.md` §9.1——64 sims 对中盘防御过浅）。
  代价：生成吞吐约 ÷3.5。变更已记入 `docs/stage-b-implementation.md` §5。
- **σ 常数【锁定】**：`σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂`，`c_visit=50`、**`c_scale=0.1`**
  （决定见 `design-deviations.md` §9.3；**代码默认值直到 2026-09-20 本次才真正落到
  `stateseq/gumbel.py:26`** —— 此前 3 天里文档写 0.1、代码仍是 `C_SCALE = 1.0`，
  `git log -S'C_SCALE = 0.1'` 为空可证。生成器 CLI 是 `default=C_SCALE`
  **不显式传 `--c_scale` 就会静默沿用旧值**，
  这正是 fix500 被降级的那个设置）。判据不是"熵要高"，而是**价值分辨力与打分幅度是否匹配**：
  c_scale=1.0 时 α≈50~80 把仅 ~0.03 的 Q 差放大成 18 logit 差，60.3% 局面目标退化为近乎确定选择
  （KL(π′‖π)=2.195 反超原始 policy 熵 1.811）；32 局同权重对抗 c_scale=0.1 胜 65.6%（21:11，0 和棋）。
- **Q 归一化【锁定】**：`gumbel.qtransform_completed` 是唯一变换，按**逐节点** completed-Q 量程归一，
  根评分 / 非根选择 / π′ 导出三处共用；**不得用全树 qbox**。
- **非根节点【锁定】**：`π_imp = softmax(ℓ + σ(completedQ))`，选择 `argmax[π_imp − N/(1+ΣN)]`。
  "无 UCB" 不等于 "不考虑访问次数"。
- **补全 Q【锁定】**：`completedQ = q`（已访问）/ `v_mix`（未访问）；`v_mix` 端点保护（零访问退化为 v̂，分母加 ε）。
- **着法选择【锁定】**：训练/生成用顺序减半最终幸存着（Gumbel 噪声提供探索，不需要 Dirichlet/温度）。
  评测/换代 arena：g=0，候选集内 argmax。
- **训练目标【锁定】**：π′ 在**全部合法着**上计算 softmax（不是只在候选集上）。
  不变量：所有合法着 completedQ 相同 ⇒ π′ = π。
- **数值安全**：非法动作 logits 填 `−3e4`（有限大负数，非 −inf），防止 `0 × (−∞) = NaN`。

## 9. 树与 R cache 关键事实（§2.3）

- 每局持有**不可变的实战根快照**（R cache bf16 ≈ 0.44 MiB/局），实战走子后根快照线性推进一次。
- **每次模拟从根快照克隆一份工作 cache**——`Mamba2.step` 原地改写传入状态，根快照与工作 cache 必须隔离。
- 树节点只存 `x = E(完整 785 维特征)`（bf16 1 KiB/节点）。E 的共享键必须包含全部输入字段（含半回合计数、全回合数、重复参考位）与模型版本。
- 叶子扩展 = 1 次 E + 路径重算 d 次 R.step + 1 次 f。

**生命周期测试（新生成器必须覆盖）**：
1. 模拟结束后根快照逐字节不变；
2. 批量槽位复用时清空该槽的 R cache、occurrence 字典与棋盘状态；
3. 生成进程在一个代次窗口内只加载一次 champion 权重；
4. 并发局间无任何 cache 串扰。

## 10. 数据格式

**v2 分片**（Stage A 人类棋谱，保留不动）：
- `*.meta.npz`：结构化数组（n_plies/tc_bucket/result/elo_mean/game_key）
- `*.actions.bin`：uint16 动作 id 池

**v3 分片**（Stage B 自对弈，新增）：
- `*.meta.bin`：v2 的 16B/局 + 扩展 16B（gen_id/ckpt_step/termination_reason/is_truncated/start_type/flags）
- `*.actions.bin`：uint16 实战着法
- `*.pipol.bin`：**变长**——每 ply 存 `u16 legal_count + legal_count × (u16 action_id + f16 prob)`
  即 π′ 在全部合法着上的目标；各 ply 偏移表随分片索引存。
- **flags（2026-09-22 启用，原保留字段）= 本局开局注入 ply 数**（0=无开局库）。
  生成器对前 `book_plies`（默认 6）ply 走 book 着法但 π′ 由搜索产生（P1-2），搜索结果按
  (seed, 开局序号, ply) 跨局共享（P1-1，manifest 记 book_memo_hit_rate）；
  训练侧据此对 book ply 的 policy 软 CE 降权（`--opening-loss-weight`，默认 0.25，P1-3）。
  旧分片 flags=0 ⇒ 无 book 段 ⇒ 行为不变。

**终局语义必须可分**：自然终局/规则申和/封顶截断在 meta 分开记录。
截断局（is_truncated=1）z 按约定记和，**mlh 损失整局剔除**。

## 11. 训练器改造要点（train/stage_b2.py）

- **数据源与监督范围**（来源分别归约损失后显式加权，不是样本条数占比）：
  - 自对弈 85%：policy = π′ 软 CE（全部合法着目标）；value = z
  - 人类棋谱 10%：policy = 实际着法 BC；value = 对局结果
  - 谜题 5%：policy = 解法首着；value = 仅强制将杀标记（独立有效位）；**mlh 禁用**
- **序列长度【锁定】**：完整 300 ply（Stage A T_max=200 截断不得继承）。
- **训练量【锁定】**：每代训练步数 = `min(3 × buffer 局数 ÷ 512, 2000)`。
- **warmup**：`min(200, 当代步数 × 10%)`；LR 每代 cosine 3e-5 → 3e-6。
- **优化器【锁定】**：AdamW β(0.9,0.999) wd 0.1；bf16 autocast；grad clip 1.0；有效 batch 512 局。
- checkpoint 纪律、metrics.jsonl 字段、原子保存、SIGTERM 处理：全部沿用 Stage A。

## 12. UCI/arena 经验（Stage A 踩坑，Stage B 仍有效）

- **Mamba-2 trunk 首调用 ~4.2s/进程**（triton kernel autotune，无法磁盘预热）；多进程并发
  autotune 互相踩踏可到 ~18s。多进程 arena 必须走 `ssm_infer_server` 单 GPU 进程
  + `ssm_uci.sh`（`UNICHESS_SSM_REMOTE=1`）纯 CPU 客户端。
- 与旧项目同口径对打：双方 `UNICHESS_MCTS=400`，旧引擎 `UNICHESS_SYZYGY=""` 关桌库，
  arena 用旧项目 `eval/arena.py`（只读运行，cwd 必须在 `/home/jeefy/UniChess`）。
- 旧项目 `/home/jeefy/UniChess` 全程只读：只 sys.path 引用其 `search/mcts.py`、`core/`、`engine/`，
  以及读取 small-champion 权重与开局库。
