# UniChessSSM AGENTS.md

> 本文件面向 AI 编码 agent。最后更新：2026-09-25（扁平化重构后）。

## 0. 先读这一节：仓库已经不是 `stateseq/` 布局了

2026-09-25 的全量重构（`rebuild` 分支）把**仓库根目录本身变成包**：`import SSM`，import 根是
`~/UniChess`（SSM / Kit / ResNet / Transformer / Server 五个仓库的共同父目录）。
本文档下文中凡是提到 `stateseq/`、`stateseq.xxx`、`tools/*.py`、`train/stage_a.py`、
`unichess_kit`、`../Kit` 的地方，说的都是**已经删除的旧代码**，git 历史里可查，不要再去创建
同名文件。真正有效的入口是：

| 旧 | 新 |
|---|---|
| `stateseq.{actions,conditions,features}` | `SSM.{actions,conditions,features}` |
| `stateseq.kit_adapter` + `stateseq.adapter` + `stateseq.depth_hist` | 合并为 `SSM/kit.py` |
| `stateseq.data.*` | `SSM/dataset/*`（`cpp/` 归到 `dataset/cpp/`） |
| `stateseq.model*` / `losses.py` / `heads.py` / `layers.py` | `SSM/model/*` |
| `stateseq.{fast_eval,gpu_server,ssm_update}` | `SSM/infer/*` |
| `stateseq.gumbel` | **已删**：黄金口径冻结在 `Kit/search/gumbel.py` |
| `train/stage_a.py`、`train/stage_b2.py` | `python -m Kit train SSM/configs/<name>.json` |
| `tools/`（70 个脚本） | **已删**：能力在 `Kit/pipelines/`（自对弈/arena/loop） |

扁平化的两条硬约束（各仓一致）：

1. **import 根是仓库目录的父目录**。`engine.py`、`tests/` 都要自己把 import 根挂进 `sys.path`
   （追加，不前插——前插会遮蔽别人）。远端 import 根 = `~/UniChess`。
2. **`Kit` 已改名为 `Kit`**（不再是 `unichess_kit`），同样是 import 根下的顶层包。
   环境变量 `UNICHESS_KIT_ROOT` 仍被接受，但语义已变成"import 根"；新名字是
   `UNICHESS_IMPORT_ROOT`。

## 1. 项目概述

UniChessSSM 是 UniChess 的**状态序列模型（state-sequence model）**：每步局面显式编码为向量
序列（白方绝对坐标，785 维无损特征），格子级 Transformer E（权重共享走两遍）→ 基础 Mamba R
（12 层）→ policy/WDL/moves-left 三头 f；训练期辅助模块 D（MLP 重建）与 g（残差动力学）不参与
推理。预热用 Lichess 人类棋谱行为克隆（**不用 Stockfish 蒸馏**）。

**权威系统设计**：`docs/state-sequence-model-design.md`（v3.0，唯一权威架构与代码手册，D1–D10
【锁定】不可改）。
**设计偏差与审计**：`docs/design-deviations.md`（c_scale=0.1 的决定在 §9）。
**Stage B 实施规格**：`docs/stage-b-implementation.md`（唯一规格来源，与 handoff 冲突时以规格为准）。

## 2. 双设备路由

| 设备 | 路径 | 用途 |
|---|---|---|
| Windows 本机 | `C:\Users\jeefy\Documents\UniChess\SSM` | 代码编辑、文档；无 torch/mamba-ssm，跑不了引擎测试 |
| 远端 5070 Ti（`jeefy@172.16.2.12`，SSH 免密） | `/home/jeefy/UniChess/SSM` | GPU 训练、评测、单测实际运行 |

- 远端 conda（与原项目共用，**复用不改动**）：
  `/home/jeefy/miniconda3/envs/unichess/bin/python`（Py 3.12，torch 2.11.0+cu128，
  mamba-ssm 2.3.2，causal-conv1d 1.7.0）。
- **Windows 陷阱**：用户名 `jeefy`（j-e-e-f-y），曾多次拼成 `jeffy` 致路径错误。
- **权限陷阱**：Windows 新建文件在远端丢 +x 执行权限（需 `chmod +x`）。
- **编码陷阱**：Windows 直接拷到远端会带 CRLF（代码等价，但 diff 会报整文件不同）；
  多行内联 python 过 ssh 必坏，写法是"本机写脚本 → scp → 远端执行"。
- git：本地 commit → push GitHub（`origin` / `gh`）→ 远端 `git pull --ff-only`。
  **勿向 `legacy-unichess` 推送**。

## 3. 目录结构（重构后）

```
SSM/
├── AGENTS.md  README.md  pyproject.toml
├── actions.py        # 1936 动作空间常量表 + 双射
├── conditions.py     # time_control / elo / color 条件（standardize_elo / ELO_MEAN / ELO_STD）
├── features.py       # 785 维局面特征编码/解码
├── kit.py            # kit 接入总入口（见 §5）
├── tasks.py          # TrainTask 实现：StageATask / StageB2Task
├── engine.py         # Server 六方法 GameEngine 插件
├── model/            # 结构与损失
│   ├── model.py      #   SeqModel（E/R/f + cond）+ forward_train + count_parameters
│   ├── losses.py     #   LossWeights 与五损失（统一归约口径）
│   ├── layers.py     #   RMSNorm / ResidualMLP
│   ├── heads.py      #   f：policy / WDL / moves-left
│   ├── e.py          #   格子级 Transformer E
│   ├── r.py          #   Mamba 主干 R（Cache / MambaTower）
│   ├── d.py          #   D 重建解码器（仅训练）
│   └── g.py          #   g 残差动力学侧枝（仅训练）
├── dataset/          # 数据管线
│   ├── gshards.py    #   v2/v3 分片（变长合法目标 + 终局元数据）
│   ├── shards.py     #   v2 分片（动作列表）
│   ├── sequences.py  #   PGN → 每半回合记录
│   ├── dataset.py    #   SequenceDataset（人类语料）
│   ├── dataset_selfplay.py  # SelfPlayDataset（v3 自对弈）+ build_book_mask
│   ├── pgns.py       #   小样本 PGN 获取与解析
│   └── cpp/          #   pgn2shards.cpp / dbg_san.cpp（多进程 PGN→分片构建器）
├── infer/            # 推理后端
│   ├── fast_eval.py  #   GPU 状态槽 + 原地单步 + CUDA graph（黄金口径）
│   ├── ssm_update.py #   mamba 内核读父槽写子槽
│   └── gpu_server.py #   跨进程共享 GPU 服务（/dev/shm + FIFO，批不变块）
├── configs/          # 训练口径（Trainer 顶层字段全在这里）
│   ├── stage_a.json  #   Stage A：人类棋谱预热
│   └── stage_b2.json #   Stage B2：人类 + 自对弈混合
├── tests/            # unittest（无 pytest，远端跑）
└── data/ runs/       # 运行时产物（gitignore）
```

## 4. 构建与测试

- 无构建步骤；远端用 conda python 直接跑。
- **远端**（cwd 必须是 SSM 仓库根，tests 是命名空间包）：
  ```bash
  cd ~/UniChess/SSM && UNICHESS_IMPORT_ROOT=~/UniChess \
    /home/jeefy/miniconda3/envs/unichess/bin/python -m unittest discover -s tests
  ```
  2026-09-25 重构后：**123 项 OK**（5 项按环境跳过）。
- **本机**：无 torch/chess/mamba-ssm，只跑语法检查（`python -m py_compile`）。
- **Kit 先过**：动 `SSM/kit.py` 或 Trainer 相关口径前，先跑
  `cd ~/UniChess && UNICHESS_IMPORT_ROOT=~/UniChess python -m unittest discover -s Kit/tests`。

## 5. kit 接入（`kit.py`）

对外符号（S 与 kit 之间的全部契约）：

- `make_evaluator(checkpoint, device, engine)`：`engine` ∈ `reference`（cat/split 重放，逐位对照用）/
  `fast`（GPU 槽池 + CUDA graph）/ `server`（连 `infer/gpu_server.py` 跨进程拼批，批不变 ⇒ 结果与
  进程数/并发无关）。
- `make_player_factory(...)` → `SsmPlayerFactory`（Gumbel 搜索；默认 `g=0`，arena 确定性口径）。
- `make_selfplay_factory(...)`、`SsmExpander` / `SsmFastExpander`、`V3Sink`（v3 分片落盘）。
- 终局裁决唯一入口 `classify_final_board`、`get_terminal_q`、`wdl_logits_to_q`、
  `hist_add` / `hist_merge` / `hist_summary`、`TERM_CODES`（顺序 = v3 meta 的取值，改了箱底数据作废）。
- `book_pipol_rng(seed, opening_idx, ply)`：开局 ply 的搜索噪声只取决于这三元组 ⇒ π′ 可跨局共享。

批量对弈配置示例：
```json
{"factory": "SSM.kit:make_player_factory", "root": "~/UniChess",
 "kwargs": {"checkpoint": "SSM/runs/champion.pt", "simulations": 256}}
```
命令行：`cd ~/UniChess && python -m Kit match <config.json> --out runs/<name>/results.jsonl`。

## 6. 训练口径（`configs/*.json`，Trainer 顶层）

- **Stage A**（`stage_a.json`）：人类 v2 分片，microbatch 32 × accum 16，AdamW `fused=True`，
  bf16 autocast（scan 部分保持 fp32），tf32 开，seed 0。
- **Stage B2**（`stage_b2.json`）：自对弈 v3 85% + 人类 10%（`w_selfplay`/`w_human` 显式写在
  loss 里），microbatch 4 × accum 16；`ckpt` 从 Stage A 的 best.pt 续训；
  `steps: 0` ⇒ 由 `StageB2Task.auto_steps` 从 buffer 局数算（`min(3×buffer÷eff, 2000)`）；
  warmup `min(200, 步数×10%)` 用 `warmup_frac`+`warmup_max` 表达（Kit 的
  `_resolve_warmup` 推导，**不要写死 200**）；损失权重 `w_v=1.0, w_r_start=w_r_end=0.1`
  （Stage A 用的是 `LossWeights` 默认值，不能继承）；`limit_games=500000`（人类库 1933 万局，
  不截断等于每步在另一个分布里抽）。
- **两个数据源的口径**：自对弈完整 300 ply（不得继承 Stage A 的 T_MAX=200）；
  `mlh_log` 开；截断局 `mlh` 损失整局剔除；book ply 的 policy 软 CE 降权 0.25。

## 7. 对拍纪律（重构期最重要的一条）

改 `model/`、`dataset/`、`infer/`、`tasks.py` 或任何训练口径后，必须能回答"与旧实现逐位一致吗"：

- 旧代码基线：tag `pre-rebuild-20260924`，远端 `/tmp/rebuild_old/`（`stateseq/` 包原样）。
- 黄金产物：`/tmp/rebuild_baseline/`（训练轨迹 `train/<name>.run{1,2}.json`、推理 `S_*.npz`、
  `positions.json`）。
- 判据：训练轨迹同 seed 同 K 步的 `backward`/`lr`，**落在旧脚本自身两次运行的差**之内
  （旧脚本不是逐位确定的，Stage A 底噪 ≈1.6e-4、Stage B2 ≈1.3e-3）；推理 `logits`/`wdl`
  必须**逐位相同**（同一 checkpoint 的 `reference` 与 `fast` 两个后端也互相对拍）。
- 2026-09-25 的结果：训练 Stage A 1.85e-4 / Stage B2 1.72e-3（均在底噪内）；
  推理 10 checkpoints × 2 引擎全部逐位一致。

## 8. Gumbel 搜索关键事实【锁定】（避免踩坑）

- **价值口径**：节点 q 一律存**该行棋方视角** `q = pW − pL ∈ [−1,1]`，和棋=0；跨边取负。
- **根节点选择**：Gumbel-Top-k 取 top-m₀=16 候选，顺序减半分配 n 次模拟（4 轮 16→8→4→2→1）。
  `n_sims` 默认 256（64 对中盘防御过浅；代价是生成吞吐 ÷3.5）。
- **σ 常数**：`σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂`，`c_visit=50`、**`c_scale=0.1`**
  （决定见 `docs/design-deviations.md` §9.3；判据是价值分辨力与打分幅度是否匹配，不是熵要高）。
- **Q 归一化**：`Kit/search/gumbel.py` 的 `qtransform_completed` 是唯一变换，按**逐节点**
  completed-Q 量程归一；不得用全树 qbox。
- **非根节点**：`π_imp = softmax(ℓ + σ(completedQ))`，选择 `argmax[π_imp − N/(1+ΣN)]`。
- **补全 Q**：`completedQ = q`（已访问）/ `v_mix`（未访问）；`v_mix` 端点保护。
- **着法选择**：训练/生成用顺序减半最终幸存着；评测/换代 arena `g=0`，候选集内 argmax。
- **训练目标**：π′ 在**全部合法着**上计算 softmax（不是只在候选集上）。
- **数值安全**：非法动作 logits 填 `−3e4`（有限大负数，非 −inf），防止 `0 × (−∞) = NaN`。

## 9. Stage B 硬约束（阶段①）

**违反即返工，不得自行调整后继续；确认的变更须记入 `docs/stage-b-implementation.md` §5 变更表。**

- **A 组算法单测全过之前，禁止生成任何正式训练数据。**
- **旧 MCTS-400 链路禁止用于数据生成**（O(T²)，实测 0.63s/步），仅作 B0 对齐基线。
- 条件输入锁定（规格 §0.2）：自对弈 Elo=2567.5 + RAPID 桶；人类混批保留原始条件。
- 架构 D1–D10 冻结；不做消融实验。
- **换代门槛【锁定】**：challenger vs champion **400 局 ≥55%**（和计 0.5），Gumbel `g=0`、
  `n=64`，配对开局 + 交换颜色；**连续 3 代失败即暂停，不放宽门槛**。
- **learner 连续性**：权重与优化器状态跨代持续，**不随换代成败回滚**；champion 仅在晋级时替换。
- 阶段推进：阶段② 首轮闭环 2k–5k 局 → 阶段③ 主循环 25k 局/代 + 最近 10 代 replay buffer
  （按代均匀采样）。单卡**分时**：生成窗口与训练窗口交替，不并行争显存。
- **显存陷阱**：`workers × concurrency` 才是预算（每进程各建 CUDA context）。
  `--engine server` 下 worker 纯 CPU、只有 GPU 服务进程显存常驻；`fast/reference` 是老口径。

## 10. 数据格式

**v2 分片**（Stage A 人类棋谱）：`*.meta.npz`（n_plies/tc_bucket/result/elo_mean/game_key）+
`*.actions.bin`（uint16 动作 id 池）。

**v3 分片**（Stage B 自对弈）：`*.meta.bin`（v2 的 16B/局 + 扩展 16B：gen_id/ckpt_step/
termination_reason/is_truncated/start_type/flags）、`*.actions.bin`、`*.pipol.bin`
（**变长**：每 ply `u16 legal_count + legal_count × (u16 action_id + f16 prob)`，即 π′ 在全部
合法着上的目标）。

**flags = 本局开局注入 ply 数**（0=无开局库）：生成器对前若干 ply 走 book 着法但 π′ 由搜索产生；
训练侧据此对 book ply 的 policy 软 CE 降权（默认 0.25）。旧分片 flags=0 ⇒ 行为不变。

**终局语义必须可分**：自然终局 / 规则申和 / 封顶截断在 meta 分开记录。截断局 z 按约定记和，
**mlh 损失整局剔除**。

## 11. 运维

- **远端是生产主机**：网站、隧道、autoloop 常驻。跑任何 GPU 任务前 `nvidia-smi` 确认占用；
  训练先小步验证吞吐再放大。
- **检查点纪律**：正式权重不容许被冒烟测试覆盖（用隔离 runs 子目录）；`train.lock` 会挡住
  同目录并发，重跑前先删 run 目录。
- **磁盘**：本目录常驻水位 ≤10 GiB，临时 ≤50 GiB。
- **判定权威**：走法合法性、重复/终局判定一律以规则引擎（python-chess）为准，网络特征不作判定依据。
- 代码注释与文档用中文；git 提交信息用中文；每个里程碑完成后 commit + push（用户已授权此节奏）。
