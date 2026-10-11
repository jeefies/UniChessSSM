# UniChessSSM AGENTS.md

> 本文件面向 AI 编码 agent。最后更新：2026-10-10（Stage B 换代大循环与三元博弈联盟落地）。

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
**Stage B 实验总录**：`docs/stage-b-experiment.md`（Round 1~5 攻坚、Champion gen4 确立、Gen 6~9 循环实录）。
**Elo 衰退根因诊断**：`docs/elo-degradation-analysis.md`（Val Policy Loss 新低与 MCTS Arena 守擂偏科根因深度剖析）。
**Stage B 实施规格**：`docs/stage-b-implementation.md`（唯一规格来源，与 handoff 冲突时以规格为准）。

## 2. 设备与连接路由

| 设备 | 地址 / 路径 | 用途 |
|---|---|---|
| Windows 本机 | `C:\Users\jeefy\Documents\UniChess\SSM` | 代码编辑、文档；无 torch/mamba-ssm，跑不了引擎测试 |
| 远端 5070 Ti（主机名 `jeefyULab`） | `jeefy@ulab.jeefy.top` → `/home/jeefy/UniChess/SSM` | GPU 训练、评测、单测实际运行 |

**远端只有一台设备，但有两条网络路由，默认走域名**（2026-10-11 实测域名可用）：

| 路由 | 地址 | 定位 |
|---|---|---|
| **域名（默认）** | `jeefy@ulab.jeefy.top`（解析到 IPv6 `2409:8a62:5c4:5170:…`） | 一切 ssh / scp / git 都先用它 |
| 内网 IP（兜底） | `jeefy@172.16.2.12`（IPv4） | 只在域名不通时才是候选 |

纪律：默认一律用 `jeefy@ulab.jeefy.top`；只有连接失败时才明确告知用户"域名不通，需要你组网/放行"，
由用户决定下一步——不要自行反复重试、不要静默改走 IP，也不要把两条路由写成两台设备。

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
  2026-09-25 重构后：**123 项 OK**（5 项按环境跳过）；2026-09-27 加 `test_kit_preset.py` 后
  **129 项 OK**；2026-10-08 加 `test_v3_sink.py` 与 `test_multigen_selfplay.py` 后 **137 项 OK**；
  2026-10-11 价值头与学习目标改造后 **188 项 OK**（新增 `test_value_tower_zeroinit` /
  `test_lambda_return` / `test_qsearch_roundtrip` / `test_value_soft_target` /
  `test_resolve_paths`）。**动 `SSM/kit.py` 或 Trainer 口径前，Kit 全量（388 项）先过。**
- **本机**：无 torch/chess/mamba-ssm，只跑语法检查（`python -m py_compile`）。
- **Kit 先过**：动 `SSM/kit.py` 或 Trainer 相关口径前，先跑
  `cd ~/UniChess && UNICHESS_IMPORT_ROOT=~/UniChess python -m unittest discover -s Kit/tests`。

## 5. kit 接入（`kit.py`）

对外符号（S 与 kit 之间的全部契约）：

- `make_evaluator(checkpoint, device, engine)`：`engine` ∈ `reference`（cat/split 重放，逐位对照用）/
  `fast`（GPU 槽池 + CUDA graph）/ `server`（连 `infer/gpu_server.py` 跨进程拼批，批不变 ⇒ 结果与
  进程数/并发无关）。**`server` 模式要求调用方自己起好 `GpuServer`**（`infer/gpu_server.py`），
  Web 服务不常驻那个进程——所以 `config.json` 只发布 `champion`（`fast`）一个预设，
  别把 `champion_server` 那类预设放回去，选了只会得到"GPU 服务未就绪"。
- `make_player_factory(checkpoint=None, *, preset=None, ..., **search_kwargs)`：`preset` 读
  `SSM/config.json`（显式传参优先）。**Server 的观战/批量对弈就是按 `preset=<arg>` 调工厂的**
  （`Server/jobs.py`），所以这个参数不是可选项；`python -m Kit match` 直接给 `checkpoint` 也能用。
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

## 9. Stage B 强化学习迭代体系与自动化循环规范

> 2026-10-10 里程碑：通过子力+步数双轮动态 $c_{\text{scale}}$ 调度与四重防和棋铁壁，Candidate gen5 在 508 局 GSPRT 对决中以 55.51% 得分率（LLR=3.47）胜出，正式晋升为现役霸主 **Champion gen4**（将杀率突破 90.94%，和棋率降至 9.06%）。随后完成自动化换代大循环（Gen 6~9），并在诊断后全面落地多模型联盟与惊讶度重要性采样（Gen 10+）。

### 9.1 核心强化学习流水线：`loop_pipeline.py`
自动化端到端大循环实现于 `SSM/tools/loop_pipeline.py`，串联三大核心阶段：
1. **生成（`step_selfplay`）**：调用 `run_gpu_server_selfplay.py`，以 GpuServer 共享显存模式拉起 8 个 Worker（并发 160 槽位），每代生成 3,000 局高质量棋谱；
2. **训练（`step_train`）**：调用 `Kit.train`，以 85% 自博弈 + 15% 人类棋谱混批训练，最近 4 代滑动窗口经验池（12,000 局）+ Surprise-based PER，留出验证集三维复合评分（策略+自博弈价值+人类价值）挑选最优检查点 `best.pt`；
3. **评测与裁决（`step_eval_raw_policy` & `step_arena`）**：双轨评测（Track 1 纯策略 0-sim 100局对决；Track 2 系统级 64-sims Gumbel成对换先五项式 GSPRT 动态早停裁决，判定是否晋升 Champion）。

### 9.2 三元多模型博弈联盟（League Self-Play）
为根治单代偏科和灾难性遗忘，自对弈生成阶段支持多模型博弈联盟池（`--league`）：
- **当前 vs 当前（50%，Worker 0~3，1,500 局）**：纯自博弈，探索策略前沿；
- **当前 vs 历史最强（25%，Worker 4~5，750 局）**：防倒退锚定；
- **当前 vs 历史霸主联盟轮换（25%，Worker 6~7，750 局）**：与 gen1/gen2/gen3 历史冠军轮流过招，吸收多样化防守流派。

### 9.3 基于信息论自信息的惊讶度重要性采样（Surprise-based PER）
彻底摒弃红黑颜色偏见，以对手历史胜率 $p$ 为基准，严格落地负对数似然自信息量指标：
$$I = -[s \log p + (1 - s) \log (1 - p)] \quad (s \in \{1, 0.5, 0\})$$
- **弱对手虐菜局（$p=0.98$）**：赢局信息量降至 0.05 截断下限，自动过滤平庸垃圾样本；翻车输局信息量打满至 3.91 强制重点复盘；
- **势均力敌自博弈（$p=0.50$）**：胜、和、负严格对称收敛于 $-\log(0.5) \approx 0.6931$；
- **强敌抗压局（$p=0.02$）**：守和与逆袭赢局信息量大幅拉高（1.96 ~ 3.91）。

### 9.5 价值头与学习目标改造（2026-10-11，已晋级一代）

针对 `docs/elo-degradation-analysis.md` §19 的"0-sim 58.5% vs 64-sim 45% 搜索-蒸馏脱节"：

- **价值塔（D1）**：`head_norm` 与 `w_v` 之间插 2×`ResidualMLP(512,768)` + 子力差辅助头，
  **零初始化** ⇒ 旧 ckpt 经 `strict=False` 加载后 step 0 逐位等价；参数 26.40M → 27.981M（D4 内）。
- **q̂ 数据（D3）**：v3 分片新增 `shard-*.qsearch.bin`（每 ply 3 B：f16 白方绝对视角
  `Σπ′·completedQ` + u8 sims）；旧分片无该文件 ⇒ 价值目标自动回退终局 z。**Kit 零改动**。
- **λ-return 目标（D2/D4）**：`g[t]=(1−λ)q̂[t+1]+λg[t+1]`（闭式一次 bmm，递推版 9.6ms→0.125ms），
  WDL 软目标用**确定性最大熵投影** `pW=max(g,0), pD=1−|g|, pL=max(−g,0)`（|g|=1 与 g=0 精确
  one-hot，λ=1 与硬标签 MC 逐位一致）。
- **不需要重新预训练**：零初始化保证连续；`load_seq_model` 改 `strict=False` 且只允许缺零初始化
  新模块（缺别的键一律响亮报错）。
- **实验结论**（逐轮记录见 `docs/value-target-experiments.md`）：
  λ=1.0（纯架构消融）43.5% < λ=0.35 46.0% < **λ=0.5 57.5%**；投影修复单独值 +45 Elo。
  Round 4（λ=0.5 + 新投影）在 Arena GSPRT 以 **55.25% / Elo +36.62 / H1** 胜出并晋级，
  旧霸主备份为 `runs/champion_gen4.pt`（进联盟池）。
- **循环开关**：`loop_pipeline.py --value-target lambda --value-lambda-end 0.5 --record-qsearch`；
  开 q̂ 记录时回放池只收带 q̂ 的代（混入老代会让价值目标在 λ-return 与纯 MC 之间漂移）。
- **两个容易踩的坑**：① λ=1 时不要做低预算 ply 降权（目标已是终局真值，与 sims 无关）；
  ② `value_q_corr` 只在同一 λ 内可比——λ<1 时目标大部分来自网络自己的搜索值，高相关性
  有很大一部分是自蒸馏，不等于价值头变强。
- **全链路时间戳真动态随机种子**：自对弈、数据 shuffle、Arena 开局全量接入 `int(time.time() * 1000) & 0x7FFFFFFF`，全局局序号逐代累加错位；
- **换代门槛【锁定】**：challenger vs champion 成对换先 GSPRT 动态早停（$\alpha=0.05, \beta=0.05, H_1: 35.0\text{ Elo}$）触发 $H_1$ 胜出或得分率 $\ge 55.0\%$ 即可晋升；
- **五代非回滚窗口（Learner 连续性与熔断机制）**：五代内有晋升则不回滚，每一代训练以**上一代最优检查点 `best.pt`** 为基座连续向前推进（深度策略演化，彻底打破在原地微调打转）；若**连续 5 代**冲击 Champion 均未突破，触发安全熔断机制，下一代重置回现役 Champion 重新热启（防止偏科与误差滚雪球）；
- **显存与多进程**：`run_gpu_server_selfplay.py` 下 Worker 纯 CPU、只有 GPU 服务进程常驻显存（4 模型并行常驻显存约 8.9 GiB）。单卡分时交替，严禁训练与自对弈抢占 GPU。

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
