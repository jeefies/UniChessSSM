# Stage B 下一步训练计划

> 记录时间：2026-09-27。上游文档：`docs/stage-b-implementation.md`（唯一规格）、
> `docs/stage-b-experiment.md`（实验史）、`docs/design-deviations.md`。
> 本文件只做**推进计划与现状盘点**；与规格冲突时以规格为准，规格要改也走规格 §5 变更表。

## 1. 结论摘要

Stage B 的搜索/分片/训练/裁决链路已被 48+74+123 项单测和 B0 审计证明无 bug，
但**换代判定一直不严**，且扁平化重构后主线循环在 S 上还跑不起来。
继续训练前必须先补齐 4 个代码缺口（§3），其中 SPI 版本不匹配已在生产环境爆掉
（Server 竞技场报 `RegistryError`）。

当前 champion = **gen1**（权重指纹 `2731b6a4aca4`，step 34），
即 `runs/champion.pt`，也是线上 Server 正在服务的权重。

## 2. 现状盘点（全部有据可查）

### 2.1 champion 与数据

| 项 | 事实 |
|---|---|
| champion | `runs/champion.pt` = `runs/stage_b_training_1500_gen1/best.pt`（step 34，指纹 `2731b6a4aca4`） |
| 起点血缘 | Stage A best.pt（step 37757）→ fix500_cs01（step 10）→ gen1（step 34） |
| 自对弈数据 | 共约 60 MB，全在 `runs/` 下；**生成工具已在扁平化时删除**（`SSM/tools/`） |

| 批次 | 局数 | 搜索口径 | provenance |
|---|---|---|---|
| `stage_b_gen_fix500_cs01` | 500 | n=64, c_scale=0.1, g=1 | ✅ 齐全（`current_teacher`，三处修复全生效） |
| `stage_b_gen_1500_gen1` | 1500 | n=64, c_scale=0.1 | ❌ 无 |
| `stage_b_gen_2500_gen2` | 2500 | n=64, c_scale=0.1 | ❌ 无 |
| `stage_b_gen_1000_gen3` | 1000 | **n=256**（6.4 plies/s） | ❌ 无 |

### 2.2 实验史（各 arena 产物）

| 对照 | 局数 | 结果 | 得分率 |
|---|---|---|---|
| gen1 vs 冻结 Stage A | 56 | 29W/4D/23L | 55.4% |
| gen2 vs Stage A | 60 | 39W/0D/21L | 65.0% |
| gen2 vs Transformer | 60 | 0W/3D/57L | **2.5%** |
| gen2 vs gen3 | 64 | gen2 32W/12D/20L | gen2 59.4% |

问题：

1. **门禁没按规格执行**。规格 §2.8 C 组锁定"challenger vs champion **400 局 ≥55%**"，
   实际用的是旧 pipeline 自写的 50% 门槛、56–64 局；gen1 那次对 Transformer 的对照
   **跑了 0 局**（31s 空跑）却没有拦住晋级。
2. **gen3 没有价值**：既没显出对 gen2 的优势，val selfplay policy CE 反而从 1.6005 退到 1.7832
   （n=256 的目标更难拟合），且 1000 局花了 4.8h。
3. **旧数据没有 provenance**，gen1 的生成命令未落日志，无法回溯口径。

### 2.3 训练侧事实

- 有效 batch 64（microbatch 4 × accum 16），T=300，bf16，microbatch 32×16 在 5070 Ti 上 OOM（14.05/15.51 GiB）。
- 每代遍历预算 `min(3×buffer÷64, 2000)`：1500 局→35 步、2500 局→116 步、1000 局→46 步。
- 观测性不足：gen1/gen3 的 `metrics.jsonl` 只落 2 行（`log_every=50` 而步数更少）。
- `configs/stage_b2.json` 是陈旧残留（`ckpt` 指 gen2 best.pt、`selfplay.dir` 指 gen3 数据），正式跑前要重写为模板。

## 3. 阻塞项（不补完，主线跑不起来）

| # | 缺口 | 位置 | 后果 |
|---|---|---|---|
| 0 | `SSM/kit.py` 声明 `KIT_SPI_VERSION = 1`，kit 是 **2** | `SSM/kit.py:47` | Server 竞技场/观战启动即 `RegistryError`；`Kit match`/`selfplay`/`loop` 全被挡 |
| 0b | `make_player_factory` 不接受 `preset=`、`checkpoint` 还是必填位参 | `SSM/kit.py:590` | Server 竞技场按 `{"preset": <arg>}` 调用（`Server/jobs.py:118`），修了 SPI 也会立刻 TypeError |
| 1 | `Kit selfplay` 没有 v3 sink 工厂 | `SSM/kit.py:744`（`V3Sink.__init__(writer, …)` 要外部传 writer，无 `done_games()`/`close()`） | 自对弈无从落盘/续跑 |
| 2 | `StageB2Task` 只吃**单个**自对弈目录 | `SSM/tasks.py:163`、`dataset_selfplay.py:68`、`gshards.py:305` | 表达不了规格锁定的"最近 10 代 replay buffer、**按代均匀采样**" |
| 3 | `Kit loop` 占位符不适配 S | `Kit/pipelines/loop.py:155` 只 glob `*.sp.bin`（R 的格式）；`phase_selfplay` 只起单个子进程 | 多代 buffer 传不进去；无 worker 分片，生成吞吐回到单进程 |

> Kit 侧 2026-09-26 起已有 `train.variants`（每代 lr × steps 枚举搜索）与
> `enumerate_generations`（只枚举前 N 代），`Kit/pipelines/loop.py` 已推进到 `05c93e0`；
> §3 的清单按当前最新版重新核对过。

## 4. 计划

### Phase 0 · 打通主线（不动任何锁定口径）

1. 修 §3 的 0 / 0b / 1 / 2 / 3，逐项配单测：
   - `SSM/kit.py`：`KIT_SPI_VERSION = 2`；`make_player_factory(checkpoint=None, *, preset=None, …, **search_kwargs)`，
     与 R/T 同一套键映射（`ckpt`→`checkpoint`、`mcts_sims`→`simulations`、`description` 忽略）；
     新增 `make_v3_sink(out_dir, gen_id, ckpt_step, …)`（内含 `V3ShardWriter`，实现 `on_game_end` /
     `done_games` / `close`）。
   - `SSM/{tasks.py,dataset_selfplay.py,dataset/gshards.py}`：自对弈来源支持 `dirs` 列表，**按代等权采样**；
     单目录行为逐位不变。
   - `Kit/AGENTS.md` 同步 SPI 现状；`Transformer/kit.py` 显式声明 `KIT_SPI_VERSION = 2`（现在靠缺省）。
   - `Kit/tests/test_registry.py`：**回归测试**——扫描 import 根下各仓 `kit.py`，用 `ast` 读声明的
     `KIT_SPI_VERSION` 并与 kit 比对（本次事故就是缺这道测试）。
2. 对拍：`Kit selfplay` 并发 1 跑 64 局，与 `/tmp/rebuild_old` 的旧生成器逐字节比
   `actions/pipol/meta`；旧工具起不来则退化为"manifest 统计口径与 fix500 一致 + 重放裁决 0 不符"。
3. 全量测试绿：Kit 269 项 + SSM 123 项 + Server 72 项。

### Phase 1 · 按规格重做一轮闭环（约半天）

| 阶段 | 口径 | 预计 |
|---|---|---|
| 自对弈 | 当前 champion，2500 局，4 worker × 24，n=64 / m0=16 / c_scale=0.1 / g=1，开局库 6 ply + pipol memo，Elo 2567.5 + RAPID | ~2h（0.36 games/s） |
| 训练 | 新模板（human 10% / 自对弈 85%），`auto_steps = min(3×2500÷64, 2000) = 116`，`log_every=10` | ~20 min |
| Arena | **400 局（pairs=200）**，g=0、n=64、配对开局交换颜色，门槛 **0.55** | ~2.2h |
| 参考局 | vs Transformer 100 局，只记录不门禁 | ~0.8h |

- **不达标不换 champion**；learner 权重与优化器状态照常延续（规格：不随成败回滚）。
- 每代输出 `provenance`（teacher_status / c_scale / n_sims / EngineSpec 身份哈希 / 生成器版本）。

### Phase 2 · 主循环

- `python -m Kit loop SSM/configs/loop_b.json`：`generations=10, games=2500, window=10`。
- 每代监控：终止原因分布 / 平均局长 / 根节点有效着法数、`val_selfplay_loss_policy`、value ECE、
  recon acc ≥90% 与 dyn rel err <1.0（规格 C#3 防坍缩）。
- **连续 3 代不晋级即暂停**，不放宽门槛。
- 门禁连续稳定通过后，`games` 提到规格的 **25k/代**（生成约 18h + arena 约 2h/代）。
- 谜题来源（w=0.05）维持 P3 不接入，只做退化报警。

## 5. 待决策点

| # | 问题 | 建议 |
|---|---|---|
| 1 | 起点：当前 champion（gen1）还是回到 Stage A best.pt 重起？ | 以当前 champion 为起点——它确实赢过 Stage A（虽然是弱证据）；重起会丢掉 34 步有效训练，且缺乏更有力的反证 |
| 2 | gen1/gen2/gen3 三批无 provenance 数据是否进新 buffer？ | 只作历史归档，新 buffer 从有 provenance 的干净批次开始 |
| 3 | 主循环每代局数 2500 还是直接 25k？ | 先 2500 跑 2–3 代验证门禁与管线，再放大 |
| 4 | 生成后端 `fast`（4 进程各 ~1 GiB）还是 `server`（跨进程拼批、批不变、worker 纯 CPU）？ | `server`：Phase 2 放大到 25k 时多进程争显存是主要风险，`server` 的批不变性也让结果可复现 |
| 5 | `train.variants` 枚举搜索要不要一上来就用？ | 先不用：门禁还没被正确执行过一次，变体搜索会放大不可解释性 |

## 6. 常用命令

```bash
cd ~/UniChess && export UNICHESS_IMPORT_ROOT=~/UniChess
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
# 测试
(cd ~/UniChess && $PY -m unittest discover -s Kit/tests -t ~/UniChess)
(cd ~/UniChess/SSM && $PY -m unittest discover -s tests)
# 自对弈（单进程分片；多进程 = 同一 seed + 不相交 first_game 区间）
$PY -m Kit selfplay <gen.json>
# 训练 / 换代
$PY -m Kit train SSM/configs/<name>.json
$PY -m Kit loop SSM/configs/loop_b.json
```
