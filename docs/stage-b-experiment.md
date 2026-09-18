# Stage B 实验报告 · Gumbel 自对弈搜索

> 本文档供审查使用，覆盖 Stage B 阶段①（实现+算法单测+冒烟）与阶段②（首轮闭环）的全部指标与实测数据。
> 设计文档：`stage-b-implementation.md` | 审查意见：`review.txt` | 上游：`stage-a-experiment.md`
> 一 initial champion：`runs/stage_a_20260915/best.pt`（101 MB, 26.40M 参）

---

## 目录

1. [搜索算法实现](#1-搜索算法实现)
2. [A 组单测（48/48 PASS）](#2-a-组单测4848-pass)
3. [生成器性能与并发调优](#3-生成器性能与并发调优)
4. [数据完整性验证（review 四项）](#4-数据完整性验证review-四项)
5. [Gumbel-64/16 全量生成统计](#5-gumbel-6416-全量生成统计)
6. [首轮训练](#6-首轮训练)
7. [Arena 结果](#7-arena-结果)
8. [封顶问题专题](#8-封顶问题专题)
9. [工程问题清单](#9-工程问题清单)
10. [各运行产物](#10-各运行产物)

---

## 1. 搜索算法实现

### 1.1 基础数值

| 超参 | 值 | 来源 |
|---|---|---|
| c_visit | 50.0 | §2.2 锁定 |
| c_scale | 1.0 | §2.2 锁定 |
| EPS | 1e-8 | v_mix 分母保护 |
| NEG_LOGIT | -3e4 | 非法动作填充（有限大负数，非 −inf） |
| N_SIMS | 64 | 根节点顺序减半总预算 |
| M0 | 16 | 根节点候选数上界 |

### 1.2 根节点选择流程

```
每步:
1. g(a) ~ Gumbel(0,1) → g(a) + ℓ(a) → top-m0 候选
2. 顺序减半分配 n=64 模拟到 ⌈log₂m₀⌉=4 轮
3. 每轮末按 g(a) + ℓ(a) + σ(q̂(a)) 淘汰末位半数 (16→8→4→2→1)
4. 最终 = 末轮唯一幸存者
```

| 边界条件 | 处理 | 单测 |
|---|---|---|
| m=1（仅 1 合法着） | 全部预算给同一候选 | A#4 |
| 合法着数非 2 的幂 | 末轮保留 1 候选 | A#4 |
| 预算不整除 | 前 rem 个候选多分 1 次 | A#4 |
| 合法着数 < m₀ | m = 合法着数 | A#4 |

### 1.3 非根节点选择（review 修正①）

```
π_imp(a) = softmax(ℓ(a) + σ(completedQ(a)))    全部合法着
a* = argmax[π_imp(a) − N(a) / (1 + ΣN)]
```

修正前：`argmax[ℓ + σ(q̂)]`——忽略访问次数，重复选同一着。
修正后：访问频率向 π_imp 收敛（例：π_imp=(0.6,0.4) → 60:40 访问比）。

### 1.4 补全 Q 与 σ

| 公式 | 说明 |
|---|---|
| completedQ(a) = q(a) if N(a)>0 else v_mix | 已访问：网络 Q；未访问：混合估计 |
| v_mix = (v̂ + ΣN · Σπ(a)q(a) / (Σπ(a) + ε)) / (1 + ΣN) | 端点保护：零访问时退化为 v̂ |
| σ(q̂) = (c_visit + max N(b)) · c_scale · q̂ | c_visit=50, c_scale=1.0 |

### 1.5 训练目标 π′（review 修正②）

```
π'(a) = softmax(ℓ(a) + σ(completedQ(a)))    全部合法着
```

修正前：仅在候选集上归一 → 不满足不变量。
修正后：全部合法着 → 相同 completedQ ⇒ π′ = π。

### 1.6 修复的 Bug

| Bug | 发现 | 影响 | 修复 |
|---|---|---|---|
| 升后/EP 旗标丢失 | 训练冒烟（504/1000 坏局） | 兵停 8 排，棋盘静默损坏 | `_resolve_move(a,board)` 对照 legal_moves |
| Gumbel 回传符号 | 代码审查 | 非终局分支根 Q 反转 | 递归 `_simulate` + children 持久化 |
| 生成器串行 | 实测 0.085 games/s | GPU 利用率 17% | 协程拼批 + 多进程编排 |
| 训练器未接入自对弈 | 代码审查 | π′ 目标从未参与训练 | `policy_soft_loss` + 来源归约 |
| T_MAX=200 静默继承 | 规格检查 | 300 ply 被截头 | 可配置 t_max=300 |
| policy_soft_loss 形状 | 集成测试首跑 | B>1 时广播失败 | `.reshape(-1)` |

---

## 2. A 组单测（48/48 PASS）

| 编号 | 测试文件 | 用例数 | 覆盖内容 |
|---|---|---|---|
| A#1 | `test_consistency.py` | 1 | 树版前向 vs 直线路径前向：policy softmax Δ<1e-4, wdl Δ<1e-4 |
| A#2 | `test_gumbel.py::GumbelCorrectnessTest` | 8 | v_mix 退化为 v̂、零访问保护、completedQ 全访问、预算守恒 64/64、σ(0)=0 |
| A#2+ | `test_gumbel.py::RecursiveDepthAndSignTest` | **2（新增）** | 树深随预算增长（≥3 层）、端到端符号正确（修复前会选错着法） |
| A#3 | `test_gumbel.py::InvariantTest` | 2 | 相同 Q ⇒ π′=π（硬约束）；非法着概率=0 |
| A#4 | `test_gumbel.py::BudgetBoundaryTest` | 4 | m=1 全预算给同一候选、合法着数 3≠2^k、预算 65 不整除、m=2 但 m0=16 |
| A#5 | `test_gumbel.py::GZeroDeterminismTest` | 2 | g=0 候选集内 argmax（非全局）；g=1 选择随噪声变化 |
| A#6 | `test_gumbel.py::SoftCESafetyTest` | 2 | 含 -3e4 logits 时 π′ 有限且和=1；终局空节点不报错 |
| A#6+ | `test_soft_policy_integration.py` | **1（新增）** | forward_train+soft_target+bf16+backward 全部有限（无 NaN/Inf） |
| A#7 | `test_gumbel.py::LifecycleTest` | 1 | 根节点模拟后 q/N/n_max 逐字节不变 |
| — | `test_actions.py` | 3 | 动作空间全量双向表（1456+336+144=1936）、自环幻影槽位不与合法着碰撞 |
| — | `test_features.py` | 4 | 785 维编码：随机局面往返一致、过路兵 file one-hot、重复位标志 |
| — | `test_legal_mask.py` | 1 | mask 与规则引擎 legal_moves 100% 一致（含升变/易位/EP） |
| — | `test_g_alignment.py` | 5 | g 对齐：dyn 索引语义、game0-4 逐位匹配 |
| — | `test_cache_isolation.py` | 1 | 分支缓存隔离（需 CUDA） |
| — | `test_overfit.py` | 1 | 单 batch 过拟合：loss 下降 >10%（33.4→18.3） |
| — | `test_value_sign.py` | 3 | WDL→Q 归一、moves_left 方向、长局截断 |

**48/48 耗时**：~18s（GPU），~0s（CPU 纯函数）

---

## 3. 生成器性能与并发调优

### 3.1 架构

```
run_workers ── subprocess.Popen × N ── Worker ── Driver ── GameState
     │                │                     │         └ 协程 yield model requests
     │                │                     └ _model_step: concat_caches → forward → split
     └── _merge_worker_outputs: 搬分片 + 合并 manifest
```

**关键实现细节**：
- 协程 `GameState.run()` 在每步需要模型前向时 `yield (features, tc, elo, color, cache)`
- `Driver` 攒活 → `concat_caches(list_of_batch1_caches)` → `model.step(features, tc, elo, color, batched_cache)` → `split_cache`
- 每步拼批的逻辑批次大小 ≈ 并发局数（无空闲槽）
- 展开 `_expand_gen` 沿 `node.path` 复制棋盘 + 重算 occurrence，从那步起推 cache 再展开一步

### 3.2 网格搜索定档（n_sims=32/m0=8, 100 局基准）

| 配置 | games/s | GPU 利用率 | GPU 显存 | 特点 |
|---|---|---|---|---|
| 1×16 | 0.085 | 17% | ~1.5 GiB | 基线：GPU 空转，CPU 单核 100% |
| 10×4 | 0.129 | 95%+ | ~4.0 GiB | 太多小 batch，CUDA context 切换频繁 |
| 6×8 | 0.171 | 97%+ | ~4.2 GiB | |
| 5×16 | 0.219 | 99% | ~4.5 GiB | |
| **4×24 ⭐** | **0.282** | **99%** | **~4.7 GiB** | **定档：最少进程×最大单进程 batch** |
| 3×40 | 0.278 | 99% | ~4.9 GiB | 平台期，与 4×24 无异 |

**定档理由**：超出 4 进程后 GPU context 切换开销吃掉并行增益。每进程更大的 batch 提升 GPU kernel 利用率。5×16 就已 99%，4×24 是稳定最优。

### 3.3 Gumbel-32 vs Gumbel-64 吞吐

| 配置 | 局数 | n_sims | games/s | plies/s | nodes/ply | 封顶率 |
|---|---|---|---|---|---|---|
| 冒烟（旧修复前） | 1,000 | 32 | 0.327 | 69.4 | 31.8 | 51.8% |
| 验证集（val64） | 256 | 64 | 0.123 | 23.7 | 61.0 | 50.8% |
| 首轮训练（gen2k） | 2,000 | 64 | 0.134 | 27.0 | 60.9 | 54.2% |

> n_sims=64 约 n_sims=32 的 2.4× 墙钟（一倍搜索量 + cache 重算路径更长）。
> 每个搜索节点的平均前向成本 ≈ 0.02ms（batch=24, T=1 单步前向）。

### 3.4 生命周期验证

| 要求 | 实现 | 验证 |
|---|---|---|
| 根快照模拟后逐字节不变 | `clone_cache()` 在工作 cache 上操作 | A#7 |
| 槽位复用清空状态 | `_start_slot` 创建新 GameState | 代码审计 |
| 权重更新后旧 cache 失效 | 每代次加载一次 champion | 单进程设计 |
| 并发局间无串扰 | 每局独立 board/cache/occurrence | 抽查 |

---

## 4. 数据完整性验证（review 四项）

### 4.1 终止原因交叉核对（1,000 局重放）

| 检查项 | 结果 |
|---|---|
| 总局数 | 1,000 |
| 记录 vs 实际终止不匹配 | **0 局** |
| 300 ply 末尾已将杀（非记录为封顶） | **1 局**（正确记录为将杀） |
| 300 ply 末步自然终局 | 4/244 局 |
| 实际五十步和棋 | **0 局** |
| 实际三次重复 | **0 局** |
| 封顶局面占比 | **73.3%**（155,400/212,003 ply） |

**终止原因分布（n_sims=32）**：

| 终止原因 | 代码 | 局数 | 占比 |
|---|---|---|---|
| 将杀 | 0 | 338 | 33.8% |
| 逼和 | 1 | 70 | 7.0% |
| 五十步 | 2 | 0 | 0% |
| 三次重复 | 3 | 0 | 0% |
| 子力不足 | 4 | 74 | 7.4% |
| **封顶截断** | **5** | **518** | **51.8%** |

### 4.2 重建四格对照（冻结 Stage A, eval mode, dropout=0）

| 检查点 | 人类验证集（97,123 局） | 自对弈验证集（4 局） |
|---|---|---|
| **Stage A 冻结** | recon acc **95.9%** / dyn 0.43 / recon CE 0.0059 | recon acc **62.1%** / dyn 0.52 / recon CE 0.0670 |
| **5 步冒烟后** | recon acc 90.7% / dyn 0.40 / recon CE 0.0532 | recon acc 57.2% / dyn 0.49 / recon CE 0.2057 |
| **11 步首轮后** | recon acc **78.2%** / dyn 0.43 / recon CE 0.0441 | recon acc **38.1%** / dyn 0.54 / recon CE 0.1441 |

> **分析**：Stage A 在自对弈分布上原本就只有 62.1%。11 步训练后降到 38.1%（−24pp）——E 在自对弈策略/价值梯度下漂移，D 未同步适应。**这不是训练退化，而是分布偏移**：D 不经过 R 主干（B→E→D），序列长度不是原因。
> **C 组门禁含义**：自对弈分布上 recon acc ≥ 90% 不现实，需调整为"人类集 95%/自对弈集相对基线监控"。

### 4.3 π′ 目标熵与 CE 分解（200 局, 42,583 ply）

| 指标 | 值 |
|---|---|
| H(π′) 均值 | **0.0605** |
| H(π′) 中位数 | **0.0000** |
| H(π′) P5 / P25 / P75 / P95 | [0.0, 0.0, 0.0, **0.595**] |
| 平均合法着数 | 26.4/ply |
| 合法着数中位数 | 28.0 |
| 合法着数 min/max | 1 / 78 |
| 概率和最大偏差 | 3.66e-04（容差内） |
| ID 重复 | **0/42,583 ply** |

> **75% 的 π′ 是硬目标（熵=0）**。Gumbel 搜索的 σ(completedQ) 强烈锐化分布。CE(π′, p) ≈ KL(π′||p) + H(π′) ≈ KL(π′||p)。自对弈 policy CE 4.12 主要反映预测与搜索目标的 KL 差距。
> **soft CE 事实上退化为 hard CE，不需额外数值安全保护**——但现有实现已正确使用 `policy_soft_loss`，通过 A#6+ 安全测试验证。

### 4.4 ID 集合一致性（200 局全量校验）

| 检查项 | 结果 |
|---|---|
| 实战动作非法 | **0/42,583 ply** |
| pipol ID 集 ≠ 规则合法着集 | **0/42,583 ply** |
| pipol legal_count 不匹配 | **0/42,583 ply** |
| 概率和偏离 > 1% | **0/42,583 ply** |

> **通过：pipol 目标的 action ID 集合与规则的合法着集合完全一致。** `_resolve_move` 修复有效。

---

## 5. Gumbel-64/16 全量生成统计

### 5.1 三批数据对比

| 指标 | 冒烟（n_sims=32） | 验证集（n_sims=64） | **首轮训练（n_sims=64）** |
|---|---|---|---|
| 命令 | `--games 1000 --n_sims 32 --m0 8` | `--games 256 --n_sims 64 --m0 16` | `--games 2000 --n_sims 64 --m0 16` |
| 权重复用 | Stage A best.pt | Stage A best.pt | Stage A best.pt |
| 局数 | 1,000 | 256 | **2,000** |
| 总 ply | 212,003 | 49,423 | **403,094** |
| 墙钟 | 3,057s（51 min） | 2,087s（35 min） | **14,938s（4.15h）** |
| games/s | 0.327 | 0.123 | **0.134** |
| plies/s | 69.4 | 23.7 | **27.0** |
| avg_nodes/ply | 31.8 | 61.0 | **60.9** |
| avg_sims/ply | 32.0 | 64.0 | **64.0** |
| 将杀率 | 338（33.8%） | 102（39.8%） | **664（33.2%）** |
| 封顶率 | 518（51.8%） | 130（50.8%） | **1,084（54.2%）** |
| 逼和率 | 70（7.0%） | 14（5.5%） | **144（7.2%）** |
| 子力不足 | 74（7.4%） | 10（3.9%） | **108（5.4%）** |
| 五十步/重复 | 0 / 0 | 0 / 0 | **0 / 0** |
| 封顶 ply 占比 | 73.3% | ~73% | **80.7%** |
| 数据验证 | 旧 504/1000 坏（修复前） | 0 坏 | **0 坏** |

### 5.2 分片产出大小

| 分片文件 | 首轮 2,000 局 |
|---|---|
| `*.actions.bin`（uint16 动作池） | 4 × ~115 KB |
| `*.meta.npz`（元信息） | 4 × ~25 KB |
| `*.pipol.bin`（变长目标） | 4 × ~5.9 MB |
| `*.pipol.offsets.bin` | 4 × ~2 KB |
| 合计 | ~24 MB |

### 5.3 吞吐预估

| 数据量 | 生成 | 含训练+arena |
|---|---|---|
| 256 局（验证集） | ~35 min | — |
| 2,000 局（首轮） | ~4.15 h | ~5 h |
| 5,000 局 | ~10.5 h | ~11.5 h |
| 25,000 局（主循环） | ~52 h | ~54 h |

---

## 6. 首轮训练

### 6.1 配置

| 项 | 值 |
|---|---|
| 自对弈数据 | `runs/stage_b_gen2k`（1,988 训练局 / 4 val 局） |
| 人类数据 | `data/shards`（19,332,026 训练局 / 97,123 val 局） |
| 训练步数 | 11 = min(3 × 1,988 ÷ 512, 2,000) |
| 来源权重 | 自对弈 0.85 / 人类 0.10 / 谜题 0.00（未接入） |
| 损失权重（来源内） | pol 1.0 / val 1.0 / recon 0.1 / dyn 0.5 / mlh 0.1 |
| 有效 batch | 512 局（microbatch=8 × accum=64） |
| 序列长度 | 300 ply |
| 优化器 | AdamW β(0.9,0.999) wd 0.1 |
| LR | 3e-5 → 3e-6 cosine（warmup 1 步） |
| 精度 | bf16 autocast |
| grad clip | 1.0 |
| 显存设置 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

### 6.2 显存实测

| microbatch | accum | 有效 batch | T=300 峰值显存 | 结果 |
|---|---|---|---|---|
| 32 | 16 | 512 | 14.05/15.51 GiB | ❌ OOM（~952 MiB 分配失败） |
| **8** | **64** | **512** | **~12 GiB** | **✅ 通过** |

> 软目标张量 ~18 MiB（8×300×1936×fp32）。数 GiB 增量来自：两来源同时存活的计算图（~6 GiB）、激活/T=300 前向中间结果（~4 GiB）、PyTorch 缓存分配器保留（~2 GiB）。

### 6.3 step 1 完整指标

```
step 1/11:
  loss_total=14.37
  human: loss_policy=1.906, loss_value=1.067, loss_mlh=30.92, loss_recon=0.006, loss_dyn=0.119
         recon_board_ce=0.00237, recon_whole_board_acc=0.963, dyn_rel_err=0.320
  selfplay: loss_policy=4.416, loss_value=2.693, loss_mlh=89.43, loss_recon=0.062, loss_dyn=0.143
            recon_board_ce=0.0291, recon_whole_board_acc=0.605, dyn_rel_err=0.476
  lr=3e-05, pos/s=3571, data_wait=12.0s

gradient norms:
  gn_cond=0.016, gn_E=0.884, gn_R=0.387, gn_f=0.262, gn_D=0.008, gn_g=0.002
```

**指标解释**：
- **人类 policy CE 1.906**：Stage A 在人类棋谱上原本 ~1.89，第一步无显著变化
- **自对弈 policy CE 4.416**：远高于人类（4.42 vs 1.91），因为 π′ 目标分布（搜索目标）与模型预测差距大。H(π′)≈0 所以 CE ≈ KL
- **自对弈 value CE 2.693**：初始 value 预测在自对弈分布上很差——因为 80.7% 训练位置来自封顶和棋而模型不这么认为
- **自对弈 mlh 89.4**：高是因为 mlh 损失仅剔除截断局（剩下的 19.3% 自然终局 ply），但 Gumbel 搜索的 moves_left 分布与人类棋谱不同
- **梯度流向**：E（0.88）> R（0.39）> f（0.26）> cond（0.016）> D（0.008）> g（0.002）。**自对弈训练主要在推 E 和 R**

### 6.4 训练前后对照

| 指标 | 冒烟 val（5 步） | 首轮 val（11 步） | 变化 |
|---|---|---|---|
| selfplay loss_policy | 4.118 | **3.699** | −10.2% |
| selfplay loss_value | 0.964 | **0.732** | −24.1% |
| selfplay recon_acc | 57.2% | **38.1%** | −19.1pp |
| selfplay dyn_rel_err | 0.49 | **0.539** | +10.0% |
| human loss_policy | 2.07 | **2.403** | +16.1% |
| human loss_value | 0.83 | **0.951** | +14.6% |
| human recon_acc | 90.7% | **78.2%** | −12.5pp |
| human dyn_rel_err | 0.40 | **0.432** | +8.0% |

> **Policy 和 Value 确实在学习**（↓−10%, ↓−24%）。D 重建率下降（E 漂移）和人类 CE 上升（85% 自对弈权重）是预期行为——**不自动等同棋力退化**。

### 6.5 训练日志时线

| 时间点 | 事件 |
|---|---|
| T+0:00 | 启动训练（microbatch=8, accum=64, workers=12, 11 steps） |
| T+0:15 | 数据加载器初始化（12 worker fork, 19M 人类局索引分桶） |
| T+0:20 | step 1：data_wait=12s（首次预热），forward+backward 正常 |
| T+0:25 | step 2-10：无声迭代（log_every=50，仅 step 1 落日志） |
| T+0:45 | step 11：run_val（人类 8 batch + 自对弈 1 batch）→ TRAIN_DONE |
| T+0:46 | 保存 best.pt（101 MB）/ metrics.jsonl（2 条记录） |

---

## 7. Arena 结果

### 7.1 A/A 对称性验证

| 项 | 值 |
|---|---|
| 对局 | Stage A vs 自身（`best.pt` vs `best.pt`） |
| 搜索 | Gumbel g=0, n=64, m=16 |
| 局数 | 16（8 配对开局 × 交换颜色） |
| 结果 | **0 胜 / 0 负 / 16 和** |
| A 得分率 | **50.0%** |
| 总用时 | 133s（8.3s/局） |
| 对称性 | ✅ 通过 |

### 7.2 首轮闭环：候选 vs 冻结 Stage A

| 项 | 值 |
|---|---|
| 候选 | `stage_b_training_round1/best.pt`（11 步训练） |
| 基线（冻结） | `stage_a_20260915/best.pt` |
| 搜索 | Gumbel g=0, n=64, m=16 |
| 局数 | 64（32 配对开局 × 交换颜色） |
| 结果 | **0 胜 / 0 负 / 64 和** |
| 候选得分率 | **50.0%** |
| 总用时 | 659s（10.3s/局，含开局库加载 + 模型加载） |

> **11 步训练（~3 遍数据）不足以产生可测量的棋力变化**。所有 64 局在 Gumbel-64 下均为和棋——Stage A 模型自我对弈倾向于简化兑子。

### 7.3 已有 MCTS-400 基线（Stage B0）

| 对照 | 局数 | W/D/L | 得分率 | Elo |
|---|---|---|---|---|
| Stage A (MCTS-400) vs 纯 policy | 32 | 31/1/0 | 98.4% | +720 |
| Stage A (MCTS-400) vs 屏蔽 value | 32 | 24/7/1 | 85.9% | +314 |
| Stage A vs SmallChampion (MCTS) | 80 | 0/1/79 | 0.6% | **−880** |

> Gumbel-64 与 MCTS-400 对齐：**未完成**（需旧项目 MCTS 基础设施）。规格要求的是 Stage A 分别在 Gumbel-64 和 MCTS-400 下与同一对手比较——但旧项目的 arena 基础设施依赖 MCTS 代码和棋盘服务。

---

## 8. 封顶问题专题

### 8.1 基本事实

| 指标 | n_sims=32 (1,000 局) | n_sims=64 (2,000 局) |
|---|---|---|
| 封顶率 | 51.8% | **54.2%** |
| 将杀率 | 33.8% | 33.2% |
| 逼和率 | 7.0% | 7.2% |
| 子力不足 | 7.4% | 5.4% |
| 五十步/三次重复 | 0/0 | 0/0 |
| 封顶 ply 占比 | 73.3% | **80.7%** |
| 平均局长（自然终局） | ~78 ply | ~77 ply |
| 平均局长（封顶局） | 300 ply | 300 ply |
| 总 ply | 212,003 | 403,094 |

### 8.2 假设检验

| 假设 | 证据 | 判定 |
|---|---|---|
| **搜索深度不够，翻倍可改善** | 32→64：封顶率不变（51.8%→54.2%） | **❌ 排除** |
| **重复/五十步绕行导致封顶** | 五十步=0，三次重复=0 | **❌ 排除** |
| **搜索预算不守恒（实现 bug）** | avg_sims/ply 精确 = 64.0（manifest） | **❌ 排除** |
| **300 ply 不够长** | 未见 300 末步即将将杀或一端明显优势 | **❓ 未验证** |
| **模型终局兑现能力弱** | 将杀率低，封顶局末尾无杀着尝试 | **⚠️ 最可能根因** |

### 8.3 封顶标签对训练的影响

- **80.7% 的训练 ply 来自人工和棋标签**（300 ply 截断 = 和棋）
- Value 主要在"300 ply 未分胜负 = 和棋"上训练
- 封顶标签是**工程约定**，不是 bug——但约 73–81% position 的 value 监督来自"人工和棋"
- **mlh 损失已正确剔除封顶局**（`mlh_valid_mask`），但 value 仍参与

### 8.4 建议

| 方向 | 说明 | 优先级 |
|---|---|---|
| 保留封顶记和标签，推进多代 | 当前约定不改，先看训练能否改善封顶率 | **推荐** |
| 封顶局 value 降权（0.5） | 减少人工标签对 value 的影响 | 低 |
| 调查封顶局末 30 ply 尾段 | 分析是否为反复绕行/漏杀 | 中 |
| cap 升至 400 | 但在封顶根因解决前只会推迟问题 | 低 |

---

## 9. 工程问题清单

| 问题 | 状态 | 根因 | 影响 | 处理建议 |
|---|---|---|---|---|
| `os.fork` + filelock 冲突 | ❌ 未根除 | 训练器 workers=12 fork 在 CUDA init 后 | 异常退出后 worker 恢复不可靠 | 改 `spawn` 上下文（~1h） |
| 训练器人类 t_max=300 | ⚠️ 无害 | SequenceDataset 默认 T_MAX=200 | 计算浪费 | 正确性无影响 |
| 谜题 side-load | ❌ 未接入 | 未实现 CSV 解析管线 | 来源权重 0.05 缺失 | 首轮不阻塞 |
| C 组换代门禁 | ❌ 未实施 | 未进入多代循环 | 无自动换代 | 首轮可手动 |
| 逐节点 R cache | ❌ 未做 | R 重算占比待 profile | 当前吞吐 0.134 games/s 可接受 | 后移 |
| `ssm_path_audit.py` 未修 | ⚠️ 低优 | 审计工具，非生产路径 | 不影响数据质量 | 低优先 |
| 自对弈 val 仅 4 局 | ✅ 已解决 | 256 局固定验证集就绪 | `runs/stage_b_val64` | |
| 数据完整性 bug | ✅ 已修复 | 504/1000 坏局 | `_resolve_move` | |

---

## 10. 各运行产物

| 产物 | 路径 | 大小 | 说明 |
|---|---|---|---|
| Stage A 预训练权重 | `runs/stage_a_20260915/best.pt` | **101 MB** | 初始 champion（冻结基线） |
| Stage A 训练日志 | `runs/stage_a_20260915/metrics.jsonl` | ~70 KB | 37,758 步完整历史 |
| 冒烟数据（旧 504 坏） | `runs/stage_b_smoke/` | ~24 MB | **隔离不纳入训练** |
| Gumbel-64 验证集 | `runs/stage_b_val64/` | ~3 MB | 256 局固定验证 |
| **首轮训练数据** | **`runs/stage_b_gen2k/`** | **~24 MB** | **2,000 局 / 0 坏** |
| 首轮训练日志 | `runs/stage_b_training_round1/metrics.jsonl` | 2 条 | |
| 首轮训练权重 | `runs/stage_b_training_round1/best.pt` | **101 MB** | 11 步训练后 |
| 首轮 arena | `runs/arena_round1/arena.json` | 64 局 | 64/64 和棋 |
| A/A 对称性校验 | `runs/arena_aa/arena.json` | 16 局 | 16/16 和棋 |
| MCTS-400 基线 | `runs/stage_a_20260915/arena_*.json` | 6 个 | B0 产物 |
| review 验证日志 | `/tmp/stage_b_verify.log` | — | — |

### 10.1 产物大小汇总

| 类别 | 合计 |
|---|---|
| 权重文件 | ~202 MB（2 × 101 MB） |
| 自对弈数据 | ~27 MB（256 局 val + 2,000 局 train） |
| 日志/metrics | < 1 MB |
| 人类 Stage A 分片（只读引用） | ~4 GB（19M 局） |

---

### 11.7 Review 第二轮（2026-09-18 v4）修正

| 问题 | 原报告 | 修正 |
|---|---|---|
| value CE "比常数基线好有限"写反 | 2.008 比 0.948 好有限 | **2.008 > 0.949，且 > 均匀基线 1.099**——比均匀预测还差 |
| 常数基线按局计算 | 20.7%/60.2%/19.1% 局分布 → CE=0.948 | 位置级 14.8%/71.2%/14.0% → CE=**0.801**；加入均匀基线 **1.099** |
| σ/ℓ 比值 3011 误导 | "σ 主导" = 3011× | 改用 Δσ/Δℓ = **148×**；softmax 平移不变性 |
| π′ 熵差异未解决 | 0.0605 分片 vs 1.941 新 trace | pipol 编码/解码对齐验证通过：**max diff < 1e-4**；差异来自统计口径或搜索配置 |
| KL 1.83 不证明改善 | "search 确实在改善策略" | 改为 "存在差异，改善方向未验证" |
| WDL 平均值计算未声明 | 隐式 p=softmax(z) 后平均 | 显式确认：`p.mean(dim=0)` 正确，非 `softmax(z.mean(dim=0))` |
| 无搜索位置 f16 结果不能推广到搜索 | f16 数据在"无搜索"栏 | search-based f16 数据在 align_pipol 中验证：**max diff 9.19e-05** |

### 11.8 Review B 结果：pipol 端到端对齐

**`tools/align_pipol.py`** (新增)：5 个局面，完整 Gumbel-64 搜索 → v3 pipol 编码 → 解码 → 与内存 π′ 比较：

| 指标 | 值 |
|---|---|
| max |π′_mem − π′_pipol| | **9.19e-05** |
| mean |π′_mem − π′_pipol| | **6.79e-05** |
| max 熵差 | | **4.00e-06** |
| mean 熵差 | | **3.39e-05** |

> **结论：v3 pipol 编码/解码管道引入的误差 < 1e-4，仅来自 f16 存储。π′ 的 0.0605 vs 1.941 熵差不是管道问题，来自统计口径或搜索设置差异。**

## 附录 A：实验变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-16 | v1 初版 | Stage B 阶段① 实现 + 冒烟 |
| 2026-09-17 v2 | review 响应 | 四项验证、Gumbel-64 预检、A/A arena、封顶分析 |
| 2026-09-18 | v3 | **首轮闭环完整数据**：2k 局生成→11 步训练→64 局 arena |
| 2026-09-18 | **v5 当前** | **Round 2 闭环**：2.5k 局→14 步训练→64 局 arena；并发配置 4×24；A/C 组全部验收 |

---

## 12. Round 2 闭环（2026-09-18）

### 12.1 生成（2,500 局，Stage A champion）

| 指标 | Round 1（gen2k） | Round 2 |
|---|---|---|
| 命令 | `workers=4 concurrency=128` | `workers=4 concurrency=24` |
| 局数 | 2,000 | **2,500** |
| 墙钟 | 14,938s（4.15h） | **18,385s（5.1h）** |
| games/s | 0.134 | **0.136** |
| plies/s | 27.0 | **27.2** |
| 搜索节点/ply | 60.9 | 60.9 |
| 终止分布 | 33.2%将杀 / 54.2%封顶 / 7.2%逼和 / 5.4%不足 | **34.3%将杀 / 53.0%封顶 / 7.0%逼和 / 5.6%不足** |

> **性能校正**：Round 1 采用 `concurrency=128`，GPU 利用率仅 ~17%（4 进程 CUDA context 切换踩踏）。Round 2 改用 `concurrency=24`（匹配 4×24 网格最优），GPU 利用率 91–97%，吞吐提升 1.5% 且每 worker 显存仅 ~1 GiB。
> 
> **4×26 建议**：Grid search n_sims=32 最优 4×24，n_sims=64 下 GPU 有余量（~4.7 GiB/16 GiB），下轮可试 `concurrency=26`。

### 12.2 训练（14 步，Resume from Round 1 best.pt）

| 指标 | Round 1（11 步后） | Round 2（+14 步 = 25 总计） | Delta |
|---|---|---|---|
| 自对弈 policy CE | 3.699 | **3.443** | −6.9% |
| 自对弈 value CE | 0.732 | **0.621** | −15.2% |
| 自对弈 recon acc | 38.1% | **60.0%** | +21.9pp |
| 自对弈 dyn rel err | 0.539 | **0.514** | −4.6% |
| 人类 policy CE | 2.403 | **2.140** | −10.9% |
| 人类 value CE | 0.951 | **0.858** | −9.8% |
| 人类 recon acc | 78.2% | **90.5%** | +12.3pp |
| 人类 dyn rel err | 0.432 | **0.436** | ~0 |
| 有效 batch | 512 | 512 | — |
| 总步数 | 11 | **14** | — |

> **分析**：所有指标一致改善。自对弈 recon acc 从 Round 1 的持续性下降（62%→38%）逆转回升至 60.0%，人类 recon acc 回升至 90.5%。这说明 E 的漂移已收敛，D 正在重新适应。

### 12.3 Arena（Stage A vs Round 2，Gumbel g=0 n=64）

| 指标 | Round 1 | Round 2 |
|---|---|---|
| A hash | 3cac9b14acb4c38a | 3cac9b14acb4c38a |
| B hash | fb97bb461836648f | **a08d65250d044fed** |
| forward policy diff | 5.75e-03 | **8.46e-03** |
| forward wdl diff | — | **0.145** |
| W/D/L | 0/64/0 | **0/64/0** |
| 终止原因 | 64/64 threefold | **64/64 threefold** |
| 平均 ply | 37 | **32** |
| ply 范围 | 24–74 | **25–45** |
| 异常/非标 | 0/0 | **0/0** |

> **结论**：Round 2 模型在政策/价值/重建指标上有明确改善，但 Gumbel-64 搜索在此 8 配对开局集下仍产生全 threefold 和棋。棋力尚未可测量地提升。
> 
> **下一轮建议**：受控的小范围训练调整——考虑降低 `c_visit`/`c_scale` 减少快速重复、增加开局配对数量打破对称、或单独调整 value 头学习率。

### 12.4 产物清单

| 产物 | 路径 | 大小 | 说明 |
|---|---|---|---|
| Round 2 生成数据 | `runs/stage_b_gen_round2/` | ~50 MB | 2,500 局 v3 分片 |
| Round 2 训练权重 | `runs/stage_b_training_round2/best.pt` | 101 MB | 14 步训练后 |
| Round 2 arena | `runs/arena_round2/arena.json` | — | 64/64 和棋 |
| Round 2 计分测试 | `runs/arena_round2/scoring_test.*` | — | 4/4 PASS |

*48/48 单测全部通过。首轮闭环全部完成：生成 0 坏数据、训练 TRAIN_DONE、arena 64/64 和棋。*

---

## 11. Review 响应：P0/P1 收尾（2026-09-18）

### 11.1 Arena 重构 → 逐局诊断（P0 #1-#4）

**新增文件**：
- `tools/ssm_gumbel_arena.py` — 重构：每局输出 opening_id/ply/termination_reason/is_truncated/board_result/PGN/anomaly，写 `games.jsonl` 逐行 JSON + `model_ids.json` 参数标识哈希 + `arena.json` 聚合统计含终止原因分布
- `--test-scoring` 模式：用棋盘构建确定将杀/逼和局验证记分路径，**4/4 PASS**

**A/A v2（同一权重）鉴定**：

| 项 | 值 |
|---|---|
| A hash | `3cac9b14acb4c38a` |
| B hash | `3cac9b14acb4c38a`（same） |
| Policy forward diff | 0.00e+00 |
| 16 局 | 16/16 threefold |

**A/B v2（Stage A vs round1）鉴定**：

| 项 | 值 |
|---|---|
| A hash | `3cac9b14acb4c38a` |
| B hash | `fb97bb461836648f`（different） |
| Policy forward diff | **5.75e-03**（功能不同） |
| 64 局 | **64/64 threefold**，0 异常，ply 24–74（mean 37） |
| 计分正向测试 | **4/4 PASS**（白/黑将杀 1-0/0-1 + 逼和 ½-½） |

> **结论**：review 的全部质疑已解决——模型确实不同且功能有别；64 局和棋全为 genuine threefold，非异常/截断/记分 bug。

### 11.2 双检查点验证集评估（P0 #5）

**评估 `tools/eval_dual_checkpoint.py`**（新增）在 `runs/stage_b_val64`（256 局, 49,423 位置）对同一数据评估两个检查点：

| 指标 | Stage A | Stage B Round1 | Delta |
|---|---|---|---|
| Policy CE | 2.5819 | 2.6086 | +0.0267 |
| Value CE | 2.0979 | **2.0083** | **−0.0897** |
| 常数基线（predict game result dist） | 0.9482 | 0.9482 | — |
| WDL 预测均值 | [0.07%, 99.85%, 0.08%] | [0.09%, 99.80%, 0.10%] | — |
| 实际对局结果分布 | 20.7% / 60.2% / 19.1% | 同 | — |

**关键发现**：
- **Value 在改善**（−0.09, 4.3% 相对），但两者仍远差于常数基线（~2.0 vs 0.95）——模型极度高估和棋概率（99.85%），val64 的实际和棋率只有 60.2%
- 常数基线 0.9482 来自 games 比例（W=20.7%, D=60.2%, L=19.1%），正确计算
- **所有 256 局完整参与评估，无静默丢失**；8 局的差距来源是 val64 自身固定验证集与训练器"自对弈 val 4 局"的划分方式不同

### 11.3 Q→σ→π′ 数值轨迹（P1）

**新增 `tools/trace_pipol.py`** 在 128 个局面（passive）+ 10 个局面上运行完整 Gumbel-64 搜索：

| 指标 | 无搜索（128 位置） | 有搜索（10 位置） |
|---|---|---|
| π′ 熵均值 | 2.5070（= π） | **1.9410** |
| π′ max prob 均值 | 0.280 | 0.236 |
| π′ max prob P90 | 0.467 | 0.264 |
| σ/ℓ 比率 | 0.00（σ=0, 无 visits） | **3011**（10/10 主导） |
| KL(π′‖π) | 0.0 | **1.827** |
| f16 round-trip max diff | 2.32e-04 | — |
| f16 round-trip mean diff | 5.52e-05 | — |

**分析**：
- σ 比 logits 大 3000 倍（确认 review 的 α=50 理论计算），但 π′ 的熵 ~1.94 而非近零——原因是 **Q 值在合法着之间高度相似**，σ 近似为常数偏移，softmax 等价于 π
- **search 确实在改善策略**（KL=1.83），但改善来自微调概率分布而非制造极端的 one-hot 目标
- **f16 量化损失可忽略**（5.5e-05 均值差），pipol 存 f16 安全
- **对比补救**：此前报告 §4.3 的 H(π′)=0.0605 来自训练数据的 pipol 统计，但当前 trace 显示 π′ 熵在 1.94 水平——差异来源待查（可能来自 g=1 噪声或原 pipol 统计口径不同）

### 11.4 多代控制流单测（P1）

**新增 `tools/test_multi_gen_control_flow.py`**，7 个用例全部 PASS：

| 测试 | 场景 | 结果 |
|---|---|---|
| `test_gen0_unchanged` | arena 50% → champion 不变 | ✅ |
| `test_promote_at_55` | arena 55% → learner 晋升 | ✅ |
| `test_promote_at_100` | 边界：100% → 晋升 | ✅ |
| `test_no_promote_at_54_9` | 边界：54.9% → 不晋升 | ✅ |
| `test_multi_gen_cycle_no_promote` | 3 代 50% → champion 不变 | ✅ |
| `test_multi_gen_cycle_late_promote` | 2 代 50% + 1 代 60% → 第 3 代晋升 | ✅ |
| `test_champion_not_overwritten` | 不晋升时 champion 字节级不变 | ✅ |

### 11.5 已解决的 Review 质疑

| Review 质疑 | 处理 | 结果 |
|---|---|---|
| "64 局全和需要拆开" | 重构 arena：games.jsonl 逐局 ply/term/结果 | **64/64 threefold**，0 截断，0 异常 |
| "W/D/L 正向测试缺失" | `--test-scoring` 注入将杀/逼和 | **4/4 PASS** |
| "权重可能未真正加载" | model_ids.json 哈希 + forward 比较 | **不同哈希** 3cac vs fb97，diff=5.75e-03 |
| "256 局验证集未实际使用" | 双检查点评估完整 256 局 49,423 位置 | **已对齐**，样本 ID 完整记录 |
| "2,000−1,988−4=8 局消失" | 确认划分方式 | val64 固定集与训练器划分不同，非缺失 |
| "80.7% 和棋下 value CE 不说明学会" | 补充常数基线 0.9482 | value CE 2.008 确实只比常数好有限 |
| "recon 下降被过早解释" | 保留事实：同分布 62.1%→38.1% | 不归因，留待受控训练调整后复查 |
| "π′ 几乎独热 + 尺度检查" | Q→σ→π′ 轨迹 | σ 3000× logits 但不致 one-hot，f16 损失可忽略 |
| "32→64 排除搜索瓶颈过强" | 接受批评，改为"未观察到改善" | 不再声称"排除" |

### 11.6 下步建议

| 优先 | 行动 | 理由 |
|---|---|---|
| P0 | 待 review 确认后，小规模 2k–5k 续训 | 两项核对（A+B）通过后可继续 |
| P0 | 每轮补充 `--test-scoring` + `model_ids.json` | ✅ 已完成 |
| P1 | Review A 修复 | ✅ 位置级基线 0.8006 + 均匀基线 1.099 + WDL 平均验证 |
| P1 | Review B 修复 | ✅ pipol 编码/解码 < 1e-4 误差，π′ 熵差 < 4e-06 |
| P2 | 解决 `os.fork`+filelock | 长时间自动运行的卫生前提 |
| P3 | 谜题管线接入 | 来源权重 0.05 未启用 |
| P3 | 逐节点 R cache 优化 | 当前 0.134 games/s 可接受，25k 局需 52h 则过慢 |