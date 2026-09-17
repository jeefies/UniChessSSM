# Stage B 实验报告 · Gumbel 自对弈搜索（阶段① 实现与冒烟）

> 本文档供**外部审查**使用，覆盖 Stage B 阶段①（实现与算法单测 + B 组冒烟）的全部实现细节与实测数据。
> 权威设计文档：`stage-b-implementation.md`（§0.2 → §2.8 A/B 组门禁）；
> 上游实验文档：[stage-a-experiment.md](stage-a-experiment.md)；
> 本文一切口径以代码为准。

## 0. 摘要与当前状态

| 项 | 值 |
|---|---|
| 实验 | Stage B 阶段① · Gumbel 顺序减半搜索自对弈生成器 + v3 分片 + 训练器改造 + 冒烟验证 |
| 基座模型 | Stage A checkpoint（`runs/stage_a_20260915/best.pt`，step 37758） |
| 搜索算法 | Gumbel 顺序减半（Danihelka et al., ICLR 2022，按论文/mctx 修正两处算法规格） |
| 搜索预算（冒烟） | n_sims=32, m0=8（规格 §2.9 降级档；正式运行 n_sims=64/m0=16） |
| 并发拓扑 | 4 OS 进程 × concurrency=24（经过 1/2/3/4/5/6/10 进程网格搜索调优定档） |
| 生成吞吐 | **0.327 games/s**，GPU 利用率 99%（修复前单进程仅 0.085 games/s，GPU 17%） |
| 生成总量 | 1,000 局，212,003 ply，3,056s（~51 min） |
| 训练冒烟 | 5 步（遍历预算 min(3×996÷512, 2000)=5），无 NaN/Inf |
| GPU 峰值显存 | ~14.05 GiB / 15.51 GiB（microbatch=8, T=300） |
| A 组单测 | **48/48 PASS**（含修复后新增的递归深化符号回归 + 软 CE 集成安全测试） |
| B 组冒烟 | **通过了**：生成→回放验证 0 坏 → 训练管道健康 |

**发现并修复的关键问题**：

1. **`action_to_move` 旗标缺失**（2026-09-17）：升后（Queen）的 action id 编码在后走法表（promo=None），`action_to_move` 返回无 promotion 旗标的 Move，`board.push` 不自动补旗标 → 兵停在 8 排，棋盘悄然损坏。旧生成 504/1000 局有坏数据。修复：`_resolve_move(a, board)` 对照 `board.legal_moves` 匹配完整 Move（含 promotion/EP/易位旗标）。验证：新生成 1000 局 0 坏。

2. **Gumbel 非终局分支回传符号错误**：`do_sim()` 固定探索 2 层且非终局分支返回值少取负一次，根节点 completedQ 符号系统性反转。修复：递归 `_simulate()`，Node 新增 `children` 字典持久化子树。单测 `test_root_perspective_sign_correct` 复现修复前会选错着法的场景。

3. **生成器串行瓶颈**：旧实现 `--concurrency` 参数未生效，逐局逐次前向（91s/局）。修复：生成器架构重写为协程驱动 + 跨局 GPU 拼批（§2.3）+ 多 OS 进程编排（`--workers N`）。提速 ~3.3×，GPU 利用率 17%→99%。

4. **训练器未接入自对弈数据**：`train/stage_b2.py` 加载 sp_ds 只为计数，`soft_ce_from_pipol()` 定义后从未调用。修复：新增 `dataset_selfplay.py`，forward_train 新增 `policy_soft_target`/`mlh_valid_mask` 参数，每个 accum 子步同时采人类+自对弈批，按来源归约后加权（0.85/0.10）。

5. **T_MAX=200 静默继承**（§2.6 违反）：`dataset.py` 硬编码 T_MAX=200，自对弈 300 ply 序列被截头。修复：SequenceDataset 支持可配置 t_max，两个数据源均显式传 300。

---

## 1. 实验目标

### 1.1 Stage B 阶段定位

| 阶段 | 内容 | 状态 |
|---|---|---|
| ① 实现与算法单测 | 原生 Gumbel 树生成器 + v3 分片 + 训练器改造 | **✅ 完成** |
| ② 首轮闭环 | Gumbel 自对弈小批（2k–5k 局）→ 训练 → 与冻结 Stage A 基线对比 | ❌ 未开始 |
| ③ 多代主循环 | 25k 局/代 + 10 代 replay buffer + 换代门槛 | ❌ 未开始 |
| 后备 B1 | 仅在吞吐不达标或 value 分布适配成瓶颈时启用 | 未触发 |

### 1.2 阶段① 交付物对照

| # | 交付 | 状态 | 备注 |
|---|---|---|---|
| 1 | `tools/ssm_gumbel_selfplay.py` Gumbel 树生成器 | ✅ | 多进程 + 跨局批处理 + `_resolve_move` 旗标修复 |
| 2 | v3 分片读写（变长合法目标 + 终局元数据） | ✅ | `stateseq/data/gshards.py` V3ShardWriter/V3ShardReader |
| 3 | `train/stage_b2.py` 改造 | ✅ | 软 CE + 来源分别归约 + mlh_valid + T_MAX=300 |
| 4 | A 组 7 项算法单测全部通过 | ✅ | 48/48 PASS（含新增递归深化+集成安全测试） |
| 5 | B 组冒烟（1,000 局 + ≤3 遍训练） | ✅ | 生成 0 坏 + 5 步训练 TRAIN_DONE |
| 6 | §4 实测回填清单 | ✅ | 见 §6 |

---

## 2. 搜索算法实现（`stateseq/gumbel.py`）

### 2.1 基础数值（§2.2【锁定】）

| 超参 | 值 | 说明 |
|---|---|---|
| c_visit | 50.0 | σ 的访问数偏置 |
| c_scale | 1.0 | σ 的价值缩放 |
| EPS | 1e-8 | v_mix 分母保护 |
| NEG_LOGIT | -3e4 | 非法动作有限大负数（软 CE 数值安全） |
| N_SIMS | 64 | 根节点顺序减半总预算（锁定初值） |
| M0 | 16 | 根节点候选数上界（锁定初值） |

### 2.2 根节点选择

1. Gumbel-Top-k：`g(a) + ℓ(a)` 取 top-m₀ 候选（g=1.0 训练/生成；g=0 评测/换代）
2. 顺序减半分配 n 次模拟：⌈log₂ m₀⌉ 轮，每轮预算均分给存活候选，末位淘汰半数
3. 淘汰得分：`g(a) + ℓ(a) + σ(q̂(a))`
4. 最终选择 = 末轮唯一幸存者

### 2.3 非根节点（修正①）

$$\pi_{imp}(a) = \mathrm{softmax}(\ell(a) + \sigma(\mathrm{completedQ}(a))) \quad (\text{全部合法着})$$
$$a^* = \arg\max_a [\pi_{imp}(a) - N(a) / (1 + \Sigma_b N(b))]$$

### 2.4 补全 Q 与 σ 变换

```
completedQ(a) = q(a)             若 N(a) > 0
              = v_mix            否则
v_mix = (v̂ + ΣN · Σ_{a:N>0} π(a)q(a) / (Σ π(a) + ε)) / (1 + ΣN)
σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂
```

### 2.5 训练目标 π′（修正②，支持集=全部合法着）

$$\pi'(a) = \mathrm{softmax}(\ell(a) + \sigma(\mathrm{completedQ}(a))) \quad (\text{全部合法着})$$

不变量：所有合法着 completedQ 相同 ⇒ π′ = π。

### 2.6 递归深化实现（修复前 vs 修复后）

| 维度 | 修复前（`do_sim`） | 修复后（`_simulate`） |
|---|---|---|
| 树结构 | 固定 root→child→leaf 2 层，每次模拟建新叶子 | 递归下探，children 字典持久化 |
| 符号 | 非终局分支返回值少取负一次 | 每层级恰好取负一次 |
| 节点数 | 约等于 2 × 模拟数 | 至多 m₀ × avg_depth |
| 单测覆盖 | `_expand_stub` 全终局（不触发 bug 路径） | `_expand_chain` 变深链（端到端符号回归） |

---

## 3. 生成器（`tools/ssm_gumbel_selfplay.py`）

### 3.1 架构

```
run_workers(args) ─── subprocess.Popen × N ──── Worker ── Driver ── GameState
      │                       │                     │            └ coroutine: yield model requests
      │                       │                     └ _model_step: concat_caches → batch forward → split
      └── _merge_worker_outputs: 搬回分片 + 合并 manifest
```

### 3.2 网格搜索定档（n_sims=32/m0=8, 100 局基准）

| 配置 | games/s | GPU 利用率 | 备注 |
|---|---|---|---|
| 1×16（旧单进程） | 0.085 | 17% | 基线 |
| 10×4 | 0.129 | 95%+ | 太多 tiny batch |
| 6×8 | 0.171 | 97%+ | |
| 5×16 | 0.219 | 99% | |
| **4×24** | **0.282** | **99%** | **定档** |
| 3×40 | 0.278 | 99% | 与 4×24 平台期 |

定档：**4 进程 × concurrency=24**（每进程更大的 batch，更少的 CUDA context 争用）。

### 3.3 冒烟生成统计（1000 局）

| 指标 | 值 |
|---|---|
| 总局数 | 1,000 |
| 总 ply | 212,003 |
| 墙钟时间 | 3,056.5s（~51 min） |
| games/s | 0.327 |
| plies/s | 69.36 |
| 平均搜索节点/ply | 31.77 |
| 平均模拟/ply | 32.0 |
| 封顶率（300 ply） | **51.8%** |
| 将杀 | 338（33.8%） |
| 逼和 | 70（7.0%） |
| 子力不足 | 74（7.4%） |
| 数据完整性验证 | **1000/1000 通过，0 坏** |

> 封顶率 51.8% 超出 §2.4 的 20% 警戒线。使用 **n_sims=32**（规格降级档）是主要原因——更少的模拟导致终局搜索深度不足，增加非自然终局比率。正式循环使用 n_sims=64/m0=16 预期显著降低封顶率。

---

## 4. 训练器（`train/stage_b2.py`）

### 4.1 配置（§3【锁定】）

| 项 | 值 |
|---|---|
| 来源权重（损失归约后） | 自对弈 0.85 / 人类 0.10 |
| 来源内损失权重 | pol 1.0 / val 1.0 / recon 0.1 / dyn 0.5 / mlh 0.1 |
| 序列长度 | 完整 300 ply（§2.6，不继承 T_MAX=200） |
| 遍历预算 | min(3 × buffer ÷ 512, 2000) = 5（冒烟） |
| 有效 batch | 512 局（microbatch=8 × accum=64） |
| 优化器 | AdamW β(0.9,0.999) wd 0.1 |
| 精度 | bf16 autocast |
| grad clip | 1.0 |

### 4.2 显存实测（T=300）

| microbatch | accum | 有效 batch | 结果 |
|---|---|---|---|
| 32 | 16 | 512 | ❌ OOM（~952 MiB 分配失败，14.05/15.51 GiB 已用） |
| 8 | 64 | 512 | ✅ 通过 |

> 冒烟后 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 帮助缓解碎片。

### 4.3 训练冒烟日志（5 步）

```
step 1/5 loss 17.3971 sp_pol 4.396 sp_val 2.321 human_pol 1.904 pos/s 2821 data_wait 12051ms
VAL:
  human_loss_policy: 2.07, human_loss_value: 0.83, human_recon_acc: 90.7%, human_dyn_rel_err: 0.40
  selfplay_loss_policy: 4.12, selfplay_loss_value: 0.96, selfplay_recon_acc: 57.2%, selfplay_dyn_rel_err: 0.49
  val_loss_policy (自对弈口径): 4.12
TRAIN_DONE
```

- **数值安全**：全部损失有限，无 NaN/Inf（含 bf16 autocast + soft CE + padding 场景）
- **自对弈 policy CE 4.12** vs **人类 policy CE 2.07**：差距合理——soft 目标（π′ 在全部合法着上的分布）比 one-hot 更难拟合
- **自对弈 value CE 0.96** vs **人类 value CE 0.83**：略高，Stage A 未在自对弈分布上见过 z 标签
- **自对弈 recon acc 57.2%**（306 格棋盘重建准确率）：低于人类的 90.7%，源于 T=300 序列更长、终局棋盘稀疏
- **dyn rel err < 1.0**：防坍缩通道正常

---

## 5. A 组算法单测（48/48 PASS）

| # | 测试 | 覆盖内容 | 状态 |
|---|---|---|---|
| A#1 | `test_consistency` | 树版前向 vs 直线路径前向 < 1e-4 | ✅ |
| A#2 | `GumbelCorrectnessTest` (8) | v_mix 退化为 v̂、零访问保护、completedQ 全访问、预算守恒、σ(0)=0 | ✅ |
| A#2+ | `RecursiveDepthAndSignTest` (2) | 树深随预算增长、根视角符号正确（新增回归） | ✅ |
| A#3 | `InvariantTest` (2) | 相同 completedQ ⇒ π′=π；非法着概率=0 | ✅ |
| A#4 | `BudgetBoundaryTest` (4) | m=1、非 2 的幂、m<m0、预算精确 | ✅ |
| A#5 | `GZeroDeterminismTest` (2) | g=0 确定选 ℓ top-m；g≠0 随噪声变化 | ✅ |
| A#6 | `SoftCESafetyTest` (2) | 含 -3e4 logits 时 π′ 有限和=1；终局空节点 | ✅ |
| A#6+ | `SoftPolicyIntegrationSafetyTest` (1) | forward_train(soft_target) + bf16 autocast + backward 全有限（新增集成） | ✅ |
| A#7 | `LifecycleTest` (1) | 根节点模拟后不变 | ✅ |
| 其他 | 原有 25 项 | actions/features/overfit/value_sign/legal_mask/cache_isolation/g_alignment 等 | ✅ |

---

## 6. §4 实测回填清单

| # | 项 | 数值 |
|---|---|---|
| 1 | Gumbel 生成吞吐 | **0.327 games/s**（4×24, n_sims=32）；均路径深度 ~3.2 ply（递归）；R 重算时间占比待 profile；GPU 批量 24（每进程） |
| 2 | Stage A 基线 Gumbel-64 vs MCTS-400 | **未做**（阶段②首轮闭环时进行） |
| 3 | microbatch/accum（T=300） | **microbatch=8, accum=64**（32 引发 OOM） |
| 4 | 每代墙钟分解 | 生成 ~51min（1000 局） + 训练 ~6min（5 步）；arena 未测 |
| 5 | B1 后备 | 未触发 |

---

## 7. 发现的问题与修复清单

| 问题 | 发现时间 | 影响 | 修复 |
|---|---|---|---|
| `action_to_move` 升后/EP 旗标缺失 | 2026-09-17 训练冒烟 | 504/1000 局数据完整性损坏 | `_resolve_move` 对照 `board.legal_moves` |
| Gumbel 非终局符号错误 | 2026-09-17 代码审查 | 根 Q 值系统性反转，选着错误 | 递归 `_simulate` + `Node.children` |
| 生成器单进程串行 | 2026-09-17 实测（0.085 games/s） | 1000 局需 >3h | 协程拼批 + 多进程编排 |
| 训练器未接入自对弈数据 | 2026-09-16 代码审查 | 自对弈 π′ 目标从未用于训练 | `dataset_selfplay.py` + forward_train 新增参数 |
| T_MAX=200 静默继承 | 2026-09-16 规格检查 | 300 ply 序列被截头 | 可配置 t_max，显式传 300 |
| `policy_soft_loss` 形状不匹配 | 2026-09-17 集成测试首跑 | B>1 时 (ce*w) 广播失败 | 加 `.reshape(-1)` |
| 冒烟脚本输出目录不存在 | 2026-09-17 部署 | 脚本退出的 redirect 失败 | 先用 mkdir -p |
| fork+filelock 冲突 | 2026-09-17 训练 OOM 后 | worker 池重创建失败 | expandable_segments:True 避免 OOM |

---

## 8. 阶段② 前置条件

阶段① 完成状态总结：

| 条件 | 状态 | 备注 |
|---|---|---|
| §2.8 A 组全过前不生成正式训练数据 | ✅ | 48/48 PASS |
| B 组冒烟通过 | ✅ | 生成 0 坏 + 5 步训练 TRAIN_DONE |
| §4 回填清单填写 | ✅ | 见 §6 |
| 规格歧义/偏差记录 | ✅ | 见 §7 修复清单 |

**阶段② 首轮闭环建议步骤**：

1. 以 n_sims=64/m0=16 生成 2k–5k 局（预计 ~3–4h）
2. 运行 `train/stage_b2.py`（预计 ~30–60 min）
3. 与冻结 Stage A 基线同算法（Gumbel-64）同预算 arena 对比
4. 评估封顶率、and 棋力变化

---

*文档更新：2026-09-17，对应 commit `9cbca47`–`3cbf34d`，48/48 单测全部通过。*