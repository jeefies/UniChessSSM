# UniChess 状态序列模型 · 实施设计文档

> 版本：v2.0（实施交接稿）  日期：2026-09-14
> 读者：负责实现的 agent / 工程师。**本文档自包含**，不依赖任何对话上下文。
> 项目性质：游玩性质的小项目——不做消融实验矩阵，只保留最小冒烟门禁。
> 一句话定位：**带历史记忆与预测性辅助训练的策略—价值模型**（D/g 只在训练时辅助表示学习，不参与推理与规划）。
> 代码落地位置：WSL Ubuntu 项目 `/home/jeefy/UniChess`（本 Windows 目录只放文档）。

**版本沿革**
- v2.0：展开为实施级文档；新增数据管线规格、模块精确结构、阶段 0 验收测试、实施约束、附录 A（否决/缓行路线）。
- v1.3：依据外部评审修正（数据源勘误、D 改 MLP、MCTS 缓存重写、残差动力学措辞、参数/成本重算、损失归一化、序列一致性）。
- v1.2：全模块残差化；g 升级为残差动力学。v1.1：E=格子级 Transformer、R=基础 Mamba、Pre-RMSNorm。

---

## 0. 目录

1. 背景与现状  2. 决策记录  3. 总体架构  4. 数据管线规格  5. 模型规格（含全部公式）  6. 损失函数  7. 梯度回传与训练稳定性  8. 分阶段训练管线  9. MCTS 集成规格  10. 验证与门禁  11. 监控  12. 风险登记册  13. 实施约束  附录 A. 否决与缓行路线  附录 B. 参考资料

---

## 1. 背景与现状

UniChess 是一个国际象棋引擎项目。现役模型为 46M 参数 ResNet（24 残差块 × 320 通道），由 Stockfish 标注数据监督训练（蒸馏路线），内部评测棋力约 1300 Elo。

**已存在且可复用的基础设施**（位于 WSL `/home/jeefy/UniChess`，详见 `claude-history/HANDOFF.md`）：
- 自对弈 autoloop 系统（actors 批量 GPU 推理 + learner + arena），5070 Ti 主机常驻；
- 修复过的 MCTS 实现（root-Q 符号、真实重复历史、唯一挂起叶、升变 target、Syzygy 残局库接入）；
- champion 门控评测（24 色对 / 400 sims / paired sign p<.05 / score>.52）；
- 已部署的对弈网站与反向隧道（unichess-server / unichess-tunnel）。
- 环境：5070 Ti 主机 conda 环境 `/home/jeefy/miniconda3/envs/unichess/bin/python`；另有共享 PRO 6000 主机（仅限空闲窗口，项目目录 `/root/autodl-tmp/fwj/UniChess`，依赖装项目本地 pylibs，**不得动共享 conda**）。

**本项目（本文档覆盖范围）**：训练一个新架构模型——状态序列化输入（每步局面显式编码为向量序列）+ 时间主干整合全历史 + 世界模型式辅助训练。人类棋谱行为克隆（BC）预热，随后接入自对弈 RL。**不用 Stockfish 蒸馏。**

---

## 2. 决策记录（已锁定，实现时不得偏离；如需变更回到文档作者确认）

| # | 决策 | 内容 |
|---|---|---|
| D1 | 预热方案 | Lichess 原始 PGN 重放生成状态序列，人类走子 BC；不用任何引擎蒸馏 |
| D2 | 条件输入 | 保留 `[time_control][elo][color]` 三个条件向量 |
| D3 | 算力分配 | 5070 Ti 常开 autoloop；PRO 6000 空闲窗口做预热大训 |
| D4 | 规模 | 目标总量 ~28M 参数（预算表 §5.6，待代码实测核算） |
| D5 | 输入 | 全历史局面状态序列；棋盘固定**白方绝对坐标**，不随走子方翻转 |
| D6 | 架构 | E=格子级 Transformer（权重共享走两遍）；R=基础 Mamba；f=policy/WDL/moves-left 三头；D=MLP 重建解码器（仅训练）；g=残差动力学侧枝（仅训练） |
| D7 | Norm | 全局 Pre-RMSNorm；禁用 BatchNorm；同形堆叠块一律残差 |
| D8 | 实验纪律 | 不做消融矩阵；仅保留冒烟门禁（§10）；L_aux 暂缓 |
| D9 | 数据配比 | 初始：全分段保留 + 线性 Elo 加权 r≈20（初始选择，可按观测调整，非铁律） |
| D10 | 动作空间 | 紧凑 (from,to)+升变编码，1936 维（§3.3 精确构造，双射单测强制） |

---

## 3. 总体架构与核心规格

### 3.1 架构总览

```
                     ┌────────── 主路径（推理 = 这条） ──────────┐

 局面序列 B₀..B_T ──► E(·) ──► x₀..x_T ──► R(·) ──► h₀..h_T ──► f(·)
 (§3.2 特征,          格子级Transformer   局面嵌入序列   基础Mamba    隐状态   ├─ policy（1936+合法mask）
  白方绝对坐标)       (64格token×2遍)                                       ├─ WDL 价值
                      ▲                                    ▲               └─ moves-left
                      │                                    │
            D 重建侧枝 │                            g 动力学侧枝 │
            MLP(x_t)→64×13                  x̂_t = sg(x_{t-1}) + predictor(g(h_{t-1}, a_t))
            （辅助监督/诊断）                  L_dyn = ‖Δ̂ − sg(x_t − x_{t-1})‖²
```

| 模块 | 功能 | 训练 | 推理 |
|---|---|---|---|
| E（≈MuZero 的 h） | 局面 → x_t ∈ R^512 | ✅ | ✅ 每节点一次 |
| R | h_t = R(h_{t-1}, x_t)，12 层 Mamba | ✅ | ✅ 每节点单步 |
| f | policy / WDL / moves-left | ✅ | ✅ |
| D | x_t → 棋盘（辅助监督与诊断） | ✅ | ❌ |
| g + predictor | (h_{t-1}, a_t) → 增量 Δ̂（表示塑形） | ✅ | ❌ |

### 3.2 局面特征 B_t（无损；规则引擎推导；白方绝对坐标）

每半回合由规则引擎（python-chess 或项目现有 movegen）导出：

| 特征 | 形状 | 说明 |
|---|---|---|
| 棋子平面 | 12 × 64 二值 | 平面顺序：白 P N B R Q K，黑 P N B R Q K；格索引 a1=0, b1=1, …, h8=63 |
| 走子方 | 1 | 白走=1 |
| 易位权 | 4 | 白王/后翼、黑王/后翼 |
| 过路兵格 | 8 | 可吃过路兵的 file one-hot，无则全零 |
| 半回合计数 | 1 | halfmove_clock / 100 |
| 全回合数 | 1 | fullmove / 200（截断到 1） |
| 重复计数 | 2 | 当前局面此前出现 0/1/≥2 次 → [is1, is2]（参考特征，判定权威永远在规则引擎） |

合计约 790 维。**无损**：由上述字段可完整恢复 FEN。

### 3.3 动作空间（1936 维，紧凑编码）

- **非升变**：所有后/马可达的有序 (from,to) 对 = 1456（后走法）+ 336（马走法）= **1792** 个 id；
- **升变**：兵到末排的 (from,to) 对（白 7→8 线、黑 2→1 线，共 16 from 格 × 3 方向 = 48 对）× {R,B,N} 三种 = **144** 个 id（升后已含于 1792 的后走法中）；
- 合计 **1936**。id 编排规则由实现方定义并**写死为常量表**，强制单元测试：动作 id ↔ (from,to,promo) 双射、覆盖全部合法着（含全部升变）、与 §3.2 的白方绝对坐标一致。
- 参考：ChessMimic 用 1968 维（构造略异，https://arxiv.org/html/2606.04473v1 ）；备选标准方案为 AlphaZero 式 8×8×73=4672（更浪费但更常见）——选用 1936 是出于 softmax 成本。

### 3.4 条件输入

`[time_control]`（离散桶 embedding：bullet/blitz/rapid/classical/correspondence/other）、`[elo]`（双方平均 Elo，标准化到零均值单位方差后线性投影）、`[color]`（2 类）。三者各投影到 R^512 后与 x_t **相加**进 R。推理时置最大 Elo 桶求最强，置目标分段得人类化风格。缺失元数据的训练样本用专用 "unknown" 桶。

---

## 4. 数据管线规格（Stage A）

### 4.1 源数据

- Lichess 开放数据库 https://database.lichess.org/ 月度标准对局文件（`.pgn.zst`）。
- 首月即可启动（单月约数亿局，按吞吐实测决定取量）；**默认保留全分段**（D9）。
- 排除：变体棋（antichess/atomic 等）、超短局（<10 ply）、无结果局以外的异常终止（abandoned 可保留但结果按实际计）。
- 元数据提取：双方 Elo、time control、结果。

### 4.2 序列构建

对每局 PGN：用规则引擎从初始局面重放，每个半回合 t 产出一条记录：

```
(B_t 特征[790], a_t 动作id[1936], legal_mask_t[1936 二值],
 result ∈ {W,D,L}（对行棋方归一）, moves_left_t = (T − t)（单位：ply）,
 cond = {time_control 桶, elo_mean, color})
```

- 每局生成**双方视角两条序列**是旧方案（走子序列模型）的做法；本方案局面显式输入，**一条序列即可**（模型在每一步都预测行棋方动作）。
- moves_left 以 ply 计；超长截断到 T_max=200。
- 存储：复用项目现有分片惯例（或新设 `data/stateseq/` 分片），每片固定局数；按 game_id 哈希划分 train/val（val 取 0.5%），**同一局不得跨 train/val**。

### 4.3 Elo 加权

线性：`w(e) = (e − e_min) / (e_max − e_min) · (r−1) + 1`，e_min/e_max 取数据分布的 P1/P99 截断，r=20（D9）。归一化见 §6。

---

## 5. 模型规格（全部公式；目标 ~28M，见 §5.6）

记号：d=512（主干维度）；T≤200；d_e=256（E 内部宽度）；heads=8。

### 5.1 Norm 与残差总则

```
RMSNorm(u) = u / sqrt(mean(u²) + 1e-6) ⊙ γ        # 无均值居中、无 β
y = u + F(RMSNorm(u))                              # 所有同形堆叠块的统一形式
```

禁用 BatchNorm（理由：批次依赖统计量、train/eval 口径不一致、自对弈期分布漂移；旧 CNN 有 BN 漂移前科）。x_t 进 R 前再过一层 RMSNorm。

### 5.2 E：格子级 Transformer（d_e=256，单块权重共享走两遍）

```
逐格输入：s_i⁽⁰⁾ = piece_emb(c_i) + pos_emb(i) + W_g·globals        i=1..64
  piece_emb: 13类×256；pos_emb: 64×256 可学；globals = §3.2 非棋子字段
TrmBlock：S ← S + MHA(RMSNorm(S))；S ← S + MLP₄ₓ(RMSNorm(S))       # Pre-RMSNorm 残差
S⁽¹⁾ = TrmBlock(S⁽⁰⁾)；S⁽²⁾ = TrmBlock(S⁽¹⁾)                       # 同一权重走两遍（K=1）
x_t  = RMSNorm( q + CrossAttn(RMSNorm(q), S⁽²⁾, S⁽²⁾) )            # q∈R^256 可学单查询
x_t ← W_proj · x_t  （256→512，如需）再 RMSNorm
```

说明：第二遍=参数不增但 E 计算量翻倍；"再思考"语义未经本场景验证，K 是可调超参（默认 1）。

### 5.3 R：基础 Mamba（12 层，d=512）

依赖官方 `mamba_ssm`（Mamba-2 块）。每块配置：`d_model=512, expand=2 (d_inner=1024), d_state=16, d_conv=4, headdim=64`。块级：`h ← h + MambaBlock(RMSNorm(h))`。**scan 部分保持 fp32**（官方 kernel 默认）。

递推视角：`h_t = Ā_t ⊙ h_{t-1} + B̄_t x_t`（时间维残差结构：上一步状态 + 本步增量）。
R 的职责：整合历史脉络/重复线索/风格上下文；**不需要记棋盘**（当前局面由输入显式给出）。

### 5.4 f：预测头

```
u_t        = h_t + MLP₂(RMSNorm(h_t))                     # 2 层共享小变换，残差
z_t^p      = W_p u_t ∈ R^1936                              # policy logits
p(a|s_t)   = softmax(z_t^p + M_t)                          # M_t：非法着 -inf
(W,D,L)_t  = softmax(W_v · RMSNorm(u_t))                   # 三分类
m̂_t        = Mish(w_m · RMSNorm(u_t))                      # 剩余 ply 预测
```

### 5.5 D：MLP 重建解码器（仅训练）

```
B̂_t = reshape( W₂·GELU(W₁·RMSNorm(x_t)) , 64, 13 )         # 512→1024→832
辅助位头：走子方(2) / 易位权(4 路独立 sigmoid) / 半回合计数分桶(16 类)
```

定位：辅助监督与诊断工具，**不是可逆性证书**，不作为进入 RL 的门槛。

### 5.6 g + predictor：残差动力学（仅训练）

```
emb_a: 1936×512 动作嵌入表
u_t   = g( [RMSNorm(h_{t-1}); emb_a(a_t)] )                # MLP 1024→1024→512，残差
x̂_t   = sg(x_{t-1}) + predictor(u_t)                       # predictor: 512→512 两层 MLP
Δ̂_t   = predictor(u_t)
```

t=0 时 h_{-1} := h_init（可学向量）。

### 5.7 参数预算（估算，以代码统计为准）

| 模块 | 配置要点 | 估算 |
|---|---|---|
| E | d_e=256 单块（attn 0.26M + MLP 0.52M）+ 嵌入/池化/投影 | ≈ 1.3M |
| R | 12 × Mamba(512, expand2) ≈ 1.7M/层 | ≈ 20M |
| f | policy 512→1936 + WDL/MLH | ≈ 1.4M |
| D | 512→1024→832 + 辅助头 | ≈ 1.5M |
| g+predictor | 1024→1024→512 + 512→512 | ≈ 1.6M |
| emb_a + 条件 | 1936×512 等 | ≈ 1.1M |
| **合计** | | **≈ 27M** |

---

## 6. 损失函数（归一化口径统一，强制）

每步 t：π_t = policy target（Stage A=人类走子 one-hot；Stage B+=MCTS 访问分布），y_t=对局结果（对行棋方归一），e=双方平均 Elo。

```
L_policy = Σ_t w(e)·CE(π_t, p(·|s_t)) / Σ_t w(e)          # 按有效权重和归一
L_value  = mean_t CE( (W,D,L)_t , y_t )
L_mlh    = mean_t Huber( m̂_t − m_t )
L_recon  = mean_{t,sq} CE( B̂_t[sq] , B_t[sq] ) + 0.3·辅助位损失   # 按格平均（不是求和）
L_dyn    = mean_t (1/d)·‖ Δ̂_t − sg(x_t − x_{t-1}) ‖²
L = w_p·L_policy + w_v·L_value + w_m·L_mlh + w_r(τ)·L_recon + w_d·L_dyn
```

初值：`w_p=1.0, w_v=0.8, w_m=0.1, w_d=0.5`；`w_r` 从 1.0 线性退火到 0.1（前 30% 训练步）。上线前按首 1000 步各分量对共享主干的梯度范数校准，使各分量贡献同数量级。
（L_aux——合法着分布/被攻击格/对手下一着——**暂缓**，Stage C 视情追加。）

---

## 7. 梯度回传与训练稳定性

### 7.1 L_dyn 梯度路径

```
∂L/∂Δ̂_t ∝ Δ̂_t − sg(x_t − x_{t-1})
   ├─► predictor / g / emb_a 参数
   ├─► h_{t-1} ─► θ_R ─►（沿序列 BPTT）─► x_{t-1} ─► θ_E     ✅ 原因侧塑形
   ├─✖ base = sg(x_{t-1})            ⛔
   └─✖ target = sg(x_t − x_{t-1})    ⛔ 答案侧保护
```

原因/答案轮转：第 t 步 x_t 在答案侧，第 t+1 步它站上原因侧收梯度——E 在每个位置被 L_dyn 塑形，同时被 L_recon/L_policy/L_value 锚定。

### 7.2 坍缩风险缓解（注意：非数学充分保证）

sg（target 与 base 双侧）+ predictor 不对称头 + recon/policy/value 的表示压力。已知残余坍缩解（E(B)=c, Δ̂=0）由后三者压制。监控指标见 §10。

### 7.3 稳定性参数

- **Stage A 全序列训练**：T≤200 直接整序列前向（Mamba 并行扫描），不使用 TBPTT。若未来训练更长流：窗口间必须携带隐状态继续，禁止片段起点静默重置；权重更新后不复用旧隐状态。
- grad clip 1.0；AdamW β(0.9,0.999)、wd 0.1；lr 1e-4 cosine→1e-5，warmup 4000 步；dropout 0.1；bf16（scan fp32）。
- 结构性稳定来源：状态序列输入 → 信用分配路径短；每步密集损失 → 深监督效应；Mamba 对角转移特征值 <1 → 数值稳定。

---

## 8. 分阶段训练管线

### 阶段 0 · 接口验证（先行，详见 §10.1 验收清单）

小批 PGN 跑通 E/R/f/D/g 与数据管线；全部单元测试通过。

### Stage A · 人类棋谱预热（28M）

- 数据：§4 管线；初始配比全分段 + 线性 Elo 加权 r≈20（D9）。
- 有效 batch：microbatch 由显存实测决定 + 梯度累积；吞吐指标 = **有效局面/秒（完整前向+反向）**。不照搬走子级论文的 batch=2048。
- 出口（冒烟门禁）：留出集 policy/value 损失持续改善；recon/dyn 曲线健康（§10.2）；能完整对弈；与现役 46M champion 打**同口径** arena（固定 sims 与固定时间双口径）。
- **不做棋力承诺**：参考论文中 28M 走子序列模型 bullet ~2000 是另一套输入/训练/评测条件，仅作量级锚点。

### Stage B · 接入自对弈

- 沿用现有 MCTS/autoloop/champion 门控；KataGo 式 playout cap randomization + forced playouts/target pruning 按象棋与算力调整（数值不照搬围棋）。
- **序列一致性（关键正确性）**：快搜索回合**留在序列中**——无 policy 标签（不贡献 L_policy），但参与 R 递推与 g 的一步预测对齐；禁止从序列删除，否则相邻监督步之间隔多着而 g 仍预测一步，训练定义错误。
- Syzygy 真值注入（≤7 子局面 value 用 WDL 真值）。
- 重复/终局判定权威永远在规则引擎（python-chess 区分 is_repetition / can_claim；网络特征不作判定依据）。
- 出口：挑战旧 champion + 固定 puzzle 集 + 运行稳定性。

### Stage C/D · 按观察到的瓶颈再加功能

优先 reanalyze、搜索预算分配、吞吐优化；Gumbel、league、DAG 转置一律后移（DAG 的局面合并不适用于历史相关的 R 状态，§9）。

---

## 9. MCTS 集成规格（工程重点）

**节点状态 = 完整 Mamba cache，不是最后一层输出向量。**

- 每层缓存 `ssm_state`（d_inner×d_state）+ `conv_state`（d_inner×(d_conv−1)）。
- 每节点字节数：`L · d_inner · (d_state + d_conv−1) · sizeof(dtype)`。
  示例（L=12, d_inner=1024, d_state=16, d_conv=4, bf16）：12×1024×19×2 ≈ 0.44 MiB/节点 → 400 节点 ≈ 178 MiB → 64 树×400 节点 ≈ 11 GiB（**按假设估算，非实测**；fp32 更高）。
- **分支隔离**：官方单步接口原地更新状态；子节点必须复制父 cache（copy-on-write）；禁止两分支共享可写 cache（搜索顺序污染输出）。
- **缓存分离**：E 输出可按局面哈希共享/去重（路径无关）；R 状态路径相关，**禁止按相同棋盘合并**（同 B_t 不同历史 → R 状态不同）。
- **缓解**：控制并发树数/sims；cache 放 CPU pinned 内存按需交换；深层节点从根前缀重放（算力换显存）。
- 节点扩展流程：规则引擎推进 B → E 单步（带 E 缓存查询）→ R 单步 → f 出 (p, v)；PUCT 沿用现有实现。
- 无搜索模式（网站人类化）：每步一次递推 + policy 采样 + Elo 条件 token。

---

## 10. 验证与门禁

### 10.1 阶段 0 验收测试（全部通过才进 Stage A）

1. 动作空间双射：1936 id ↔ (from,to,promo) 全覆盖、含全部升变、与坐标约定一致；
2. 特征往返：随机合法局面 B → 特征 → 手工解码一致；
3. **整序列 vs 逐步递推一致性**：同一棋谱整段前向与逐步单步前向的 logits 差 < 1e-4（fp32 口径）；
4. 分支缓存隔离：同一节点复制出的两个子树互不影响输出；
5. 合法 mask：随机局面 mask 与规则引擎 legal_moves 完全一致；
6. 单 batch 过拟合：固定一小 batch 训练百步，总损失显著下降、无 NaN；
7. 价值符号与 moves_left 方向的单元测试。

### 10.2 冒烟门禁（训练中持续）

| 指标 | 口径 |
|---|---|
| recon | per-square 与**整盘完全一致率**双记录（0.999⁶⁴≈93.8%，勿混淆）；诊断用，不设硬门槛 |
| dyn 健康 | 相对误差 `E‖Δ̂−Δ‖²/(E‖Δ‖²+ε)` 显著 <1 且持续下降（优于"恒预测零变化"基线）；嵌入方差非零；Δ̂ 随动作变化 |
| 非法着 | 推理有 mask 恒为 0（无信息量）；记录 **mask 前非法倾向**（argmax 非法频率）作诊断 |
| puzzle | 分层准确率建档，供阶段间对比 |

---

## 11. 监控

分模块 grad norm（E/R/f/D/g 各自）、五损失分量曲线、recon 双口径、dyn 相对误差、mask 前非法倾向、arena（固定 sims / 固定时间双口径）。grad norm 突增 >10× 中位数 → 告警并回滚检查点。

---

## 12. 风险登记册

| 风险 | 等级 | 缓解 |
|---|---|---|
| 组合无完全一致的公开棋力验证 | 中 | 28M 小规模 + 门禁；不达标回退纯 Transformer 主干（E/管线不变，只换 R） |
| MCTS 缓存显存 | 中高 | §9 公式预估 + 并发控制 + CPU pinned/前缀重放 |
| 训练成本估计偏差 | 中 | 以实测有效局面/秒为准 |
| 多损失失衡 | 中 | §6 统一归约 + 梯度范数校准 |
| 表示坍缩 | 中 | §7.2 缓解组合 + §10.2 相对误差监控 |
| 自对弈分布漂移 | 中 | replay 窗口 + champion 门控（已有） |

---

## 13. 实施约束（交接给实现方）

1. 代码落在 WSL `/home/jeefy/UniChess` 新目录（建议 `model/stateseq/` + `tools/stateseq_*`），**不得修改**现有 autoloop/server/tunnel 的运行配置；
2. 5070 Ti 上现有服务（autoloop actors/learner、网站、隧道）持续运行，新训练先小步验证吞吐再放大；
3. PRO 6000 为共享主机：仅用空闲窗口、项目目录内操作、依赖装项目本地 pylibs；
4. 不确定处回到本文档作者确认，不要自行变更 D1–D10。

---

## 附录 A · 否决与缓行路线（含理由）

**A.1 已否决（有明确理由，不建议重开）**

| 路线 | 否决理由 |
|---|---|
| Stockfish 蒸馏（继续现有路线） | 项目目标即摆脱蒸馏；预热改用人类棋谱 BC |
| 字符级 PGN token | Karvonen Chess-GPT 实证：50M 字符级仅 ~1300–1500 Elo，token 效率低一个量级 |
| 纯 UCI 走子序列 + 纯 Mamba | 状态追踪=精确回忆任务，SSM 有损压缩结构性错配；Toshniwal：全注意力是追踪的必要条件；且 ~170 步序列吃不到线性复杂度红利（本方案改为显式局面输入后 Mamba 才重新成立） |
| 经典 LSTM/GRU 主干 | 长链梯度衰减，棋谱状态追踪实证远弱于注意力 |
| RWKV / xLSTM 主干 | 无任何公开象棋棋力实证，风险不对等 |
| Jamba 式混合（当前） | 实现复杂度高；先基础 Mamba，留作 R 不达标时的后备 |
| E-linear 无损线性投影 | 被格子级 Transformer 取代：局面内棋子关联需要注意力建模 |
| D 用 cross-attention 解码 | K/V 仅单 latent 时 softmax 恒为 1，退化为按 query 查表；改 MLP 更简 |
| searchless_chess BC 数据集 | 标签口径有争议（疑为引擎 oracle 动作），且为 FEN→动作记录、不含完整序列/条件信息 |
| BatchNorm | 批次依赖统计、train/eval 不一致、自对弈期漂移 |
| 指数 Elo 加权（r=200）/ 过滤低分局 | dual-capability 实证：摧毁追踪多样性，非法着率翻倍（对走子序列模型；本方案降为可调初始值） |
| 时间维折中（近期 k 步精确 + 远期摘要） | 用户明确否决；全历史完整输入 |
| GNN（棋子为节点） | 无公开强棋力实证，工程生态成本高 |
| NNUE | 评估函数而非策略模型，物种不同；可作对照组不进主线 |
| 4096/4672 全铺开动作空间 | softmax 成本；选 1936 紧凑编码 |
| 投降机制 | 采用 KataGo 式不投降 + 访问数退火替代，避免弱模型投降偏置污染标签 |

**A.2 缓行（非否决，Stage C/D 再评估）**

| 路线 | 缓行理由 |
|---|---|
| Gumbel MCTS | 低 sims 策略改进保证，Stage B 稳定后再引入 |
| League/exploiter 对手池 | 先单 champion 门控跑通 |
| DAG 转置表 | 历史相关的 R 状态禁止按局面合并；至多合并 E 缓存 |
| L_aux（合法着/攻击格/对手下一着辅助头） | 首版简化，Stage C 视瓶颈追加 |
| 潜在空间 rollout（g 参与推理） | 远期期权；规则引擎即完美动力学，当前无必要 |
| 模型放大（>28M） | 待 28M 门禁与瓶颈证据 |
| MoE / 多 token 预测头 | 与主干正交的增强，首版不引入 |

## 附录 B · 参考资料

**本会话检索核验**：
- Dual-Capability Bottleneck（2026-03）https://arxiv.org/html/2603.29761v1 —— 人类棋谱 BC 配方、Elo 加权、T/Q 框架、Pre-RMSNorm、28M/120M 训练配置
- Chessformer（2026-05）https://arxiv.org/html/2605.19091v1 —— 格子 token、GAB、注意力策略头、Lc0 +100 Elo
- ChessMimic（2026-06）https://arxiv.org/html/2606.04473v1 —— 紧凑动作空间、FEN token 化
- KataGo https://arxiv.org/pdf/1902.10565 与 https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md —— playout cap randomization、forced playouts、辅助目标
- Lc0 https://lczero.org/blog/2024/02/transformer-progress/ 与 https://github.com/leelachesszero/lc0/releases —— transformer 化 +270 Elo、WDL/moves-left 头
- Drama https://arxiv.org/html/2410.08893v1 —— Mamba 世界模型
- Chess-GPT 世界模型探针 https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html

**评审文件提供（未逐一亲自核验）**：
- SPR https://arxiv.org/html/2007.05929v3 —— 动力学仅作辅助训练的先例（本方案 g 的定位）
- EfficientZero https://arxiv.org/html/2111.00210v2 ；MuZero https://arxiv.org/abs/1911.08265
- Chess-World-Model https://arxiv.org/html/2605.30100v1 —— Mamba-2/3 象棋状态追踪基准
- Recurrent-Depth https://arxiv.org/abs/2502.05171 ；SimSiam https://arxiv.org/abs/2011.10566
- Lichess Database https://database.lichess.org/ ；python-chess https://python-chess.readthedocs.io/en/latest/core.html

**凭文献记忆（实现前需复核）**：Gumbel AlphaZero（ICLR 2022）；Jamba（AI21）；Mamba-2（SSD, Dao & Gu 2024）。
