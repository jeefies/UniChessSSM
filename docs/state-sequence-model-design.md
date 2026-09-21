# UniChess SSM 状态序列模型系统设计与框架说明书 (v3.0)

> **版本**：v3.0（框架权威技术手册）  
> **最后更新**：2026-09-21  
> **文档性质**：本文档为 UniChessSSM 项目的**唯一权威框架说明书**。  
> **范围界定**：本文档**严格仅保留两类核心信息**：
> 1. **框架真实的设计结构与代码实现**（Architecture, Dataflow, Feature Specs, Model Topologies, Loss Formulations, Gumbel Search Engine, Shard Protocols, Training & Inference Pipelines, Code Layout）；
> 2. **当前阶段存在的问题与技术债务**（Known Issues, Technical Debt, Numerical Constraints, Traps & Methodological Lessons）。
>
> *(注：所有分阶段历史实施计划、定性/定量基准实验结果、消融矩阵与探索推演，均已剥离至专用文档：`docs/stage-a-experiment.md`、`docs/stage-b-experiment.md`、`docs/stage-b-implementation.md`、`docs/stage-b-handoff.md` 与 `docs/offline-explorations.md`。原 `docs/design-deviations.md` 的所有已确认偏差与改进均已完整审核并合入本文档。)*

---

## 1. 核心架构设计与系统总览

UniChessSSM 是一个针对国际象棋的**状态序列模型（State-Sequence Model, SSM）**。区别于将局面序列化为字符 token 的文本生成模型或仅看单步画面的传统卷积网络，UniChessSSM 将整局历史中每一步棋盘状态显式编码为密集向量，通过双遍格子级 Transformer 提取单步局部拓扑空间嵌入，随后送入单向 12 层 Mamba-2 状态空间时序主干处理长程依赖与时空上下文，最终由轻量级三头多任务预测层输出策略与价值评估。在训练期间，辅以浅层棋盘重构模块（Model D）与残差动力学表征预测模块（Model g），在无引擎蒸馏的前提下实现高效自主表征塑造。

### 1.1 架构数据流图 (Dataflow)

```
                       ┌────────── 主前向路径（训练 + 推理共用） ──────────┐

  棋盘状态 B_t ────────► E(·) ─────► x_t ∈ R^512 ──(+)──► Pre-RMSNorm ──► R(·) ──► h_t ∈ R^512 ──► f(·)
  (785 维无损特征,       格子级Transformer         ▲                     12层Mamba-2       ├─ Policy logits (1936 + mask)
   白方绝对坐标)         (64格token×2遍)          │                     (逐层conv+ssm)     ├─ WDL 胜平负概率三分类
                                                   │                                       └─ Moves-Left 剩余步数
                         条件嵌入 c_t ──────────────┘
                         (time_control + elo + color)

                         ┌────────── 辅助监督侧枝（仅训练期激活，推理完全丢弃） ──────────┐
                         │                                                                  │
                         ▼ D 模块 (MLP 重建侧枝)                                            ▼ g 模块 (残差动力学侧枝)
                         B̂_t = D(x_t)                                                       Δ̂_t = predictor(g(h_{t-1}, emb_a(a_t)))
                         (64×13 棋子重构 + 3类全局辅助位)                                   x̂_t = sg(x_{t-1}) + Δ̂_t
                         L_recon 辅助空间表征锚定                                            L_dyn = ‖Δ̂_t − sg(x_t − x_{t-1})‖²
```

### 1.2 核心设计决策 (D1–D10 架构约束)

根据系统锁定规范，以下十项核心决策构成项目的不可变基石：

- **D1（预热方案）**：使用 Lichess 真实人类棋谱行为克隆（Behavior Cloning, BC）预热，严禁使用 Stockfish 或其他传统国际象棋引擎进行特征与分数蒸馏。
- **D2（条件输入）**：保留 `time_control`（时间控制）、`elo`（选手评分等级）、`color`（走子方视角）三类条件特征嵌入，使模型具备风格与强度的可控条件先验。
- **D3（算力与调度规范）**：自对弈生成与强化学习训练单卡分时交替，互不争抢显存；严格遵循各计算设备的隔离约束。
- **D4（模型参数规模）**：整体规模严格受控在 ~26.4M–28M 参数量，主干兼顾推理低延迟与表征容量。
- **D5（白方绝对坐标系）**：输入特征严格采用白方绝对坐标（a1..h8 拓扑恒定），棋盘不随当前走子方翻转；走子方信息由显式特征位提供。
- **D6（模块拓扑分配）**：
  - 格子级 Transformer $E$：单层权重共享循环两遍（$K=2$），无跨局面历史；
  - 基础主干 $R$：12 层 Mamba-2 状态空间模型（d_model=512, expand=2, d_state=16）；
  - 预测头 $f$：policy（1936 维）/ WDL 价值（3 分类）/ moves-left（1 标量）；
  - 辅助模块 $D$：浅层 MLP 局面重建解码器（训练期）；
  - 辅助模块 $g$：残差动力学增量预测器（训练期）。
- **D7（归一化与残差范式）**：全局采用 Pre-RMSNorm 架构；**全模型严禁使用 BatchNorm**（防止批次统计量泄露、自对弈分布漂移与 train/eval 不一致）；同形堆叠块严格残差连接。
- **D8（工程实验纪律）**：主线工程不做组合爆炸的消融矩阵，仅维持最小可验证的冒烟门禁，聚焦快速闭环。
- **D9（数据分布与加权）**：人类棋谱保留全分段覆盖，配合线性 Elo 权重聚焦高质量决策。
- **D10（紧凑动作空间）**：采用 1936 维紧凑动作空间（后/马合法有序对 + 升变特判），通过全量双射保证空间完备性与高效 softmax。

---

## 2. 局面与动作表征规范

### 2.1 785 维局面特征编码（白方绝对坐标）

每个半回合局面由规则引擎推导并编码为 $B_t \in \mathbb{R}^{785}$ 的密集张量。该编码是**确定性且完全无损的**（由该特征可精确逆向恢复 FEN 与全部棋规状态）：

| 索引区间 | 维度 | 字段定义与物理含义 | 取值范围与编码规范 |
|---|---|---|---|
| `[0, 768)` | 768 (12×64) | 12 个棋子平面的占用位棋盘 | 严格二值 $\{0, 1\}$。平面顺序：白 P, N, B, R, Q, K (0..5)，黑 P, N, B, R, Q, K (6..11)；格序按 a1=0, b1=1, ..., h8=63 绝对排列 |
| `[768]` | 1 | 走子方标记 (Side to move) | 白方走子 = 1.0，黑方走子 = 0.0 |
| `[769, 773)` | 4 | 四方易位权 (Castling rights) | 白王翼、白后翼、黑王翼、黑后翼，二值 $\{0, 1\}$ |
| `[773, 781)` | 8 | 吃过路兵列 (En-passant target file) | 列 a..h 的 8 维 One-Hot（仅当场上存在**合法可执行**吃过路兵走法时置 1，否则全零） |
| `[781]` | 1 | 半回合计数器 (Halfmove clock) | `halfmove_clock / 100.0`（截断上限 1.0，用于五十步和棋规则追踪） |
| `[782]` | 1 | 完整回合数 (Fullmove number) | `fullmove / 200.0`（截断上限 1.0） |
| `[783, 785)` | 2 | 局面重复计数历史 (Repetition references) | `[is_rep_1, is_rep_ge2]`，此前出现 1 次 / $\ge 2$ 次（供网络参考；终局判定权威严格在规则引擎） |

#### 零分配高速位棋盘提取器 (`encode_board_fast`)
为解决 Python 字典遍历与动态对象创建的高昂开销，核心编码器底层（`stateseq/features.py`）采用零堆分配（Zero-Allocation）位棋盘直接解包算法：
- 直接提取 `chess.Board` 的 `occupied_co` 及其各兵种 `uint64` 掩码；
- 使用预分配固定缓冲区 `_MASKS_SCRATCH` 与 `np.unpackbits` 对 64 位整数进行矢量化解包；
- 易位权与过路兵通过位运算提取，支持外部传入预分配张量 `out` 原地覆写；
- 编码单次延迟压降至 **8.97 $\mu\text{s}$**（相比通用实现的 49.92 $\mu\text{s}$ 提速 **5.57×**，吞吐达 11.1 万局面/秒），保证单步 MCTS / Gumbel 仿真中 CPU 瓶颈彻底消除，且输出与标准 785 维特征张量保持**逐位严格浮点一致**。

### 2.2 1936 维紧凑动作空间与双射协议

为降低全展开 4672 维动作空间对 softmax 与注意力头带来的冗余计算与显存开销，UniChessSSM 严格采用 1936 维紧凑动作空间：
$$\mathcal{A}_{\text{total}} = 1456 (\text{后走法}) + 336 (\text{马走法}) + 144 (\text{升变走法}) = 1936$$

1. **后走法有序对 (1456)**：包含车、象、后及兵的常规直斜进移动，以及王的两格横向移动（王车易位天然编码在王横移两格的有序对中，如白王 e1→g1、e1→c1，无需额外特判分支）。
2. **马走法有序对 (336)**：覆盖全盘所有合法马跃步。
3. **升变走法特判 (144)**：白兵 7→8 线、黑兵 2→1 线，共 16 个出发格 × 3 个名义方向（直推、左吃、右吃）× 3 种兵种 $\{R, B, N\} = 144$ 种动作（升后走法已完全被后走法 1456 空间覆盖，不重复计入）。

`stateseq/actions.py` 将上述规则固化为全局只读双射映射常量表（`ACTION_TO_MOVE` 与 `MOVE_TO_ACTION`），并受自动化单测严格看护，确保双向单射无空洞、无重叠。

### 2.3 多源条件输入嵌入 ($c_t$)

每个半回合的条件向量 $c_t \in \mathbb{R}^{512}$ 由三路特征独立线性投影后相加构成：
$$c_t = \mathrm{emb}_{tc}(b^{tc}) + W_e \cdot \hat{e} + \mathrm{emb}_{color}(j_t)$$
- **时间控制桶 ($\mathrm{emb}_{tc}$)**：7 类别 Embedding（bullet, blitz, rapid, classical, correspondence, other, unknown），依据对局总时限与加秒综合折算。
- **选手评分等级 ($W_e$)**：双方平均 Elo 经全量数据集 P1/P99 截断并归一化为零均值单位方差标量 $\hat{e} = (e - \mu) / \sigma$，经 1×512 线性变换投射。
- **走子方视角 ($\mathrm{emb}_{color}$)**：当前半回合走子方（2 类 Embedding）。**必须按每步实际走子方动态逐步切换**，严禁使用对局首步值静态绑定，确保时序展开中黑白方条件对称。

---

## 3. 模型各模块详细拓扑与数学规格

全模型实测参数量为 **26.40M**（完全符合 D4 预算要求）。

| 模块名称 | 源码定位 | 参数量 | 推理时是否启用 | 核心功能与维度 |
|---|---|---|---|---|
| **E (格子级 Transformer)** | `stateseq/model_e.py` | 1.14M | **是**（每步 1 次） | 785 维特征 $\to$ 64 格 Token $\to$ 2 遍权重共享 Transformer $\to$ 单查询聚合 $\to x_t \in \mathbb{R}^{512}$ |
| **R (Mamba-2 主干)** | `stateseq/model_r.py` | 19.25M | **是**（每步单步递推） | 12 层 Mamba-2，d_model=512, expand=2，隐状态时序更新 $h_t \in \mathbb{R}^{512}$ |
| **f (预测头群)** | `stateseq/heads.py` | 1.52M | **是**（每步输出） | policy logits (1936), WDL 价值 (3), moves-left 剩余步数 (1) |
| **D (MLP 重建侧枝)** | `stateseq/model_d.py` | 1.39M | **否**（仅训练） | $x_t \to 64\times 13$ 棋盘类别重构 + 走子方/易位权/半回合辅助预测 |
| **g + predictor (残差动力学)** | `stateseq/model_g.py` | 3.09M | **否**（仅训练） | $(h_{t-1}, a_t) \to \Delta \hat{t} \in \mathbb{R}^{512}$，对齐潜在状态差分 $\text{sg}(x_t - x_{t-1})$ |

### 3.1 编码器 E：格子级 Transformer（权重共享双遍，K=2）

编码器 $E$ 旨在将单步全局特征 $B_t$ 转化为高内聚的单一局面嵌入向量 $x_t \in \mathbb{R}^{512}$，不依赖任何历史记忆：
1. **格子 Token 初始构造**：
   $$s_i^{(0)} = \mathrm{piece\_emb}(c_i) + \mathrm{pos\_emb}(i) + W_g \cdot \mathrm{globals}_t \quad (i = 0 \dots 63)$$
   每个格子 $i$ 包含当前格棋子类别嵌入（13 类，含空格）、可学习绝对坐标嵌入 $\mathrm{pos\_emb}(i) \in \mathbb{R}^{256}$，以及 17 维全局状态（走子方、易位、过路兵、半回合数、重复标记）经线性投影 $W_g$ 注入的偏置。
2. **权重共享双遍前向 (K=2)**：
   $$S^{(1)} = \mathrm{TrmBlock}(S^{(0)}), \quad S^{(2)} = \mathrm{TrmBlock}(S^{(1)})$$
   单层 TransformerBlock 内部包含 Pre-RMSNorm 的 8 头多头自注意力（MHA）与 4 倍隐藏宽度的 MLP 残差块。同一组网络权重串行前向计算两次，使得全盘棋子交互感受野在不增加网络参数量的前提下得到倍增。
3. **可学习单查询全局汇聚 (Cross-Attention Pooling)**：
   $$x_t = \mathrm{RMSNorm}\left( W_{\text{proj}} \cdot \mathrm{CrossAttn}\left(q_{\text{learn}}, S^{(2)}, S^{(2)}\right) \right) \in \mathbb{R}^{512}$$
   利用单一可学习的查询向量 $q_{\text{learn}} \in \mathbb{R}^{256}$ 对 64 个格子的上下文特征进行注意力加权汇聚，经线性投影放大至 512 维，经 Pre-RMSNorm 后输出。

### 3.2 时序主干 R：12 层 Mamba-2 状态空间模型

主干网络 $R$ 负责建模整局走子的因果发展、战术积累与重复局面历史上下文：
- **基础超参**：层数 $L = 12$，$d_{\text{model}} = 512$，$d_{\text{inner}} = 1024$（$\text{expand}=2$），状态维度 $d_{\text{state}} = 16$，局部卷积核宽 $d_{\text{conv}} = 4$，头维度 $\text{headdim} = 64$。
- **数值精度规范**：官方 `mamba_ssm` kernel 内部并行关联扫描（Selective Scan）必须**全程保持 float32 数值精度**，防止深度时序累积下溢或梯度爆炸。
- **递推双模态一致性**：
  - **训练期**：整条序列并行关联扫描（Parallel Scan），单次前向完成 $T$ 步计算；
  - **推理/自对弈期**：逐步单步递推 `step(x_t, cache)`。单步内部通过原地维护每个块的 `(conv_state, ssm_state)` 推进状态，显存占用恒定为 $O(1)$。
  - **浮点扰动界**：官方 GPU Triton/CUDA kernel 处于并行 Scan 与逐步 decode 之间存在硬件层固有微小计算顺序差异。在 fp32 下，单层累积误差约为 $\sim 1\text{e-}5$，12 层累积叠加 policy 头放大后 logits 最大标量差界定在 $\le 2\text{e-}4$。**系统严格断言：推理实际消费的 Masked Softmax 策略概率绝对差必须 $< 1\text{e-}4$**（实测 $< 1.2\text{e-}5$），WDL 与 Moves-Left 的 raw logits 绝对差严格 $< 1\text{e-}4$。

### 3.3 预测头群 f

时序输出隐向量 $h_t \in \mathbb{R}^{512}$ 首先通过共享的两层残差适配层：
$$u_t = h_t + \mathrm{MLP}_2(\mathrm{RMSNorm}(h_t))$$
随后分流至三个多任务头：
1. **策略头 (Policy Head)**：
   $$z_t^p = W_p \cdot u_t \in \mathbb{R}^{1936}, \quad p(a \mid s_t) = \mathrm{softmax}(z_t^p + M_t)$$
   其中 $M_t$ 为规则引擎输出的非法着法掩码。**数值安全硬约束：非法动作处填充掩码必须使用有限大负数 $-3\times 10^4$（严禁使用 $-\infty$）**，彻底避免 Softmax 反向传播梯度与交叉熵计算中触发 $0 \times (-\infty) = \mathrm{NaN}$。
2. **WDL 胜平负价值头 (Value Head)**：
   $$(P_W, P_D, P_L)_t = \mathrm{softmax}(W_v \cdot \mathrm{RMSNorm}(u_t)) \in \Delta^2$$
   输出当前**行棋方视角**的胜、和、负三项分类概率。节点标量价值严格定义为期望胜负分：
   $$Q = P_W - P_L \in [-1.0, 1.0]$$
3. **剩余步数头 (Moves-Left Head, MLH)**：
   $$\hat{m}_t = \mathrm{Mish}(W_m \cdot \mathrm{RMSNorm}(u_t)) \in \mathbb{R}_{\ge 0}$$
   预测距本局真实终止剩余的半回合步数（ply）。

### 3.4 训练期辅助侧枝：Model D 与 Model g

两模块仅在训练反向传播中提供辅助梯度流，模型导出推理与自对弈树搜索时完全不参与计算：
1. **Model D（局面重构解码器）**：
   $$\hat{B}_t = \mathrm{reshape}\left( W_2 \cdot \mathrm{GELU}(W_1 \cdot \mathrm{RMSNorm}(x_t)), 64, 13 \right)$$
   由局面向量 $x_t$ 逆向重建 64 格的棋子分类，辅以走子方（2 分类）、易位权（4 路独立 BCE）与半回合分桶预测。用于对 $E$ 施加显式信息保留压力，确保空间表征不坍缩。
2. **Model g（残差动力学侧枝）**：
   包含动作嵌入表 $\mathrm{emb}_a \in \mathbb{R}^{1936 \times 512}$ 与增量预测器：
   $$u_t^g = g\left( [\mathrm{RMSNorm}(h_{t-1}); \mathrm{emb}_a(a_t)] \right), \quad \hat{\Delta}_t = \mathrm{predictor}(u_t^g)$$
   网络预测一步动作转移引起的潜在空间差分：
   $$\mathcal{L}_{\text{dyn}} = \frac{1}{d} \left\| \hat{\Delta}_t - \text{sg}(x_t - x_{t-1}) \right\|_2^2$$
   **阻止梯度反传保护机制（Stop-Gradient, sg）**：目标项 $\text{sg}(x_t - x_{t-1})$ 必须双侧切断梯度（答案侧绝对保护），使得动力学损失只沿 $h_{t-1} \to \theta_R \to x_{t-1} \to \theta_E$ 反向传播，仅对前置状态产生结构化动力学塑形，坚决避免平凡坍缩解（即 $E(B) \equiv \text{const}, \hat{\Delta} \equiv 0$）。

---

## 4. 多任务联合损失体系与数值口径

训练总损失严格定义为五项归一化分量的加权和：
$$\mathcal{L} = w_p \mathcal{L}_{\text{policy}} + w_v \mathcal{L}_{\text{value}} + w_m \mathcal{L}_{\text{mlh}} + w_r(\tau) \mathcal{L}_{\text{recon}} + w_d \mathcal{L}_{\text{dyn}}$$

### 4.1 各分量精确公式与掩码约束

1. **策略损失 ($\mathcal{L}_{\text{policy}}$)**：
   $$\mathcal{L}_{\text{policy}} = \frac{\sum_t w(e) \cdot \mathrm{CrossEntropy}(\pi_t, p(\cdot \mid s_t))}{\sum_t w(e)}$$
   - 在 Stage A（人类棋谱）中，$\pi_t$ 为实际走法的 One-Hot 标签；
   - 在 Stage B（自对弈）中，$\pi_t$ 为 Gumbel 搜索导出的软目标概率分布 $\pi'$；
   - 损失按有效权重和严格归一；非法着法 logits 经 $-3\times 10^4$ 屏蔽。
2. **价值损失 ($\mathcal{L}_{\text{value}}$)**：
   $$\mathcal{L}_{\text{value}} = \frac{1}{T} \sum_t \mathrm{CrossEntropy}((P_W, P_D, P_L)_t, y_t)$$
   $y_t \in \{W, D, L\}$ 为对局终局结果转换至当前行棋方视角的 One-Hot 标签（零和对齐）。
3. **剩余步数损失 ($\mathcal{L}_{\text{mlh}}$)**：
   标准模式下采用绝对步数 Huber 损失（$\delta=1.0$）：
   $$\mathcal{L}_{\text{mlh}} = \frac{1}{\sum_t \mathbf{1}_{\text{valid}}} \sum_{t: \text{valid}} \mathrm{Huber}_\delta(\hat{m}_t - m_t)$$
   - **截断与无效样本剔除**：**截断局（`is_truncated=1`）及谜题样本必须从 MLH 损失中严格剔除**（`mlh_valid=0`），防止伪造和棋导致的步数标签系统性污染。
   - **可选 Log-Huber 变换模式 (`--mlh-log`)**：当启用该开关时，对预测与真实目标施加 $\log(1 + \mathrm{ReLU}(\cdot))$ 空间映射并采用 $\delta=0.5$，将开局阶段高达 100+ plies 的残差梯度模长压降 88% 以上，彻底解除 MLH 对主干表征梯度的过度劫持。
4. **局面重建损失 ($\mathcal{L}_{\text{recon}}$)**：
   $$\mathcal{L}_{\text{recon}} = \frac{1}{64} \sum_{\text{sq}=0}^{63} \mathrm{CE}(\hat{B}_t[\text{sq}], B_t[\text{sq}]) + 0.3 \cdot \mathcal{L}_{\text{aux}}$$
   严格按全盘 64 格平均计算（严禁写成 64 格求和，避免权重失衡）。
5. **残差动力学损失 ($\mathcal{L}_{\text{dyn}}$)**：
   $$\mathcal{L}_{\text{dyn}} = \frac{1}{T \cdot d} \sum_t \left\| \hat{\Delta}_t - \text{sg}(x_t - x_{t-1}) \right\|_2^2$$
   诊断指标 `dyn_rel_err` 必须在有效位置掩码 `pos_mask` 下统计：$\frac{\mathbb{E}\|\hat{\Delta} - \Delta\|^2}{\mathbb{E}\|\Delta\|^2 + \epsilon}$，健康阈值应显著 $< 1.0$（实测 Stage A 稳定于 0.26）。

### 4.2 权重配置与退火日程

- 锁定配置：$w_p = 1.0, w_v = 0.8, w_m = 0.1, w_d = 0.5$；
- 重建权重 $w_r(\tau)$：在前 30% 训练步内由 $1.0$ 线性退火至 $0.1$，随后保持 $0.1$。
- **多源数据混合权重**：在 Stage B 混合训练中，自对弈数据占 85%（软策略标签 + 最终结果 $z$），人类棋谱占 10%（人类走子 BC + 真实结果），谜题数据占 5%（战术首步 + 强制将杀标记，MLH 剔除）。三类数据源在各自子批次独立计算归一化损失后再按权重加权求和，**不存在单源梯度主导问题**。

---

## 5. Gumbel-Top-k 顺序减半树搜索引擎

Stage B 采用无经典 UCB 探索公式的 Gumbel 顺序减半（Sequential Halving）启发式树搜索（实现于 `stateseq/gumbel.py` 与 `tools/ssm_gumbel_selfplay.py`）。该算法在浅层模拟（如 $n=64$）下具有严格的策略改进保证。

### 5.1 搜索超参锁定标准

- **根节点初始候选集容量**：$m_0 = 16$（当合法着法不足 16 时取全部合法着法）；
- **总模拟预算**：$n = 64$ 次前向递推；
- **顺序减半轮次**：固定 4 轮迭代（候选集沿 $16 \to 8 \to 4 \to 2 \to 1$ 逐轮减半，模拟次数分配为 $32 \to 16 \to 8 \to 8$）；
- **探索常数**：$c_{\text{visit}} = 50$；
- **价值尺度缩放常数【锁定】**：**$c_{\text{scale}} = 0.1$**（严禁恢复为历史遗留值 1.0）。

### 5.2 核心数学机制

#### 1. 根节点 Gumbel 噪声初始化
对根节点所有合法动作 $a \in \mathcal{A}_{\text{legal}}$，提取网络前向策略 logits $\ell(a)$，独立采样标准 Gumbel 噪声 $g(a) \sim \mathrm{Gumbel}(0, 1)$。依据 $g(a) + \ell(a)$ 选取 Top-$m_0$ 动作作为初始候选集 $S_0$。

#### 2. 补全 Q 估计 (Completed Q) 与端点保护
对于树中任意未访问的分支动作，不可直接置零，必须采用混淆值（Mixed Value）进行插值补全：
$$v_{\text{mix}} = \frac{v_{\text{root}} + \sum_{b \in \mathcal{A}_{\text{visited}}} N(b) Q(b)}{1 + \sum_{b \in \mathcal{A}_{\text{visited}}} N(b)}$$
$$\bar{Q}(a) = \begin{cases} Q(a) & \text{若 } N(a) > 0 \\ v_{\text{mix}} & \text{若 } N(a) = 0 \end{cases}$$
该公式具备严格的端点保护特性：在零访问极端情况下平滑退化为先验估值 $v_{\text{root}}$，分母加 1 避免零除。

#### 3. 逐节点局部 completed-Q 归一化 (`qtransform_completed`)【锁定】
为消除全树全局跨节点 Q 值极差将局部细微差距过度压缩的缺陷，**归一化量程必须严格按当前节点内部全部合法动作的 completed-Q 取局部 min-max**：
$$q_{\min} = \min_{b \in \mathcal{A}} \bar{Q}(b), \quad q_{\max} = \max_{b \in \mathcal{A}} \bar{Q}(b), \quad \text{span} = q_{\max} - q_{\min}$$
$$q_{\text{norm}}(a) = \begin{cases} \frac{\bar{Q}(a) - q_{\min}}{\text{span}} & \text{若 } \text{span} > 1\text{e-}6 \\ 0.5 & \text{若 } \text{span} \le 1\text{e-}6 \end{cases}$$
`gumbel.qtransform_completed` 是**全系统唯一**的 Q 值变换函数，由**根节点淘汰打分、非根节点分支选择、以及最终策略目标 $\pi'$ 导出三处完全共用**，确保搜索行为与训练监督目标同构。

#### 4. 尺度变换 $\sigma(\hat{q})$ 与 $c_{\text{scale}}=0.1$ 的必要性
价值调整项定义为：
$$\sigma(\hat{q}(a)) = (c_{\text{visit}} + \max_b N(b)) \cdot c_{\text{scale}} \cdot q_{\text{norm}}(a)$$
当历史采用 $c_{\text{scale}}=1.0$ 时，系数常年高达 $50 \sim 80$，将仅 $\sim 0.03$ 的细微价值差异剧烈放大为 18+ logit 差，导致 60.3% 的局面策略目标退化为绝对确定的 One-Hot 分布；而采用 $c_{\text{scale}}=0.1$ 时，系数回落至 $5 \sim 8$，使网络先验与价值探索保持健康平衡。

#### 5. 非根节点选择法则
非根节点不使用 UCB 上置信界，而是基于改进策略分布与访问计数的显式惩罚匹配：
$$\pi_{\text{imp}}(a) = \mathrm{softmax}\left( \ell(a) + \sigma(q_{\text{norm}}(a)) \right)$$
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{legal}}} \left[ \pi_{\text{imp}}(a) - \frac{N(a)}{1 + \sum_b N(b)} \right]$$

#### 6. 变长合法策略监督目标 $\pi'$ 导出
搜索结束时，训练目标 $\pi'$ **必须在全部合法动作集合上计算 Softmax**（不仅限于被搜索采样的候选集），保证未搜索分支仍保留合法的先验概率反向梯度：
$$\pi'(a) = \mathrm{softmax}\left( \ell(a) + \sigma(q_{\text{norm}}(a)) \right)_{a \in \mathcal{A}_{\text{legal}}}$$
该公式天然满足恒等不变性：若所有合法动作的 Completed-Q 严格相等，则 $\pi' \equiv \pi$。

---

## 6. 时序缓存与树搜索生命周期管理

在带状态主干的 Mamba-2 模型上运行树搜索，显存管理与状态隔离是工程成败的关键。

### 6.1 R Cache 拓扑与分支隔离

Mamba-2 模型的单步递推状态包含每一层的卷积状态与 SSM 状态：
- 每层状态：$\mathrm{ssm\_state} \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}}$，$\mathrm{conv\_state} \in \mathbb{R}^{d_{\text{inner}} \times (d_{\text{conv}}-1)}$；
- 单个节点 R Cache 尺寸（$L=12, \text{bfloat16}$）：
  $$12 \times 1024 \times (16 + 3) \times 2 \text{ Bytes} \approx 0.44 \text{ MiB / 节点}$$
- **原地改写破坏性约束**：官方 `Mamba2.step()` 底层会对传入的 `conv_state` 与 `ssm_state` 张量进行**原地（In-place）破坏性覆写**。
- **快照与隔离机制**：
  - 每局自对弈维护唯一的**实战根节点不可变快照**（Root Cache Snapshot）；
  - 每轮模拟从根快照浅拷贝克隆一份独立工作张量（Working Cache）；
  - 树内节点仅缓存 $x = E(B_t)$ 向量（bf16 仅 1 KiB）；叶子扩展时沿搜索路径单向重放递推，严禁任何跨分支可写 Cache 共享。

### 6.2 槽位复用与并发安全

自对弈生成器（`tools/ssm_gumbel_selfplay.py`）采用预分配固定槽位（Slots）的批处理并发架构：
- **生命周期清空准则**：某对局终止时，其所在槽位的 R Cache、棋盘实例、哈希 occurrence 计数器必须**完全重置为零**；
- **并发独立性**：各并发槽位间张量前向操作严格批量独立，防止任何跨局隐状态交叉污染；
- **权重只读隔离**：每个代次的自对弈进程在启动时单次加载 Champion 权重至 GPU，对弈期间保持 `torch.no_grad()` 与只读冻结。

---

## 7. 数据分片协议规范 (v2 与 v3)

项目数据管线严禁将数十亿步的中间状态全部以明文张量持久化存储，而是采用**紧凑二进制动作流 + 结构化元数据**存储范式。

### 7.1 v2 分片协议（Stage A 人类棋谱专用，只读冻结）

- `*.meta.npz`：每局 16 字节定长 Numpy 结构化数组，字段包含：`n_plies` (u16), `tc_bucket` (u8), `result` (u8), `elo_mean` (f32), `game_key` (u64)。
- `*.actions.bin`：全局平铺的 uint16 紧凑动作 ID 池。训练时由数据加载器多进程重放棋盘并在线提取 785 维特征。

### 7.2 v3 分片协议（Stage B 强化学习自对弈专用）

v3 分片支持变长搜索软策略分布 $\pi'$、丰富终局因果分析与严格的数据溯源：

```
分片文件群结构：
├── shard_00000.meta.npz       # 结构化数组：定长 56 字节/局
├── shard_00000.actions.bin    # 实战动作序列：uint16 紧凑 ID 平铺
├── shard_00000.pipol.bin      # 变长 π' 稀疏目标：二进制紧凑编码
└── shard_00000.pipol.offsets.bin # 各局各步在 pipol.bin 中的起始偏移索引
```

#### 1. 变长策略目标编码 (`*.pipol.bin`)
每个半回合仅持久化当前局面的合法动作搜索结果：
$$\text{存储单元} = \underbrace{\text{u16 } K}_{\text{合法动作数}} + \sum_{k=1}^K \left( \underbrace{\text{u16 } a_k}_{\text{紧凑动作 ID}} + \underbrace{\text{f16 } p_k}_{\text{搜索导出概率 } \pi'(a_k)} \right)$$
相比全量 1936 维密集存储，存储空间压降 **96.5%** 以上。

#### 2. 终局因果判定权威与唯一入口 (`adapter.classify_final_board`)
为彻底杜绝规则申和提前退出的语义不一致问题，系统固化 `adapter.classify_final_board` 为终局裁决的**唯一权威入口**：
- **终局原因分类枚举**：`checkmate`（将杀）、`stalemate`（逼和）、`insufficient_material`（子力不足）、`threefold`（三次重复，包含当前步走完即满足申和条件的局面）、`fifty_move`（五十步规则）、`truncated`（达到 300 ply 步数硬封顶）。
- **截断与训练标记**：若为封顶截断局，元数据中 `is_truncated = 1`，终局价值 $z$ 按和棋（0.0）记录，**但其 `mlh_valid` 标志必须置为 0（训练时彻底剔除出剩余步数损失）**。

---

## 8. 代码库模块映射与工程拓扑

核心源码位于 `stateseq/`、`train/`、`tools/` 与 `tests/`，层次边界严格隔离：

```
UniChessSSM/
├── stateseq/                   # 核心模型与算法 Python 包
│   ├── actions.py              # 1936 动作空间常量表、双射与合法掩码
│   ├── features.py             # 785 维局面特征编码/解码，含 encode_board_fast 位棋盘加速
│   ├── conditions.py           # time_control / elo / color 多源条件嵌入投影层
│   ├── model_e.py              # 格子级 Transformer E (d_e=256, 权重共享双遍)
│   ├── model_r.py              # 12 层 Mamba-2 状态空间主干 (mamba_ssm 包装)
│   ├── model_d.py              # MLP 局面重建模块 (仅训练辅助)
│   ├── model_g.py              # 残差动力学侧枝与 predictor (仅训练辅助)
│   ├── heads.py                # policy (1936), WDL (3), moves-left (1) 预测头
│   ├── losses.py               # 统一多任务损失定义 (含 pos_mask 保护与可选 Log-Huber)
│   ├── gumbel.py               # 纯 CPU/Numpy Gumbel-Top-k 顺序减半搜索核心算法
│   ├── model.py                # SSMModel 全架构总装集成
│   └── data/
│       ├── shards.py           # v2 分片数据读写
│       └── gshards.py          # v3 变长策略分片读写与元数据校验
├── train/
│   ├── stage_a.py              # Stage A 人类棋谱 BC 预训练器
│   └── stage_b2.py             # Stage B2 强化学习自对弈训练器 (85/10/5 混合监督)
├── tools/                      # 离线工具、自对弈生成器与评测 Arena
│   ├── ssm_gumbel_selfplay.py  # 多进程原生 Gumbel 树搜索自对弈生成器 (支持 --openings)
│   ├── ssm_gumbel_arena.py     # 换代对抗评测 Arena (含 --sprt 早停检验与逐局诊断)
│   ├── ssm_infer_server.py     # 单 GPU Unix Domain Socket 推理服务器
│   ├── ssm_uci.py / ssm_uci.sh # 跨进程纯 CPU UCI 引擎客户端
│   ├── ssm_path_audit.py       # 跨链路前向逐位精度对拍工具
│   └── repair_v3_meta.py       # 历史分片元数据就地修复工具
└── tests/                      # 自动化回归单元测试集
    ├── test_actions.py         # 动作空间完备性与双射测试
    ├── test_features.py        # 特征编码与逆向解码往返一致性
    ├── test_gumbel.py          # Gumbel 搜索算法、Completed-Q 量程与目标一致性测试
    ├── test_gshards_v3.py      # v3 分片二进制序列化往返测试
    ├── test_termination_classify.py # 终局原因判定唯一入口一致性测试
    └── test_sprt_arena.py      # 换代 Arena Wald SPRT 序贯检验逻辑测试
```

---

## 9. 当前阶段已知问题与技术债务

本节系统梳理当前版本在算法、工程链路与算力协同中已识别的客观缺陷、历史陷阱与技术债务，供后续迭代针对性攻坚。

### 9.1 自对弈多进程显存并发陷阱 (Worker OOM Trap)

- **现象**：在远端 15.51 GiB 显存的 5070 Ti 机器上运行 `ssm_gumbel_selfplay.py` 时，若设置 `--workers 3 --concurrency 128`，所有 Worker 进程瞬间崩溃并抛出 `torch.OutOfMemoryError`。
- **根因**：生成器中 `--concurrency` 参数的语义是**每个 Worker 进程各自独立的并发局数**，而非全局总量。每个子进程均创建了独立的 PyTorch CUDA Context（底噪开销 ~500 MiB）及 128 个完整槽位的张量分配。3 个 Worker 各占约 5.1 GiB，瞬间突破 15.51 GiB 总显存上限。
- **现行边界与防线**：
  - 显存预算必须按 $\text{Workers} \times \text{Concurrency}$ 计算；
  - 实测安全并发配置：`--workers 4 --concurrency 24`（合计 96 并发，显存稳定占用 ~11.8 GiB，GPU 算力利用率 97%）；
  - 失败时生成器故意不合并半成品分片，防止破损数据混入正式训练集。

### 9.2 历史分片数据定级：旧版 Teacher 策略不可用作搜索目标

- **背景与状态**：
  - `stage_b_smoke`、`stage_b_val64`、`stage_b_gen2k`、`stage_b_gen_round2` 四个历史分片生成于早期 Adapter 修复（commit `4835b79`）之前。
  - 当时搜索存在三大缺陷：WDL 差分作用于未经 Softmax 的原始 logits、使用了未标准化的原始 Elo、终局叶子由网络估值而非规则真值接入。
  - `stage_b_gen_fix500` 虽修复了 Adapter，但使用了过度激进的 `c_scale=1.0`（60% 目标处于伪确定性坍缩态）。
- **处置方案与技术债务**：
  - 上述分片的 `actions` 序列、终局结果 $z$ 及经 `repair_v3_meta.py` 修复后的终局元数据**完全可用**；
  - **上述分片的 `pi_prime` 字段已被正式定级为 `legacy_teacher`，严禁作为最新策略改进的监督目标**；
  - 当前待办任务依赖以 $c_{\text{scale}}=0.1$ 重新生成的标准数据集（`stage_b_gen_fix500_cs01`）。

### 9.3 基础网络战术盲区（两步杀漏算与低价值分辨率）

- **现象**：Stage A 预训练权重在面对简单战术（如强制两步将杀 Mate-in-2）时，纯 Policy 预测经常下出严重缓着甚至漏杀。
- **根因分析**：
  - 预热完全基于 Lichess 人类快棋对局（超快棋占比高，人类在复杂战术中存在大量错漏）；
  - 模型未引入任何战术引擎离线强化；
  - 价值头在复杂局面下的标量辨识度有限，在 $c_{\text{scale}}=1.0$ 时容易将 $\pm 0.03$ 的价值噪声误判为胜负手。
- **缓解方向**：依赖 Stage B 自对弈中 Gumbel 树搜索对局部战术走法的强制展开筛选，并在后续阶段按规划注入 5% 高质量残局与战术谜题。

### 9.4 终局裁决平局偏置与和棋奖励妥协

- **权衡记录**：系统当前严格执行国际象棋规则裁决（可申和的三次重复与五十步规则直接判定为和棋，结果 $z=0.0$）。
- **潜在风险**：早期自对弈模型在优势局面下可能由于缺乏强烈的破和驱动，反复运子导致进入三次重复求和。目前三次重复占自然终局的 38.0%。系统已明确暂不引入启发式和棋惩罚，避免在模型棋力尚不稳定时引入人为奖励偏置，保持纯净零和博弈属性。

### 9.5 历史提交哈希悬空与文档陈旧引用

- **状态记录**：由于早期远端仓库执行过 Git Rebase，早期文档广泛记录的提交哈希（如 `8d14a58`, `784dc64`, `154d673`, `f1146ac`, `2c12930`, `8331930`）已成为悬空提交（不可从 HEAD 祖先追溯）。
- **代码状态**：所有对应缺陷的真实代码修复均已完整合入主分支并受单元测试保护（真实对应提交分别为 `6cde89b`, `dd620d5`, `b423ede`, `4835b79`, `45fc1b6`, `ee0801b`）。设计说明书已彻底解耦对悬空哈希的依赖，仅以当前实际源码与行为合约为准。

---
*(说明书完结)*
