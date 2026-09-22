"""Gumbel 顺序减半搜索（Danihelka et al., ICLR 2022）——纯逻辑核心，不依赖 torch / python-chess。

规格来源：`docs/stage-b-implementation.md` §2.2【锁定】。
本模块只处理"节点 + 选择规则 + 预算分配"的算术，所有浮点运算在 numpy fp32 下完成，
因此可以在无 GPU、无棋盘的条件下做精确的算法单测（§2.8 A#2/#3/#4/#5）。

约定：
- `Node.q` 为**该行棋方视角**的价值标量 q = P(win) − P(loss) ∈ [−1, 1]，和棋 = 0。
  与 WDL 三头换算 `q = wdl[0] − wdl[2]` 同构，与 Stage A 训练标签（行棋方归一）一致。
- 跨边取负（零和）：父节点上"动作 a 的价值" = −(子节点 q)。
- σ(q̂) 的 q̂ 是**该节点自身 completed Q 集合**的 min−max 归一值（2026-09-19 规格变更，
  对齐 mctx `qtransform_completed_by_mix_value`）。此前用整棵树的 qbox，且该 qbox 以
  父视角 root.q 起算、用子视角 child.q 扩展——视角混用且跨节点，会改变 ℓ 与 Q 的相对
  权重。根评分 / 非根选择 / π′ 导出共用 `qtransform_completed` 一个函数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- 锁定超参（§2.2 / §3）----
C_VISIT = 50.0          # σ 的访问数偏置
C_SCALE = 0.1           # σ 的价值缩放（2026-09-20 由 1.0 改，见 design-deviations.md §9.3）
EPS = 1e-8              # 分母保护
NEG_LOGIT = -3e4        # 非法动作的**有限**大负数（软 CE 数值安全，§2.6 / A#6）
N_SIMS = 256            # 根节点顺序减半总预算
M0 = 16                 # 根节点候选数上界

# 终止原因编码（§2.5 v3 meta.termination_reason）
TERM_CHECKMATE = 0
TERM_STALEMATE = 1
TERM_FIFTY_MOVE = 2
TERM_THREEFOLD = 3
TERM_INSUFFICIENT = 4
TERM_TRUNCATED = 5
TERM_CODES = ("checkmate", "stalemate", "fifty_move", "threefold",
              "insufficient_material", "truncated")


# ---------------- 基础数值 ----------------

def softmax(x: np.ndarray) -> np.ndarray:
    """稳定 softmax，fp32 输出，行和 = 1。"""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x, dtype=np.float32)
    return e / e.sum(dtype=np.float64).astype(np.float32)


def sigma(q_hat: np.ndarray | float, n_max: int | np.ndarray,
          c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> np.ndarray:
    """σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂。"""
    return (c_visit + np.asarray(n_max, dtype=np.float32)) * c_scale * np.asarray(q_hat, np.float32)


def normalize_q(q: np.ndarray | float, q_min: float, q_max: float) -> np.ndarray:
    """min−max 归一：q̂ = (q − q_min) / (q_max − q_min + eps) ∈ [0, 1]。

    ⚠️ 2026-09-19 起 q_min/q_max **必须来自当前节点自身的 completed Q 集合**
    （见 `qtransform_completed`）。此前传入的是整棵树的 min/max，且该统计以父视角的
    `root.q` 起算、用子视角的 `child.q` 扩展——视角混用 + 跨节点量程，会改变
    ℓ 与 Q 在打分中的相对权重（σ 的分母被其他节点撑大后，价值差异被压缩）。
    """
    span = float(q_max) - float(q_min) + EPS
    return (np.asarray(q, np.float32) - np.float32(q_min)) / np.float32(span)


# ---------------- 节点 ----------------

@dataclass
class Node:
    """搜索树节点。所有数组维度对齐 `legal`。

    q：该行棋方视角价值。terminal=True 时 legal 为空、q 为规则真值（行棋方 −1 负 / 0 和 / +1 胜）。
    """

    legal: np.ndarray                      # (n,) int64 合法动作 id
    logits: np.ndarray                     # (n,) fp32 policy logits（仅合法集合，已掩码）
    q: float                               # 行棋方视角价值
    depth: int = 0                         # 距根的层数（根 = 0）
    action: int | None = None              # 从父节点到本节点的动作（根为 None）
    path: tuple[int, ...] = ()             # 从根到本节点的动作序列（含本节点动作）
    terminal: bool = False                 # 终局局面（无合法着）
    n: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    q_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    root_key: int = 0                      # 本搜索内的根分组键（用于跨局拼批）
    children: dict = field(default_factory=dict)  # action_id(int) -> Node，跨模拟持久化子树（递归变深必需）

    @property
    def is_terminal(self) -> bool:
        """终局局面：显式 terminal 标记或无合法着。"""
        return bool(self.terminal) or self.legal.size == 0

    @property
    def n_total(self) -> int:
        return int(self.n.sum()) if self.n.size else 0

    @property
    def n_max(self) -> int:
        return int(self.n.max()) if self.n.size else 0

    def record_child(self, edge_idx: int, child_value: float) -> None:
        """累加一条边的访问：child_value 已转换为**本节点行棋方视角**。"""
        if self.n.size == 0:
            n = len(self.legal)
            self.n = np.zeros(n, np.int64)
            self.q_sum = np.zeros(n, np.float32)
        self.n[edge_idx] += 1
        self.q_sum[edge_idx] += np.float32(child_value)

    def q_edge(self, edge_idx: int) -> float | None:
        """某条边的已访问均值（本节点行棋方视角）；未访问返回 None。"""
        if self.n.size == 0 or self.n[edge_idx] == 0:
            return None
        return float(self.q_sum[edge_idx] / self.n[edge_idx])


# ---------------- completed Q 与策略 ----------------

def policy_probs(node: Node) -> np.ndarray:
    """π = softmax(ℓ)，在全部合法着上。终局节点返回空。"""
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    return softmax(node.logits)


def v_mix(node: Node, c_visit: float = C_VISIT) -> float:
    """v_mix = (v̂ + Σ_b N(b) · Σ_{a:N(a)>0} π(a)q(a)/(Σ_{a:N(a)>0} π(a)+ε)) / (1 + Σ_b N(b))。

    端点保护：无访问时退化为 v̂；分母加 ε（§2.2 review 意见）。
    q(a) 取该边的已访问均值（本节点行棋方视角），与 v̂ 同视角。
    """
    v_hat = float(node.q)
    if node.n.size == 0 or node.n_total == 0:
        return v_hat
    pi = policy_probs(node)
    visited = np.flatnonzero(node.n > 0)
    num = float(np.dot(pi[visited], node.q_sum[visited] / node.n[visited].astype(np.float32)))
    den = float(pi[visited].sum()) + EPS
    n_tot = float(node.n_total)
    return (v_hat + n_tot * (num / den)) / (1.0 + n_tot)


def completed_q(node: Node) -> np.ndarray:
    """completedQ(a) = q(a) 若 N(a)>0，否则 v_mix。返回按 `legal` 对齐的 fp32 向量。"""
    if node.terminal or node.legal.size == 0:
        return np.zeros(0, np.float32)
    vm = v_mix(node)
    out = np.full(len(node.legal), vm, np.float32)
    if node.n.size:
        visited = np.flatnonzero(node.n > 0)
        out[visited] = (node.q_sum[visited] / node.n[visited].astype(np.float32))
    return out


def qtransform_completed(node: Node, c_visit: float = C_VISIT,
                         c_scale: float = C_SCALE) -> np.ndarray:
    """**唯一的 Q→打分变换**：σ(q̂)，q̂ 为该节点自身 completed Q 集合的 min−max 归一。

    根评分（顺序减半淘汰）、非根选择、π′ 导出三处共用本函数，保证同一节点上
    "选择依据"与"监督目标"完全一致。对齐 mctx 的
    `qtransform_completed_by_mix_value`：先在当前节点补全全部合法动作的 Q，
    再对**这一组**值重缩放，而不是用跨节点/跨视角的全树 qbox。
    """
    cq = completed_q(node)
    if cq.size == 0:
        return cq
    q_hat = normalize_q(cq, float(cq.min()), float(cq.max()))
    return sigma(q_hat, node.n_max, c_visit, c_scale)


def improved_policy(node: Node, c_visit: float = C_VISIT,
                    c_scale: float = C_SCALE) -> np.ndarray:
    """π_imp = softmax(ℓ + σ(completedQ))，在全部合法着上。"""
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    return softmax(node.logits + qtransform_completed(node, c_visit, c_scale))


def pi_prime(node: Node, c_visit: float = C_VISIT,
             c_scale: float = C_SCALE) -> np.ndarray:
    """训练目标 π′(a) = softmax(ℓ + σ(completedQ(a)))，在**全部合法着**上（§2.2 修正②）。"""
    # 按规格定义，π′ 与 π_imp 的公式完全相同；独立成函数以区分"选择用"与"监督用"。
    return improved_policy(node, c_visit, c_scale)


def select_action(node: Node, c_visit: float = C_VISIT,
                  c_scale: float = C_SCALE) -> int:
    """非根节点确定性选择（§2.2 修正①）：a* = argmax[π_imp − N/(1+ΣN)]。"""
    pi_imp = improved_policy(node, c_visit, c_scale)
    if node.n.size == 0:
        return int(node.legal[np.argmax(pi_imp)])
    frac = node.n.astype(np.float32) / np.float32(1 + node.n_total)
    return int(node.legal[np.argmax(pi_imp - frac)])


def gumbel_topm(node: Node, m0: int = M0, rng: np.random.Generator | None = None,
                g: float = 1.0) -> list[tuple[int, float]]:
    """根节点候选：按 g·Gumbel(0,1) + ℓ 取 top-m（无放回 Gumbel-Top-k）。

    m = min(m0, 合法着数)；g=0 时关闭噪声（候选选择退化为按 ℓ 取 top-m）。
    返回 [(action_id, gumbel_noise), ...]，按得分降序。
    """
    if node.terminal or node.logits.size == 0:
        return []
    rng = rng if rng is not None else np.random.default_rng()
    m = min(m0, len(node.legal))
    noise = np.zeros(len(node.legal), np.float32)
    if g != 0.0:
        u = rng.random(len(node.legal), dtype=np.float32)
        noise = -np.log(-np.log(u + 1e-20), dtype=np.float32).astype(np.float32)
        noise = (g * noise)
    score = noise + node.logits.astype(np.float32)
    order = np.argsort(-score, kind="stable")[:m]
    return [(int(node.legal[i]), float(noise[i])) for i in order]


# ---------------- 顺序减半调度 ----------------

@dataclass
class _Candidate:
    action: int
    noise: float
    child: Node | None = None      # expand(root, action) 的结果；终局局面临时为 terminal Node


def _n_rounds(m: int) -> int:
    """⌈log₂ m⌉；m=1 时仍需 1 轮把预算全部投入该唯一候选（用于充实 π′ 的树信息）。"""
    return max(1, math.ceil(math.log2(max(m, 2)))) if m >= 2 else 1


def order_halving(
    root: Node,
    expand,
    n_sims: int = N_SIMS,
    m0: int = M0,
    g: float = 1.0,
    seed: int | np.random.Generator = 0,
    qmin: float | None = None,
    qmax: float | None = None,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
) -> dict:
    """根节点顺序减半（§2.2【锁定】）。

    expand(parent_node, action_id) -> Node | None
        返回从 parent 走 action 到达的子节点（q 为子局面行棋方视角价值，legal/logits 齐全）；
        None 表示该着不存在（调用方应保证 parent.legal 内的动作都能展开）。

    qmin/qmax：**仅作诊断统计**（全树见到的 q 范围），2026-09-19 起不再参与任何打分——
        归一化已改为逐节点（`qtransform_completed`）。保留入参是为了兼容既有调用与
        轨迹工具；传入值不影响搜索结果。
    返回 dict：{action, noise, sims_used, rounds, budget_check, survivors_per_round, qmin, qmax,
                 n_nodes, n_terminal, tree}
    """
    if root.terminal or root.legal.size == 0:
        return {"action": None, "noise": 0.0, "sims_used": 0, "rounds": 0,
                "budget_check": True, "survivors_per_round": [], "qmin": None,
                "qmax": None, "n_nodes": 0, "n_terminal": 0, "tree": None}

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    cands = gumbel_topm(root, m0=m0, rng=rng, g=g)
    m = len(cands)
    rounds = _n_rounds(m)
    surv = [_Candidate(action=a, noise=ns) for a, ns in cands]

    # 预算分配：每轮 n_sims/rounds 次模拟均分给存活候选，余数前置（保证总和恰好 = n_sims）
    base, rem = divmod(n_sims, rounds)
    budget_per_round = [base + (1 if i < rem else 0) for i in range(rounds)]

    q_seen: list[float] = []
    if qmin is None or qmax is None:
        qmin = root.q
        qmax = root.q
    qmin = float(qmin); qmax = float(qmax)
    n_nodes = 0
    n_terminal = 0
    tree: list[Node] = [root]

    def maybe_extend(q: float) -> None:
        nonlocal qmin, qmax
        if q < qmin:
            qmin = q
        if q > qmax:
            qmax = q
        q_seen.append(q)

    def _simulate(node: Node) -> float:
        """递归下探一次模拟：返回该节点**自身行棋方视角**的价值。

        非根节点确定性选择（select_action）挑一个动作；若该动作对应子节点尚未展开，
        expand 一次作为本次模拟的新叶子；否则递归深入已存在的子节点——这样树深度随同一
        候选获得的模拟预算自然增长（而不是每次模拟都固定只探两层，review 修正）。
        每一层的回传都恰好取负一次（父子行棋方相邻取负，零和），使得任意深度的叶子
        价值经过奇数次/偶数次取负后，最终仍严格等于"该层节点自身视角"——这是修正前
        do_sim() 的符号 bug（非终局分支曾少取负一次，导致根节点 Q 符号系统性反转）。
        """
        nonlocal n_nodes, n_terminal
        if node.is_terminal:
            return float(node.q)
        a = select_action(node, c_visit, c_scale)
        edge_idx = int(np.flatnonzero(node.legal == a)[0])
        key = int(a)
        child = node.children.get(key)
        if child is None:
            child = expand(node, a)
            if child is None:
                raise ValueError(f"expand 返回 None：动作 {a} 无法展开")
            node.children[key] = child
            n_nodes += 1
            tree.append(child)
            if child.is_terminal:
                n_terminal += 1
            maybe_extend(child.q)
            val = -float(child.q)
        else:
            val = -_simulate(child)
        node.record_child(edge_idx, val)
        return val

    def do_sim_root(c: _Candidate) -> None:
        """对候选 c 做 1 次模拟并记账到根（根视角价值）。"""
        nonlocal n_nodes, n_terminal
        if c.child is None:
            c.child = expand(root, c.action)
            if c.child is None:
                raise ValueError(f"expand 返回 None：动作 {c.action} 无法从根展开")
            n_nodes += 1
            tree.append(c.child)
            if c.child.is_terminal:
                n_terminal += 1
            maybe_extend(c.child.q)
            val = -float(c.child.q)
        elif c.child.is_terminal:
            val = -float(c.child.q)
        else:
            val = -_simulate(c.child)
        idx = int(np.flatnonzero(root.legal == c.action)[0])
        root.record_child(idx, val)

    survivors_log: list[int] = []
    sims_used = 0
    for r, budget in enumerate(budget_per_round):
        if len(surv) == 1:
            # 唯一候选：把剩余预算全部投入（预算守恒），不再淘汰
            budget = sum(budget_per_round[r:])
        per_base, per_rem = divmod(budget, len(surv))
        for i, c in enumerate(surv):
            k = per_base + (1 if i < per_rem else 0)
            for _ in range(k):
                do_sim_root(c)
            sims_used += k
        # 顺序减半：按 g + ℓ + σ(q̂) 淘汰末位一半（σ 前先做本树 min−max 归一，§2.2）
        l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
        s_root = qtransform_completed(root, c_visit, c_scale)
        s_map = {int(a): float(x) for a, x in zip(root.legal, s_root)}
        scored = sorted(
            ((c.noise + l_root[c.action] + s_map[c.action], c) for c in surv),
            key=lambda t: -t[0],
        )
        keep = max(1, (len(surv) + 1) // 2)
        surv = [c for _, c in scored[:keep]]
        survivors_log.append(len(surv))

    return {
        "action": int(surv[0].action),
        "noise": float(surv[0].noise),
        "sims_used": int(sims_used),
        "rounds": rounds,
        "budget_check": sims_used == n_sims,
        "survivors_per_round": survivors_log,
        "qmin": qmin,
        "qmax": qmax,
        "n_nodes": n_nodes,
        "n_terminal": n_terminal,
        "tree": tree,
    }


# ---------------- 目标导出 ----------------

def export_pi_prime(
    node: Node,
    c_visit: float = C_VISIT,
    c_scale: float = C_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """导出训练目标：返回 (legal_action_ids int64, pi_prime fp32)，支持集 = 全部合法着。

    **尺度必须与搜索使用的同一份配置一致**：默认值只是兜底，调用方（生成器/arena）
    应显式传入 cfg.c_visit / cfg.c_scale；否则会出现「按 c_scale=0.1 分配访问、
    却按 1.0 写训练目标」的错配。
    """
    ids = node.legal.astype(np.int64)
    probs = pi_prime(node, c_visit, c_scale)
    return ids, probs
