"""UniChessKit 接入：S（状态序列模型）的 BatchEvaluator / Expander / Player。

与 T/R 的区别：S 是**有状态**模型——每个局面的前向依赖整盘历史的 Mamba cache。
kit 的搜索只通过 ``Leaf.parent_handle`` / ``NodeEval.handle`` 与状态交接，状态管理全在本模块：

- ``SsmEvaluator``：一次批量 ``SeqModel.step``。负载 = (feats785, tc, elo, color, cache)，
  cache 为 batch=1 的状态；torch 调用序列与原 ``tools/ssm_gumbel_arena.py::ArenaModel.step``（git 历史）
  / 批量驱动器 ``_model_step`` 完全相同（同批大小下逐位一致）。
- ``SsmExpander``（ReplayStore）：句柄 = 该步搜索根的 ``RootState``（根 cache + occurrence）。
  展开深度 d 的叶子时从根 cache 重放 d−1 步再评估 1 次——与 arena/生成器的
  ``_expand_child`` / ``_expand_gen`` 同口径（encode-before-increment，重放逐步 occurrence+1）。
  P3 埋点实测 64 sims 平均深度约 3，重放开销可接受；换 SlabCacheStore 留给 P4。
- ``SsmPlayer``：**懒追赶**——``choose`` 时把尚未步进的局面（B₀、开局各局面、对手走后的局面…B_t）
  按序逐个步进，最后一次输出即根评估。这与 S arena「每 ply 双方模型各进一步」等价：
  每个模型的状态只取决于它按序看到的局面序列，与何时步进无关。``observe`` 因此不需要前向。

批量对弈配置示例（``python -m unichess_kit.match``）::

    {"factory": "stateseq.kit_adapter:make_player_factory",
     "root": "/home/jeefy/UniChess/SSM",
     "kwargs": {"checkpoint": "runs/stage_b/gen3.pt", "simulations": 256}}

注意 kit 的 ``max_plies`` 按整盘（含开局）计，S arena 的 ``--max-plies`` 只计开局之后。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import chess
import numpy as np
import torch

from unichess_kit.api import EvalRequest, GameStart, MoveDecision, NodeEval, SearchBudget, immediate
from unichess_kit.search.gumbel import Gumbel, GumbelConfig, softmax

from .actions import move_to_action
from .adapter import classify_final_board, encode_board, wdl_logits_to_q
from .conditions import TimeControlBucket
from .data.gshards import META_V3_DTYPE, encode_v3_pipol, make_game_key
from .data.sequences import _board_key
from .depth_hist import hist_merge
from .gumbel import C_SCALE, C_VISIT, M0, N_SIMS, TERM_CODES
from .model import SeqModel

KIT_SPI_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_seq_model(checkpoint, device: str = "cuda") -> tuple:
    """→ (eval 模式的 SeqModel, 解析后的绝对路径)。"""
    path = _resolve(checkpoint).resolve()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    seq = SeqModel(dropout=0.0)
    seq.load_state_dict(ckpt.get("model", ckpt))
    seq.to(device).eval()
    return seq, path


def _concat_caches(caches: Sequence) -> list:
    """N 份 batch=1 的 Cache 沿 batch 维拼成一份 batch=N（torch.cat 总是新张量，不改入参）。"""
    return [(torch.cat([c[li][0] for c in caches], dim=0),
             torch.cat([c[li][1] for c in caches], dim=0))
            for li in range(len(caches[0]))]


def _split_cache(cache: list, n: int) -> list:
    out = [[] for _ in range(n)]
    for conv, ssm in cache:
        for i in range(n):
            out[i].append((conv[i:i + 1], ssm[i:i + 1]))
    return out


def legal_moves_and_ids(board: chess.Board) -> tuple:
    """合法着（``board.legal_moves`` 顺序，滤掉动作空间外的着）与对应动作 id。
    顺序与 arena 的 ``_legal_actions_of`` 相同，Gumbel 的并列取首因此一致。"""
    moves, ids = [], []
    for m in board.legal_moves:
        a = move_to_action(m)
        if a is not None:
            moves.append(m)
            ids.append(a)
    return moves, np.asarray(ids, dtype=np.int64)


# ------------------------------------------------------------------ 前向

class ReferenceStateAPI:
    """参考实现的状态句柄 API（SsmPlayer 用）：句柄 = cache，负载 = (feats, tc, elo, color, cache)，
    搜索根 = RootState，展开走 ReplayStore。快速实现见 ``fast_eval.SsmFastEvaluator``。
    子类需提供 ``initial_cache()`` 与 ``evaluate()``。"""

    def root_state(self):
        return self.initial_cache()

    def child(self, parent, feats, tc, elo, color, root_occ=None, path_keys=()):
        return (feats, int(tc), float(elo), int(color), parent)

    def hold(self, state) -> None:
        pass

    def release(self, state) -> None:
        pass

    def make_root(self, ply: int, state, occurrence: dict) -> "RootState":
        return RootState(ply=ply, cache=state, occurrence=occurrence)

    def make_expander(self):
        return SsmExpander(self)


class SsmEvaluator(ReferenceStateAPI):
    """``SeqModel.step`` 的批量前向。结果 = (logits[1936] fp32, wdl[3], 新 cache(batch=1))。"""

    def __init__(self, seq: SeqModel, device: str, model_key: str):
        self.seq = seq
        self.device = device
        self.model_key = model_key
        self._tc_cache: dict = {}
        self._elo_cache: dict = {}
        self.n_forwards = 0

    @classmethod
    def from_checkpoint(cls, checkpoint, device: str = "cuda") -> "SsmEvaluator":
        seq, path = load_seq_model(checkpoint, device)
        return cls(seq, device, f"S:{path}:{device}")

    def initial_cache(self):
        return self.seq.initial_cache(1, device=self.device, dtype=torch.float32)

    def _scalar_tensor(self, cache: dict, kind: str, val, n: int) -> torch.Tensor:
        key = (val, n)
        t = cache.get(key)
        if t is None:
            t = (torch.full((n,), int(val), dtype=torch.long, device=self.device)
                 if kind == "long" else
                 torch.full((n,), float(val), dtype=torch.float32, device=self.device))
            cache[key] = t
        return t

    @torch.no_grad()
    def evaluate(self, payloads: Sequence) -> list:
        n = len(payloads)
        self.n_forwards += n
        feats = np.stack([p[0] for p in payloads]).astype(np.float32)
        tc = [p[1] for p in payloads]
        elo = [p[2] for p in payloads]
        color = [p[3] for p in payloads]
        cache = _concat_caches([p[4] for p in payloads])
        # 以下与 ArenaModel.step 逐条相同
        f_t = torch.from_numpy(feats).float().to(self.device)
        if len(set(tc)) == 1:
            tc_t = self._scalar_tensor(self._tc_cache, "long", tc[0], n)
        else:
            tc_t = torch.from_numpy(np.asarray(tc, dtype=np.int64)).to(self.device)
        if len(set(elo)) == 1:
            elo_t = self._scalar_tensor(self._elo_cache, "float", elo[0], n)
        else:
            elo_t = torch.from_numpy(np.asarray(elo, dtype=np.float32)).to(self.device)
        color_t = torch.from_numpy(np.asarray(color, dtype=np.int64)).to(self.device)
        logits, wdl, _mlh, _x, cache_new = self.seq.step(f_t, tc_t, elo_t, color_t, cache)
        logits, wdl = logits.cpu().numpy(), wdl.cpu().numpy()
        caches = _split_cache(cache_new, n)
        return [(logits[i], wdl[i], caches[i]) for i in range(n)]


def encode_payload(board: chess.Board, occurrence: int, cache) -> tuple:
    feats, tc_val, elo_std, color = encode_board(board, occurrence)
    return (np.asarray(feats, dtype=np.float32).reshape(-1), int(tc_val), float(elo_std),
            int(color), cache)


def node_eval_from_output(board: chess.Board, logits: np.ndarray, wdl: np.ndarray,
                          handle) -> NodeEval:
    """前向输出 → NodeEval：logits 取合法着子集（fp32，与 arena 的 Node 构造逐位相同）。"""
    moves, ids = legal_moves_and_ids(board)
    lg = logits[ids].astype(np.float32) if len(ids) else np.zeros(0, np.float32)
    priors = softmax(lg) if len(ids) else np.zeros(0, np.float32)
    return NodeEval(moves=moves, priors=priors, value=wdl_logits_to_q(wdl), handle=handle,
                    logits=lg)


# ------------------------------------------------------------------ 状态（ReplayStore）

@dataclass(frozen=True, eq=False)
class RootState:
    """一步搜索的根状态快照：根局面已步进后的 cache 与 occurrence（均不可被修改）。"""
    ply: int                 # 根局面的 move_stack 长度
    cache: list
    occurrence: dict


class SsmExpander:
    """ReplayStore：每个叶子从根 cache 重放路径（不含最后一着）再评估叶子本身。

    多个叶子按步同步重放（每拍一个 EvalRequest），Gumbel 每次只给 1 个叶子。
    """

    def __init__(self, evaluator: SsmEvaluator):
        self.evaluator = evaluator

    def expand(self, leaves: Sequence):
        states = []
        for leaf in leaves:
            root = leaf.parent_handle
            if not isinstance(root, RootState):
                raise ValueError("S 的叶子必须带根状态句柄（根评估由 SsmPlayer 提供）")
            path = leaf.board.move_stack[root.ply:]
            if not path:
                raise ValueError("叶子与根同一局面：S 的根评估不经 Expander")
            board = leaf.board.copy()
            for _ in path:
                board.pop()
            states.append({"root": root, "board": board, "path": path,
                           "cache": root.cache, "occ": dict(root.occurrence)})
        # 重放：第 k 步把各叶子路径上的第 k 个着法后的局面推进 cache（最后一着留给评估）
        depth = max(len(s["path"]) for s in states)
        for k in range(depth - 1):
            active = [s for s in states if k < len(s["path"]) - 1]
            payloads = []
            for s in active:
                s["board"].push(s["path"][k])
                key = _board_key(s["board"])
                payloads.append(encode_payload(s["board"], s["occ"].get(key, 0), s["cache"]))
                s["occ"][key] = s["occ"].get(key, 0) + 1
            outs = yield EvalRequest(self.evaluator, payloads)
            for s, (_, _, cache) in zip(active, outs):
                s["cache"] = cache
        payloads = []
        for s in states:
            s["board"].push(s["path"][-1])
            key = _board_key(s["board"])
            payloads.append(encode_payload(s["board"], s["occ"].get(key, 0), s["cache"]))
        outs = yield EvalRequest(self.evaluator, payloads)
        return [node_eval_from_output(s["board"], lg, wd, s["root"])
                for s, (lg, wd, _) in zip(states, outs)]


class SsmFastExpander:
    """SlabStore（P4）：句柄 = 父节点的 ``NodeState``（GPU 槽）。每个叶子 1 次前向：
    父槽复制到新槽后原地单步；父节点若已被池淘汰，先经 ``evaluator.ensure`` 重算。
    occurrence = 根处计数 + 根到父节点路径上的出现次数（与 ReplayStore 的逐步 +1 相同）。"""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def expand(self, leaves: Sequence):
        ev = self.evaluator
        prepared = []
        for leaf in leaves:
            parent = leaf.parent_handle
            if parent is None or getattr(parent, "root_occ", None) is None:
                raise ValueError("S 的叶子必须带父节点状态句柄（根评估由 SsmPlayer 提供）")
            key = _board_key(leaf.board)
            occ = parent.root_occ.get(key, 0) + parent.path_keys.count(key)
            feats, tc, elo, color = encode_board(leaf.board, occ)
            prepared.append((leaf, parent, key, np.asarray(feats, dtype=np.float32).reshape(-1),
                             tc, elo, color))
        yield from ev.ensure([p[1] for p in prepared])      # 返回时父节点各被钉住一次
        nodes = [ev.child(parent, f, tc, elo, color, parent.root_occ, parent.path_keys + (key,))
                 for _, parent, key, f, tc, elo, color in prepared]
        for p in prepared:
            ev.pool.unpin(p[1])         # child 已另行钉住父节点，直到前向完成
        outs = yield EvalRequest(ev, nodes)
        return [node_eval_from_output(p[0].board, lg, wd, node)
                for p, (lg, wd, node) in zip(prepared, outs)]


# ------------------------------------------------------------------ Player

class SsmPlayer:
    """S 的对局参与者：懒追赶步进完整历史 + Gumbel 顺序减半搜索（arena 默认 g=0）。"""

    def __init__(self, name: str, evaluator: SsmEvaluator, cfg: GumbelConfig):
        self.name = name
        self.evaluator = evaluator
        self.cfg = cfg
        self.search = Gumbel(evaluator.make_expander(), cfg)
        self._reset()

    def _reset(self):
        self.start_fen: Optional[str] = None
        self.seed = 0
        self.cache = None         # 当前局面的状态句柄（参考实现 = cache，快速实现 = NodeState）
        self.occurrence: dict = {}
        self.stepped = 0          # 已步进的局面数：B_0..B_{stepped-1}
        self.last = None          # 最后一次步进的 (logits, wdl)
        self.expand_hist: list = []

    def new_game(self, start: GameStart):
        self._reset()
        self.start_fen = start.fen
        self.seed = int(start.seed)
        self._set_state(self.evaluator.root_state())
        return immediate(None)

    def _set_state(self, state) -> None:
        ev = self.evaluator
        old, self.cache = self.cache, state
        if state is not None:
            ev.hold(state)
        ev.release(old)

    def _catch_up(self, board: chess.Board):
        """按序步进 B_stepped..B_t（B_t = board）。encode-before-increment，与 arena 同口径。"""
        stack = board.move_stack
        t = len(stack)
        if self.stepped > t + 1:
            raise RuntimeError(f"{self.name}: 局面回退（已步进 {self.stepped}，当前 ply {t}）")
        if self.stepped == t + 1:
            return            # 同一局面重复 choose：沿用已步进状态
        first = self.stepped
        replay = chess.Board(self.start_fen) if self.start_fen else chess.Board()
        for mv in stack[:first]:
            replay.push(mv)            # replay = B_first
        for k in range(first, t + 1):
            if k > first:
                replay.push(stack[k - 1])
            key = _board_key(replay)
            feats, tc, elo, color = encode_board(replay, self.occurrence.get(key, 0))
            payload = self.evaluator.child(self.cache, np.asarray(feats, dtype=np.float32)
                                           .reshape(-1), tc, elo, color)
            (out,) = yield EvalRequest(self.evaluator, [payload])
            lg, wd, state = out
            self._set_state(state)
            self.occurrence[key] = self.occurrence.get(key, 0) + 1
            self.last = (lg, wd)
        self.stepped = t + 1

    def _root(self, board: chess.Board):
        """追赶到当前局面，返回 (根 NodeEval, 根合法着的动作 id)。"""
        ply = len(board.move_stack)
        yield from self._catch_up(board)
        root_state = self.evaluator.make_root(ply, self.cache, dict(self.occurrence))
        root = node_eval_from_output(board, self.last[0], self.last[1], root_state)
        return root, legal_moves_and_ids(board)[1]

    def _search(self, board: chess.Board, root: NodeEval, rng, simulations):
        res = yield from self.search.search(board, root=root, rng=rng, simulations=simulations)
        if res.move is None:
            raise RuntimeError(f"{self.name}: 搜索未给出着法 @ {board.fen()}")
        self.expand_hist = hist_merge(self.expand_hist, res.stats.get("expand_hist"))
        return res

    def choose(self, board: chess.Board, budget: SearchBudget):
        root, _ = yield from self._root(board)
        rng = np.random.default_rng(self.seed + len(board.move_stack) * 100003)
        res = yield from self._search(board, root, rng, budget.simulations)
        return MoveDecision(res.move, source="search",
                            info={"q": float(root.value), "sims": res.stats["sims_used"],
                                  "max_depth": res.stats.get("max_depth", 0),
                                  "expand_hist": list(res.stats.get("expand_hist") or [])})

    def observe(self, board: chess.Board, move: chess.Move):
        return immediate(None)

    def close(self) -> None:
        self._set_state(None)
        self.last = None


class SsmPlayerFactory:
    def __init__(self, name: str, evaluator: SsmEvaluator, cfg: GumbelConfig):
        self.name = name
        self.evaluator = evaluator
        self.cfg = cfg

    def __call__(self) -> SsmPlayer:
        return SsmPlayer(self.name, self.evaluator, self.cfg)


ENGINES = ("server", "fast", "reference")


def make_evaluator(checkpoint, device: str = "cuda", engine: str = "fast", *,
                   server_dir: Optional[str] = None, **fast_kw):
    """engine：

    - "server"：连到共享 GPU 服务（``gpu_server``，需 server_dir）；本进程不碰 GPU，多进程跨进程拼批，
      批不变 ⇒ 结果逐位可复现、与进程数 / 并发无关（P4，多进程工具默认）；
    - "fast"：同进程 GPU 槽池 + CUDA graph（P4）；
    - "reference"：原 cat/split + 重放实现（与 P3 及更早的 S arena 逐位对照用）。
    """
    if engine == "server":
        if not server_dir:
            raise ValueError('engine="server" 需要 server_dir（由启动方的 GpuServer 提供）')
        from .gpu_server import remote_evaluator
        return remote_evaluator(server_dir, checkpoint, chunk=fast_kw.get("server_chunk", 0))
    if engine == "fast":
        from .fast_eval import SsmFastEvaluator
        return SsmFastEvaluator.from_checkpoint(checkpoint, device, **fast_kw)
    if engine == "reference":
        return SsmEvaluator.from_checkpoint(checkpoint, device)
    raise ValueError(f"未知 engine={engine!r}，可选 {ENGINES}")


def _engine_kw(engine, pool_slots, cuda_graphs, server_dir, server_chunk=0) -> dict:
    if engine == "fast":
        return {"pool_slots": pool_slots, "cuda_graphs": cuda_graphs}
    if engine == "server":
        return {"server_dir": server_dir, "server_chunk": server_chunk}
    return {}


def _gumbel_cfg(evaluator, simulations, m0, g, c_visit, c_scale, parallel) -> GumbelConfig:
    if parallel is None:        # 快速实现默认轮内并发；参考实现保持原串行次序（逐位对照）
        parallel = bool(getattr(evaluator, "fast", False))
    return GumbelConfig(simulations=simulations, m0=m0, g=g, c_visit=c_visit, c_scale=c_scale,
                        parallel=parallel)


def make_player_factory(checkpoint, *, name: str = "S", device: str = "cuda",
                        simulations: int = N_SIMS, m0: int = M0, g: float = 0.0,
                        c_visit: float = C_VISIT, c_scale: float = C_SCALE,
                        engine: str = "fast", parallel: Optional[bool] = None,
                        pool_slots: int = 2048, cuda_graphs: bool = True,
                        server_dir: Optional[str] = None,
                        server_chunk: int = 0) -> SsmPlayerFactory:
    """加载一次权重，返回每局一个 SsmPlayer 的工厂。默认 g=0（arena 口径，确定性）。"""
    evaluator = make_evaluator(checkpoint, device, engine,
                               **_engine_kw(engine, pool_slots, cuda_graphs, server_dir,
                                            server_chunk))
    cfg = _gumbel_cfg(evaluator, simulations, m0, g, c_visit, c_scale, parallel)
    return SsmPlayerFactory(name, evaluator, cfg)


# ------------------------------------------------------------------ 自对弈（Stage B 生成器）

def book_pipol_rng(seed: int, opening_idx: int, ply: int) -> np.random.Generator:
    """开局 ply 的搜索噪声 RNG：只取决于 (seed, 开局序号, ply)，与具体局次无关。

    同一条开局在所有局中的第 ply 个局面完全相同（book 着法序列一致 ⇒ 棋盘、occurrence、
    R cache 全部一致），因此该 RNG 保证这些局的搜索树逐位一致，π′ 可跨局共享。
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(opening_idx), int(ply)]))


class SelfPlayShared:
    """同一工厂（= 同一 worker 进程）内各局共享：book ply 的 π′ 缓存与生成统计。

    pipol_memo：(seed, book_id, ply) → (ids uint16, π′ fp32, stats)。并发局若在同一 ply
    竞态（都未命中）会各自搜索，二者输入一致、浮点结果可能因拼批差 ~1e-6——这是计算优化而非
    正确性依赖，复用到的 π′ 始终是该局面的一次合法搜索结果（与原生成器相同）。
    """

    def __init__(self):
        self.pipol_memo: dict = {}
        self.n_nodes = 0
        self.n_terminal = 0
        self.sims = 0
        self.max_depth = 0
        self.budget_violations = 0
        self.book_memo_hits = 0
        self.book_memo_misses = 0
        self.expand_hist: list = []

    def add_search(self, stats: dict) -> None:
        self.n_nodes += int(stats["n_nodes"])
        self.n_terminal += int(stats["n_terminal"])
        self.sims += int(stats["sims_used"])
        self.max_depth += int(stats["max_depth"])


class SsmSelfPlayer(SsmPlayer):
    """自对弈 Player：执双方（单一 cache，每个局面只步进一次——懒追赶天然如此）。

    与原生成器 ``GameState`` 逐条对应：每局一条随机数流
    ``default_rng(SeedSequence(seed, spawn_key=(index,)))``（= spawn(num_games)[index]），
    跨 ply 连续消耗；book ply 走 book 着法，π′ 用 ``book_pipol_rng`` 搜索并按
    (seed, book_id, ply) 跨局共享；训练目标放在 ``decision.info``（pi_ids / pi）交给 V3Sink。
    """

    def __init__(self, name: str, evaluator: SsmEvaluator, cfg: GumbelConfig,
                 shared: SelfPlayShared):
        super().__init__(name, evaluator, cfg)
        self.shared = shared

    def new_game(self, start: GameStart):
        if not start.both_sides:
            raise ValueError("SsmSelfPlayer 只用于自对弈（GameStart.both_sides=True）")
        if start.fen:
            raise ValueError("自对弈从标准初始局面开始")
        self._reset()
        self.seed = int(start.seed)
        self._set_state(self.evaluator.root_state())
        self.rng = np.random.default_rng(
            np.random.SeedSequence(self.seed, spawn_key=(int(start.index),)))
        self.book = [chess.Move.from_uci(u) for u in start.book]
        self.book_id = start.book_id
        return immediate(None)

    def _search_pi(self, board, root, ids, rng, simulations):
        res = yield from self._search(board, root, rng, simulations)
        _, probs = res.pi_prime(self.cfg)
        stats = {k: res.stats[k] for k in ("n_nodes", "n_terminal", "sims_used", "max_depth")}
        if stats["sims_used"] != (simulations or self.cfg.simulations):
            self.shared.budget_violations += 1
        self.shared.expand_hist = hist_merge(self.shared.expand_hist, res.stats.get("expand_hist"))
        return res, ids.astype(np.uint16), probs.astype(np.float32), stats

    def choose(self, board: chess.Board, budget: SearchBudget):
        ply = len(board.move_stack)
        root, ids = yield from self._root(board)
        if ply < len(self.book):
            mv = self.book[ply]
            if move_to_action(mv) is None:
                raise RuntimeError(f"开局着法 {mv} 在 {board.fen()} 上无法映射到 action")
            key = (self.seed, self.book_id, ply) if self.book_id is not None else None
            cached = self.shared.pipol_memo.get(key) if key is not None else None
            if cached is not None:
                pi_ids, pi, stats = cached
                self.shared.book_memo_hits += 1
            else:
                rng = book_pipol_rng(self.seed, self.book_id, ply) if key is not None else self.rng
                _, pi_ids, pi, stats = yield from self._search_pi(board, root, ids, rng,
                                                                  budget.simulations)
                self.shared.book_memo_misses += 1
                if key is not None:
                    self.shared.pipol_memo[key] = (pi_ids, pi, stats)
            self.shared.add_search(stats)
            return MoveDecision(mv, source="book", info={"pi_ids": pi_ids, "pi": pi})
        res, pi_ids, pi, stats = yield from self._search_pi(board, root, ids, self.rng,
                                                            budget.simulations)
        self.shared.add_search(stats)
        return MoveDecision(res.move, source="search", info={"pi_ids": pi_ids, "pi": pi})


class SsmSelfPlayerFactory:
    def __init__(self, name: str, evaluator: SsmEvaluator, cfg: GumbelConfig):
        self.name = name
        self.evaluator = evaluator
        self.cfg = cfg
        self.shared = SelfPlayShared()

    def __call__(self) -> SsmSelfPlayer:
        return SsmSelfPlayer(self.name, self.evaluator, self.cfg, self.shared)


def make_selfplay_factory(checkpoint=None, *, evaluator=None,
                          name: str = "S", device: str = "cuda", simulations: int = N_SIMS,
                          m0: int = M0, g: float = 1.0, c_visit: float = C_VISIT,
                          c_scale: float = C_SCALE, engine: str = "fast",
                          parallel: Optional[bool] = None, pool_slots: int = 2048,
                          cuda_graphs: bool = True,
                          server_dir: Optional[str] = None,
                          server_chunk: int = 0) -> SsmSelfPlayerFactory:
    """自对弈工厂（默认 g=1）。给 evaluator 时复用已加载的模型，否则从 checkpoint 加载。"""
    if evaluator is None:
        evaluator = make_evaluator(checkpoint, device, engine,
                                   **_engine_kw(engine, pool_slots, cuda_graphs, server_dir,
                                                server_chunk))
    cfg = _gumbel_cfg(evaluator, simulations, m0, g, c_visit, c_scale, parallel)
    return SsmSelfPlayerFactory(name, evaluator, cfg)


def pipol_byte_offsets(per_ply_actions: list) -> np.ndarray:
    """pipol 变长目标的字节偏移表（含末尾哨兵）。"""
    off = [0]
    for acts in per_ply_actions:
        off.append(off[-1] + 2 + len(acts) * 4)
    return np.array(off, dtype=np.int32)


class V3Sink:
    """kit RecordSink → v3 分片（actions + pipol + 扩展 meta），字段与原生成器逐项相同。"""

    def __init__(self, writer, *, gen_id: int, ckpt_step: int = 0, elo: float = 2567.5,
                 tc_bucket=TimeControlBucket.RAPID):
        self.writer = writer
        self.gen_id = int(gen_id)
        self.ckpt_step = int(ckpt_step)
        self.elo = float(elo)
        self.tc_bucket = tc_bucket
        self.term_reason_counts = [0] * len(TERM_CODES)
        self.truncated_games = 0
        self.games = 0
        self.plies = 0

    def on_game_end(self, record: dict, board: chess.Board, decisions) -> None:
        result, reason, is_truncated = classify_final_board(board)
        term = TERM_CODES.index(reason)
        actions = []
        for d in decisions:
            a = move_to_action(d.move)
            if a is None:
                raise RuntimeError(f"着法 {d.move} 无法映射到 action（第 {record['game']} 局）")
            actions.append(a)
        pipol_actions = [d.info["pi_ids"] for d in decisions]
        pipol_probs = [d.info["pi"] for d in decisions]
        meta = np.zeros((), dtype=META_V3_DTYPE)
        meta["n_plies"] = len(actions)
        meta["tc_bucket"] = int(self.tc_bucket)
        meta["result"] = result
        meta["elo_missing"] = 0
        meta["elo_mean"] = self.elo
        meta["game_key"] = make_game_key(f"selfplay_gen{self.gen_id}", int(record["game"]))
        meta["gen_id"] = self.gen_id
        meta["ckpt_step"] = self.ckpt_step
        meta["termination_reason"] = term
        meta["is_truncated"] = 1 if is_truncated else 0
        # flags = 本局开局注入 ply 数（训练侧据此对 book ply 的 policy 损失降权，§P1-3）
        meta["flags"] = int(record["book_plies"])
        self.writer.add(meta, np.array(actions, dtype=np.uint16),
                        encode_v3_pipol(pipol_actions, pipol_probs),
                        pipol_byte_offsets(pipol_actions))
        self.games += 1
        self.plies += len(actions)
        self.term_reason_counts[term] += 1
        if is_truncated:
            self.truncated_games += 1
