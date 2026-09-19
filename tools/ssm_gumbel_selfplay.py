"""Stage B Gumbel 自对弈生成器（规格 §2.4 / §2.3 / §2.5）。

并发局数：128（初值，按显存/CPU 实测调）——真实实现为跨局 GPU 拼批（§2.3）：
树内逻辑（选择/淘汰/备份）按局在 CPU 端用生成器/协程串行推进，每当某局需要一次模型前向
（根节点步进，或搜索树内沿路径重算一步）就 yield 出请求；驱动器把所有"当前活跃局"的
待处理请求拼成一个批次一次性上 GPU，再把结果分发回各自的生成器——一局终局立即用队列中
下一局补位（槽位复用），使批大小长期维持在 concurrency 附近。
每代局数：首轮闭环 2k–5k；主循环 25k/代。
输出：v3 分片（actions + pipol + 扩展 meta）+ manifest（含 gen 节生成统计）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import move_to_action
from stateseq.conditions import TimeControlBucket
from stateseq.data.sequences import _board_key
from stateseq.adapter import (
    encode_board, wdl_logits_to_q, get_terminal_q, classify_final_board,
)
from stateseq.gumbel import (
    C_SCALE,
    C_VISIT,
    Node,
    TERM_CODES,
    _Candidate,
    _n_rounds,
    export_pi_prime,
    gumbel_topm,
    qtransform_completed,
    select_action,
)
from stateseq.data.gshards import META_V3_DTYPE, V3ShardWriter, encode_v3_pipol, make_game_key

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------- 配置 -------------------------

@dataclass
class SelfPlayConfig:
    ckpt: str
    out_dir: str
    tag: str
    num_games: int = 2000
    concurrency: int = 128
    n_sims: int = 64
    m0: int = 16
    seed: int = 42
    c_visit: float = C_VISIT
    c_scale: float = C_SCALE
    max_plies: int = 300
    gen_id: int = 1
    ckpt_step: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    elo: float = 2567.5
    tc_bucket: TimeControlBucket = TimeControlBucket.RAPID
    gumbel_g: float = 1.0  # Gumbel 噪声尺度；评测/换代 arena 用 g=0


def _legal_actions_of(board: chess.Board) -> list[int]:
    legal = []
    for m in board.legal_moves:
        a = move_to_action(m)
        if a is not None:
            legal.append(a)
    return legal


def _resolve_move(action: int, board: chess.Board) -> chess.Move | None:
    """从 action id 还原合法着（逐 board.legal_moves 匹配，补齐 promotion/EP/易位旗标）。

    action_to_move 对升后（Queen）返回 promotion=None，board.push 不会自动补旗标，
    兵停留在 8 排导致棋盘悄然损坏——这是 504/1000 局数据完整性事故的根因（2026-09-17）。
    """
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


# ------------------------- 模型封装（批量前向） -------------------------

class ModelWrapper:
    """封装 champion 模型：initial_cache + 批量单步前向（跨局/跨候选拼批，§2.3）。"""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        from stateseq.model import SeqModel
        self.seq = SeqModel(dropout=0.0)
        self.seq.load_state_dict(state_dict)
        self.seq.to(device).eval()

    def initial_cache(self, batch_size: int = 1):
        return self.seq.initial_cache(batch_size, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def step_batch(self, features: np.ndarray, tc: list[int], elo: list[float],
                    color: list[int], cache):
        """features (N,785)；返回 numpy (logits, wdl, mlh, x) + 新 batched cache。"""
        f_t = torch.from_numpy(features).float().to(self.device)
        tc_t = torch.tensor(tc, dtype=torch.long, device=self.device)
        elo_t = torch.tensor(elo, dtype=torch.float32, device=self.device)
        color_t = torch.tensor(color, dtype=torch.long, device=self.device)
        logits, wdl, mlh, x, cache_new = self.seq.step(f_t, tc_t, elo_t, color_t, cache)
        return (logits.detach().cpu().numpy(), wdl.detach().cpu().numpy(),
                mlh.detach().cpu().numpy(), x.detach().cpu().numpy(), cache_new)


def concat_caches(caches: list) -> list:
    """把 N 份 batch=1 的 Cache 沿 batch 维拼成一份 batch=N 的 Cache。"""
    n_layers = len(caches[0])
    out = []
    for li in range(n_layers):
        conv = torch.cat([c[li][0] for c in caches], dim=0)
        ssm = torch.cat([c[li][1] for c in caches], dim=0)
        out.append((conv, ssm))
    return out


def split_cache(cache: list, n: int) -> list:
    """把 batch=N 的 Cache 拆回 N 份 batch=1（视图切片；model_r.step 内部会 clone，无需再拷贝）。"""
    out = [[] for _ in range(n)]
    for conv, ssm in cache:
        for i in range(n):
            out[i].append((conv[i:i + 1], ssm[i:i + 1]))
    return out


def _compute_pipol_byte_offsets(per_ply_actions: list[np.ndarray]) -> np.ndarray:
    """计算 pipol 变长目标的字节偏移表（含末尾哨兵）。"""
    off = [0]
    for acts in per_ply_actions:
        off.append(off[-1] + 2 + len(acts) * 4)
    return np.array(off, dtype=np.int32)


# ------------------------- 单局状态机（生成器驱动，供跨局拼批） -------------------------

class GameState:
    """单局状态：真实棋盘 + 不可变根 cache 快照，仅在实战落子时推进（§2.3）。

    所有方法均以生成器形式编写：每当需要一次模型前向就 ``yield (features, tc, elo,
    color, cache)``，由外部驱动器 ``send()`` 回填 ``(logits, wdl, mlh, x, cache_new)``。
    这样同一时刻多局的"下一步请求"可以被驱动器攒成一个批次一次性上 GPU。
    """

    def __init__(self, game_idx: int, model: ModelWrapper, cfg: SelfPlayConfig,
                 seed_seq: np.random.SeedSequence):
        self.game_idx = game_idx
        self.model = model
        self.cfg = cfg
        self.board = chess.Board()
        self.root_cache = model.initial_cache(1)
        self.occurrence: dict[tuple, int] = {}
        self.actions: list[int] = []
        self.pipol_actions: list[np.ndarray] = []
        self.pipol_probs: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed_seq)
        self.n_nodes_total = 0
        self.n_terminal_total = 0
        self.sims_total = 0
        self.max_depth_total = 0
        self.budget_violations = 0

    def _update_occurrence(self) -> None:
        key = _board_key(self.board)
        self.occurrence[key] = self.occurrence.get(key, 0) + 1

    # ---- 顶层：整局 ----

    def run(self):
        for _ in range(self.cfg.max_plies):
            cont = yield from self._play_ply()
            if not cont:
                break
        return None

    def _play_ply(self):
        if self.board.is_game_over(claim_draw=True):
            return False
        key = _board_key(self.board)
        occ = self.occurrence.get(key, 0)
        legal_actions = _legal_actions_of(self.board)
        if not legal_actions:
            return False
        color = 1 if self.board.turn == chess.WHITE else 0
        feats_root, tc_root, elo_root, _ = encode_board(self.board, occ)
        logits_np, wdl_np, mlh_np, x_np, cache_new = yield (
            feats_root, int(tc_root), float(elo_root), color, self.root_cache)
        self.root_cache = cache_new
        self._update_occurrence()
        q = wdl_logits_to_q(wdl_np)
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_full = np.full(1936, -3e4, dtype=np.float32)
        logits_full[legal_arr] = logits_np[legal_arr]
        root_node = Node(legal=legal_arr, logits=logits_full[legal_arr], q=q, depth=0, path=())

        result = yield from self._order_halving_gen(root_node)
        if result["action"] is None:
            return False
        chosen = result["action"]
        self.n_nodes_total += result["n_nodes"]
        self.n_terminal_total += result["n_terminal"]
        self.sims_total += result["sims_used"]
        self.max_depth_total += int(result["max_depth"])
        if result["sims_used"] != self.cfg.n_sims:
            self.budget_violations += 1

        self.actions.append(int(chosen))
        ids, probs = export_pi_prime(root_node)
        self.pipol_actions.append(ids.astype(np.uint16))
        self.pipol_probs.append(probs.astype(np.float32))

        move = _resolve_move(chosen, self.board)
        if move is None:  # 搜索只在合法着上选择，走到这里说明动作编解码已损坏
            raise RuntimeError(f"选中动作 {chosen} 在 {self.board.fen()} 上不合法")
        self.board.push(move)
        return True

    # ---- 展开：沿 node.path 从根快照重算，再展开一步（§2.3 路径重算） ----

    def _expand_gen(self, node: Node, action: int):
        board = self.board.copy()
        cache = self.root_cache
        occ = dict(self.occurrence)
        for a in node.path:
            move = _resolve_move(a, board)
            if move is None:
                raise RuntimeError(f"路径重放动作 {a} 在 {board.fen()} 上不合法")
            board.push(move)
            key = _board_key(board)
            color = 1 if board.turn == chess.WHITE else 0
            feats, tc_val, elo_std, _ = encode_board(board, occ.get(_board_key(board), 0))
            _, _, _, _, cache = yield (feats, int(tc_val), float(elo_std), color, cache)
            occ[key] = occ.get(key, 0) + 1

        move = _resolve_move(action, board)
        if move is None:
            raise RuntimeError(f"动作 {action} 在 {board.fen()} 上不合法")
        board.push(move)
        terminal_by_rule = board.is_game_over(claim_draw=True)
        legal_actions = [] if terminal_by_rule else _legal_actions_of(board)
        key = _board_key(board)
        color = 1 if board.turn == chess.WHITE else 0
        new_path = node.path + (action,)
        if terminal_by_rule or not legal_actions:
            q_term = get_terminal_q(board)
            return Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32),
                        q=q_term, depth=node.depth + 1, path=new_path, terminal=True)

        feats, tc_val, elo_std, _ = encode_board(board, occ.get(_board_key(board), 0))
        logits_np, wdl_np, mlh_np, x_np, _ = yield (
            feats, int(tc_val), float(elo_std), color, cache)
        occ[key] = occ.get(key, 0) + 1
        q = wdl_logits_to_q(wdl_np)
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_full = np.full(1936, -3e4, dtype=np.float32)
        logits_full[legal_arr] = logits_np[legal_arr]
        return Node(legal=legal_arr, logits=logits_full[legal_arr], q=q,
                    depth=node.depth + 1, path=new_path)

    def _simulate_gen(self, node: Node, qbox: list, counters: dict):
        if node.is_terminal:
            return float(node.q)
        a = select_action(node, self.cfg.c_visit, self.cfg.c_scale)
        edge_idx = int(np.flatnonzero(node.legal == a)[0])
        key = int(a)
        child = node.children.get(key)
        if child is None:
            child = yield from self._expand_gen(node, a)
            node.children[key] = child
            counters["n_nodes"] += 1
            if child.depth > counters["max_depth"]:
                counters["max_depth"] = child.depth
            if child.is_terminal:
                counters["n_terminal"] += 1
            if child.q < qbox[0]:
                qbox[0] = child.q
            if child.q > qbox[1]:
                qbox[1] = child.q
            val = -float(child.q)
        else:
            val = -(yield from self._simulate_gen(child, qbox, counters))
        node.record_child(edge_idx, val)
        return val

    def _order_halving_gen(self, root: Node):
        cfg = self.cfg
        if root.is_terminal:
            return {"action": None, "qmin": None, "qmax": None, "n_nodes": 0,
                    "n_terminal": 0, "sims_used": 0, "max_depth": 0}

        m0 = min(cfg.m0, len(root.legal))
        cands = gumbel_topm(root, m0=m0, rng=self.rng, g=self.cfg.gumbel_g)
        m = len(cands)
        rounds = _n_rounds(m)
        surv = [_Candidate(action=a, noise=ns) for a, ns in cands]
        base, rem = divmod(cfg.n_sims, rounds)
        budget_per_round = [base + (1 if i < rem else 0) for i in range(rounds)]

        qbox = [root.q, root.q]  # [qmin, qmax]
        counters = {"n_nodes": 0, "n_terminal": 0, "max_depth": 0}

        def do_sim_root(c: _Candidate):
            if c.child is None:
                child = yield from self._expand_gen(root, c.action)
                c.child = child
                counters["n_nodes"] += 1
                if child.depth > counters["max_depth"]:
                    counters["max_depth"] = child.depth
                if child.is_terminal:
                    counters["n_terminal"] += 1
                if child.q < qbox[0]:
                    qbox[0] = child.q
                if child.q > qbox[1]:
                    qbox[1] = child.q
                val = -float(child.q)
            elif c.child.is_terminal:
                val = -float(c.child.q)
            else:
                val = -(yield from self._simulate_gen(c.child, qbox, counters))
            idx = int(np.flatnonzero(root.legal == c.action)[0])
            root.record_child(idx, val)

        sims_used = 0
        for r, budget in enumerate(budget_per_round):
            if len(surv) == 1:
                budget = sum(budget_per_round[r:])
            per_base, per_rem = divmod(budget, len(surv))
            for i, c in enumerate(surv):
                k = per_base + (1 if i < per_rem else 0)
                for _ in range(k):
                    yield from do_sim_root(c)
                    sims_used += 1
            if len(surv) == 1:
                break
            l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
            s_root_vals = qtransform_completed(root, cfg.c_visit, cfg.c_scale)
            s_map = {int(a): float(x) for a, x in zip(root.legal, s_root_vals)}
            scored = sorted(((c.noise + l_root[c.action] + s_map[c.action], c) for c in surv),
                            key=lambda t: -t[0])
            keep = max(1, (len(surv) + 1) // 2)
            surv = [c for _, c in scored[:keep]]

        return {"action": int(surv[0].action), "qmin": qbox[0], "qmax": qbox[1],
                "n_nodes": counters["n_nodes"], "n_terminal": counters["n_terminal"],
                "sims_used": sims_used, "max_depth": counters["max_depth"]}

    def result(self) -> tuple[int, int, bool]:
        """→ (result 白视角 0/1/2, termination_reason 编码, is_truncated)。

        统一走 `adapter.classify_final_board`（口径 = 对局循环的 claim_draw=True），
        只有规则未终局才是 300 ply 封顶截断。
        """
        result, reason, is_truncated = classify_final_board(self.board)
        return result, TERM_CODES.index(reason), is_truncated


# ------------------------- 驱动器：跨局拼批 + 槽位复用 -------------------------

class Driver:
    def __init__(self, model: ModelWrapper, cfg: SelfPlayConfig, writer: V3ShardWriter):
        self.model = model
        self.cfg = cfg
        self.writer = writer
        seed_seq = np.random.SeedSequence(cfg.seed)
        self.child_seeds = seed_seq.spawn(cfg.num_games)
        self.next_idx = 0
        self.slots: list[dict | None] = [None] * cfg.concurrency
        self.games_done = 0
        self.total_plies = 0
        self.total_nodes = 0
        self.total_terminal = 0
        self.total_sims = 0
        self.total_max_depth = 0
        self.budget_violations = 0
        self.term_reason_counts = [0] * len(TERM_CODES)
        self.truncated_games = 0

    def _new_game(self) -> GameState | None:
        if self.next_idx >= self.cfg.num_games:
            return None
        game = GameState(self.next_idx, self.model, self.cfg, self.child_seeds[self.next_idx])
        self.next_idx += 1
        return game

    def _start_slot(self, i: int) -> None:
        game = self._new_game()
        if game is None:
            self.slots[i] = None
            return
        gen = game.run()
        try:
            req = gen.send(None)
        except StopIteration:
            self._finish_game(game)
            self._start_slot(i)
            return
        self.slots[i] = {"game": game, "gen": gen, "req": req}

    def _finish_game(self, game: GameState) -> None:
        result, term_reason, is_truncated = game.result()
        meta = np.zeros((), dtype=META_V3_DTYPE)
        meta["n_plies"] = len(game.actions)
        meta["tc_bucket"] = int(self.cfg.tc_bucket)
        meta["result"] = result
        meta["elo_missing"] = 0
        meta["elo_mean"] = self.cfg.elo
        meta["game_key"] = make_game_key(f"selfplay_gen{self.cfg.gen_id}", game.game_idx)
        meta["gen_id"] = self.cfg.gen_id
        meta["ckpt_step"] = self.cfg.ckpt_step
        meta["termination_reason"] = term_reason
        meta["is_truncated"] = 1 if is_truncated else 0
        pipol = encode_v3_pipol(game.pipol_actions, game.pipol_probs)
        poff = _compute_pipol_byte_offsets(game.pipol_actions)
        self.writer.add(meta, np.array(game.actions, dtype=np.uint16), pipol, poff)

        self.games_done += 1
        self.total_plies += len(game.actions)
        self.total_nodes += game.n_nodes_total
        self.total_terminal += game.n_terminal_total
        self.total_sims += game.sims_total
        self.total_max_depth += game.max_depth_total
        self.budget_violations += game.budget_violations
        self.term_reason_counts[term_reason] += 1
        if is_truncated:
            self.truncated_games += 1

    def _model_step(self, reqs: list[tuple]) -> list[tuple]:
        features = np.stack([r[0] for r in reqs]).astype(np.float32)
        tc = [r[1] for r in reqs]
        elo = [r[2] for r in reqs]
        color = [r[3] for r in reqs]
        caches = [r[4] for r in reqs]
        batched_cache = concat_caches(caches)
        logits, wdl, mlh, x, cache_new = self.model.step_batch(features, tc, elo, color, batched_cache)
        per_item = split_cache(cache_new, len(reqs))
        return [(logits[i], wdl[i], mlh[i], x[i], per_item[i]) for i in range(len(reqs))]

    def run(self, progress_every: int = 20) -> None:
        t0 = time.time()
        last_reported = 0
        for i in range(self.cfg.concurrency):
            self._start_slot(i)
        while any(s is not None for s in self.slots):
            active = [(i, s) for i, s in enumerate(self.slots) if s is not None]
            reqs = [s["req"] for _, s in active]
            responses = self._model_step(reqs)
            for (i, s), resp in zip(active, responses):
                try:
                    req = s["gen"].send(resp)
                    s["req"] = req
                except StopIteration:
                    self._finish_game(s["game"])
                    self._start_slot(i)
            if self.games_done - last_reported >= progress_every:
                last_reported = self.games_done
                elapsed = time.time() - t0
                print(f"[{elapsed:7.1f}s] 已完成 {self.games_done}/{self.cfg.num_games} 局，"
                      f"batch={len(active)}，{self.games_done / max(elapsed, 1e-6):.3f} games/s")


# ------------------------- 生成主循环 -------------------------

def generate(cfg: SelfPlayConfig) -> dict:
    writer = V3ShardWriter(cfg.out_dir, cfg.tag)
    model = ModelWrapper(cfg.ckpt, cfg.device)
    driver = Driver(model, cfg, writer)

    t0 = time.time()
    driver.run()
    writer.flush()
    elapsed = time.time() - t0

    stats = {
        "games": driver.games_done,
        "plies": driver.total_plies,
        "elapsed_s": elapsed,
        "games_per_s": driver.games_done / max(elapsed, 1e-6),
        "plies_per_s": driver.total_plies / max(elapsed, 1e-6),
        "avg_search_nodes_per_ply": driver.total_nodes / max(driver.total_plies, 1),
        "avg_sims_per_ply": driver.total_sims / max(driver.total_plies, 1),
        "avg_max_tree_depth": driver.total_max_depth / max(driver.total_plies, 1),
        "budget_violations": driver.budget_violations,
        "termination_reason_counts": dict(zip(TERM_CODES, driver.term_reason_counts)),
        "truncated_rate": driver.truncated_games / max(driver.games_done, 1),
        "concurrency": cfg.concurrency,
        "n_sims": cfg.n_sims,
        "m0": cfg.m0,
        "gen_id": cfg.gen_id,
        "ckpt_step": cfg.ckpt_step,
    }
    manifest_path = os.path.join(cfg.out_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        manifest = {}
    manifest["gen"] = stats
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    print(f"生成完毕：{driver.games_done} 局，{elapsed:.1f}s，"
          f"{stats['games_per_s']:.3f} games/s，{stats['plies_per_s']:.2f} plies/s，"
          f"封顶率 {stats['truncated_rate']:.1%}")
    return stats


# ------------------------- 多进程编排：跨核并行（CPU 侧才是当前瓶颈） -------------------------
#
# 单进程内的 Driver 只把"同一进程内并发局"的模型前向拼批，跨局树逻辑（棋盘复制/路径重算/
# 合法着生成）仍是单进程单核 Python，GPU 因而长期低利用率（实测 17%）而 CPU 只用满 1/20 核。
# 这里改为起 N 个独立 OS 进程（各自独立 CUDA context，避免 CUDA fork 后不安全的问题），
# 每个进程内部仍用上面的单进程 Driver 逻辑，各分到 games/N 局、独立 tag 与解耦的随机种子；
# 全部结束后把各进程产出的分片文件搬回顶层目录并合并 manifest。

def _merge_worker_outputs(out_dir: str, worker_dirs: list[str], wall_elapsed: float) -> dict:
    combined_shards: list[str] = []
    total_games = total_steps = total_skipped = 0
    total_plies = total_nodes = total_terminal = total_sims = 0
    total_max_depth = 0
    budget_violations = 0
    term_counts = [0] * len(TERM_CODES)
    truncated_games = 0
    last_cfg: dict = {}

    for wd in worker_dirs:
        mpath = os.path.join(wd, "manifest.json")
        with open(mpath, encoding="utf-8") as fh:
            wm = json.load(fh)
        for shard in wm.get("shards", []):
            for ext in (".actions.bin", ".meta.npz", ".pipol.bin", ".pipol.offsets.bin"):
                src = os.path.join(wd, shard + ext)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(out_dir, shard + ext))
            combined_shards.append(shard)
        total_games += wm.get("games", 0)
        total_steps += wm.get("steps", 0)
        total_skipped += wm.get("skipped", 0)
        gen = wm.get("gen", {})
        total_plies += gen.get("plies", 0)
        total_nodes += gen.get("avg_search_nodes_per_ply", 0.0) * gen.get("plies", 0)
        total_sims += gen.get("avg_sims_per_ply", 0.0) * gen.get("plies", 0)
        total_max_depth += gen.get("avg_max_tree_depth", 0.0) * gen.get("plies", 0)
        budget_violations += gen.get("budget_violations", 0)
        for k, v in gen.get("termination_reason_counts", {}).items():
            term_counts[TERM_CODES.index(k)] += v
        truncated_games += round(gen.get("truncated_rate", 0.0) * gen.get("games", 0))
        last_cfg = {k: gen.get(k) for k in ("n_sims", "m0", "gen_id", "ckpt_step") if k in gen}
        shutil.rmtree(wd, ignore_errors=True)

    stats = {
        "games": total_games,
        "plies": total_plies,
        "elapsed_s": wall_elapsed,
        "games_per_s": total_games / max(wall_elapsed, 1e-6),
        "plies_per_s": total_plies / max(wall_elapsed, 1e-6),
        "avg_search_nodes_per_ply": total_nodes / max(total_plies, 1),
        "avg_sims_per_ply": total_sims / max(total_plies, 1),
        "avg_max_tree_depth": total_max_depth / max(total_plies, 1),
        "budget_violations": budget_violations,
        "termination_reason_counts": dict(zip(TERM_CODES, term_counts)),
        "truncated_rate": truncated_games / max(total_games, 1),
        "workers": len(worker_dirs),
        **last_cfg,
    }
    manifest = {"shards": combined_shards, "months": [], "games": total_games,
                "steps": total_steps, "skipped": total_skipped, "gen": stats}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return stats


def run_workers(args: argparse.Namespace) -> None:
    n = args.workers
    seed_seq = np.random.SeedSequence(args.seed)
    worker_seeds = seed_seq.spawn(n)
    base, rem = divmod(args.games, n)
    counts = [base + (1 if i < rem else 0) for i in range(n)]

    os.makedirs(args.out, exist_ok=True)
    worker_dirs = [os.path.join(args.out, f"_w{i}") for i in range(n)]
    procs = []
    t0 = time.time()
    for i, (wdir, n_games) in enumerate(zip(worker_dirs, counts)):
        if n_games == 0:
            continue
        os.makedirs(wdir, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--ckpt", args.ckpt, "--out", wdir, "--tag", f"{args.tag}-w{i}",
               "--games", str(n_games), "--concurrency", str(args.concurrency),
               "--n_sims", str(args.n_sims), "--m0", str(args.m0),
               "--seed", str(int(worker_seeds[i].generate_state(1)[0])),
               "--gen_id", str(args.gen_id), "--ckpt_step", str(args.ckpt_step),
               "--g", str(args.g)]
        log_path = os.path.join(wdir, "worker.log")
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        procs.append((proc, log_fh, wdir))
        print(f"[worker {i}] 启动 pid={proc.pid} games={n_games}", flush=True)

    failed = []
    for i, (proc, log_fh, wdir) in enumerate(procs):
        rc = proc.wait()
        log_fh.close()
        print(f"[worker {i}] 退出码 {rc}", flush=True)
        if rc != 0:
            failed.append((i, wdir))
    if failed:
        raise RuntimeError(f"worker 进程失败：{failed}；查看各自 worker.log 排查后重试，"
                           f"不合并已产出的部分分片（避免正式数据混入未验证的失败批次）")

    elapsed = time.time() - t0
    stats = _merge_worker_outputs(args.out, [wd for _, _, wd in procs], elapsed)
    print(f"[全部 worker 完成] {n} 进程，{stats['games']} 局，{elapsed:.1f}s，"
          f"{stats['games_per_s']:.3f} games/s，{stats['plies_per_s']:.2f} plies/s，"
          f"封顶率 {stats['truncated_rate']:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="stage_b")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_id", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 OS 进程数（跨核；每进程独立 CUDA context）。>1 时委派给 run_workers。")
    ap.add_argument("--ckpt_step", type=int, default=0)
    ap.add_argument("--g", type=float, default=1.0,
                    help="Gumbel 噪声尺度；1.0 训练/生成，0.0 评测/换代 arena")
    args = ap.parse_args()

    if args.workers > 1:
        run_workers(args)
        return

    cfg = SelfPlayConfig(
        ckpt=args.ckpt,
        out_dir=args.out,
        tag=args.tag,
        num_games=args.games,
        concurrency=args.concurrency,
        n_sims=args.n_sims,
        m0=args.m0,
        seed=args.seed,
        gen_id=args.gen_id,
        ckpt_step=args.ckpt_step,
        gumbel_g=args.g,
    )
    generate(cfg)


if __name__ == "__main__":
    main()
