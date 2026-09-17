"""Stage B Gumbel 自对弈生成器（规格 §2.4 / §2.3 / §2.5）。

并发局数：128（初值，按显存/CPU 实测调）。
每代局数：首轮闭环 2k–5k；主循环 25k/代。
输出：v3 分片（actions + pipol + 扩展 meta）+ manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.conditions import TimeControlBucket
from stateseq.features import encode
from stateseq.gumbel import Node, order_halving, export_pi_prime, C_VISIT, C_SCALE, N_SIMS, M0, TERM_CODES
from stateseq.data.gshards import V3ShardWriter, encode_v3_pipol, META_V3_DTYPE
from stateseq.model_r import clone_cache

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


# ------------------------- 模型封装 -------------------------

class ModelWrapper:
    """封装 champion 模型，提供单步推理与 R cache 管理。"""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        from stateseq.model import SeqModel
        self.seq = SeqModel(dropout=0.0)
        self.seq.load_state_dict(state_dict)
        self.seq.to(device).eval()

    @torch.no_grad()
    def step(self, features: np.ndarray, tc: int, elo: float, color: int, cache):
        """单步递推，返回 (logits, wdl, mlh, x, cache_new)。"""
        f_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        tc_t = torch.tensor([tc], dtype=torch.long, device=self.device)
        elo_t = torch.tensor([elo], dtype=torch.float32, device=self.device)
        color_t = torch.tensor([color], dtype=torch.long, device=self.device)
        logits, wdl, mlh, x, cache_new = self.seq.step(f_t, tc_t, elo_t, color_t, cache)
        return logits.cpu().numpy()[0], wdl.cpu().numpy()[0], mlh.cpu().numpy()[0], x.cpu().numpy()[0], cache_new


# ------------------------- 搜索树 -------------------------

class SearchTree:
    """管理单局的搜索树与 R cache。"""

    def __init__(self, model: ModelWrapper, cfg: SelfPlayConfig):
        self.model = model
        self.cfg = cfg
        self.root_cache = model.seq.initial_cache(1, device=cfg.device, dtype=torch.float32)
        self.board = chess.Board()
        self.occurrence: dict[str, int] = {}
        self.actions: list[int] = []
        self.pipol_actions: list[np.ndarray] = []
        self.pipol_probs: list[np.ndarray] = []

    def _board_key(self) -> str:
        return self.board.fen().split(" ")[0]

    def _encode(self) -> np.ndarray:
        key = self._board_key()
        occ = self.occurrence.get(key, 0)
        return encode(self.board, occurrence=occ)

    def _update_occurrence(self) -> None:
        key = self._board_key()
        self.occurrence[key] = self.occurrence.get(key, 0) + 1

    def _legal_actions(self) -> list[int]:
        from stateseq.actions import move_to_action
        legal = []
        for m in self.board.legal_moves:
            a = move_to_action(m)
            if a is not None:
                legal.append(a)
        return legal

    def _do_model_step(self, features: np.ndarray, legal_actions: list[int], cache) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, object]:
        logits_np, wdl_np, mlh_np, x_np, cache_new = self.model.step(
            features, int(self.cfg.tc_bucket), self.cfg.elo,
            1 if self.board.turn == chess.WHITE else 0, cache
        )
        q = float(wdl_np[0] - wdl_np[2])
        return logits_np, q, x_np, mlh_np, cache_new

    def _expand(self, parent_node: Node, action: int, work_cache) -> Node | None:
        """从 parent_node 沿 action 扩展子节点（使用 work_cache，不修改 root_cache）。"""
        features = self._encode()
        legal_actions = self._legal_actions()
        logits_np, q, x_np, mlh_np, _ = self._do_model_step(features, legal_actions, work_cache)
        logits_full = np.full(1936, -3e4, dtype=np.float32)
        logits_full[legal_actions] = logits_np[legal_actions]
        child_legal = np.array(legal_actions, dtype=np.int64)
        child_logits = logits_full[legal_actions]
        return Node(legal=child_legal, logits=child_logits, q=q, depth=parent_node.depth + 1)

    def play_move(self, rng: np.random.Generator) -> bool:
        """执行一步搜索并走子，返回 False 表示对局结束。"""
        if self.board.is_game_over(claim_draw=True):
            return False

        features = self._encode()
        self._update_occurrence()
        legal_actions = self._legal_actions()
        if not legal_actions:
            return False

        logits_np, q, x_np, mlh_np, cache_new = self._do_model_step(features, legal_actions, self.root_cache)
        self.root_cache = cache_new
        logits_full = np.full(1936, -3e4, dtype=np.float32)
        logits_full[legal_actions] = logits_np[legal_actions]

        node = Node(legal=np.array(legal_actions, dtype=np.int64),
                    logits=logits_full[legal_actions], q=q, depth=0)

        def _expand_fn(parent_node, action):
            work_cache = clone_cache(self.root_cache)
            return self._expand(parent_node, action, work_cache)

        res = order_halving(node, _expand_fn, n_sims=self.cfg.n_sims,
                            m0=min(self.cfg.m0, len(legal_actions)),
                            g=1.0, seed=int(rng.integers(0, 2**31)),
                            c_visit=self.cfg.c_visit, c_scale=self.cfg.c_scale)

        if res["action"] is None:
            return False

        chosen = res["action"]
        self.actions.append(int(chosen))

        ids, probs = export_pi_prime(node, res["qmin"], res["qmax"])
        self.pipol_actions.append(ids.astype(np.uint16))
        self.pipol_probs.append(probs.astype(np.float32))

        from stateseq.actions import action_to_move
        move = action_to_move(chosen)
        if move is None:
            return False
        self.board.push(move)
        return True

    def result(self) -> tuple[int, int, bool]:
        if self.board.is_checkmate():
            r = 0 if self.board.turn == chess.BLACK else 2
            return r, TERM_CODES.index("checkmate"), False
        if self.board.is_stalemate():
            return 1, TERM_CODES.index("stalemate"), False
        if self.board.is_fifty_moves():
            return 1, TERM_CODES.index("fifty_move"), False
        if self.board.is_repetition(3):
            return 1, TERM_CODES.index("threefold"), False
        if self.board.is_insufficient_material():
            return 1, TERM_CODES.index("insufficient_material"), False
        return 1, TERM_CODES.index("truncated"), True


# ------------------------- 生成主循环 -------------------------

def _compute_pipol_byte_offsets(per_ply_actions: list[np.ndarray]) -> np.ndarray:
    """计算 pipol 变长目标的字节偏移表（含末尾哨兵）。"""
    off = [0]
    for acts in per_ply_actions:
        off.append(off[-1] + 2 + len(acts) * 4)
    return np.array(off, dtype=np.int32)


def generate(cfg: SelfPlayConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    writer = V3ShardWriter(cfg.out_dir, cfg.tag)
    t0 = time.time()
    games_done = 0

    model = ModelWrapper(cfg.ckpt, cfg.device)

    for i in range(cfg.num_games):
        tree = SearchTree(model, cfg)
        for ply in range(cfg.max_plies):
            if not tree.play_move(rng):
                break
        result, term_reason, is_truncated = tree.result()
        meta = np.zeros((), dtype=META_V3_DTYPE)
        meta["n_plies"] = len(tree.actions)
        meta["tc_bucket"] = int(cfg.tc_bucket)
        meta["result"] = result
        meta["elo_missing"] = 0
        meta["elo_mean"] = cfg.elo
        meta["gen_id"] = cfg.gen_id
        meta["ckpt_step"] = cfg.ckpt_step
        meta["termination_reason"] = term_reason
        meta["is_truncated"] = 1 if is_truncated else 0
        pipol = encode_v3_pipol(tree.pipol_actions, tree.pipol_probs)
        poff = _compute_pipol_byte_offsets(tree.pipol_actions)
        writer.add(meta, np.array(tree.actions, dtype=np.uint16), pipol, poff)
        games_done += 1

    writer.flush()
    elapsed = time.time() - t0
    print(f"生成完毕：{games_done} 局，{elapsed:.1f}s，{games_done / max(elapsed, 1e-6):.2f} games/s")


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
    args = ap.parse_args()

    cfg = SelfPlayConfig(
        ckpt=args.ckpt,
        out_dir=args.out,
        tag=args.tag,
        num_games=args.games,
        concurrency=args.concurrency,
        n_sims=args.n_sims,
        m0=args.m0,
        seed=args.seed,
    )
    generate(cfg)


if __name__ == "__main__":
    main()
