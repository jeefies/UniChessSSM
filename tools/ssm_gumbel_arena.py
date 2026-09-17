"""最小 arena（review §六-第二步）：两个 checkpoint 用 Gumbel g=0/n=64 对弈。

用法：
  python tools/ssm_gumbel_arena.py --ckpt-a runs/champion.pt --ckpt-b runs/challenger.pt \\
      --out runs/arena_ab --games 64 --pairs 4

--games: 总局数（每人每色各一半）。--pairs: 配对开局数（从开局库轮流取）。
默认--g 0 --n_sims 64 --m0 16。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import action_to_move, move_to_action
from stateseq.conditions import TimeControlBucket
from stateseq.features import encode
from stateseq.gumbel import (
    C_SCALE, C_VISIT, N_SIMS, M0, TERM_CODES,
    Node, _Candidate, _n_rounds, completed_q, export_pi_prime,
    gumbel_topm, select_action, sigma,
)
from stateseq.model import SeqModel
from stateseq.model_r import clone_cache

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _board_key(board: chess.Board) -> str:
    return board.fen().split(" ")[0]


def _legal_actions_of(board: chess.Board) -> list[int]:
    return [a for m in board.legal_moves if (a := move_to_action(m)) is not None]


def _resolve_move(action: int, board: chess.Board):
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


class ArenaModel:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        self.seq = SeqModel(dropout=0.0)
        self.seq.load_state_dict(state_dict)
        self.seq.to(device).eval()

    def initial_cache(self):
        return self.seq.initial_cache(1, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, features: np.ndarray, tc_bucket: int, elo: float, color: int, cache):
        f_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        tc_t = torch.tensor([tc_bucket], dtype=torch.long, device=self.device)
        elo_t = torch.tensor([elo], dtype=torch.float32, device=self.device)
        color_t = torch.tensor([color], dtype=torch.long, device=self.device)
        logits, wdl, mlh, x, cache_new = self.seq.step(f_t, tc_t, elo_t, color_t, cache)
        return logits.cpu().numpy()[0], wdl.cpu().numpy()[0], mlh.cpu().numpy()[0], x.cpu().numpy()[0], cache_new


def play_one_game(model_w: ArenaModel, model_b: ArenaModel, cfg) -> dict:
    models = {chess.WHITE: model_w, chess.BLACK: model_b}
    board = chess.Board()
    root_cache = {color: model.initial_cache() for color, model in models.items()}
    occur = {}
    actions = []

    max_plies = cfg.max_plies
    tc = int(cfg.tc_bucket)
    elo = cfg.elo
    n_sims = cfg.n_sims
    m0 = cfg.m0
    c_visit = cfg.c_visit
    c_scale = cfg.c_scale

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        turn = board.turn
        model = models[turn]
        cache = root_cache[turn]

        key = _board_key(board)
        occ = occur.get(key, 0)
        feats = encode(board, occurrence=occ)
        legal_actions = _legal_actions_of(board)
        if not legal_actions:
            break
        color_val = 1 if turn == chess.WHITE else 0
        logits_np, wdl_np, mlh_np, x_np, cache_new = model.step(feats, tc, elo, color_val, cache)
        root_cache[turn] = cache_new
        occur[key] = occ + 1

        q = float(wdl_np[0] - wdl_np[2])
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_full = np.full(1936, -3e4, dtype=np.float32)
        logits_full[legal_arr] = logits_np[legal_arr]
        root = Node(legal=legal_arr, logits=logits_full[legal_arr], q=q, depth=0, path=())

        cands = gumbel_topm(root, m0=m0, rng=np.random.default_rng(), g=0.0)
        if not cands:
            break
        m = len(cands)
        rounds = _n_rounds(m)
        surv = [_Candidate(action=a, noise=ns) for a, ns in cands]
        base_b, rem_b = divmod(n_sims, rounds)
        budget = [base_b + (1 if i < rem_b else 0) for i in range(rounds)]
        qbox = [root.q, root.q]

        for r, bgt in enumerate(budget):
            if len(surv) == 1:
                bgt = sum(budget[r:])
            pb, pr = divmod(bgt, len(surv))
            for i, c in enumerate(surv):
                k = pb + (1 if i < pr else 0)
                for _ in range(k):
                    child = _expand_search(root, c.action, model, models, root_cache, board, occur, qbox)
                    idx = int(np.flatnonzero(root.legal == c.action)[0])
                    root.record_child(idx, -float(child.q))
            if len(surv) == 1:
                break
            l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
            cq = completed_q(root, qbox[0], qbox[1])
            sv = sigma(cq, root.n_max, c_visit, c_scale)
            sm = {int(a): float(x) for a, x in zip(root.legal, sv)}
            scored = sorted(((c.noise + l_root[c.action] + sm[c.action], c) for c in surv), key=lambda t: -t[0])
            surv = [c for _, c in scored[:max(1, (len(surv) + 1) // 2)]]

        chosen = surv[0].action
        actions.append(int(chosen))
        mv = _resolve_move(chosen, board)
        if mv is None:
            break
        board.push(mv)

    result_white = 0 if board.is_checkmate() and board.turn == chess.BLACK else (
        2 if board.is_checkmate() else 1)
    return {"actions": actions, "result": result_white, "n_plies": len(actions)}


def _expand_search(root, action, model_active, models, root_caches, board, occur, qbox):
    """在 arena 搜索中展开一个子节点。简化版——只展开 1 层，不走递归树。"""
    mv = _resolve_move(action, board)
    if mv is None:
        return Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32), q=0.0, terminal=True)
    board.push(mv)
    key = _board_key(board)
    occ = occur.get(key, 0)
    feats = encode(board, occurrence=occ)
    turn = board.turn
    model = models[turn]
    cache = root_caches[turn]
    color_val = 1 if turn == chess.WHITE else 0
    tc = 2
    elo = 2567.5
    logits_np, wdl_np, _, _, _ = model.step(feats, tc, elo, color_val, cache)
    board.pop()
    q = float(wdl_np[0] - wdl_np[2])
    if q < qbox[0]:
        qbox[0] = q
    if q > qbox[1]:
        qbox[1] = q
    legal_arr = np.array(_legal_actions_of(board), dtype=np.int64)
    logits_full = np.full(1936, -3e4, dtype=np.float32)
    logits_full[legal_arr] = logits_np[legal_arr]
    return Node(legal=legal_arr, logits=logits_full[legal_arr], q=q, depth=root.depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_a = ArenaModel(args.ckpt_a)
    model_b = ArenaModel(args.ckpt_b)

    cfg = lambda: None
    cfg.n_sims = args.n_sims
    cfg.m0 = args.m0
    cfg.max_plies = args.max_plies
    cfg.tc_bucket = TimeControlBucket.RAPID
    cfg.elo = 2567.5
    cfg.c_visit = C_VISIT
    cfg.c_scale = C_SCALE

    half = args.games // 2
    results = []  # [(white_ckpt, black_ckpt, result: 0/1/2)]

    print(f"A={args.ckpt_a} vs B={args.ckpt_b}")
    print(f"Gumbel g=0 n={args.n_sims} m0={args.m0}")
    print(f"games={args.games} ({half} each color)")

    t0 = time.time()
    for g in range(half):
        gd = play_one_game(model_a, model_b, cfg)
        results.append(("A", "B", gd["result"]))
        if (g + 1) % 8 == 0:
            print(f"  [{time.time()-t0:.0f}s] game {g+1}/{half} (B vs W)")

    for g in range(half):
        gd = play_one_game(model_b, model_a, cfg)
        # 翻转结果：model_b 走白时的结果 vs model_a 走黑
        r = gd["result"]
        flipped = 0 if r == 2 else 2 if r == 0 else 1
        results.append(("B", "A", flipped))
        if (g + 1) % 8 == 0:
            print(f"  [{time.time()-t0:.0f}s] game {half+g+1}/{args.games} (swapped)")

    elapsed = time.time() - t0
    wins_a = sum(1 for w, b, r in results if r == 0 and w == "A")
    wins_b = sum(1 for w, b, r in results if r == 0 and w == "B")
    draws = sum(1 for _, _, r in results if r == 1)
    score_a = wins_a + 0.5 * draws

    manifest = {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "total_games": len(results),
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "score_a": score_a,
        "score_a_percent": score_a / max(len(results), 1) * 100,
        "n_sims": args.n_sims, "m0": args.m0,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.out, "arena.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    print(f"用时 {elapsed:.0f}s")


if __name__ == "__main__":
    main()