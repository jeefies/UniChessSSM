"""Arena evaluation script for SSM vs Transformer / ResNet.

Supports:
- Opponent type: 'transformer' (TransformerEngine) or 'resnet' (UniChessEngine)
- SSM engine using native Gumbel search (order_halving) with full historical R cache and occurrence tracking
- 32 opening pairs (64 games) by default, swapped white/black for fair evaluation
- Multiprocessing worker pool with round-robin opening pair assignment
- Evaluation metrics: Win/Draw/Loss rates, score percentage, Elo difference (with 95% confidence interval),
  and termination reason distribution.

Usage:
  python tools/ssm_eval_vs_transformer.py \
      --ssm-ckpt runs/stage_b_training_fix500_cs01/best.pt \
      --opponent-type transformer \
      --opponent-ckpt /home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt \
      --games 64 --workers 2 --out runs/arena_ssm_vs_transformer
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

SSM_ROOT = Path(__file__).resolve().parents[1]
if str(SSM_ROOT) not in sys.path:
    sys.path.insert(0, str(SSM_ROOT))

from stateseq.actions import move_to_action
from stateseq.adapter import (
    classify_final_board,
    encode_board,
    get_terminal_q,
    wdl_logits_to_q,
)
from stateseq.data.sequences import _board_key
from stateseq.gumbel import C_SCALE, C_VISIT, Node, order_halving
from stateseq.model import SeqModel
from stateseq.model_r import clone_cache

# 32 standard opening sequences for paired evaluation
OPENINGS_32 = [
    "e4 e5 Nf3 Nc6 Bb5",                 # Ruy Lopez
    "d4 d5 c4 e6",                       # QGD
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",     # Sicilian Najdorf / Open
    "d4 Nf6 c4 g6 Nc3 Bg7",              # King's Indian
    "e4 e6 d4 d5",                       # French
    "d4 Nf6 c4 e6 Nf3 Bb4+",             # Bogo-Indian
    "e4 c6 d4 d5",                       # Caro-Kann
    "c4 e5",                             # English
    "Nf3 Nf6 c4 g6",                     # Reti / King's Indian setup
    "d4 d5 c4 c6",                       # Slav
    "e4 d5 exd5 Qxd5 Nc3 Qa5",           # Scandinavian
    "d4 Nf6 c4 c5",                      # Benoni
    "e4 e5 Nf3 Nf6",                     # Petroff
    "d4 e6 c4 Bb4+",                     # Nimzo-adjacent
    "e4 e5 Nf3 Nc6 Bc4",                 # Italian
    "d4 g6 c4 Bg7",                      # Modern / King's Indian
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4",         # Sicilian Kan/Taimanov
    "d4 d5 Nf3 Nf6 c4",                  # Catalan / QGD setup
    "e4 e5 f4",                          # King's Gambit
    "d4 f5",                             # Dutch
    "c4 c5 Nf3 Nf6 d4 cxd4 Nxd4",        # Symmetrical English
    "Nf3 d5 g3 Nf6 Bg2",                 # King's Indian Attack
    "e4 c5 Nc3 Nc6 g3",                  # Closed Sicilian
    "d4 Nf6 Bg5",                        # Trompowsky
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4",        # Scotch
    "d4 d5 Bf4",                         # London System
    "e4 c5 c3",                          # Alapin Sicilian
    "d4 Nf6 c4 e6 Nc3 Bb4",              # Nimzo-Indian
    "e4 e5 Nf3 d6",                      # Philidor
    "d4 d5 Nc3 Nf6 Bg5",                 # Richter-Veresov
    "b3 e5 Bb2 Nc6",                     # Nimzo-Larsen
    "e4 c5 Nf3 Nc6 Bb5",                 # Rossolimo Sicilian
]


def _legal_actions_of(board: chess.Board) -> list[int]:
    return [a for m in board.legal_moves if (a := move_to_action(m)) is not None]


def _resolve_move(action: int, board: chess.Board) -> chess.Move | None:
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


def _result_str(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "*"
    if outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def compute_elo_diff(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Compute Elo difference and 95% confidence margin from match score.

    SSM score = (wins + 0.5 * draws) / total
    Returns: (elo_diff, elo_margin_95)
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / total

    # Clamp score to avoid log(0)
    clamped_score = min(max(score, 1e-4), 1.0 - 1e-4)
    elo = -400.0 * math.log10(1.0 / clamped_score - 1.0)

    # Standard error of score
    # Var(S) = [W*(1-s)^2 + D*(0.5-s)^2 + L*(0-s)^2] / total^2
    w_term = wins * ((1.0 - score) ** 2)
    d_term = draws * ((0.5 - score) ** 2)
    l_term = losses * ((0.0 - score) ** 2)
    variance = (w_term + d_term + l_term) / (total * total)
    std_err = math.sqrt(variance)

    # Delta method for Elo: d(Elo)/ds = 400 / (ln(10) * s * (1 - s))
    factor = 400.0 / (math.log(10.0) * clamped_score * (1.0 - clamped_score))
    margin_95 = 1.96 * std_err * factor
    return elo, margin_95


class SSMArenaEngine:
    """SSM engine wrapper with Gumbel search and persistent R-cache across the game."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        n_sims: int = 64,
        m0: int = 16,
        c_visit: float = C_VISIT,
        c_scale: float = C_SCALE,
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.n_sims = n_sims
        self.m0 = m0
        self.c_visit = c_visit
        self.c_scale = c_scale

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        self.seq = SeqModel(dropout=0.0)
        self.seq.load_state_dict(sd)
        self.seq.to(device).eval()

    def initial_cache(self, b: int = 1):
        return self.seq.initial_cache(b, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, feats, tc, elo, color, cache):
        f_t = torch.from_numpy(feats).float().to(self.device)
        tc_t = torch.tensor(tc, dtype=torch.long, device=self.device)
        elo_t = torch.tensor(elo, dtype=torch.float32, device=self.device)
        color_t = torch.tensor(color, dtype=torch.long, device=self.device)
        logits, wdl, mlh, x, cache_new = self.seq.step(f_t, tc_t, elo_t, color_t, cache)
        return (
            logits.cpu().numpy(),
            wdl.cpu().numpy(),
            mlh.cpu().numpy(),
            x.cpu().numpy(),
            cache_new,
        )

    def expand_child(
        self,
        board: chess.Board,
        cache,
        occur: dict,
        node: Node,
        action: int,
    ) -> Node:
        b_copy = board.copy()
        cache_copy = clone_cache(cache)
        occ_copy = dict(occur)
        new_path = node.path + (action,)

        for a in node.path:
            mv = _resolve_move(a, b_copy)
            if mv is None:
                raise RuntimeError(f"Replay action {a} illegal on {b_copy.fen()}")
            b_copy.push(mv)
            key = _board_key(b_copy)
            feats, tc_val, elo_std, color = encode_board(b_copy, occ_copy.get(key, 0))
            _, _, _, _, cache_copy = self.step(
                np.asarray(feats, dtype=np.float32).reshape(1, -1),
                [int(tc_val)],
                [float(elo_std)],
                [int(color)],
                cache_copy,
            )
            occ_copy[key] = occ_copy.get(key, 0) + 1

        mv = _resolve_move(action, b_copy)
        if mv is None:
            raise RuntimeError(f"Action {action} illegal on {b_copy.fen()}")
        b_copy.push(mv)
        if b_copy.is_game_over(claim_draw=True) or not list(b_copy.legal_moves):
            return Node(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                get_terminal_q(b_copy),
                depth=node.depth + 1,
                action=action,
                path=new_path,
                terminal=True,
            )

        key = _board_key(b_copy)
        feats, tc_val, elo_std, color = encode_board(b_copy, occ_copy.get(key, 0))
        lc, wc, _, _, _ = self.step(
            np.asarray(feats, dtype=np.float32).reshape(1, -1),
            [int(tc_val)],
            [float(elo_std)],
            [int(color)],
            cache_copy,
        )
        q_c = wdl_logits_to_q(wc[0])
        legal_c = _legal_actions_of(b_copy)
        lc_np = lc[0]
        lc_masked = np.full(1936, -3e4, dtype=np.float32)
        lc_masked[legal_c] = lc_np[legal_c]
        return Node(
            np.array(legal_c, dtype=np.int64),
            lc_masked[np.array(legal_c)].astype(np.float32),
            q_c,
            depth=node.depth + 1,
            action=action,
            path=new_path,
        )

    def select_move(
        self,
        board: chess.Board,
        cache,
        occur: dict,
        seed: int = 0,
    ) -> tuple[chess.Move | None, str | None]:
        legal_actions = _legal_actions_of(board)
        if not legal_actions:
            return None, "no_legal_moves"

        # Advance root board once to get logits & wdl
        key = _board_key(board)
        feats, tc_val, elo_std, color = encode_board(board, occur.get(key, 0))
        feats_np = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        logits_np, wdl_np, _, _, _ = self.step(
            feats_np, [int(tc_val)], [float(elo_std)], [int(color)], cache
        )

        q_root = wdl_logits_to_q(wdl_np[0])
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_legal = logits_np[0][legal_arr].astype(np.float32)

        root = Node(legal=legal_arr.copy(), logits=logits_legal.copy(), q=q_root)

        def expand_fn(node: Node, action: int) -> Node:
            return self.expand_child(board, cache, occur, node, action)

        result = order_halving(
            root,
            expand_fn,
            n_sims=self.n_sims,
            m0=self.m0,
            g=0.0,
            seed=seed,
            c_visit=self.c_visit,
            c_scale=self.c_scale,
        )

        chosen_action = result.get("action")
        if chosen_action is None:
            return None, "order_halving_returned_none"

        mv = _resolve_move(int(chosen_action), board)
        if mv is None:
            return None, f"action_{chosen_action}_resolves_to_none"
        return mv, None


def create_opponent_engine(
    opponent_type: str,
    ckpt_path: str,
    device: str = "cuda",
    mcts_sims: int = 0,
    precision: str = "fp16",
):
    """Factory creating TransformerEngine or UniChessEngine (ResNet)."""
    if opponent_type.lower() == "transformer":
        transformer_root = "/home/jeefy/UniChess/Transformer"
        if transformer_root not in sys.path:
            sys.path.insert(0, transformer_root)
        from engine.engine import TransformerEngine

        return TransformerEngine(
            ckpt_path,
            device=device,
            precision=precision,
            mcts_sims=mcts_sims,
            syzygy_path=None,
            book_path=None,
            temperature=0.0,
        )
    elif opponent_type.lower() in ("resnet", "unichess"):
        resnet_root = "/home/jeefy/UniChess/ResNet"
        if resnet_root not in sys.path:
            sys.path.insert(0, resnet_root)
        from engine.engine import UniChessEngine

        use_half = (precision in ("fp16", "bf16"))
        return UniChessEngine(
            ckpt_path,
            device=device,
            half=use_half,
            mcts_sims=mcts_sims,
            syzygy_path=None,
            book_path=None,
            temperature=0.0,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def play_one_game(
    ssm_engine: SSMArenaEngine,
    opponent_engine,
    ssm_color: chess.Color,
    opening_san: str | None = None,
    max_plies: int = 200,
    seed: int = 0,
) -> dict:
    """Play a single game between SSM and Opponent.

    ssm_color: chess.WHITE if SSM plays White, chess.BLACK if SSM plays Black.
    """
    board = chess.Board()
    occur: dict = {}
    ssm_cache = ssm_engine.initial_cache(1)
    actions: list[chess.Move] = []
    anomaly: str | None = None

    def _advance_ssm():
        nonlocal ssm_cache
        key = _board_key(board)
        feats, tc_val, elo_std, color = encode_board(board, occur.get(key, 0))
        feats_np = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        _, _, _, _, ssm_cache = ssm_engine.step(
            feats_np, [int(tc_val)], [float(elo_std)], [int(color)], ssm_cache
        )
        occur[key] = occur.get(key, 0) + 1

    # Play opening sequence if given
    if opening_san:
        for token in opening_san.split():
            _advance_ssm()
            mv = board.parse_san(token)
            board.push(mv)
            actions.append(mv)

    for ply in range(len(actions), max_plies):
        if board.is_game_over(claim_draw=True):
            break

        turn = board.turn
        if turn == ssm_color:
            ply_seed = seed + ply * 100003
            mv, err = ssm_engine.select_move(board, ssm_cache, occur, seed=ply_seed)
            if err:
                anomaly = f"SSM move error: {err}"
                break
            _advance_ssm()
            board.push(mv)
            actions.append(mv)
        else:
            # Opponent to move
            _advance_ssm()  # SSM tracks opponent's position in its R-cache
            mv = opponent_engine.play(board)
            if mv is None or mv not in board.legal_moves:
                anomaly = f"Opponent produced illegal move: {mv}"
                break
            board.push(mv)
            actions.append(mv)

    our_result, term_reason, is_truncated = classify_final_board(board)
    result_str = _result_str(board)

    # Map game result to SSM perspective:
    # our_result: 0 = White win, 1 = Draw, 2 = Black win
    if our_result == 1:
        ssm_result = 1  # Draw
    elif our_result == 0:
        ssm_result = 0 if ssm_color == chess.WHITE else 2  # SSM win or loss
    else:
        ssm_result = 2 if ssm_color == chess.WHITE else 0  # SSM loss or win

    game_pgn = chess.pgn.Game.from_board(board)
    return {
        "ssm_color": "white" if ssm_color == chess.WHITE else "black",
        "n_plies": len(actions),
        "termination_reason": term_reason,
        "is_truncated": is_truncated,
        "board_result": result_str,
        "ssm_result": ssm_result,  # 0: SSM Win, 1: Draw, 2: SSM Loss
        "anomaly": anomaly,
        "pgn": str(game_pgn) if game_pgn is not None else "",
    }


def _worker_process_fn(
    worker_id: int,
    assigned_pairs: list[tuple[int, int, str]],
    args: argparse.Namespace,
    result_queue: mp.Queue,
):
    try:
        ssm = SSMArenaEngine(
            args.ssm_ckpt,
            device=args.device,
            n_sims=args.n_sims,
            m0=args.m0,
            c_visit=args.c_visit,
            c_scale=args.c_scale,
        )
        opp = create_opponent_engine(
            args.opponent_type,
            args.opponent_ckpt,
            device=args.device,
            mcts_sims=args.opponent_mcts,
            precision=args.opponent_precision,
        )

        for pair_idx, oi, opening_san in assigned_pairs:
            # Game 1: SSM White, Opponent Black
            g1_seed = args.seed + worker_id * 10000 + pair_idx * 2
            gd1 = play_one_game(
                ssm, opp, chess.WHITE, opening_san=opening_san,
                max_plies=args.max_plies, seed=g1_seed
            )
            gd1["pair_idx"] = pair_idx
            gd1["game_idx"] = pair_idx * 2
            gd1["worker_id"] = worker_id

            # Game 2: Opponent White, SSM Black
            g2_seed = args.seed + worker_id * 10000 + pair_idx * 2 + 1
            gd2 = play_one_game(
                ssm, opp, chess.BLACK, opening_san=opening_san,
                max_plies=args.max_plies, seed=g2_seed
            )
            gd2["pair_idx"] = pair_idx
            gd2["game_idx"] = pair_idx * 2 + 1
            gd2["worker_id"] = worker_id

            result_queue.put(("game_pair", (pair_idx, [gd1, gd2])))

        result_queue.put(("worker_done", worker_id))
    except Exception as e:
        import traceback
        result_queue.put(("worker_error", (worker_id, str(e), traceback.format_exc())))


def main():
    parser = argparse.ArgumentParser(description="SSM vs Transformer / ResNet Arena Evaluation")
    parser.add_argument("--ssm-ckpt", required=True, help="Path to SSM checkpoint (.pt)")
    parser.add_argument(
        "--opponent-type",
        default="transformer",
        choices=["transformer", "resnet"],
        help="Opponent engine architecture ('transformer' or 'resnet')",
    )
    parser.add_argument(
        "--opponent-ckpt",
        default="/home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt",
        help="Path to opponent checkpoint (.pt)",
    )
    parser.add_argument("--opponent-mcts", type=int, default=0, help="Opponent MCTS simulations (0 = direct policy)")
    parser.add_argument("--opponent-precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--games", type=int, default=64, help="Total games to play (must be even, pairs = games / 2)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--n-sims", type=int, default=64, help="SSM Gumbel simulations per move")
    parser.add_argument("--m0", type=int, default=16, help="SSM Gumbel top-m0 candidates")
    parser.add_argument("--c-scale", type=float, default=0.1, help="SSM Gumbel c_scale")
    parser.add_argument("--c-visit", type=float, default=50.0, help="SSM Gumbel c_visit")
    parser.add_argument("--max-plies", type=int, default=200, help="Max plies per game")
    parser.add_argument("--seed", type=int, default=20260921, help="RNG seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/arena_ssm_vs_opponent", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    num_pairs = max(1, args.games // 2)
    total_planned_games = num_pairs * 2
    n_openings = len(OPENINGS_32)

    print("=" * 60)
    print("UniChess Arena: SSM vs Opponent Evaluation")
    print(f"  SSM Model:        {args.ssm_ckpt}")
    print(f"  SSM Search:       Gumbel n={args.n_sims}, m0={args.m0}, c_scale={args.c_scale}")
    print(f"  Opponent Type:    {args.opponent_type.upper()}")
    print(f"  Opponent Model:   {args.opponent_ckpt}")
    print(f"  Opponent Search:  {'MCTS ' + str(args.opponent_mcts) if args.opponent_mcts > 0 else 'Direct Policy'}")
    print(f"  Games:            {total_planned_games} ({num_pairs} opening pairs, swapped)")
    print(f"  Workers:          {args.workers}")
    print("=" * 60, flush=True)

    t0 = time.time()
    n_workers = min(args.workers, num_pairs)
    worker_pairs: list[list[tuple[int, int, str]]] = [[] for _ in range(n_workers)]
    for pair_idx in range(num_pairs):
        oi = pair_idx % n_openings
        opening_san = OPENINGS_32[oi]
        worker_pairs[pair_idx % n_workers].append((pair_idx, oi, opening_san))

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = []
    for wid in range(n_workers):
        p = ctx.Process(
            target=_worker_process_fn,
            args=(wid, worker_pairs[wid], args, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)
        print(f"[worker {wid}] Started (PID: {p.pid}, pairs: {len(worker_pairs[wid])})", flush=True)

    received_pairs: dict[int, list[dict]] = {}
    completed_workers = 0

    while completed_workers < n_workers:
        try:
            msg_type, payload = result_queue.get(timeout=1.0)
        except Exception:
            dead = [i for i, p in enumerate(workers) if not p.is_alive()]
            if dead and completed_workers + len(dead) >= n_workers:
                while not result_queue.empty():
                    msg_type, payload = result_queue.get_nowait()
                    if msg_type == "game_pair":
                        pair_idx, games = payload
                        received_pairs[pair_idx] = games
                    elif msg_type == "worker_done":
                        completed_workers += 1
                break
            continue

        if msg_type == "game_pair":
            pair_idx, games = payload
            received_pairs[pair_idx] = games
            total_done = sum(len(g) for g in received_pairs.values())
            print(
                f"  [{time.time() - t0:5.0f}s] Games: {total_done:3d}/{total_planned_games} "
                f"(Pairs: {len(received_pairs):2d}/{num_pairs})",
                flush=True,
            )
        elif msg_type == "worker_done":
            completed_workers += 1
        elif msg_type == "worker_error":
            wid, err_msg, tb = payload
            print(f"[worker {wid}] ERROR: {err_msg}\n{tb}", file=sys.stderr, flush=True)
            completed_workers += 1

    for p in workers:
        p.join(timeout=5.0)

    # Flatten and sort games by game_idx
    all_games = []
    for p_idx in sorted(received_pairs.keys()):
        all_games.extend(received_pairs[p_idx])
    all_games.sort(key=lambda g: g["game_idx"])

    # Aggregate statistics
    wins = sum(1 for g in all_games if g["ssm_result"] == 0)
    draws = sum(1 for g in all_games if g["ssm_result"] == 1)
    losses = sum(1 for g in all_games if g["ssm_result"] == 2)
    total_played = len(all_games)
    score = wins + 0.5 * draws
    score_pct = (score / max(1, total_played)) * 100.0
    elo_diff, elo_margin = compute_elo_diff(wins, draws, losses)

    # Color breakdown
    white_games = [g for g in all_games if g["ssm_color"] == "white"]
    black_games = [g for g in all_games if g["ssm_color"] == "black"]

    w_wins = sum(1 for g in white_games if g["ssm_result"] == 0)
    w_draws = sum(1 for g in white_games if g["ssm_result"] == 1)
    w_losses = sum(1 for g in white_games if g["ssm_result"] == 2)

    b_wins = sum(1 for g in black_games if g["ssm_result"] == 0)
    b_draws = sum(1 for g in black_games if g["ssm_result"] == 1)
    b_losses = sum(1 for g in black_games if g["ssm_result"] == 2)

    term_counts: dict[str, int] = {}
    for g in all_games:
        t = g["termination_reason"]
        term_counts[t] = term_counts.get(t, 0) + 1

    elapsed = time.time() - t0
    summary = {
        "ssm_ckpt": args.ssm_ckpt,
        "opponent_type": args.opponent_type,
        "opponent_ckpt": args.opponent_ckpt,
        "opponent_mcts": args.opponent_mcts,
        "total_games": total_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_pct": score_pct,
        "elo_diff": elo_diff,
        "elo_margin_95": elo_margin,
        "white": {"games": len(white_games), "wins": w_wins, "draws": w_draws, "losses": w_losses},
        "black": {"games": len(black_games), "wins": b_wins, "draws": b_draws, "losses": b_losses},
        "termination": term_counts,
        "elapsed_sec": elapsed,
        "games_per_sec": total_played / max(1e-3, elapsed),
    }

    # Save output files
    with open(os.path.join(args.out, "arena_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "games.jsonl"), "w", encoding="utf-8") as fh:
        for g in all_games:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("Arena Evaluation Finished:")
    print(f"  Total Games:      {total_played} in {elapsed:.1f}s ({summary['games_per_sec']:.2f} games/s)")
    print(f"  SSM Score:        {score:.1f}/{total_played} ({score_pct:.1f}%)")
    print(f"  SSM W / D / L:    {wins} / {draws} / {losses}")
    print(f"  Elo Difference:   {elo_diff:+.1f} +/- {elo_margin:.1f} (95% CI)")
    print(f"  As White:         +{w_wins} ={w_draws} -{w_losses}")
    print(f"  As Black:         +{b_wins} ={b_draws} -{b_losses}")
    print(f"  Terminations:     {term_counts}")
    print(f"  Summary saved to: {os.path.join(args.out, 'arena_summary.json')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
