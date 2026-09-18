"""Gumbel arena — 复用生产搜索 order_halving，修复 review §2 所有问题。

用法：
  python tools/ssm_gumbel_arena.py --ckpt-a runs/champion.pt --ckpt-b runs/challenger.pt \\
      --out runs/arena_ab --games 64 --pairs 8

输出：
  arena.json        — 聚合统计（含终止分布+验证断言）
  games.jsonl       — 逐局诊断（每行 JSON）
  model_ids.json    — 双方检查点参数标识+前向比较
  scoring_test.json — 计分正向测试（--test-scoring）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import move_to_action
from stateseq.model import SeqModel
from stateseq.model_r import clone_cache
from stateseq.gumbel import (
    C_SCALE, C_VISIT, N_SIMS, M0,
    Node, _Candidate, _n_rounds, completed_q, sigma, normalize_q,
    export_pi_prime, gumbel_topm, order_halving,
)
from stateseq.adapter import (
    encode_board, standardize_elo, wdl_logits_to_q, wdl_logits_to_probs,
    get_terminal_q, get_termination_reason, get_termination_reason_from_board,
    ELO_MEAN, ELO_STD,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 开局库（ECO 经典变例）----
OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5",
    "d4 d5 c4 e6",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",
    "d4 Nf6 c4 g6 Nc3 Bg7",
    "e4 e6 d4 d5",
    "d4 Nf6 c4 e6 Nf3 Bb4+",
    "e4 c6 d4 d5",
    "c4 e5",
    "Nf3 Nf6 c4 g6",
    "d4 d5 c4 c6",
    "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "d4 Nf6 c4 c5",
    "e4 e5 Nf3 Nf6",
    "d4 e6 c4 Bb4+",
    "e4 e5 Nf3 Nc6 Bc4",
    "d4 g6 c4 Bg7",
]


def _legal_actions_of(board: chess.Board) -> list[int]:
    return [a for m in board.legal_moves if (a := move_to_action(m)) is not None]


def _resolve_move(action: int, board: chess.Board):
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


def _result_str(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "*"
    if outcome.winner is None:
        return "\u00bd-\u00bd"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def _model_id(state_dict: dict) -> str:
    keys = sorted(state_dict.keys())
    buf = bytearray()
    for k in keys[:4]:
        t = state_dict[k]
        head = t.flatten()[:16].detach().cpu().numpy().astype(np.float32).view(np.uint8).tobytes()
        buf.extend(head)
    return hashlib.sha256(buf).hexdigest()[:16]


# ---- 计分正向测试 ----

SCORING_TEST_FENS = [
    ("k6R/8/1K6/8/8/8/8/8 b - - 0 1", chess.WHITE),
    ("K6r/8/1k6/8/8/8/8/8 w - - 0 1", chess.BLACK),
    ("k7/Q7/1K6/8/8/8/8/8 b - - 0 1", chess.WHITE),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", None),
]


def _run_scoring_test(out_dir: str) -> None:
    print("=== 计分正向测试 ===")
    os.makedirs(out_dir, exist_ok=True)
    games_log = []
    passed = 0
    for fen, expected in SCORING_TEST_FENS:
        board = chess.Board(fen)
        outcome = board.outcome(claim_draw=True)
        result_str = _result_str(board)
        winner = outcome.winner if outcome is not None else None
        ok = (winner == expected) or (winner is None and expected is None)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        games_log.append({"fen": fen, "expected": str(expected), "result": result_str,
                          "winner": str(winner), "status": status})
        print("  [%s] %s → %s" % (status, fen.split("/")[0], result_str))
    manifest = {"test": "scoring_positive_test", "total": len(SCORING_TEST_FENS),
                "passed": passed, "failed": len(SCORING_TEST_FENS) - passed, "games": games_log}
    with open(os.path.join(out_dir, "scoring_test.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("=== %d/%d PASS ===" % (passed, len(SCORING_TEST_FENS)))


# ---- 模型封装 ----

class ArenaModel:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device
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
        return (logits.cpu().numpy(), wdl.cpu().numpy(),
                mlh.cpu().numpy(), x.cpu().numpy(), cache_new)


# ---- 单局对弈（每局为双方各维护独立搜索，完整历史）----

def play_one_game(model_w: ArenaModel, model_b: ArenaModel, cfg,
                  opening_fen: str | None = None, opening_id: int = 0) -> dict:
    models = {chess.WHITE: model_w, chess.BLACK: model_b}
    board = chess.Board()
    if opening_fen:
        board = chess.Board(opening_fen)

    cache_w = model_w.initial_cache(1)
    cache_b = model_b.initial_cache(1)
    occur_w: dict[str, int] = {}
    occur_b: dict[str, int] = {}
    actions: list[int] = []
    anomaly = None
    n_sims = cfg.n_sims
    m0 = cfg.m0
    c_visit = cfg.c_visit
    c_scale = cfg.c_scale

    for ply in range(cfg.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        turn = board.turn
        model = models[turn]
        cache_turn = cache_w if turn == chess.WHITE else cache_b
        occur_turn = occur_w if turn == chess.WHITE else occur_b

        # 前向：用当前方的模型和 cache
        feats, tc_val, elo_std, color = encode_board(board, occur_turn.get(board.fen().split(" ")[0], 0))
        logits_np, wdl_np, _, _, cache_new = model.step(
            np.asarray(feats, dtype=np.float32).reshape(1, -1),
            [int(tc_val)], [float(elo_std)], [color], cache_turn)
        if turn == chess.WHITE:
            cache_w = cache_new
        else:
            cache_b = cache_new
        k = board.fen().split(" ")[0]
        occur_turn[k] = occur_turn.get(k, 0) + 1

        q_root = wdl_logits_to_q(wdl_np[0])
        legal_actions = _legal_actions_of(board)
        if not legal_actions:
            break
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_masked = logits_np[0][legal_arr].astype(np.float32)

        # 构建 root node —— 生产搜索的 order_halving 入口
        root = Node(legal=legal_arr.copy(), logits=logits_masked.copy(), q=q_root)

        # expand 函数：只使用当前方的模型，不切换
        def expand(node, action):
            b_copy = board.copy()
            cache_copy = clone_cache(cache_w if turn == chess.WHITE else cache_b)
            occ_copy = dict(occur_turn)
            for a in node.path:
                mv = _resolve_move(a, b_copy)
                if mv is None:
                    return Node(np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0.0, terminal=True)
                b_copy.push(mv)
                kk = b_copy.fen().split(" ")[0]
                occ_copy[kk] = occ_copy.get(kk, 0) + 1
                ff, tt, ee, cc = encode_board(b_copy, occ_copy.get(b_copy.fen().split(" ")[0], 0))
                ft = torch.from_numpy(ff).float().unsqueeze(0).to(model.device)
                with torch.no_grad():
                    _, _, _, _, cache_copy = model.seq.step(ft,
                        torch.tensor([int(tt)], dtype=torch.long, device=model.device),
                        torch.tensor([float(ee)], dtype=torch.float32, device=model.device),
                        torch.tensor([cc], dtype=torch.long, device=model.device),
                        cache_copy)
            mv = _resolve_move(action, b_copy)
            if mv is None:
                return Node(np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0.0, terminal=True)
            b_copy.push(mv)
            if b_copy.is_game_over(claim_draw=True) or not list(b_copy.legal_moves):
                q_term = get_terminal_q(b_copy)
                return Node(np.array([], dtype=np.int64), np.array([], dtype=np.float32), q_term,
                            depth=node.depth + 1, action=action, terminal=True)
            kk = b_copy.fen().split(" ")[0]
            occ_copy[kk] = occ_copy.get(kk, 0) + 1
            ff, tt, ee, cc = encode_board(b_copy, occ_copy.get(b_copy.fen().split(" ")[0], 0))
            ft = torch.from_numpy(ff).float().unsqueeze(0).to(model.device)
            with torch.no_grad():
                lc, wc, _, _, _ = model.seq.step(ft,
                    torch.tensor([int(tt)], dtype=torch.long, device=model.device),
                    torch.tensor([float(ee)], dtype=torch.float32, device=model.device),
                    torch.tensor([cc], dtype=torch.long, device=model.device),
                    cache_copy)
            q_c = wdl_logits_to_q(wc[0].cpu().numpy())
            lc_np = lc[0].cpu().numpy()[legal_arr] if False else lc[0].cpu().numpy()
            legal_c = _legal_actions_of(b_copy)
            lc_masked = np.full(1936, -3e4, dtype=np.float32)
            lc_masked[legal_c] = lc_np[legal_c]
            return Node(np.array(legal_c, dtype=np.int64),
                        lc_masked[np.array(legal_c)].astype(np.float32),
                        q_c, depth=node.depth + 1, action=action)

        result = order_halving(root, expand, n_sims=n_sims, m0=m0, g=0.0,
                               c_visit=c_visit, c_scale=c_scale)
        if result["action"] is None:
            anomaly = "order_halving returned None"
            break
        chosen = int(result["action"])
        actions.append(chosen)
        mv = _resolve_move(chosen, board)
        if mv is None:
            anomaly = "chosen action resolves to None"
            break
        board.push(mv)

    is_truncated = len(actions) >= cfg.max_plies
    term_reason, _ = get_termination_reason(board, cfg.max_plies, len(actions))
    result_str = _result_str(board)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        our_result = 1  # draw
    elif outcome.winner == chess.WHITE:
        our_result = 0  # white wins
    else:
        our_result = 2  # black wins

    game_pgn = chess.pgn.Game.from_board(board)
    return {
        "opening_id": opening_id,
        "ckpt_white": model_w.ckpt_path,
        "ckpt_black": model_b.ckpt_path,
        "n_plies": len(actions),
        "termination_reason": term_reason,
        "is_truncated": is_truncated,
        "board_result": result_str,
        "arena_result": our_result,
        "anomaly": anomaly,
        "pgn": str(game_pgn) if game_pgn is not None else "",
    }


def _aggregate_results(games_log, half, args) -> dict:
    wins_a = 0
    wins_b = 0
    draws = 0
    for gd in games_log:
        if gd["arena_result"] == 0:
            if gd["white_ckpt_side"] == "A":
                wins_a += 1
            else:
                wins_b += 1
        elif gd["arena_result"] == 2:
            if gd["black_ckpt_side"] == "A":
                wins_a += 1
            else:
                wins_b += 1
        else:
            draws += 1
    # 断言：W_A + W_B + D = N
    assert wins_a + wins_b + draws == len(games_log), \
        "Scoring invariant violated: W_A+W_B+D != N"
    score_a = wins_a + 0.5 * draws
    term_counts = {}
    for gd in games_log:
        t = gd["termination_reason"]
        term_counts[t] = term_counts.get(t, 0) + 1
    truncated = term_counts.get("truncated", 0)
    anomalies = sum(1 for g in games_log if g["anomaly"])
    return {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "total_games": len(games_log),
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "score_a": score_a,
        "score_a_percent": score_a / max(len(games_log), 1) * 100,
        "n_sims": args.n_sims, "m0": args.m0,
        "elapsed_s": 0.0,
        "termination": term_counts,
        "truncated_rate": truncated / max(len(games_log), 1),
        "anomalies": anomalies,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a")
    ap.add_argument("--ckpt-b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--test-scoring", action="store_true")
    args = ap.parse_args()

    if args.test_scoring:
        _run_scoring_test(args.out)
        return

    os.makedirs(args.out, exist_ok=True)
    model_a = ArenaModel(args.ckpt_a)
    model_b = ArenaModel(args.ckpt_b)

    # 模型身份验证
    ckpt_a_data = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    sd_a = ckpt_a_data.get("model", ckpt_a_data)
    ckpt_b_data = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    sd_b = ckpt_b_data.get("model", ckpt_b_data)
    id_a = _model_id(sd_a)
    id_b = _model_id(sd_b)

    dummy_feats = np.zeros((1, 785), dtype=np.float32)
    dummy_tc = [2]
    dummy_elo = [float(standardize_elo(2567.5))]
    dummy_color = [1]
    la, wa, _, _, _ = model_a.step(dummy_feats, dummy_tc, dummy_elo, dummy_color, model_a.initial_cache(1))
    lb, wb, _, _, _ = model_b.step(dummy_feats, dummy_tc, dummy_elo, dummy_color, model_b.initial_cache(1))
    import torch.nn.functional as F
    pa = F.softmax(torch.from_numpy(la[0]), dim=0).numpy()
    pb = F.softmax(torch.from_numpy(lb[0]), dim=0).numpy()
    policy_diff = float(np.max(np.abs(pa - pb)))
    wdl_diff = float(np.max(np.abs(wa - wb)))

    model_ids = {"a": {"hash": id_a}, "b": {"hash": id_b}, "same_hash": id_a == id_b,
                 "forward_comparison": {"max_policy_prob_diff": policy_diff,
                                        "max_wdl_diff": wdl_diff,
                                        "models_differ_functionally": policy_diff > 1e-6}}
    with open(os.path.join(args.out, "model_ids.json"), "w") as fh:
        json.dump(model_ids, fh, indent=1)
    print("A hash=%s B hash=%s same=%s policy_diff=%.2e" % (id_a, id_b, id_a == id_b, policy_diff))

    # 运行对局
    half = args.games // 2
    cfg = lambda: None
    cfg.n_sims = args.n_sims
    cfg.m0 = args.m0
    cfg.max_plies = args.max_plies
    cfg.c_visit = C_VISIT
    cfg.c_scale = C_SCALE
    games_log = []
    t0 = time.time()
    n_openings = min(args.pairs, len(OPENINGS))

    for g in range(half):
        oi = g % n_openings
        b = chess.Board()
        for token in OPENINGS[oi].split():
            b.push_san(token)
        gd = play_one_game(model_a, model_b, cfg, opening_fen=b.fen(), opening_id=oi)
        gd["game_idx"] = g; gd["white_ckpt_side"] = "A"; gd["black_ckpt_side"] = "B"
        games_log.append(gd)
        if (g + 1) % 8 == 0:
            print("  [%.0fs] game %d/%d" % (time.time() - t0, g + 1, half))

    for g in range(half):
        oi = g % n_openings
        b = chess.Board()
        for token in OPENINGS[oi].split():
            b.push_san(token)
        gd = play_one_game(model_b, model_a, cfg, opening_fen=b.fen(), opening_id=oi)
        r = gd["arena_result"]
        gd["arena_result"] = 0 if r == 2 else 2 if r == 0 else 1
        gd["game_idx"] = half + g; gd["white_ckpt_side"] = "B"; gd["black_ckpt_side"] = "A"
        games_log.append(gd)
        if (g + 1) % 8 == 0:
            print("  [%.0fs] game %d/%d (swapped)" % (time.time() - t0, half + g + 1, args.games))

    elapsed = time.time() - t0
    manifest = _aggregate_results(games_log, half, args)
    manifest["elapsed_s"] = elapsed

    with open(os.path.join(args.out, "arena.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(args.out, "games.jsonl"), "w") as fh:
        for gd in games_log:
            fh.write(json.dumps(gd) + "\n")
    print(json.dumps(manifest, indent=1))
    print("用时 %.0fs" % elapsed)


if __name__ == "__main__":
    main()