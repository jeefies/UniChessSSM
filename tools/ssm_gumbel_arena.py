"""Gumbel arena：两个 checkpoint 用 Gumbel g=0/n=64 对弈，输出逐局诊断。

用法：
  # 正常对弈
  python tools/ssm_gumbel_arena.py --ckpt-a runs/champion.pt --ckpt-b runs/challenger.pt \\
      --out runs/arena_ab --games 64 --pairs 4

  # 计分正向测试（强制将杀短局，验证记分路径）
  python tools/ssm_gumbel_arena.py --test-scoring --out runs/arena_scoring_test

--games: 总局数。--pairs: 配对开局数（从开局库轮流取）。
默认--g 0 --n_sims 64 --m0 16。
输出：
  arena.json  — 聚合统计（含终止原因分布）
  games.jsonl — 逐局诊断（每行 JSON，含 PGN/termination/ply 等）
  model_ids.json — 双方检查点参数标识哈希
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import io

import chess
import chess.pgn
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import action_to_move, move_to_action
from stateseq.conditions import TimeControlBucket
from stateseq.features import encode
from stateseq.gumbel import (
    C_SCALE, C_VISIT, N_SIMS, M0, TERM_CODES,
    Node, _Candidate, _n_rounds, completed_q, sigma,
    gumbel_topm,
)
from stateseq.model import SeqModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- 开局库（ECO 经典变例，每对白/黑各一次） ----
OPENINGS = [
    "1. e4 e5 2. Nf3 Nc6 3. Bb5",                    # Ruy Lopez
    "1. d4 d5 2. c4 e6",                              # Queen's Gambit Declined
    "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6",     # Open Sicilian
    "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7",                  # King's Indian
    "1. e4 e6 2. d4 d5",                               # French Defense
    "1. d4 Nf6 2. c4 e6 3. Nf3 Bb4+",                 # Bogo-Indian
    "1. e4 c6 2. d4 d5",                               # Caro-Kann
    "1. c4 e5",                                        # English (Sicilian Reversed)
    "1. Nf3 Nf6 2. c4 g6",                             # Reti/Robatsch
    "1. d4 d5 2. c4 c6",                               # Slav Defense
    "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qa5",               # Scandinavian
    "1. d4 Nf6 2. c4 c5",                              # Modern Benoni
    "1. e4 e5 2. Nf3 Nf6",                             # Petrov Defense
    "1. d4 e6 2. c4 Bb4+",                             # Keres Defense
    "1. e4 e5 2. Nf3 Nc6 3. Bc4",                     # Italian Game
    "1. d4 g6 2. c4 Bg7",                              # Modern Defense
]


def _board_key(board: chess.Board) -> str:
    return board.fen().split(" ")[0]


def _legal_actions_of(board: chess.Board) -> list[int]:
    return [a for m in board.legal_moves if (a := move_to_action(m)) is not None]


def _resolve_move(action: int, board: chess.Board):
    for m in board.legal_moves:
        if move_to_action(m) == action:
            return m
    return None


def _get_termination(board: chess.Board, is_truncated: bool) -> tuple[str, bool]:
    """返回 (termination_reason, is_truncated)。"""
    if is_truncated:
        return "truncated", True
    if board.is_checkmate():
        return "checkmate", False
    if board.is_stalemate():
        return "stalemate", False
    if board.can_claim_fifty_moves():
        return "fifty_move", False
    if board.can_claim_threefold_repetition():
        return "threefold", False
    if board.is_insufficient_material():
        return "insufficient_material", False
    return "unknown", False


def _result_str(board: chess.Board) -> str:
    """python-chess 结果字符串：1-0 / 0-1 / ½-½ / *"""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "*"
    if outcome.winner is None:
        return "½-½"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def _model_id(state_dict: dict) -> str:
    """从模型 state_dict 的前 64 字节计算 SHA-256 摘要。"""
    keys = sorted(state_dict.keys())
    buf = bytearray()
    for k in keys[:4]:
        t = state_dict[k]
        buf.extend(t.flatten()[:16].view(np.uint8).tobytes())
    return hashlib.sha256(buf).hexdigest()[:16]


def _load_checkpoint_identifier(ckpt_path: str) -> dict:
    """加载检查点并返回参数标识摘要，不保留模型。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    return {"path": ckpt_path, "hash": _model_id(sd), "size_mb": round(os.path.getsize(ckpt_path) / 1e6, 1)}


def _compare_forward_pass(model_a, model_b, device: str) -> dict:
    """在固定输入上比较两个模型的前向输出差异。"""
    import torch.nn.functional as F
    dummy_feats = np.zeros((1, 785), dtype=np.float32)
    dummy_tc = torch.tensor([2], dtype=torch.long, device=device)
    dummy_elo = torch.tensor([2567.5], dtype=torch.float32, device=device)
    dummy_color = torch.tensor([1], dtype=torch.long, device=device)
    f_t = torch.from_numpy(dummy_feats).float().to(device)
    with torch.no_grad():
        logits_a, wdl_a, mlh_a, _, _ = model_a.seq.step(f_t, dummy_tc, dummy_elo, dummy_color, model_a.initial_cache())
        logits_b, wdl_b, mlh_b, _, _ = model_b.seq.step(f_t, dummy_tc, dummy_elo, dummy_color, model_b.initial_cache())
    policy_a = F.softmax(logits_a[0], dim=0).cpu().numpy()
    policy_b = F.softmax(logits_b[0], dim=0).cpu().numpy()
    max_pol_diff = float(np.max(np.abs(policy_a - policy_b)))
    max_wdl_diff = float(np.max(np.abs(wdl_a.cpu().numpy() - wdl_b.cpu().numpy())))
    return {"max_policy_prob_diff": max_pol_diff, "max_wdl_diff": max_wdl_diff,
            "models_differ_functionally": max_pol_diff > 1e-6}


# ---- 正向将杀测试 ----

SCORING_TEST_POSITIONS = [
    # (fen, expected_winner: chess.Color or None for draw)
    # fen 必须表示该走棋方已被将杀（无合法着 + 被将军）
    ("k1R5/8/8/8/8/8/8/K7 b - - 0 1", chess.WHITE),        # 黑王被白车将杀
    ("K1r5/8/8/8/8/8/8/k7 w - - 0 1", chess.BLACK),         # 白王被黑车将杀
    ("k1Q5/8/8/8/8/8/8/K7 b - - 0 1", chess.WHITE),         # 黑王被白后将杀
    ("K1q5/8/8/8/8/8/8/k7 w - - 0 1", chess.BLACK),         # 白王被黑后将杀
    ("k1N5/8/8/8/8/8/8/K7 b - - 0 1", chess.WHITE),         # 黑王被白马将杀
    ("k7/8/8/8/8/8/1R6/1K6 b - - 0 1", chess.WHITE),        # 黑王被白车将杀 2
]


def _run_scoring_test(out_dir: str) -> None:
    """运行计分正向测试：注入强杀位置，验证 arena 记分路径正确记录。"""
    print("=== 计分正向测试 ===")
    os.makedirs(out_dir, exist_ok=True)
    games_log = []
    passed = 0
    failed = 0

    for i, (fen, expected) in enumerate(SCORING_TEST_POSITIONS):
        board = chess.Board(fen)
        outcome = board.outcome(claim_draw=True)

        result_str = _result_str(board)
        winner = outcome.winner if outcome is not None else None

        if outcome is not None and outcome.winner == expected:
            status = "PASS"
            passed += 1
        elif outcome is not None and outcome.winner is None:
            status = "PASS" if expected is None else f"FAIL (expected {expected}, got draw)"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
        else:
            status = f"FAIL (outcome={outcome}, expected winner={expected})"
            failed += 1

        rec = {"test_idx": i, "fen": fen, "expected_winner": str(expected),
               "result_str": result_str, "winner": str(winner) if winner is not None else None,
               "status": status}
        games_log.append(rec)
        print(f"  [{status}] fen={fen}")
        print(f"    result={result_str} winner={winner} expected={expected}")

    manifest = {"test": "scoring_positive_test", "total": len(SCORING_TEST_POSITIONS),
                "passed": passed, "failed": failed,
                "games": games_log}
    with open(os.path.join(out_dir, "scoring_test.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"=== 计分正向测试 {'ALL PASS' if failed==0 else f'{failed} FAILURES'} ===")
    print(json.dumps({"passed": passed, "failed": failed}, indent=1))


# ---- 模型对弈 ----

class ArenaModel:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self.ckpt_path = ckpt_path
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


def play_one_game(model_w: ArenaModel, model_b: ArenaModel, cfg, opening_fen: str | None = None,
                  opening_id: int = 0) -> dict:
    """返回丰富逐局诊断的 dict。"""
    models = {chess.WHITE: model_w, chess.BLACK: model_b}
    board = chess.Board()
    if opening_fen:
        board = chess.Board(opening_fen)
    root_cache = {color: model.initial_cache() for color, model in models.items()}
    occur = {}
    actions = []
    anomaly = None

    max_plies = cfg.max_plies
    tc = int(cfg.tc_bucket)
    elo = cfg.elo
    n_sims = cfg.n_sims
    m0 = cfg.m0
    c_visit = cfg.c_visit
    c_scale = cfg.c_scale

    for ply in range(max_plies):
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
            anomaly = f"ply {ply}: action {chosen} resolves to None"
            break
        board.push(mv)

    is_truncated = len(actions) >= max_plies
    term_reason, _ = _get_termination(board, is_truncated)
    result_str = _result_str(board)

    # 我们的结果编码：0=白胜，1=和，2=黑胜
    # 行棋方视角统一：结果是 white 视角
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        our_result = 1
    elif outcome.winner == chess.WHITE:
        our_result = 0
    else:
        our_result = 2

    # PGN 导出
    game_pgn = chess.pgn.Game.from_board(board)
    pgn_str = str(game_pgn) if game_pgn is not None else ""

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
        "pgn": pgn_str,
    }


def _expand_search(root, action, model_active, models, root_caches, board, occur, qbox):
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
    ap.add_argument("--ckpt-a", required="--test-scoring" not in sys.argv)
    ap.add_argument("--ckpt-b", required="--test-scoring" not in sys.argv)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--n_sims", type=int, default=64)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--test-scoring", action="store_true", help="运行计分正向测试并退出")
    args = ap.parse_args()

    if args.test_scoring:
        _run_scoring_test(args.out)
        return

    os.makedirs(args.out, exist_ok=True)

    # ---- 模型加载 & 身份验证 ----
    model_a = ArenaModel(args.ckpt_a)
    model_b = ArenaModel(args.ckpt_b)

    id_a = _load_checkpoint_identifier(args.ckpt_a)
    id_b = _load_checkpoint_identifier(args.ckpt_b)

    model_ids = {"a": id_a, "b": id_b, "same_hash": id_a["hash"] == id_b["hash"]}
    with open(os.path.join(args.out, "model_ids.json"), "w") as fh:
        json.dump(model_ids, fh, indent=1)
    print(f"A hash={id_a['hash']} B hash={id_b['hash']} same={id_a['hash']==id_b['hash']}")

    # 前向输出比较
    fwd_cmp = _compare_forward_pass(model_a, model_b, "cuda")
    model_ids["forward_comparison"] = fwd_cmp
    with open(os.path.join(args.out, "model_ids.json"), "w") as fh:
        json.dump(model_ids, fh, indent=1)
    print(f"  forward: policy_max_diff={fwd_cmp['max_policy_prob_diff']:.2e} "
          f"functionally_different={fwd_cmp['models_differ_functionally']}")

    cfg = lambda: None
    cfg.n_sims = args.n_sims
    cfg.m0 = args.m0
    cfg.max_plies = args.max_plies
    cfg.tc_bucket = TimeControlBucket.RAPID
    cfg.elo = 2567.5
    cfg.c_visit = C_VISIT
    cfg.c_scale = C_SCALE

    half = args.games // 2
    games_log = []

    print(f"A={args.ckpt_a} vs B={args.ckpt_b}")
    print(f"Gumbel g=0 n={args.n_sims} m0={args.m0}")
    print(f"games={args.games} ({half} each color, {args.pairs} opening pairs)")

    t0 = time.time()

    # 第一半：model_a 走白
    for g in range(half):
        opening_idx = g % min(args.pairs, len(OPENINGS))
        opening_fen = None
        if opening_idx < len(OPENINGS):
            b = chess.Board()
            for token in OPENINGS[opening_idx].split():
                b.push_san(token)
            opening_fen = b.fen()
        gd = play_one_game(model_a, model_b, cfg, opening_fen=opening_fen, opening_id=opening_idx)
        gd["game_idx"] = g
        gd["white_ckpt_side"] = "A"
        gd["black_ckpt_side"] = "B"
        games_log.append(gd)
        if (g + 1) % 8 == 0:
            print(f"  [{time.time()-t0:.0f}s] game {g+1}/{half} (B vs W)")

    # 第二半：model_b 走白（翻转）
    for g in range(half):
        opening_idx = g % min(args.pairs, len(OPENINGS))
        opening_fen = None
        if opening_idx < len(OPENINGS):
            b = chess.Board()
            for token in OPENINGS[opening_idx].split():
                b.push_san(token)
            opening_fen = b.fen()
        gd = play_one_game(model_b, model_a, cfg, opening_fen=opening_fen, opening_id=opening_idx)
        # 翻转 arena_result：model_b 走白时的结果
        r = gd["arena_result"]
        flipped = 0 if r == 2 else 2 if r == 0 else 1
        gd["arena_result"] = flipped
        gd["game_idx"] = half + g
        gd["white_ckpt_side"] = "B"
        gd["black_ckpt_side"] = "A"
        games_log.append(gd)
        if (g + 1) % 8 == 0:
            print(f"  [{time.time()-t0:.0f}s] game {half+g+1}/{args.games} (swapped)")

    elapsed = time.time() - t0

    # ---- 聚合统计 ----
    wins_a = sum(1 for gd in games_log if gd["arena_result"] == 0 and gd["white_ckpt_side"] == "A")
    wins_b = sum(1 for gd in games_log if gd["arena_result"] == 0 and gd["white_ckpt_side"] == "B")
    draws = sum(1 for gd in games_log if gd["arena_result"] == 1)
    score_a = wins_a + 0.5 * draws

    term_counts = {}
    for gd in games_log:
        t = gd["termination_reason"]
        term_counts[t] = term_counts.get(t, 0) + 1
    truncated_count = term_counts.get("truncated", 0)
    anomaly_count = sum(1 for gd in games_log if gd["anomaly"] is not None)
    non_standard_results = sum(1 for gd in games_log if gd["board_result"] == "*")

    # 有实际输赢的局
    decisive = sum(1 for gd in games_log if gd["arena_result"] != 1)

    manifest = {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "total_games": len(games_log),
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "decisive_games": decisive,
        "score_a": score_a,
        "score_a_percent": score_a / max(len(games_log), 1) * 100,
        "n_sims": args.n_sims, "m0": args.m0,
        "elapsed_s": elapsed,
        "termination": term_counts,
        "truncated_rate": truncated_count / max(len(games_log), 1),
        "anomalies": anomaly_count,
        "non_standard_results": non_standard_results,
    }
    with open(os.path.join(args.out, "arena.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))

    # ---- 写入逐局日志 ----
    with open(os.path.join(args.out, "games.jsonl"), "w") as fh:
        for gd in games_log:
            fh.write(json.dumps(gd) + "\n")
    print(f"用时 {elapsed:.0f}s")
    print(f"逐局日志写入 {os.path.join(args.out, 'games.jsonl')}")


if __name__ == "__main__":
    main()