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
import math
import os
import sys
import time

import chess
import chess.pgn
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.actions import move_to_action
from stateseq.data.sequences import _board_key
from stateseq.model import SeqModel
from stateseq.model_r import clone_cache
from stateseq.gumbel import C_SCALE, C_VISIT, Node, order_halving
from stateseq.adapter import (
    encode_board, standardize_elo, wdl_logits_to_q,
    get_terminal_q, classify_final_board,
)

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


# ---- 单局对弈（每方独立模型 + 完整历史 cache/occurrence）----

def _expand_child(model: ArenaModel, board: chess.Board, cache, occur: dict,
                  node: Node, action: int) -> Node:
    """从本方根快照重放 node.path，再展开 action（与生成器 `_expand_gen` 同口径）。

    - board/cache/occur 均为**根局面**的快照（该方模型对完整历史推进后的状态）；
    - occurrence 统一 encode-before-increment（与根节点及训练重放一致）；
    - 路径重放动作必须合法（规则引擎权威），否则抛 RuntimeError（不得伪造终局）。
    """
    b_copy = board.copy()
    cache_copy = clone_cache(cache)
    occ_copy = dict(occur)
    new_path = node.path + (action,)
    for a in node.path:
        mv = _resolve_move(a, b_copy)
        if mv is None:
            raise RuntimeError(f"路径重放动作 {a} 在 {b_copy.fen()} 上不合法")
        b_copy.push(mv)
        key = _board_key(b_copy)
        feats, tc_val, elo_std, color = encode_board(b_copy, occ_copy.get(key, 0))
        _, _, _, _, cache_copy = model.step(
            np.asarray(feats, dtype=np.float32).reshape(1, -1),
            [int(tc_val)], [float(elo_std)], [int(color)], cache_copy)
        occ_copy[key] = occ_copy.get(key, 0) + 1
    mv = _resolve_move(action, b_copy)
    if mv is None:
        raise RuntimeError(f"动作 {action} 在 {b_copy.fen()} 上不合法")
    b_copy.push(mv)
    if b_copy.is_game_over(claim_draw=True) or not list(b_copy.legal_moves):
        return Node(np.array([], dtype=np.int64), np.array([], dtype=np.float32),
                    get_terminal_q(b_copy), depth=node.depth + 1, action=action,
                    path=new_path, terminal=True)
    key = _board_key(b_copy)
    feats, tc_val, elo_std, color = encode_board(b_copy, occ_copy.get(key, 0))
    lc, wc, _, _, _ = model.step(
        np.asarray(feats, dtype=np.float32).reshape(1, -1),
        [int(tc_val)], [float(elo_std)], [int(color)], cache_copy)
    q_c = wdl_logits_to_q(wc[0])
    legal_c = _legal_actions_of(b_copy)
    lc_np = lc[0]
    lc_masked = np.full(1936, -3e4, dtype=np.float32)
    lc_masked[legal_c] = lc_np[legal_c]
    return Node(np.array(legal_c, dtype=np.int64),
                lc_masked[np.array(legal_c)].astype(np.float32),
                q_c, depth=node.depth + 1, action=action, path=new_path)


def play_one_game(model_w: ArenaModel, model_b: ArenaModel, cfg,
                  opening_san: str | None = None, opening_id: int = 0) -> dict:
    """单局对弈：双方模型各自对**完整历史**推进 cache/occurrence，搜索复用生产 order_halving。

    - 每 ply 双方模型都前进一步——每方的根快照等于"该检查点单独下完这盘棋"的 R 状态；
    - occurrence 全局面共享（口径 = `stateseq/data/sequences.py::_board_key`）；
    - 开局着法同样经过模型步进（不得凭空跳开局，否则历史缺失）；
    - 终局原因：棋盘优先（`get_termination_reason_from_board`），未终局记 truncated。
    """
    models = {chess.WHITE: model_w, chess.BLACK: model_b}
    caches = {chess.WHITE: model_w.initial_cache(1), chess.BLACK: model_b.initial_cache(1)}
    board = chess.Board()
    occur: dict = {}
    actions: list[int] = []
    anomaly = None
    n_sims = cfg.n_sims
    m0 = cfg.m0
    # σ 展幅常数**绑定在模型侧**：换色时随模型一起交换，支持「同权重、不同尺度」对照。
    # ArenaModel 未显式设置时回落到 cfg（同一份共享配置），行为与旧版一致。
    scales = {
        chess.WHITE: (getattr(model_w, "c_visit", cfg.c_visit), getattr(model_w, "c_scale", cfg.c_scale)),
        chess.BLACK: (getattr(model_b, "c_visit", cfg.c_visit), getattr(model_b, "c_scale", cfg.c_scale)),
    }

    def _advance():
        """当前局面：双方模型各前进一步，返回行棋方 (logits, wdl)。"""
        key = _board_key(board)
        feats, tc_val, elo_std, color = encode_board(board, occur.get(key, 0))
        feats_np = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        mover_logits = None
        mover_wdl = None
        for side in (chess.WHITE, chess.BLACK):
            lg, wd, _, _, new_cache = models[side].step(
                feats_np, [int(tc_val)], [float(elo_std)], [int(color)], caches[side])
            caches[side] = new_cache
            if side == board.turn:
                mover_logits, mover_wdl = lg, wd
        occur[key] = occur.get(key, 0) + 1
        return mover_logits, mover_wdl

    if opening_san:
        # 先编码当前局面再落子（encode-before-move），与生成器 / 训练重放同口径：
        # 序列是 B₀,B₁,…，每个局面恰好进 R 一次。此前写成 push→advance，导致初始局面
        # B₀ 从未入 R、而最后一个开局局面被重复步进两次（occurrence 也多记一次）。
        for token in opening_san.split():
            _advance()
            board.push_san(token)

    for ply in range(cfg.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        turn = board.turn

        logits_np, wdl_np = _advance()
        q_root = wdl_logits_to_q(wdl_np[0])
        legal_actions = _legal_actions_of(board)
        if not legal_actions:
            break
        legal_arr = np.array(legal_actions, dtype=np.int64)
        logits_legal = logits_np[0][legal_arr].astype(np.float32)

        root = Node(legal=legal_arr.copy(), logits=logits_legal.copy(), q=q_root)

        side = turn

        def expand(node, action):
            return _expand_child(models[side], board, caches[side], occur, node, action)

        c_visit, c_scale = scales[side]
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

    # 终局裁决与生成器共用唯一入口（claim_draw=True 口径；未终局 = 走满 max_plies）
    our_result, term_reason, is_truncated = classify_final_board(board)
    result_str = _result_str(board)

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


def _aggregate_results(games_log, half, args, sprt_info=None) -> dict:
    wins_a = 0
    wins_b = 0
    draws = 0
    for gd in games_log:
        if gd["arena_result"] == 0:
            wins_a += 1
        elif gd["arena_result"] == 2:
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
    res = {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "c_visit": args.c_visit, "c_scale_a": args.c_scale_a, "c_scale_b": args.c_scale_b,
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
    if sprt_info:
        res["sprt"] = sprt_info
    return res


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
    ap.add_argument("--c_visit", type=float, default=C_VISIT, help="双方共用的 c_visit")
    ap.add_argument("--c_scale_a", type=float, default=C_SCALE, help="A 侧 c_scale")
    ap.add_argument("--c_scale_b", type=float, default=C_SCALE, help="B 侧 c_scale")
    ap.add_argument("--sprt", action="store_true", help="启用 Wald SPRT 早期停止（针对落后候选）")
    ap.add_argument("--sprt-min-games", type=int, default=64, help="SPRT 判决最少局数（成对边界）")
    ap.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT Type I error (False Positive) bound")
    ap.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT Type II error (False Negative) bound")
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

    # 运行对局：成对开局与颜色互换
    # pair i 包含 2 局：第 1 局 A 白 B 黑，第 2 局 B 白 A 黑（复用相同开局）
    num_pairs = args.games // 2
    cfg = lambda: None
    cfg.n_sims = args.n_sims
    cfg.m0 = args.m0
    cfg.max_plies = args.max_plies
    cfg.c_visit = args.c_visit
    cfg.c_scale = C_SCALE
    model_a.c_visit = model_b.c_visit = args.c_visit
    model_a.c_scale = args.c_scale_a
    model_b.c_scale = args.c_scale_b
    games_log = []
    t0 = time.time()
    n_openings = min(args.pairs, len(OPENINGS))

    # SPRT 参数与阈值
    # H0: p <= 0.50 vs H1: p >= 0.55
    p0, p1 = 0.50, 0.55
    bound_b = math.log(args.sprt_beta / (1.0 - args.sprt_alpha))  # ~ -2.944 (落后时提前拒绝 H1)
    log_p1_p0 = math.log(p1 / p0)
    log_1p1_1p0 = math.log((1.0 - p1) / (1.0 - p0))

    sprt_stopped = False
    sprt_info = None

    for pair_idx in range(num_pairs):
        oi = pair_idx % n_openings
        opening_san = OPENINGS[oi]

        # 局 1: A 白 B 黑
        g1_idx = len(games_log)
        gd1 = play_one_game(model_a, model_b, cfg, opening_san=opening_san, opening_id=oi)
        gd1["game_idx"] = g1_idx
        gd1["pair_idx"] = pair_idx
        gd1["white_ckpt_side"] = "A"
        gd1["black_ckpt_side"] = "B"
        games_log.append(gd1)

        # 局 2: B 白 A 黑
        g2_idx = len(games_log)
        gd2 = play_one_game(model_b, model_a, cfg, opening_san=opening_san, opening_id=oi)
        r2 = gd2["arena_result"]
        # 对齐到 A 视角：原结果 0(白胜/B胜) -> 2(A负), 2(黑胜/A胜) -> 0(A胜), 1 -> 1
        gd2["arena_result"] = 0 if r2 == 2 else 2 if r2 == 0 else 1
        gd2["game_idx"] = g2_idx
        gd2["pair_idx"] = pair_idx
        gd2["white_ckpt_side"] = "B"
        gd2["black_ckpt_side"] = "A"
        games_log.append(gd2)

        if len(games_log) % 8 == 0 or len(games_log) == args.games:
            print("  [%.0fs] games %d/%d (pair %d/%d)" % (
                time.time() - t0, len(games_log), args.games, pair_idx + 1, num_pairs))

        # 成对边界处检查 SPRT 早停（候选为 A）
        current_n = len(games_log)
        if args.sprt and current_n >= args.sprt_min_games and current_n < args.games:
            s_a = sum(1.0 if g["arena_result"] == 0 else 0.5 if g["arena_result"] == 1 else 0.0
                      for g in games_log)
            llr = s_a * log_p1_p0 + (current_n - s_a) * log_1p1_1p0
            if llr <= bound_b:
                saved_games = args.games - current_n
                saved_pct = saved_games / args.games * 100.0
                sprt_stopped = True
                sprt_info = {
                    "early_stopped": True,
                    "stop_reason": "sprt_reject_h1",
                    "llr": float(llr),
                    "llr_bound": float(bound_b),
                    "alpha": float(args.sprt_alpha),
                    "beta": float(args.sprt_beta),
                    "p0": float(p0),
                    "p1": float(p1),
                    "games_played": current_n,
                    "games_planned": args.games,
                    "compute_saved_games": saved_games,
                    "compute_saved_percent": saved_pct,
                }
                print(f"\n[SPRT Early Stop Triggered] LLR={llr:.3f} <= bound={bound_b:.3f} at game {current_n}/{args.games}")
                print(f"Candidate rejected early. Compute saved: {saved_games} games ({saved_pct:.1f}%)\n")
                break

    elapsed = time.time() - t0
    manifest = _aggregate_results(games_log, len(games_log) // 2, args, sprt_info=sprt_info)
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