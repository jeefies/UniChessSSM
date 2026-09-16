"""SSM UCI 适配器映射自测（跑 arena 前必须全过）。

  a. 随机合法着往返 ≥10000 个随机局面（含升变/吃过路兵/王车易位/黑方）：
     move_to_action(mv) -> FROM_ACTION 反查 -> 4096 索引 -> 回到唯一动作，一致性 100%。
  b. 同局面双引擎冒烟：同一批 FEN 分别过旧 net（small-champion）与 SSMAdapter，
     检查 adapter policy 在合法着上归一、无 NaN、wdl 三者和为 1。
  c. orient 核对：黑方走子局面，确认 mirror 方向与旧 core/encoding.py 一致
     （对比旧 net 对 board 与 mirrored board 的策略分布置换关系）。

远端用法：
    python tools/ssm_uci_test.py \
        --ckpt runs/stage_a_20260915/best.pt \
        --old-ckpt /home/jeefy/UniChess/runs/autoloop/models/small-champion.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("UNICHESS_ROOT", "/home/jeefy/UniChess"))

import chess  # noqa: E402

from stateseq.actions import FROM_ACTION, move_to_action  # noqa: E402

from core.encoding import orient_move  # noqa: E402  旧项目只读引用
from core.moves import PROMO_PIECES, move_to_index  # noqa: E402


def random_position(rng: np.random.Generator, max_plies: int = 120) -> chess.Board:
    b = chess.Board()
    for _ in range(rng.integers(0, max_plies)):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over(claim_draw=False):
            break
        b.push(moves[rng.integers(0, len(moves))])
    return b


def test_a_roundtrip(n_positions: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_moves = 0
    bad = 0
    special = {"promo": 0, "ep": 0, "castle": 0, "black": 0}
    for _ in range(n_positions):
        b = random_position(rng)
        if b.turn == chess.BLACK:
            special["black"] += 1
        for mv in list(b.legal_moves):
            n_moves += 1
            a = move_to_action(mv)
            frm, to, promo = FROM_ACTION[a]
            if frm != mv.from_square or to != mv.to_square:
                bad += 1
                continue
            expect_promo = None if mv.promotion in (None, chess.QUEEN) else mv.promotion
            if promo != expect_promo:
                bad += 1
                continue
            # 4096 索引（orient 坐标系）-> 该 (from,to) 下的唯一动作反查
            om = orient_move(mv, b.turn)
            idx = om.from_square * 64 + om.to_square
            frm2, to2 = divmod(idx, 64)
            if (frm2, to2) != (frm, to):
                bad += 1
                continue
            # 同 (from,to) 的所有合法着必须全部映射到同一索引（保证 mcts 反查一致）
            sibs = [m for m in b.legal_moves
                    if orient_move(m, b.turn).from_square == om.from_square
                    and orient_move(m, b.turn).to_square == om.to_square]
            if any(move_to_action(m) != a for m in sibs):
                bad += 1
            # 动作 -> 走法（python-chess Move 相等只看 from/to/promotion）
            if chess.Move(frm, to, promotion=promo) != mv:
                bad += 1
            if mv.promotion:
                special["promo"] += 1
            if b.is_en_passant(mv):
                special["ep"] += 1
            if b.is_castling(mv):
                special["castle"] += 1
    print(f"[a] 往返一致 {n_moves - bad}/{n_moves}（{n_positions} 局面，特殊着法 {special}）")
    assert bad == 0, f"[a] 失败 {bad} 处"


def _collect_legal_moves(board: chess.Board) -> list[chess.Move]:
    return list(board.legal_moves)


def test_bc(adapter, old_evaluate_batch, fens: list[str]) -> None:
    import torch

    boards = [chess.Board(f) for f in fens]
    pol_a, promo_a, wdl_a = adapter.evaluate_batch(boards)
    pol_o, promo_o, wdl_o = old_evaluate_batch(boards)

    n_nan = int(np.isnan(pol_a).sum() + np.isnan(wdl_a).sum())
    wdl_sum_err = float(np.abs(wdl_a.sum(axis=1) - 1.0).max())

    # adapter policy 在「orient 后合法着索引集」上归一
    norm_err = 0.0
    legal_mass = 0.0
    for i, b in enumerate(boards):
        idxs = {move_to_index(orient_move(mv, b.turn)) for mv in b.legal_moves}
        mass = float(pol_a[i][list(idxs)].sum())
        legal_mass = max(legal_mass, mass)
        norm_err = max(norm_err, abs(mass - 1.0))
    print(f"[b] NaN={n_nan}  wdl_sum_err={wdl_sum_err:.2e}  "
          f"合法着质量={legal_mass:.6f}  norm_err={norm_err:.2e}")
    assert n_nan == 0 and wdl_sum_err < 1e-5 and norm_err < 1e-4

    # orient 核对：黑方局面，旧 net 在 board 上的策略 ≈ 旧 net 在 mirror(board) 上
    # 的策略做 square_mirror 置换；即 p_b[idx(orient mv)] == p_w[idx(mv_w)]，其中
    # mv_w 是 mirror 棋盘上对应的白方走法。这验证 orient 方向约定（c 块），
    # 同时打印旧 net 与 SSM adapter 的策略重叠度作参考（不要求一致）。
    overlaps, perm_err = [], []
    for i, b in enumerate(boards):
        if b.turn != chess.BLACK:
            continue
        mb = b.mirror()
        # 旧 net 在黑方原始棋盘与镜像白方棋盘上的策略
        po_b = pol_o[i]
        po_w = old_evaluate_batch([mb])[0][0]
        perm = np.zeros_like(po_w)
        for idx in np.flatnonzero(po_w > 0):
            f0, t0 = divmod(idx, 64)
            perm[chess.square_mirror(f0) * 64 + chess.square_mirror(t0)] = po_w[idx]
        if po_b.sum() > 0:
            perm_err.append(float(np.abs(po_b - perm / max(perm.sum(), 1e-9)).max()))
        # SSM vs 旧 net 重叠（仅参考，不同模型不必接近）
        if pol_a[i].sum() > 0:
            a_top = set(np.argsort(-pol_a[i])[:5].tolist())
            o_top = set(np.argsort(-po_b)[:5].tolist())
            overlaps.append(len(a_top & o_top) / 5)
    print(f"[c] orient 置换一致性: 最大偏差 {max(perm_err):.4f}（越小越一致，n={len(perm_err)}）")
    print(f"    SSM vs 旧 net top-5 策略重叠（仅参考）: {np.mean(overlaps):.2f}")
    assert perm_err and max(perm_err) < 0.35, "orient 方向与旧 encoding 不一致"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(HERE) / "runs" / "stage_a_20260915" / "best.pt"))
    ap.add_argument("--old-ckpt", default="/home/jeefy/UniChess/runs/autoloop/models/small-champion.pt")
    ap.add_argument("--n-positions", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    t0 = time.time()
    test_a_roundtrip(args.n_positions, args.seed)
    print(f"    （耗时 {time.time() - t0:.1f}s）")

    import torch  # noqa: E402

    sys.path.insert(0, os.path.join(HERE, "tools"))
    from ssm_uci import SSMAdapter  # noqa: E402

    # 旧 champion 评估（只读引用旧项目 engine）
    sys.path.insert(0, os.environ.get("UNICHESS_ROOT", "/home/jeefy/UniChess"))
    from engine.engine import UniChessEngine  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_eng = UniChessEngine(args.old_ckpt, device=device, syzygy_path=None)

    adapter = SSMAdapter(args.ckpt, device=device)

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        "8/2P5/8/8/8/7k/8/K7 w - - 0 1",                      # 升变
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",              # 易位权（黑走）
        "8/8/8/k2Pp3/8/8/8/K7 w - e6 0 2",                    # 吃过路兵
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "2r1kr2/8/8/8/8/8/3PPP2/R3K2R w KQ - 0 1",
    ]
    t0 = time.time()
    test_bc(adapter, old_eng.evaluate_batch, fens)
    dt = time.time() - t0
    print(f"    双引擎冒烟耗时 {dt:.1f}s（含首次 CUDA 初始化）")

    # 每步耗时粗测：adapter 在带历史棋盘上 400 sims MCTS
    from search.mcts import MCTS, MCTSConfig  # noqa: E402

    mcts = MCTS(adapter.evaluate_batch, MCTSConfig(simulations=400, batch_size=128))
    b = chess.Board()
    for u in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
              "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "e8g8", "c2c3", "d7d6"]:
        b.push_uci(u)
    t0 = time.time()
    mv, root = mcts.best_move(b.copy(stack=True))
    dt = time.time() - t0
    print(f"[timing] 中盘 400 sims: {dt:.2f}s/步 -> bestmove {mv.uci()} "
          f"root_N={int(root.N.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
