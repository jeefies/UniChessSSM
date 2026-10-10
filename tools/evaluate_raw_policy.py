"""纯网络策略（Raw Policy / 0-sim）对弈评估工具。

用于纯粹检验两个模型的神经网络直觉大局观与模式识别质量（不依赖任何 MCTS 搜索外挂与战术惩罚）。
成对开局交换黑白，纯依靠网络 Policy logits 的 argmax 进行确定性走子。
严格遵循 SSM 隐藏状态递推法则：每走一步双边状态同步步进，开局至残局完全对齐。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np

IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from Kit.rules.referee import StandardReferee, TRUNCATED
from Kit.rules.openings import OpeningBook
from Kit import stats
from SSM.kit import make_evaluator, encode_board, _board_key, legal_moves_and_ids


class RawSsmAgent:
    """无搜索纯策略代理，严格跟踪单盘对局的隐藏状态。"""
    def __init__(self, evaluator, name: str = "agent"):
        self.evaluator = evaluator
        self.name = name
        self.cache = None
        self.occurrence: dict = {}
        self.last_logits = None

    def cleanup(self):
        if self.cache is not None:
            self.evaluator.release(self.cache)
            self.cache = None
        self.occurrence.clear()
        self.last_logits = None

    def reset(self):
        self.cleanup()
        init_state = self.evaluator.root_state()
        if init_state is not None:
            self.evaluator.hold(init_state)
        self.cache = init_state

    def step(self, board: chess.Board):
        key = _board_key(board)
        feats, tc, elo, color = encode_board(board, self.occurrence.get(key, 0))
        payload = self.evaluator.child(
            self.cache,
            np.asarray(feats, dtype=np.float32).reshape(-1),
            tc, elo, color
        )
        (out,) = self.evaluator.evaluate([payload])
        logits, wd, state = out
        old = self.cache
        self.cache = state
        self.evaluator.hold(state)
        if old is not None:
            self.evaluator.release(old)
        self.occurrence[key] = self.occurrence.get(key, 0) + 1
        self.last_logits = logits
        return logits

    def pick_move(self, board: chess.Board) -> chess.Move | None:
        moves, ids = legal_moves_and_ids(board)
        if not moves:
            return None
        logits = self.last_logits
        move_logits = logits[ids]
        best_idx = int(np.argmax(move_logits))
        return moves[best_idx]


def evaluate_raw_game(agent_a: RawSsmAgent, agent_b: RawSsmAgent,
                      opening: list[str], a_is_white: bool, max_plies: int = 200) -> dict:
    agent_a.reset()
    agent_b.reset()

    board = chess.Board()
    referee = StandardReferee(max_plies=max_plies)

    # 1. 步进初始盘面
    agent_a.step(board)
    agent_b.step(board)

    # 2. 步进开局序列
    moves = []
    for uci in opening:
        mv = chess.Move.from_uci(uci)
        board.push(mv)
        moves.append(uci)
        agent_a.step(board)
        agent_b.step(board)

    # 3. 对弈主循环
    while True:
        v = referee.verdict(board)
        if v is not None:
            break

        is_a_turn = (board.turn == chess.WHITE) if a_is_white else (board.turn == chess.BLACK)
        acting_agent = agent_a if is_a_turn else agent_b

        mv = acting_agent.pick_move(board)
        if mv is None:
            break

        moves.append(mv.uci())
        board.push(mv)

        # 双方同步递进 SSM 隐藏状态
        agent_a.step(board)
        agent_b.step(board)

    v = referee.verdict(board)
    if v is None:
        v = TRUNCATED

    white_score = 1.0 if v.result == "1-0" else (0.5 if v.result == "1/2-1/2" else 0.0)
    a_score = white_score if a_is_white else (1.0 - white_score)

    agent_a.cleanup()
    agent_b.cleanup()

    return {
        "a_score": a_score,
        "result": v.result,
        "termination": v.termination,
        "plies": len(board.move_stack),
        "moves": moves
    }


def main():
    parser = argparse.ArgumentParser(description="Raw Policy (0-sim) Match Evaluation")
    parser.add_argument("--a-ckpt", required=True, help="模型 A 检查点")
    parser.add_argument("--b-ckpt", required=True, help="模型 B 检查点")
    parser.add_argument("--pairs", type=int, default=50, help="开局对数（总局数 = 2 * pairs）")
    parser.add_argument("--openings", default=os.path.join(IMPORT_ROOT, "SSM/data/openings_200.txt"))
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    print(f"[Raw Policy Match] 启动纯网络策略对决（0 树搜索，纯 Policy 直觉）：")
    print(f"  A (Challenger): {args.a_ckpt}")
    print(f"  B (Baseline):   {args.b_ckpt}")
    print(f"  总局数: {args.pairs * 2} 局（{args.pairs} 对成对开局交换黑白）")

    ev_a = make_evaluator(args.a_ckpt, device="cuda", engine="fast")
    ev_b = make_evaluator(args.b_ckpt, device="cuda", engine="fast")
    agent_a = RawSsmAgent(ev_a, name="A")
    agent_b = RawSsmAgent(ev_b, name="B")

    book = OpeningBook.from_file(args.openings) if os.path.exists(args.openings) else OpeningBook.bundled()
    lines = book.lines[:args.pairs]

    t0 = time.time()
    records = []
    w, l, d = 0, 0, 0

    for p_idx, line in enumerate(lines):
        # Game 1: A 执白
        r1 = evaluate_raw_game(agent_a, agent_b, line, a_is_white=True, max_plies=args.max_plies)
        records.append(r1)
        if r1["a_score"] == 1.0: w += 1
        elif r1["a_score"] == 0.0: l += 1
        else: d += 1

        # Game 2: A 执黑
        r2 = evaluate_raw_game(agent_a, agent_b, line, a_is_white=False, max_plies=args.max_plies)
        records.append(r2)
        if r2["a_score"] == 1.0: w += 1
        elif r2["a_score"] == 0.0: l += 1
        else: d += 1

        if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(lines):
            cur_n = len(records)
            print(f"  已完成 {cur_n}/{args.pairs*2} 局: A 胜 {w} / 负 {l} / 和 {d} (得分率: {(w+0.5*d)/cur_n*100:.1f}%)")

    elapsed = time.time() - t0
    total = len(records)
    score_a = (w + 0.5 * d) / total
    elo, lo, hi = stats.elo_with_error(w, d, l)
    terms = Counter(r["termination"] for r in records)

    print("\n" + "=" * 50)
    print(f"【纯网络策略对弈结果】(Raw Policy / 0-sim, 耗时 {elapsed:.1f}s)")
    print(f"  总战绩: {w} 胜 / {l} 负 / {d} 和 (得分率: {score_a*100:.2f}%)")
    print(f"  纯网络相对 Elo: {elo:+.2f} (95% CI: [{lo:+.1f}, {hi:+.1f}])")
    print(f"  优势概率 (LOS): {stats.los(w, d, l)*100:.2f}%")
    print(f"  终局构成:")
    for k, v in terms.most_common():
        print(f"    - {k}: {v} ({v/total*100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
