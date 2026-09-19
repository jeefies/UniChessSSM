"""统一模型适配器（review 2026-09-18 §7：修复组件 A）。

负责：
1. Elo 标准化：所有入口用同一 (elo-mean)/std 变换
2. WDL logits → 概率 → Q：`softmax(wdl_logits)[0]-softmax(wdl_logits)[2]`
3. 终局真值：`board.outcome(claim_draw=True)` 直接产 q=+1/0/-1，不走网络

所有生成器、arena、评估脚本统一个个入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .features import encode
from .conditions import TimeControlBucket

# Stage A 人类数据拟合的 Elo 统计量（与 conditions.EloStandardizer 的通用默认值不同）。
# 自对弈固定条件 2567.5 统一用此组 mean/std。
ELO_MEAN = 1656.1
ELO_STD = 390.9


def standardize_elo(elo: float | np.ndarray) -> float | np.ndarray:
    """Elo → 标准化值：(elo - mean) / std。mean/std 来自 Stage A 人类数据。"""
    return (np.asarray(elo, dtype=np.float64) - ELO_MEAN) / ELO_STD


def wdl_logits_to_q(wdl_logits: np.ndarray) -> float:
    """WDL logits → 价值标量 q = P(W) − P(L) ∈ [−1, 1]。"""
    wdl = np.asarray(wdl_logits, dtype=np.float64)
    wdl -= wdl.max()
    probs = np.exp(wdl) / np.exp(wdl).sum()
    return float(probs[0] - probs[2])


def wdl_logits_to_probs(wdl_logits: np.ndarray) -> np.ndarray:
    """WDL logits → 概率向量 [P(W), P(D), P(L)]。"""
    wdl = np.asarray(wdl_logits, dtype=np.float64)
    wdl -= wdl.max()
    probs = np.exp(wdl) / np.exp(wdl).sum()
    return probs.astype(np.float32)


def get_terminal_q(board: chess.Board) -> float:
    """终局真值：当前行棋方视角的 +1（本方胜）/ 0（和）/ −1（本方负）。
    
    将杀后当前行棋方是被将死者 → 返回 -1；对手视角取负即 +1。
    逼和/五十步/重复/不足 → 和棋 0。
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.0
    if outcome.winner is None:
        return 0.0
    # outcome.winner 是胜方颜色
    # 当前行棋方视角：若 board.turn == outcome.winner → +1，否则 −1
    return 1.0 if board.turn == outcome.winner else -1.0


def get_termination_reason(board: chess.Board, max_plies: int, ply: int) -> tuple[str, bool]:
    """统一终止原因与截断标记。ply 从 0 开始计数。
    
    返回 (reason_str, is_truncated):
      "checkmate" / "stalemate" / "fifty_move" / "threefold" / "insufficient_material" / "truncated"
    """
    if ply >= max_plies:
        return "truncated", True
    if board.is_checkmate():
        return "checkmate", False
    if board.is_stalemate():
        return "stalemate", False
    if board.is_fifty_moves():
        return "fifty_move", False
    if board.is_repetition(3):
        return "threefold", False
    if board.is_insufficient_material():
        return "insufficient_material", False
    return "unknown", False


# python-chess Termination → v3 meta 的 termination_reason 口径（§2.5）。
# 五次重复/七十五步是强制终局，与"可申和"的三次重复/五十步同因，归入同一编码。
_TERMINATION_TO_REASON = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.FIFTY_MOVES: "fifty_move",
    chess.Termination.SEVENTYFIVE_MOVES: "fifty_move",
    chess.Termination.THREEFOLD_REPETITION: "threefold",
    chess.Termination.FIVEFOLD_REPETITION: "threefold",
}


# 从分片读取时重构终止原因（不依赖 max_plies，用实际 n_plies 判断）
def get_termination_reason_from_board(board: chess.Board) -> tuple[str, bool]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "unknown", False
    return _TERMINATION_TO_REASON.get(outcome.termination, "unknown"), False


def classify_final_board(board: chess.Board) -> tuple[int, str, bool]:
    """终局裁决唯一入口 → (result 白视角 0 胜/1 和/2 负, termination_reason, is_truncated)。

    口径必须与对局循环的退出条件 `is_game_over(claim_draw=True)` 一致：该语义把"下一着
    可申和"的三次重复/五十步也算终局，而 `board.is_repetition(3)` / `is_fifty_moves()`
    是**严格**判定，两者差一 ply。曾因此把 78% 的规则申和局错记为"300 ply 封顶截断"
    （2026-09-19 实测 gen2k：218 条 truncated 中 173 条实为申和），连带污染封顶率统计与
    mlh 有效位。这里统一以 `outcome(claim_draw=True)` 为准，**只有规则未终局**（= 走满
    max_plies）才是真截断。
    """
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 1, "truncated", True
    if outcome.winner is None:
        result = 1
    else:
        result = 0 if outcome.winner == chess.WHITE else 2
    reason = _TERMINATION_TO_REASON.get(outcome.termination)
    if reason is None:  # 标准国际象棋不应出现（variant 终止）
        raise ValueError(f"未知终止原因 {outcome.termination} @ {board.fen()}")
    return result, reason, False


def encode_board(board: chess.Board, occurrence: int = 0) -> np.ndarray:
    """编码 785 维特征 + 标准化条件 + tc/color。返回 (feats, tc, elo_std, color)。"""
    feats = encode(board, occurrence=occurrence)
    tc = int(TimeControlBucket.RAPID)
    elo_std = standardize_elo(2567.5)
    color = 1 if board.turn == chess.WHITE else 0
    return feats, tc, elo_std, color