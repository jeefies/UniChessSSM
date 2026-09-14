"""序列构建（设计文档 §4.2）：每局一条序列，每半回合 t 产出一条 StepRecord。

(B_t 特征[785], a_t 动作id, legal_mask_t[1936], result ∈ {W,D,L}（对行棋方归一）,
 moves_left_t = (T − t) ply（截断 T_max=200）, cond = {tc_bucket, elo_mean, color})

重复计数特征：沿局统计当前局面此前出现次数（0/1/≥2 → [is1,is2]）；判定权威永远在规则引擎。
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from ..actions import legal_mask, move_to_action
from ..conditions import time_control_bucket
from ..features import FEATURE_DIM, encode
from ..losses import RESULT_DRAW, RESULT_LOSS, RESULT_WIN

T_MAX = 200
RESULT_TO_LABEL = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}  # 白视角：胜/和/负


@dataclass
class StepRecord:
    features: np.ndarray      # (785,) float32
    action: int               # 动作 id
    legal_mask: np.ndarray    # (1936,) bool
    result: int               # 0 胜/1 和/2 负（对行棋方归一）
    moves_left: int           # 剩余 ply，截断 200
    tc_bucket: int
    elo_mean: float           # 双方平均 Elo；缺失填 1500（权重另置 1.0）
    elo_missing: bool
    color: int                # 0 黑走 / 1 白走


def _board_key(board: chess.Board) -> tuple:
    """局面重复判定键：棋子布置 + 走子方 + 易位权 + 合法过路兵（等价于 FEN 前三段+ep，裁判口径）。"""
    return (
        tuple(sorted(board.piece_map().items())),
        board.turn,
        board.castling_rights,
        board.ep_square if board.has_legal_en_passant() else None,
    )


def game_to_sequence(game: chess.pgn.Game, meta: dict) -> list[StepRecord]:
    """一局 PGN → StepRecord 列表（半回合粒度，T = 总 ply）。"""
    board = game.board()
    moves = list(game.mainline_moves())
    total = len(moves)
    elo_mean = meta["elo_mean"] if meta["elo_mean"] is not None else 1500.0
    elo_missing = meta["elo_mean"] is None
    tc_bucket = int(time_control_bucket(meta["time_control"]))
    result_w = RESULT_TO_LABEL[meta["result"]]  # 白视角

    occurrences: dict[tuple, int] = {}
    records: list[StepRecord] = []
    for t, move in enumerate(moves):
        key = _board_key(board)
        prior = occurrences.get(key, 0)
        occurrences[key] = prior + 1

        mover_result = result_w if board.turn == chess.WHITE else (2 - result_w)  # 对行棋方归一
        records.append(
            StepRecord(
                features=encode(board, occurrence=prior),
                action=move_to_action(move),
                legal_mask=legal_mask(board),
                result=mover_result,
                moves_left=min(total - t, T_MAX),
                tc_bucket=tc_bucket,
                elo_mean=elo_mean,
                elo_missing=elo_missing,
                color=1 if board.turn == chess.WHITE else 0,
            )
        )
        board.push(move)
    return records
