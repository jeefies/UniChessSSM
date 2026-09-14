"""局面特征 B_t：785 维无损编码（设计文档 §3.2，D5；文档称"约 790"，实际 785）。

白方绝对坐标，不随走子方翻转。布局（float32，顺序固定）：
  [0, 768)   棋子平面 12×64 二值：白 P N B R Q K（0..5）、黑 P N B R Q K（6..11）；格序 a1=0..h8=63
  [768]      走子方：白=1 黑=0
  [769,773)  易位权：白王翼、白后翼、黑王翼、黑后翼
  [773,781)  过路兵 file one-hot（仅当存在合法吃过路兵；否则全零）
  [781]      半回合计数 halfmove_clock / 100
  [782]      全回合数 fullmove / 200（截断到 1）
  [783,785)  重复计数：当前局面此前出现 1 次/≥2 次 → [is1, is2]（参考特征；判定权威在规则引擎）

无损：由上述字段可完整恢复 FEN。判定（重复/终局）一律以规则引擎为准，本特征不作判定依据。
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

FEATURE_DIM = 785
PIECE_PLANES = 12
GLOBALS_DIM = 17  # side1 + castling4 + ep8 + halfmove1 + fullmove1 + rep2（进 E 的 globals）

# 平面序：白 P N B R Q K = 0..5，黑 = 6..11
_PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2, (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8, (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}


@dataclass(frozen=True)
class BoardFields:
    """解码出的局面字段（用于往返测试）。"""

    pieces: dict[int, tuple[bool, int]]  # 格 -> (是否白, 棋子类型)
    side_white: bool
    castling: tuple[bool, bool, bool, bool]  # 白王翼/白后翼/黑王翼/黑后翼
    ep_file: int | None                    # 可吃过路兵的 file（0..7），无则 None
    halfmove: int
    fullmove: int
    rep_is1: bool
    rep_is2: bool


def encode(board: chess.Board, occurrence: int = 0) -> np.ndarray:
    """局面 → 785 维特征。

    occurrence：当前局面在本局此前出现的次数（0/1/≥2），由调用方沿棋谱统计（参考特征）。
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 棋子平面
    for sq, piece in board.piece_map().items():
        plane = _PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        feat[plane * 64 + sq] = 1.0

    # 走子方 / 易位权
    feat[768] = 1.0 if board.turn == chess.WHITE else 0.0
    feat[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    feat[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    feat[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    feat[772] = float(board.has_queenside_castling_rights(chess.BLACK))

    # 过路兵 file（仅当存在合法吃过路兵）
    if board.ep_square is not None and board.has_legal_en_passant():
        feat[773 + chess.square_file(board.ep_square)] = 1.0

    # 计数
    feat[781] = min(board.halfmove_clock, 100) / 100.0
    feat[782] = min(board.fullmove_number / 200.0, 1.0)

    # 重复计数
    feat[783] = 1.0 if occurrence == 1 else 0.0
    feat[784] = 1.0 if occurrence >= 2 else 0.0
    return feat


def decode(feat: np.ndarray) -> BoardFields:
    """785 维特征 → 局面字段（往返测试用；棋盘几何由规则引擎重建）。"""
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    assert feat.shape[0] == FEATURE_DIM, f"特征维度 {feat.shape[0]} != {FEATURE_DIM}"

    pieces: dict[int, tuple[bool, int]] = {}
    for plane in range(PIECE_PLANES):
        for sq in range(64):
            if feat[plane * 64 + sq] > 0.5:
                white = plane < 6
                pieces[sq] = (white, (plane % 6) + 1)  # PAWN=1..KING=6

    ep_file = None
    for f in range(8):
        if feat[773 + f] > 0.5:
            ep_file = f

    return BoardFields(
        pieces=pieces,
        side_white=feat[768] > 0.5,
        castling=tuple(feat[769 + i] > 0.5 for i in range(4)),  # type: ignore[arg-type]
        ep_file=ep_file,
        halfmove=int(round(float(feat[781]) * 100)),
        fullmove=max(1, int(round(float(feat[782]) * 200))),
        rep_is1=feat[783] > 0.5,
        rep_is2=feat[784] > 0.5,
    )


def globals_vector(feat: np.ndarray) -> np.ndarray:
    """提取进 E 的 globals 段（17 维：非棋子字段），输入可为任意批量形状 (..., 785)。"""
    feat = np.asarray(feat, dtype=np.float32)
    return feat[..., PIECE_PLANES * 64:]
