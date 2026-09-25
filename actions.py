"""1936 维紧凑动作空间（设计文档 §3.3，D10）。

- 非升变：后可达有序 (from,to) 对 1456 + 马可达 336 = 1792；
- 升变：兵到末排 (白 7→8、黑 2→1，16 个 from 格 × 3 名义方向) × {R,B,N} = 144（升后已含于后走法）；
- 王车易位 = 王两格移动的 (from,to) 对，天然落在"后走法"集合中（e1→g1 沿横线）。
- 格索引：a1=0, b1=1, …, h8=63（白方绝对坐标，D5）。

id 编排（写死规则，构建即断言计数）：
  [0, 1456)      后走法：from 升序 → 8 方向固定序 → 距离升序
  [1456, 1792)   马走法：from 升序 → 8 跳固定序
  [1792, 1936)   升变：from 格升序（白 a7..h7 后黑 a2..h2）→ 方向（直进/左吃/右吃）→ 升变子 R,B,N
"""

from __future__ import annotations

import chess
import numpy as np

NUM_ACTIONS = 1936
NUM_QUEEN_MOVES = 1456
NUM_KNIGHT_MOVES = 336
NUM_PROMOTION_MOVES = 144
NUM_PROMO_PIECES = 3  # R, B, N（升后=Q 走后走法）
PROMO_PIECES = (chess.ROOK, chess.BISHOP, chess.KNIGHT)

# 8 方向：直横斜（固定序，仅影响 id 编排，不影响正确性）
_QUEEN_DIRS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)
_KNIGHT_JUMPS = (
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
)

# 升变名义方向：delta = 目标格 - 起始格（白 +7/+8/+9，黑 -9/-8/-7）
_PROMO_DELTAS_WHITE = (8, 7, 9)   # 直进、左吃(file-1)、右吃(file+1)
_PROMO_DELTAS_BLACK = (-8, -9, -7)


def _sq(file_: int, rank: int) -> int:
    return rank * 8 + file_


def _on_board(file_: int, rank: int) -> bool:
    return 0 <= file_ < 8 and 0 <= rank < 8


def _build_tables() -> tuple[dict, list[tuple[int, int, int | None]]]:
    """构建 (from,to,promo) → action id 映射及反向表；构建时断言 1456/336/144。"""
    to_action: dict[tuple[int, int, int | None], int] = {}
    from_action: list[tuple[int, int, int | None]] = []

    def _add(frm: int, to: int, promo: int | None) -> None:
        key = (frm, to, promo)
        assert key not in to_action, f"重复动作 {key}"
        to_action[key] = len(from_action)
        from_action.append(key)

    # 后走法
    for frm in range(64):
        f0, r0 = frm % 8, frm // 8
        for df, dr in _QUEEN_DIRS:
            f, r = f0 + df, r0 + dr
            while _on_board(f, r):
                _add(frm, _sq(f, r), None)
                f += df
                r += dr
    assert len(from_action) == NUM_QUEEN_MOVES, f"后走法数 {len(from_action)} != {NUM_QUEEN_MOVES}"

    # 马走法
    for frm in range(64):
        f0, r0 = frm % 8, frm // 8
        for df, dr in _KNIGHT_JUMPS:
            f, r = f0 + df, r0 + dr
            if _on_board(f, r):
                _add(frm, _sq(f, r), None)
    assert len(from_action) == NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES, (
        f"非升变数 {len(from_action)} != {NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES}"
    )

    # 升变：from 格升序 = a7..h7（48..55）后 a2..h2（8..15）；方向序与 PROMO_PIECES 对齐文档。
    # 名义方向在边线不成立（a 线无 file-1 吃、h 线无 file+1 吃，且不会越出 rank），
    # 为保持 16×3=48 的固定 id 布局，幻影方向用自环 (frm,frm) 占位——from==to 永远非法，
    # 不会与任何合法着或后/马走法碰撞（单测兜底）。
    promo_from_squares = list(range(48, 56)) + list(range(8, 16))
    for frm in promo_from_squares:
        deltas = _PROMO_DELTAS_WHITE if frm >= 48 else _PROMO_DELTAS_BLACK
        for delta in deltas:
            to = frm + delta
            if not (0 <= to < 64) or abs(to % 8 - frm % 8) != (0 if delta in (8, -8) else 1):
                to = frm  # 幻影槽位：自环占位
            for promo in PROMO_PIECES:
                _add(frm, to, promo)
    assert len(from_action) == NUM_ACTIONS, f"总数 {len(from_action)} != {NUM_ACTIONS}"
    return to_action, from_action


TO_ACTION: dict[tuple[int, int, int | None], int]
FROM_ACTION: list[tuple[int, int, int | None]]
TO_ACTION, FROM_ACTION = _build_tables()


def move_to_action(move: chess.Move) -> int:
    """python-chess Move → 动作 id。升后走 (from,to) 的后走法 id；升 R/B/N 走升变 id。"""
    promo = move.promotion
    if promo == chess.QUEEN:
        promo = None  # 升后已含于后走法
    key = (move.from_square, move.to_square, promo)
    if key not in TO_ACTION:
        raise ValueError(f"非法/未覆盖动作: {move.uci()} (key={key})")
    return TO_ACTION[key]


def action_to_move(action: int) -> chess.Move:
    """动作 id → python-chess Move（仅 (from,to,promo)；易位/吃过路兵的旗标由规则引擎补）。"""
    frm, to, promo = FROM_ACTION[action]
    return chess.Move(frm, to, promotion=promo)


def legal_mask(board: chess.Board) -> np.ndarray:
    """当前局面的 1936 维合法动作掩码（bool），与规则引擎 legal_moves 完全一致（验收 #5）。"""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for mv in board.legal_moves:
        mask[move_to_action(mv)] = True
    return mask
