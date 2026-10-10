import sys
sys.path.insert(0, "/home/jeefy/UniChess")
import json
import chess
from Kit.rules.fast import outcome as fast_outcome, copy_board
from Kit.search.gumbel import _material_score

# 读取 Game 41
with open("/home/jeefy/UniChess/SSM/runs/match_gen4_vs_champ_gen1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        if g.get("game") == 41:
            game41 = g
            break

board = chess.Board()
for uci in game41["opening"]:
    board.push_uci(uci)

moves = game41["moves"]
white = game41.get('white', 'A')
black = 'B' if white == 'A' else 'A'
print(f"Game 41 总步数: {len(moves)}, 双方: 白={white} 黑={black}")

# 我们重放到倒数第 15 步开始观察
start_idx = len(moves) - 15
for i, uci in enumerate(moves):
    mv = chess.Move.from_uci(uci)
    
    if i >= start_idx:
        turn_str = "白(B)" if board.turn == chess.WHITE else "黑(A)"
        w_mat = _material_score(board, chess.WHITE)
        b_mat = _material_score(board, chess.BLACK)
        
        # 模拟走这步棋
        child = copy_board(board)
        child.push(mv)
        out = fast_outcome(child, claim_draw=True)
        is_rep2 = child.is_repetition(2)
        is_rep3 = child.is_repetition(3)
        
        # 计算行棋方净子力
        cur_col = board.turn
        opp_col = not cur_col
        net_mat = _material_score(board, cur_col) - _material_score(board, opp_col)
        
        print(f"\n[步 {i+1}] {turn_str} 走出 {uci}:")
        print(f"  当前棋盘子力: 白={w_mat}, 黑={b_mat}, 行棋方净优势={net_mat}")
        print(f"  走完后: is_rep(2)={is_rep2}, is_rep(3)={is_rep3}")
        print(f"  fast_outcome(claim_draw=True): {out}")
        if out:
            print(f"  -> 触发终局! termination={out.termination}")
        
    board.push(mv)

print(f"\n终局判定: result={board.result(claim_draw=True)}, is_repetition(3)={board.is_repetition(3)}")
