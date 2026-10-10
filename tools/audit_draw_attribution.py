import sys
sys.path.insert(0, "/home/jeefy/UniChess")
import json
import chess
from Kit.search.gumbel import _material_score

path = "/home/jeefy/UniChess/SSM/runs/match_gen4_vs_champ_gen1.jsonl"

a_draws_as_dominant = 0  # A 优势成和 (A丢分)
b_draws_as_dominant = 0  # B 优势成和 (B丢分, A得分)
equal_draws = 0          # 均势成和

a_draws_details = []
b_draws_details = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        if g.get("type") == "header" or g.get("result") != "1/2-1/2":
            continue
        
        # 重放整局，找出整局过程中以及终局时的最大子力优势
        board = chess.Board()
        for uci in g.get("opening", []):
            board.push_uci(uci)
        
        max_a_net = -999
        max_b_net = -999
        
        white_is_a = (g.get("white") == "A")
        
        for uci in g.get("moves", []):
            board.push_uci(uci)
            w_m = _material_score(board, chess.WHITE)
            b_m = _material_score(board, chess.BLACK)
            a_m = w_m if white_is_a else b_m
            opp_m = b_m if white_is_a else w_m
            net_a = a_m - opp_m
            if net_a > max_a_net:
                max_a_net = net_a
            if -net_a > max_b_net:
                max_b_net = -net_a

        # 终局子力差
        final_w = _material_score(board, chess.WHITE)
        final_b = _material_score(board, chess.BLACK)
        final_a_net = (final_w - final_b) if white_is_a else (final_b - final_w)

        term = g.get("termination")
        
        if final_a_net >= 3 or max_a_net >= 4:
            a_draws_as_dominant += 1
            a_draws_details.append({
                "game": g.get("game"), "term": term, "final_a_net": final_a_net,
                "max_a_net": max_a_net, "plies": len(g.get("moves", []))
            })
        elif final_a_net <= -3 or max_b_net >= 4:
            b_draws_as_dominant += 1
            b_draws_details.append({
                "game": g.get("game"), "term": term, "final_b_net": -final_a_net,
                "max_b_net": max_b_net, "plies": len(g.get("moves", []))
            })
        else:
            equal_draws += 1

print(f"=== 140 局和棋深层原因归因 ===")
print(f"1. A 方曾获大优/终局大优却成和 (A错失胜机): {a_draws_as_dominant} 局 ({a_draws_as_dominant/140*100:.1f}%)")
print(f"2. B 方(初代基准)大优却被 A 顽强守和 (A成功守和防守加分): {b_draws_as_dominant} 局 ({b_draws_as_dominant/140*100:.1f}%)")
print(f"3. 全程势均力敌兑光正常和棋 (理论均势和棋): {equal_draws} 局 ({equal_draws/140*100:.1f}%)")

print("\nA 方大优成和的类型分布:")
terms_a = {}
for d in a_draws_details:
    terms_a[d["term"]] = terms_a.get(d["term"], 0) + 1
print(terms_a)

print("\nB 方大优被 A 顽强守和的类型分布:")
terms_b = {}
for d in b_draws_details:
    terms_b[d["term"]] = terms_b.get(d["term"], 0) + 1
print(terms_b)
