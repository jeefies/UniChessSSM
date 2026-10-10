import json
import collections
import glob
import sys

paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("/home/jeefy/UniChess/SSM/runs/stage_b_gen_8000_round4/_w*/.games.jsonl")
reasons = collections.Counter()
results = collections.Counter()
plies = []

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            g = json.loads(line)
            res = g.get("result")
            results[res] += 1
            reason = g.get("reason", "unknown")
            reasons[reason] += 1
            plies.append(g.get("plies", 0))

total = sum(results.values())
if total == 0:
    print("目前尚无对局记录完成")
    sys.exit(0)

print(f"=== 自对弈统计 (总局数: {total}) ===")
print("胜负结果分布 (0=白胜, 1=和棋, 2=黑胜):", dict(results))
draw_count = results.get(1, 0)
win_loss = total - draw_count
print(f"分出胜负率: {win_loss/total*100:.2f}% (分出胜负: {win_loss} 局, 和棋: {draw_count} 局, 和棋率: {draw_count/total*100:.2f}%)")
print(f"平均回合深度: {sum(plies)/len(plies):.1f} ply (中位数: {sorted(plies)[len(plies)//2]})")
print("终局原因分布 (reason):")
for k, v in reasons.most_common():
    print(f"  - {k}: {v} ({v/total*100:.2f}%)")
