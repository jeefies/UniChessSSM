import json
import collections

reasons = collections.Counter()
results = collections.Counter()
plies = []

path = "/home/jeefy/UniChess/SSM/runs/stage_b_gen_round3_10000/.games.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        res = g.get("result")
        results[res] += 1
        reason = g.get("reason", "unknown")
        reasons[reason] += 1
        plies.append(g.get("plies", 0))

total = sum(results.values())
# Kit V3 sink: result: 0 = White win, 1 = Draw, 2 = Black win (or 0=Draw, 1=White, 2=Black)
# 让我们看看 reason 与 result 的对应关系
print(f"=== 自对弈统计 (总局数: {total}) ===")
print("结果分布 (数值编码):", dict(results))
print(f"平均回合深度: {sum(plies)/len(plies):.1f} ply (中位数: {sorted(plies)[len(plies)//2]})")
print("终局原因分布 (reason):")
for k, v in reasons.most_common():
    print(f"  - {k}: {v} ({v/total*100:.2f}%)")
