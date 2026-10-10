import json
import collections
import chess

def audit_file(path):
    print(f"\n=======================================================")
    print(f"审计对决文件: {path}")
    print(f"=======================================================")
    
    total = 0
    results = collections.Counter()
    draw_reasons = collections.Counter()
    
    with open(path, "r", encoding="utf-8") as f:
        printed = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            if game.get("type") == "header":
                continue
            res = game.get("result")
            results[res] += 1
            total += 1
            
            if res == "1/2-1/2":
                if not printed:
                    print("和棋对局完整字段:")
                    print(json.dumps(game, indent=2))
                    printed = True
                term = game.get("termination", "unknown")
                draw_reasons[term] += 1

    print(f"总局数: {total}, 结果分布: {dict(results)}")
    print(f"和棋总数: {results.get('1/2-1/2', 0)} ({results.get('1/2-1/2', 0)/total*100:.1f}%)")
    print(f"和棋类型分布:")
    for k, v in draw_reasons.most_common():
        print(f"  - {k}: {v} ({v/results.get('1/2-1/2', 1)*100:.1f}%)")

if __name__ == "__main__":
    audit_file("/home/jeefy/UniChess/SSM/runs/match_gen4_vs_champ_gen1.jsonl")
