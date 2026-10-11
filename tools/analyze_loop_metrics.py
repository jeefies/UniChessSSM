#!/usr/bin/env python3
"""
analyze_loop_metrics.py
深度解析 Gen 6 ~ Gen 8 的训练 loss、0-sim、Arena 对决及数据分布
"""
import os
import json
import re
from pathlib import Path

RUNS_DIR = Path("/home/jeefy/UniChess/SSM/runs/loop_stage_b")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze SSM loop metrics")
    parser.add_argument("--gen", type=int, default=None, help="Specific generation to analyze")
    parser.add_argument("--last", type=int, default=5, help="Number of recent generations to analyze")
    args = parser.parse_args()

    print("=" * 60)
    print("           SSM 代际循环深入数据诊断报告")
    print("=" * 60)

    all_gen_dirs = sorted([d for d in RUNS_DIR.glob("gen_*") if d.is_dir()])
    gen_ids = []
    for d in all_gen_dirs:
        m = re.match(r"gen_(\d+)", d.name)
        if m:
            gen_ids.append(int(m.group(1)))

    if args.gen is not None:
        target_gens = [args.gen]
    else:
        target_gens = gen_ids[-args.last:] if gen_ids else []

    for gen in target_gens:
        gen_dir = RUNS_DIR / f"gen_{gen:04d}"
        if not gen_dir.exists():
            continue

        print(f"\n################### Generation {gen} ###################")
        
        # 1. 自对弈统计
        sp_manifest = gen_dir / "selfplay" / "manifest.json"
        if sp_manifest.exists():
            try:
                with open(sp_manifest, "r", encoding="utf-8") as f:
                    mf = json.load(f)
                games = mf.get("total_games", 0)
                white_wins = mf.get("white_wins", 0)
                black_wins = mf.get("black_wins", 0)
                draws = mf.get("draws", 0)
                print(f"[自对弈生成 3000局]")
                print(f"  总对局: {games} | 白胜: {white_wins} ({white_wins/max(1,games)*100:.1f}%) | 黑胜: {black_wins} ({black_wins/max(1,games)*100:.1f}%) | 和棋: {draws} ({draws/max(1,games)*100:.1f}%)")
            except Exception as e:
                print(f"  自对弈解析异常: {e}")

        # 2. 训练日志与 Loss
        train_log = gen_dir / "train.log"
        if train_log.exists():
            try:
                with open(train_log, "r", encoding="utf-8") as f:
                    t_lines = f.readlines()
                val_lines = [l.strip() for l in t_lines if "val/" in l or "val_loss" in l or "validate" in l.lower() or "score=" in l]
                best_lines = [l.strip() for l in t_lines if "best" in l.lower()]
                print(f"[强化训练 Loss 走势]")
                if val_lines:
                    print("  末期验证记录:")
                    for vl in val_lines[-4:]:
                        print(f"    {vl}")
                if best_lines:
                    print("  最优保存记录:")
                    for bl in best_lines[-2:]:
                        print(f"    {bl}")
            except Exception as e:
                print(f"  训练日志解析异常: {e}")

        # 3. Track 1: 纯策略 0-sim
        eval_raw_log = gen_dir / "eval_raw_policy.log"
        if eval_raw_log.exists():
            try:
                with open(eval_raw_log, "r", encoding="utf-8") as f:
                    raw_lines = [l.strip() for l in f if "得分率:" in l or "对决结果" in l or "总计" in l]
                print(f"[Track 1: 纯策略网络 0-sim 对决]")
                for rl in raw_lines:
                    print(f"    {rl}")
            except Exception as e:
                print(f"  0-sim 解析异常: {e}")

        # 4. Track 2: Arena 64-sims MCTS 对决
        arena_log = gen_dir / "arena.log"
        if arena_log.exists():
            try:
                with open(arena_log, "r", encoding="utf-8") as f:
                    a_lines = [l.strip() for l in f if "[match]" in l]
                
                white_wins = sum(1 for l in a_lines if "1-0" in l and "A 执白" in l)
                white_losses = sum(1 for l in a_lines if "0-1" in l and "A 执白" in l)
                black_wins = sum(1 for l in a_lines if "0-1" in l and "A 执黑" in l)
                black_losses = sum(1 for l in a_lines if "1-0" in l and "A 执黑" in l)
                draws = sum(1 for l in a_lines if "1/2-1/2" in l)
                total = len(a_lines)

                # 终局原因
                reasons = {}
                plies = []
                for l in a_lines:
                    m = re.search(r"\(([^,]+),\s*(\d+)\s*ply", l)
                    if m:
                        r = m.group(1).strip()
                        p = int(m.group(2).strip())
                        reasons[r] = reasons.get(r, 0) + 1
                        plies.append(p)

                avg_ply = sum(plies) / max(1, len(plies))
                print(f"[Track 2: Arena 64-sim MCTS 对决 (vs Champion gen4)]")
                print(f"  总对局数: {total}")
                score = (white_wins + black_wins + 0.5 * draws) / max(1, total)
                print(f"  Candidate(A) 总成绩: {white_wins+black_wins} 胜 / {white_losses+black_losses} 负 / {draws} 和 | 得分率: {score*100:.2f}%")
                print(f"  A 执白局: {white_wins} 胜 / {white_losses} 负 (得分率: {(white_wins+0.5*(draws/2))/(max(1, white_wins+white_losses+draws/2))*100:.1f}%)")
                print(f"  A 执黑局: {black_wins} 胜 / {black_losses} 负 (得分率: {(black_wins+0.5*(draws/2))/(max(1, black_wins+black_losses+draws/2))*100:.1f}%)")
                print(f"  平均对局步数: {avg_ply:.1f} ply")
                print(f"  终局原因分布: {reasons}")
            except Exception as e:
                print(f"  Arena 解析异常: {e}")

if __name__ == "__main__":
    main()
