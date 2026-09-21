#!/usr/bin/env python3
"""Autonomous 7-hour loop pipeline for UniChessSSM.

Orchestration stages:
1. Stage 1: Monitor current generation (runs/stage_b_gen_1500_gen1) until completion and verify manifest.json.
2. Stage 2: Automatically launch Gen-1 training (train/stage_b2.py):
   - --data data/shards
   - --selfplay runs/stage_b_gen_1500_gen1
   - --ckpt runs/stage_b_training_fix500_cs01/best.pt
   - --out runs/stage_b_training_1500_gen1
   - --workers 4 --threads 4 --mem-fraction 0.45 --mlh-log
   - --microbatch 8 --accum 16
   - Logs metrics and waits for completion.
3. Stage 3: Automatically launch Dual Arena:
   - Part A: Arena vs Baseline / Previous Champion (tools/ssm_gumbel_arena.py)
     Candidate: runs/stage_b_training_1500_gen1/best.pt
     Baseline: runs/stage_a_20260915/best.pt (also evaluates vs fix500_cs01 if desired)
     64 games, --workers 4, --sprt --sprt-min-games 32, out runs/arena_1500_gen1_vs_baseline.
   - Part B: Arena vs Transformer (tools/ssm_eval_vs_transformer.py)
     SSM: runs/stage_b_training_1500_gen1/best.pt
     Transformer: /home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt
     64 games, --workers 4, out runs/arena_1500_gen1_vs_transformer.
4. Stage 4: Analyze results:
   - If winrate >= 50% on baseline, promote runs/stage_b_training_1500_gen1/best.pt to new Champion (runs/champion.pt).
   - Log comprehensive summary to runs/pipeline_summary.json and console.
   - If time permits, automatically scale up to Gen-2 (e.g. 2500 games) and continue loop.
   - If regression occurs (<45%), analyze cause and revert or adjust hyperparameters.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [AutoPipeline] {msg}"
    print(formatted, flush=True)


def run_command(cmd: list[str], log_file_path: Path | None = None, cwd: Path | None = None) -> int:
    log(f"Running command: {' '.join(str(x) for x in cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_fh = None
    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file_path, "a", encoding="utf-8")
        log(f"Redirecting command output to {log_file_path}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if log_fh is None else log_fh,
            stderr=subprocess.STDOUT if log_fh is not None else subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            bufsize=1,
        )

        if log_fh is None:
            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout, flush=True)
            if stderr:
                print(stderr, file=sys.stderr, flush=True)
            return proc.returncode
        else:
            ret = proc.wait()
            return ret
    finally:
        if log_fh is not None:
            log_fh.close()


def stage1_wait_for_generation(gen_dir: Path, poll_interval_s: int = 30) -> dict:
    log(f"=== Stage 1: Monitoring selfplay generation in {gen_dir} ===")
    manifest_path = gen_dir / "manifest.json"

    start_time = time.time()
    last_log_time = 0.0

    while True:
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                games_count = manifest.get("games", 0)
                shards = manifest.get("shards", [])
                if games_count > 0 and len(shards) > 0:
                    log(f"Generation complete! manifest.json verified: games={games_count}, shards={len(shards)}")
                    return manifest
            except Exception as e:
                log(f"manifest.json exists but error reading ({e}), waiting...")

        # Periodic status logging
        now = time.time()
        if now - last_log_time >= 120:
            last_log_time = now
            worker_logs = list(gen_dir.glob("_w*/worker.log"))
            statuses = []
            for wlog in sorted(worker_logs):
                wname = wlog.parent.name
                try:
                    with open(wlog, "r", encoding="utf-8") as wf:
                        lines = [line.strip() for line in wf if line.strip()]
                        if lines:
                            statuses.append(f"{wname}: {lines[-1]}")
                except Exception:
                    pass
            elapsed_min = (now - start_time) / 60.0
            log(f"Still generating (elapsed {elapsed_min:.1f}m)... " + " | ".join(statuses))

        time.sleep(poll_interval_s)


def stage2_run_training(
    python_bin: str,
    root_dir: Path,
    selfplay_dir: Path,
    init_ckpt: Path,
    out_dir: Path,
    workers: int = 4,
    threads: int = 4,
    mem_fraction: float = 0.45,
    microbatch: int = 8,
    accum: int = 16,
) -> Path:
    log(f"=== Stage 2: Launching Gen-1 Training ===")
    log(f"Selfplay data: {selfplay_dir}")
    log(f"Init ckpt: {init_ckpt}")
    log(f"Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train.log"

    cmd = [
        python_bin,
        str(root_dir / "train" / "stage_b2.py"),
        "--data",
        str(root_dir / "data" / "shards"),
        "--selfplay",
        str(selfplay_dir),
        "--ckpt",
        str(init_ckpt),
        "--out",
        str(out_dir),
        "--workers",
        str(workers),
        "--threads",
        str(threads),
        "--mem-fraction",
        str(mem_fraction),
        "--mlh-log",
        "--microbatch",
        str(microbatch),
        "--accum",
        str(accum),
    ]

    ret = run_command(cmd, log_file_path=train_log, cwd=root_dir)
    if ret != 0:
        raise RuntimeError(f"Training failed with exit code {ret}! Check {train_log}")

    best_pt = out_dir / "best.pt"
    if not best_pt.is_file():
        # Fallback to latest.pt if best.pt wasn't created
        latest_pt = out_dir / "latest.pt"
        if latest_pt.is_file():
            log(f"best.pt not found, using latest.pt as fallback")
            shutil.copy(latest_pt, best_pt)
        else:
            raise FileNotFoundError(f"Neither best.pt nor latest.pt found in {out_dir}")

    log(f"Training completed successfully! Model saved at {best_pt}")
    return best_pt


def stage3_run_arenas(
    python_bin: str,
    root_dir: Path,
    candidate_ckpt: Path,
    baseline_ckpt: Path,
    transformer_ckpt: Path,
    arena_baseline_out: Path,
    arena_transformer_out: Path,
    workers: int = 4,
    games: int = 64,
    sprt_min_games: int = 32,
) -> tuple[dict, dict]:
    log(f"=== Stage 3: Launching Dual Arena Evaluations ===")

    # --- Part A: SSM vs Baseline / Previous Champion ---
    log(f"[Arena Part A] SSM Candidate vs Baseline: {candidate_ckpt.name} vs {baseline_ckpt.name}")
    arena_baseline_out.mkdir(parents=True, exist_ok=True)
    arena_baseline_log = arena_baseline_out / "arena.log"

    cmd_a = [
        python_bin,
        str(root_dir / "tools" / "ssm_gumbel_arena.py"),
        "--ckpt-a",
        str(candidate_ckpt),
        "--ckpt-b",
        str(baseline_ckpt),
        "--out",
        str(arena_baseline_out),
        "--games",
        str(games),
        "--workers",
        str(workers),
        "--sprt",
        "--sprt-min-games",
        str(sprt_min_games),
        "--c_scale_a",
        "0.1",
        "--c_scale_b",
        "0.1",
        "--n_sims",
        "64",
        "--m0",
        "16",
    ]

    ret_a = run_command(cmd_a, log_file_path=arena_baseline_log, cwd=root_dir)
    log(f"Arena Part A finished with code {ret_a}")

    summary_a = {}
    arena_a_json = arena_baseline_out / "arena.json"
    if arena_a_json.is_file():
        try:
            with open(arena_a_json, "r", encoding="utf-8") as f:
                summary_a = json.load(f)
        except Exception as e:
            log(f"Error loading {arena_a_json}: {e}")

    # --- Part B: SSM Candidate vs Transformer ---
    log(f"[Arena Part B] SSM Candidate vs Transformer: {candidate_ckpt.name} vs {transformer_ckpt.name}")
    arena_transformer_out.mkdir(parents=True, exist_ok=True)
    arena_transformer_log = arena_transformer_out / "arena.log"

    summary_b = {}
    if transformer_ckpt.is_file():
        cmd_b = [
            python_bin,
            str(root_dir / "tools" / "ssm_eval_vs_transformer.py"),
            "--ssm-ckpt",
            str(candidate_ckpt),
            "--opponent-type",
            "transformer",
            "--opponent-ckpt",
            str(transformer_ckpt),
            "--games",
            str(games),
            "--workers",
            str(workers),
            "--n-sims",
            "64",
            "--m0",
            "16",
            "--c-scale",
            "0.1",
            "--out",
            str(arena_transformer_out),
        ]
        ret_b = run_command(cmd_b, log_file_path=arena_transformer_log, cwd=root_dir)
        log(f"Arena Part B finished with code {ret_b}")

        arena_b_json = arena_transformer_out / "arena_summary.json"
        if arena_b_json.is_file():
            try:
                with open(arena_b_json, "r", encoding="utf-8") as f:
                    summary_b = json.load(f)
            except Exception as e:
                log(f"Error loading {arena_b_json}: {e}")
    else:
        log(f"Warning: Transformer checkpoint {transformer_ckpt} not found! Skipping Part B.")

    return summary_a, summary_b


def stage4_analyze_and_advance(
    root_dir: Path,
    candidate_ckpt: Path,
    summary_baseline: dict,
    summary_transformer: dict,
    pipeline_summary_path: Path,
) -> dict:
    log(f"=== Stage 4: Result Analysis & Champion Promotion ===")

    score_pct = summary_baseline.get("score_a_percent", 0.0)
    wins_a = summary_baseline.get("wins_a", 0)
    wins_b = summary_baseline.get("wins_b", 0)
    draws = summary_baseline.get("draws", 0)
    total_games = summary_baseline.get("total_games", 0)

    promoted = False
    status = "normal"
    champion_path = root_dir / "runs" / "champion.pt"

    log(f"Baseline Matchup Result: {wins_a}W - {draws}D - {wins_b}L ({score_pct:.1f}%) over {total_games} games")

    if summary_transformer:
        tf_score_pct = summary_transformer.get("score_percentage", 0.0)
        tf_wins = summary_transformer.get("wins", 0)
        tf_draws = summary_transformer.get("draws", 0)
        tf_losses = summary_transformer.get("losses", 0)
        log(f"Transformer Matchup Result: {tf_wins}W - {tf_draws}D - {tf_losses}L ({tf_score_pct:.1f}%)")

    if score_pct >= 50.0:
        promoted = True
        status = "promoted"
        log(f"🎉 Candidate achieved {score_pct:.1f}% >= 50.0% winrate! PROMOTING TO NEW CHAMPION!")
        shutil.copy(candidate_ckpt, champion_path)
        log(f"Champion updated at {champion_path}")
    elif score_pct < 45.0:
        status = "regressed"
        log(f"⚠️ Regression detected: score {score_pct:.1f}% < 45.0%. Champion remains unchanged.")
    else:
        status = "retained"
        log(f"Candidate score {score_pct:.1f}% within neutral window [45%, 50%). Champion remains unchanged.")

    analysis = {
        "timestamp": datetime.datetime.now().isoformat(),
        "candidate_ckpt": str(candidate_ckpt),
        "promoted_to_champion": promoted,
        "status": status,
        "baseline_arena": {
            "score_pct": score_pct,
            "wins_candidate": wins_a,
            "wins_baseline": wins_b,
            "draws": draws,
            "total_games": total_games,
            "sprt": summary_baseline.get("sprt_info"),
        },
        "transformer_arena": summary_transformer,
    }

    with open(pipeline_summary_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    log(f"Pipeline summary written to {pipeline_summary_path}")

    return analysis


def stage5_launch_gen2_if_permitted(
    python_bin: str,
    root_dir: Path,
    champion_ckpt: Path,
    gen2_dir: Path,
    games: int = 2500,
    workers: int = 4,
    concurrency: int = 24,
) -> None:
    log(f"=== Stage 5: Scaling up to Gen-2 ({games} selfplay games) ===")
    gen2_dir.mkdir(parents=True, exist_ok=True)
    gen2_log = root_dir / "runs" / "stage_b_gen_2500_gen2.log"

    cmd = [
        python_bin,
        str(root_dir / "tools" / "ssm_gumbel_selfplay.py"),
        "--ckpt",
        str(champion_ckpt),
        "--out",
        str(gen2_dir),
        "--games",
        str(games),
        "--workers",
        str(workers),
        "--concurrency",
        str(concurrency),
        "--c_scale",
        "0.1",
        "--n_sims",
        "64",
        "--m0",
        "16",
        "--seed",
        "20260925",
        "--gen_id",
        "2",
    ]

    log(f"Starting Gen-2 selfplay generation...")
    ret = run_command(cmd, log_file_path=gen2_log, cwd=root_dir)
    log(f"Gen-2 selfplay exited with returncode {ret}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UniChessSSM 7-hour autonomous loop pipeline")
    parser.add_argument("--root", type=str, default="/home/jeefy/UniChess/SSM", help="Repository root path")
    parser.add_argument(
        "--python",
        type=str,
        default="/home/jeefy/miniconda3/envs/unichess/bin/python",
        help="Path to Python interpreter",
    )
    parser.add_argument(
        "--gen1-dir",
        type=str,
        default="runs/stage_b_gen_1500_gen1",
        help="Directory where Gen-1 selfplay is currently running",
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="runs/stage_b_training_fix500_cs01/best.pt",
        help="Checkpoint to initialize Gen-1 training",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default="runs/stage_a_20260915/best.pt",
        help="Baseline checkpoint for arena evaluation",
    )
    parser.add_argument(
        "--transformer-ckpt",
        type=str,
        default="/home/jeefy/UniChess/Transformer/runs/stratified_middlegame_curriculum/best_model.pt",
        help="Transformer model checkpoint",
    )
    parser.add_argument(
        "--auto-gen2",
        action="store_true",
        default=True,
        help="Automatically trigger Gen-2 selfplay if Gen-1 loop completes",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    python_bin = args.python
    gen1_dir = (root_dir / args.gen1_dir).resolve()
    init_ckpt = (root_dir / args.init_ckpt).resolve()
    baseline_ckpt = (root_dir / args.baseline_ckpt).resolve()
    transformer_ckpt = Path(args.transformer_ckpt).resolve()

    gen1_train_out = root_dir / "runs" / "stage_b_training_1500_gen1"
    arena_baseline_out = root_dir / "runs" / "arena_1500_gen1_vs_baseline"
    arena_transformer_out = root_dir / "runs" / "arena_1500_gen1_vs_transformer"
    pipeline_summary = root_dir / "runs" / "pipeline_summary.json"
    gen2_dir = root_dir / "runs" / "stage_b_gen_2500_gen2"

    log("=" * 60)
    log("UniChessSSM Autonomous 7-Hour Loop Pipeline Starting")
    log(f"Root: {root_dir}")
    log(f"Python: {python_bin}")
    log(f"Gen1 selfplay dir: {gen1_dir}")
    log(f"Initial training ckpt: {init_ckpt}")
    log(f"Baseline evaluation ckpt: {baseline_ckpt}")
    log("=" * 60)

    # Stage 1: Monitor Gen-1 selfplay
    stage1_wait_for_generation(gen1_dir)

    # Stage 2: Train Gen-1
    candidate_best_pt = stage2_run_training(
        python_bin=python_bin,
        root_dir=root_dir,
        selfplay_dir=gen1_dir,
        init_ckpt=init_ckpt,
        out_dir=gen1_train_out,
        workers=4,
        threads=4,
        mem_fraction=0.45,
        microbatch=8,
        accum=16,
    )

    # Stage 3: Dual Arena Evaluations
    summary_baseline, summary_transformer = stage3_run_arenas(
        python_bin=python_bin,
        root_dir=root_dir,
        candidate_ckpt=candidate_best_pt,
        baseline_ckpt=baseline_ckpt,
        transformer_ckpt=transformer_ckpt,
        arena_baseline_out=arena_baseline_out,
        arena_transformer_out=arena_transformer_out,
        workers=4,
        games=64,
        sprt_min_games=32,
    )

    # Stage 4: Result Analysis & Champion Promotion
    analysis = stage4_analyze_and_advance(
        root_dir=root_dir,
        candidate_ckpt=candidate_best_pt,
        summary_baseline=summary_baseline,
        summary_transformer=summary_transformer,
        pipeline_summary_path=pipeline_summary,
    )

    # Stage 5: Scale to Gen-2 if requested
    if args.auto_gen2:
        champion_to_use = root_dir / "runs" / "champion.pt"
        if not champion_to_use.is_file():
            champion_to_use = candidate_best_pt
        stage5_launch_gen2_if_permitted(
            python_bin=python_bin,
            root_dir=root_dir,
            champion_ckpt=champion_to_use,
            gen2_dir=gen2_dir,
            games=2500,
            workers=4,
            concurrency=24,
        )

    log("=" * 60)
    log("Auto-loop pipeline execution sequence finished.")
    log("=" * 60)


if __name__ == "__main__":
    main()
