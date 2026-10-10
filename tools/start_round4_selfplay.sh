#!/usr/bin/env bash
set -e
export UNICHESS_IMPORT_ROOT=/home/jeefy/UniChess
export PYTHONPATH=/home/jeefy/UniChess:$PYTHONPATH

OUT_DIR="/home/jeefy/UniChess/SSM/runs/stage_b_gen_8000_round4"
mkdir -p "$OUT_DIR"

/home/jeefy/miniconda3/envs/unichess/bin/python -u /home/jeefy/UniChess/SSM/tools/run_gpu_server_selfplay.py \
    --ckpt /home/jeefy/UniChess/SSM/runs/champion.pt \
    --opp-ckpt /home/jeefy/UniChess/SSM/runs/champion_gen2.pt \
    --out "$OUT_DIR" \
    --total-games 8000 \
    --first-game 0 \
    --workers 6 \
    --concurrency 24 \
    --simulations 64 \
    --m0 16 \
    --c-scale 0.02 \
    --twofold-penalty 1.0 \
    --stalemate-penalty 1.0 \
    --insufficient-penalty 1.0 \
    --contempt 0.5 \
    --temp-plies 15 \
    --temperature 1.0 \
    --min-book-plies 6 \
    --pcr-rate 0.5 \
    --pcr-fast-sims 16 > /home/jeefy/UniChess/SSM/runs/stage_b_gen_8000_round4.log 2>&1
