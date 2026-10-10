#!/usr/bin/env bash
set -e
export UNICHESS_IMPORT_ROOT=/home/jeefy/UniChess
export PYTHONPATH=/home/jeefy/UniChess:$PYTHONPATH
cd /home/jeefy/UniChess

OUT_DIR=/home/jeefy/UniChess/SSM/runs/loop_stage_b
mkdir -p "$OUT_DIR"

/home/jeefy/miniconda3/envs/unichess/bin/python -u /home/jeefy/UniChess/SSM/tools/loop_pipeline.py \
    --generations 4 \
    --games 3000 \
    --out "$OUT_DIR" \
    --champion /home/jeefy/UniChess/SSM/runs/champion.pt \
    --opp /home/jeefy/UniChess/SSM/runs/champion_gen3.pt \
    --workers 8 \
    --concurrency 20 > "$OUT_DIR/loop.log" 2>&1
