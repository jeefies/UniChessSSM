#!/usr/bin/env bash
set -e
export UNICHESS_IMPORT_ROOT=/home/jeefy/UniChess
export PYTHONPATH=/home/jeefy/UniChess:$PYTHONPATH
cd /home/jeefy/UniChess

LOG_PATH=/home/jeefy/UniChess/SSM/runs/stage_b_training_round4.log
echo "Starting Stage B6 training at $(date)..." > "$LOG_PATH"

/home/jeefy/miniconda3/envs/unichess/bin/python -u -m Kit train SSM/configs/stage_b6_round4.json >> "$LOG_PATH" 2>&1
echo "Stage B6 training completed at $(date)." >> "$LOG_PATH"
