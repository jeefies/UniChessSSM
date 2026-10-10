#!/usr/bin/env bash
set -e
export UNICHESS_IMPORT_ROOT=/home/jeefy/UniChess
export PYTHONPATH=/home/jeefy/UniChess:$PYTHONPATH
cd /home/jeefy/UniChess

/home/jeefy/miniconda3/envs/unichess/bin/python -u -m Kit train /home/jeefy/UniChess/SSM/configs/stage_b7_round5.json > /home/jeefy/UniChess/SSM/runs/stage_b_training_round5.log 2>&1
