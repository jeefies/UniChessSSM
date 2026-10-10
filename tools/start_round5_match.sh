#!/usr/bin/env bash
set -e
export UNICHESS_IMPORT_ROOT=/home/jeefy/UniChess
export PYTHONPATH=/home/jeefy/UniChess:$PYTHONPATH
cd /home/jeefy/UniChess

OUT_FILE=/home/jeefy/UniChess/SSM/runs/match_gen6_vs_champ_gen4.jsonl
rm -f "$OUT_FILE"

/home/jeefy/miniconda3/envs/unichess/bin/python -u -m Kit match /home/jeefy/UniChess/SSM/configs/match_round5_vs_champ_gen4.json --out "$OUT_FILE" > /home/jeefy/UniChess/SSM/runs/match_gen6_vs_champ_gen4.log 2>&1
