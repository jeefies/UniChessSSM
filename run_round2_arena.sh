#!/usr/bin/env bash
set -e -o pipefail
cd /home/jeefy/UniChess/SSM
export PYTHONUNBUFFERED=1
mkdir -p runs/arena_round2
/home/jeefy/miniconda3/envs/unichess/bin/python \
  tools/ssm_gumbel_arena.py \
  --ckpt-a runs/stage_a_20260915/best.pt \
  --ckpt-b runs/stage_b_training_round2/best.pt \
  --out runs/arena_round2 \
  --games 64 --pairs 8 \
  --n_sims 64 --m0 16 \
  > runs/arena_round2/arena.log 2>&1
echo "ARENA_DONE" >> runs/arena_round2/arena.log