#!/usr/bin/env bash
# Gen-3 换代 arena：challenger（Gen-3 best.pt）vs champion（Gen-2 best.pt），
# 配对开局 + 交换颜色，Gumbel g=0。n_sims 跟随 2026-09-22 变更取 256。
set -e -o pipefail
cd /home/jeefy/UniChess/SSM
export PYTHONUNBUFFERED=1
OUT=runs/arena_1000_gen3_vs_gen2
mkdir -p "$OUT"
/home/jeefy/miniconda3/envs/unichess/bin/python \
  tools/ssm_gumbel_arena.py \
  --ckpt-a runs/stage_b_training_2500_gen2/best.pt \
  --ckpt-b runs/stage_b_training_1000_gen3/best.pt \
  --out "$OUT" \
  --games 64 --pairs 8 \
  --n_sims 256 --m0 16 \
  --seed 20260922 \
  > "$OUT/arena.log" 2>&1
echo "ARENA_DONE" >> "$OUT/arena.log"
