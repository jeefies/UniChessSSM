#!/usr/bin/env bash
set -e -o pipefail
cd /home/jeefy/UniChessSSM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs/stage_b_training_round2
/home/jeefy/miniconda3/envs/unichess/bin/python \
  train/stage_b2.py \
  --data data/shards \
  --selfplay runs/stage_b_gen_round2 \
  --out runs/stage_b_training_round2 \
  --ckpt runs/stage_b_training_round1/best.pt \
  --microbatch 8 --accum 64 \
  --w-selfplay 0.85 --w-human 0.10 \
  > runs/stage_b_training_round2/train.log 2>&1
echo "TRAIN_DONE" >> runs/stage_b_training_round2/train.log