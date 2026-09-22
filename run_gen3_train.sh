#!/usr/bin/env bash
# Gen-3 训练：1k 局 @256 sims 自对弈数据（含开局 book ply 搜索 π′）→ 从 Gen-2 best.pt 续训。
# 显存安全档位（此前 OOM 教训）：microbatch 4 × accum 16 × workers 4 × mem-fraction 0.55。
# P1-3：book ply（meta flags）policy 软 CE 权重 0.25（默认值即 0.25，此处显式写出）。
set -e -o pipefail
cd /home/jeefy/UniChess/SSM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
OUT=runs/stage_b_training_1000_gen3
mkdir -p "$OUT"
/home/jeefy/miniconda3/envs/unichess/bin/python \
  train/stage_b2.py \
  --data data/shards \
  --selfplay runs/stage_b_gen_1000_gen3 \
  --out "$OUT" \
  --ckpt runs/stage_b_training_2500_gen2/best.pt \
  --microbatch 4 --accum 16 \
  --workers 4 --threads 4 --mem-fraction 0.55 \
  --limit-games 500000 \
  --w-selfplay 0.85 --w-human 0.10 \
  --opening-loss-weight 0.25 \
  --mlh-log \
  > "$OUT/train.log" 2>&1
echo "TRAIN_DONE" >> "$OUT/train.log"
