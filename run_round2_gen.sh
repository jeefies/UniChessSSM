#!/usr/bin/env bash
set -e -o pipefail
cd /home/jeefy/UniChessSSM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
mkdir -p runs/stage_b_gen_round2
rm -f runs/stage_b_gen_round2/gen.log runs/stage_b_gen_round2/_w*/worker.log 2>/dev/null
/home/jeefy/miniconda3/envs/unichess/bin/python \
  tools/ssm_gumbel_selfplay.py \
  --ckpt runs/stage_a_20260915/best.pt \
  --out runs/stage_b_gen_round2 \
  --tag round2 --games 2500 --gen_id 2 \
  --ckpt_step 11 --g 1.0 \
  --concurrency 24 --workers 4 \
  > runs/stage_b_gen_round2/gen.log 2>&1
echo "GEN_DONE" >> runs/stage_b_gen_round2/gen.log