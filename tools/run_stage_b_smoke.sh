#!/usr/bin/env bash
# Stage B 冒烟：1,000 局 Gumbel 自对弈生成 + ≤3 遍训练
#
# 用法：
#   bash tools/run_stage_b_smoke.sh
#
# 环境变量：
#   GAMES       默认 1000
#   CONCURRENCY 默认 1
#   CKPT        默认 runs/stage_a_20260915/best.pt
#   PYTHON      默认 /home/jeefy/miniconda3/envs/unichess/bin/python

set -u
set -o pipefail

SSM=/home/jeefy/UniChessSSM
RUNS=$SSM/runs/stage_b_smoke
PY=${PYTHON:-/home/jeefy/miniconda3/envs/unichess/bin/python}
GAMES=${GAMES:-1000}
CONCURRENCY=${CONCURRENCY:-1}
CKPT=${CKPT:-runs/stage_a_20260915/best.pt}
TAG=smoke

mkdir -p "$RUNS"

# 清理旧输出
rm -f "$RUNS"/shard-smoke-*
rm -f "$RUNS"/manifest.json

echo "[$(date)] 开始生成 $GAMES 局自对弈（concurrency=$CONCURRENCY）"
cd "$SSM" || exit 1

"$PY" tools/ssm_gumbel_selfplay.py \
  --ckpt "$CKPT" \
  --out "$RUNS" \
  --tag "$TAG" \
  --games "$GAMES" \
  --concurrency "$CONCURRENCY" \
  --seed 42 \
  > "$RUNS/generate.log" 2>&1

if [ $? -ne 0 ]; then
  echo "[$(date)] 生成失败，见 $RUNS/generate.log"
  exit 1
fi

echo "[$(date)] 生成完成，开始训练冒烟（≤3 遍）"
"$PY" train/stage_b2.py \
  --data data/shards \
  --selfplay "$RUNS" \
  --out "$RUNS/train_smoke" \
  --ckpt "$CKPT" \
  --microbatch 32 --accum 16 --workers 12 \
  --steps 20 \
  >> "$RUNS/generate.log" 2>&1

if [ $? -ne 0 ]; then
  echo "[$(date)] 训练失败，见 $RUNS/generate.log"
  exit 1
fi

echo "[$(date)] 冒烟完成，日志：$RUNS/generate.log"
