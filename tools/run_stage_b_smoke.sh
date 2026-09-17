#!/usr/bin/env bash
set -u
set -o pipefail

SSM=/home/jeefy/UniChessSSM
RUNS=$SSM/runs/stage_b_smoke
PY=${PYTHON:-/home/jeefy/miniconda3/envs/unichess/bin/python}
GAMES=${GAMES:-1000}
CONCURRENCY=${CONCURRENCY:-16}
N_SIMS=${N_SIMS:-32}
M0=${M0:-8}
CKPT=${CKPT:-runs/stage_a_20260915/best.pt}
TAG=smoke

mkdir -p "$RUNS"
rm -f "$RUNS"/shard-smoke-*
rm -f "$RUNS"/manifest.json

echo "[$(date)] 开始生成 $GAMES 局自对弈（concurrency=$CONCURRENCY）" | tee -a "$RUNS/generate.log"
cd "$SSM" || exit 1

echo "[$(date)] 启动自对弈生成器" | tee -a "$RUNS/generate.log"
PYTHONUNBUFFERED=1 "$PY" tools/ssm_gumbel_selfplay.py \
  --ckpt "$CKPT" \
  --out "$RUNS" \
  --tag "$TAG" \
  --games "$GAMES" \
  --concurrency "$CONCURRENCY" \
  --n_sims "$N_SIMS" \
  --m0 "$M0" \
  --seed 42 \
  >> "$RUNS/generate.log" 2>&1

status=$?
echo "[$(date)] 生成器退出状态: $status" | tee -a "$RUNS/generate.log"

if [ $status -ne 0 ]; then
  echo "[$(date)] 生成失败，见 $RUNS/generate.log"
  exit 1
fi

echo "[$(date)] 开始训练冒烟（≤3 遍）" | tee -a "$RUNS/generate.log"
PYTHONUNBUFFERED=1 "$PY" train/stage_b2.py \
  --data data/shards \
  --selfplay "$RUNS" \
  --out "$RUNS/train_smoke" \
  --ckpt "$CKPT" \
  --microbatch 32 --accum 16 --workers 12 \
  >> "$RUNS/generate.log" 2>&1

status=$?
echo "[$(date)] 训练器退出状态: $status" | tee -a "$RUNS/generate.log"

if [ $status -ne 0 ]; then
  echo "[$(date)] 训练失败，见 $RUNS/generate.log"
  exit 1
fi

echo "[$(date)] 冒烟完成，日志：$RUNS/generate.log"
