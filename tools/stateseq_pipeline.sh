#!/usr/bin/env bash
# Stage A 编排：月片到位即构建分片，全部构建完成后自动启动 1 epoch 正式训练。
# 用法：setsid nohup bash tools/stateseq_pipeline.sh > data/pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for month in 2026-08 2026-07 2026-06; do
  raw="data/raw/lichess_standard_${month}.pgn.zst"
  while [ ! -f "${raw}.done" ]; do
    sleep 120
  done
  if [ -f "data/shards/.built_${month}" ]; then
    echo "已构建 $month，跳过"
    continue
  fi
  echo "== 构建 $month 分片 $(date)"
  if $PY tools/stateseq_build_shards.py --pgn "$raw" --month "$month" --out data/shards >> data/pipeline.log 2>&1; then
    touch "data/shards/.built_${month}"
    echo "== $month 构建完成 $(date)"
  else
    echo "== $month 构建失败，退出（可重跑本脚本续作）"
    exit 1
  fi
done

echo "== 全部数据就绪，启动 Stage A 训练 $(date)"
RUN=runs/stage_a_20260915
if [ -f "$RUN/latest.pt" ]; then RESUME="--resume"; else RESUME=""; fi
setsid nohup "$PY" train/stage_a.py --data data/shards --out "$RUN" \
  --microbatch 32 --accum 16 --workers 12 \
  --save-every 1000 --val-every 1000 --log-every 50 \
  $RESUME > "$RUN.log" 2>&1 < /dev/null &
disown
echo "TRAINING_LAUNCHED pid=$!"
