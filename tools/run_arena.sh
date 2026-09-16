#!/usr/bin/env bash
# Stage A vs 旧 small champion 正式 arena（pairs=50，建议 nohup 后台跑）
# 用法: nohup bash tools/run_arena.sh > runs/stage_a_20260915/arena.out 2>&1 &
set -u
PAIRS=${PAIRS:-50}
OUT=${OUT:-/home/jeefy/UniChessSSM/runs/stage_a_20260915/arena_vs_smallchampion.json}
cd /home/jeefy/UniChess   # arena.py 的 sys.path 与 ./unichess_gpu.sh 相对路径都依赖这里
export UNICHESS_MCTS=400
export UNICHESS_CKPT=runs/autoloop/models/small-champion.pt
export UNICHESS_SYZYGY=""   # 公平性：关闭旧引擎 Syzygy
exec /home/jeefy/miniconda3/envs/unichess/bin/python eval/arena.py \
  --engine-a /home/jeefy/UniChessSSM/tools/ssm_uci.sh \
  --engine-b ./unichess_gpu.sh \
  --name-a SSM-StageA --name-b SmallChampion \
  --pairs "$PAIRS" --workers 4 \
  --openings /home/jeefy/UniChess/eval/openings.txt \
  --movetime 1.0 --out "$OUT"
