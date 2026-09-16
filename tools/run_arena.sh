#!/usr/bin/env bash
# Stage A vs 旧 small champion 正式 arena（pairs=50，建议 nohup 后台跑）
# 用法: nohup bash tools/run_arena.sh > runs/stage_a_20260915/arena.out 2>&1 &
set -u
PAIRS=${PAIRS:-50}
OUT=${OUT:-/home/jeefy/UniChessSSM/runs/stage_a_20260915/arena_vs_smallchampion.json}
SSM=/home/jeefy/UniChessSSM
RUNS=$SSM/runs/stage_a_20260915
SOCK=${UNICHESS_SSM_SOCK:-/tmp/unichess-ssm-infer.sock}
READY=$SOCK.ready
PIDFILE=$SOCK.pid

cd /home/jeefy/UniChess   # arena.py 的 sys.path 与 ./unichess_gpu.sh 相对路径都依赖这里

# 单 GPU 推理服务器：4 个 UCI 客户端共享，避免每进程 Mamba autotune 并发踩踏
if [ ! -S "$SOCK" ]; then
  rm -f "$SOCK" "$READY"
  nohup /home/jeefy/miniconda3/envs/unichess/bin/python "$SSM/tools/ssm_infer_server.py" \
      --ckpt "$SSM/runs/stage_a_20260915/best.pt" --sock "$SOCK" --ready "$READY" \
      > "$RUNS/infer_server.log" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 120); do [ -f "$READY" ] && break; sleep 1; done
  if [ ! -f "$READY" ]; then echo "infer server 启动超时，见 $RUNS/infer_server.log"; exit 1; fi
fi
cleanup() {
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
  rm -f "$SOCK" "$READY" "$PIDFILE"
}
trap cleanup EXIT

export UNICHESS_SSM_REMOTE=1
export UNICHESS_SSM_SOCK="$SOCK"
export UNICHESS_MCTS=400
export UNICHESS_CKPT=runs/autoloop/models/small-champion.pt
export UNICHESS_SYZYGY=""   # 公平性：关闭旧引擎 Syzygy
/home/jeefy/miniconda3/envs/unichess/bin/python eval/arena.py \
  --engine-a "$SSM/tools/ssm_uci.sh" \
  --engine-b ./unichess_gpu.sh \
  --name-a SSM-StageA --name-b SmallChampion \
  --pairs "$PAIRS" --workers 4 \
  --openings /home/jeefy/UniChess/eval/openings.txt \
  --movetime 1.0 --out "$OUT"
