#!/usr/bin/env bash
# 评审待办④：SSM 自对照 arena —— MCTS-400(baseline) vs 变体（纯 policy / 屏蔽 value）
#
# 用法（在 /home/jeefy/UniChessSSM 下）：
#   MODE=pure   PAIRS=16 nohup bash tools/run_arena_selfplay.sh > runs/stage_a_20260915/arena_mcts_vs_pure.out 2>&1 &
#   MODE=neutral PAIRS=16 nohup bash tools/run_arena_selfplay.sh > runs/stage_a_20260915/arena_mcts_vs_neutral.out 2>&1 &
#
# MODE:
#   pure    -> engine-b = tools/ssm_uci_pure.sh（UNICHESS_PURE_POLICY=1，不搜索）
#   neutral -> engine-b = tools/ssm_uci_neutral.sh（UNICHESS_NEUTRAL_WDL=1，叶值中性）
# 双方共用同一 ssm_infer_server 实例（多客户端交错已验证）；结果 JSON 存 runs/。
set -u
MODE=${MODE:?用法: MODE=pure|neutral PAIRS=N bash tools/run_arena_selfplay.sh}
PAIRS=${PAIRS:-16}
SSM=/home/jeefy/UniChessSSM
RUNS=$SSM/runs/stage_a_20260915
SOCK=${UNICHESS_SSM_SOCK:-/tmp/unichess-ssm-infer.sock}
READY=$SOCK.ready
PIDFILE=$SOCK.pid

case "$MODE" in
  pure)   ENGINE_B=$SSM/tools/ssm_uci_pure.sh;   NAME_B=SSM-PurePolicy ;;
  neutral) ENGINE_B=$SSM/tools/ssm_uci_neutral.sh; NAME_B=SSM-NeutralWDL ;;
  *) echo "未知 MODE=$MODE（只支持 pure|neutral）"; exit 1 ;;
esac
OUT=${OUT:-$RUNS/arena_selfplay_mcts_vs_$MODE.json}

cd /home/jeefy/UniChess   # arena.py 的 sys.path 相对路径依赖这里

# GPU 可能与 stateseq_heldout_eval.py 等任务共享：expandable_segments 减少碎片浪费
# （OOM 实测 reserved-unallocated 达 3.25 GiB）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/jeefy/miniconda3/envs/unichess/bin/python
server_up() {
  [ -S "$SOCK" ] && "$PY" -c "import socket; s=socket.socket(socket.AF_UNIX); s.connect('$SOCK'); s.close()" 2>/dev/null
}
if ! server_up; then
  rm -f "$SOCK" "$READY"
  nohup "$PY" "$SSM/tools/ssm_infer_server.py" \
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
"$PY" /home/jeefy/UniChess/eval/arena.py \
  --engine-a "$SSM/tools/ssm_uci.sh" \
  --engine-b "$ENGINE_B" \
  --name-a SSM-MCTS400 --name-b "$NAME_B" \
  --pairs "$PAIRS" --workers 4 \
  --openings /home/jeefy/UniChess/eval/openings.txt \
  --movetime 1.0 --out "$OUT"
