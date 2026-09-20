#!/usr/bin/env bash
# 远端子进程监控：检测 gen2k 完成后自动执行后续步骤
set -u -o pipefail
SSM=/home/jeefy/UniChessSSM
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
MONITOR_LOG=/home/jeefy/UniChessSSM/runs/stage_b_gen2k/monitor.log

exec > "$MONITOR_LOG" 2>&1

echo "[$(date)] 监控启动，等待生成完成..."

while true; do
  if [ -f "$SSM/runs/stage_b_gen2k/manifest.json" ]; then
    echo "[$(date)] manifest 已出现，等待 worker 进程退出..."
    while pgrep -f "ssm_gumbel_selfplay.*gen2k" > /dev/null 2>&1; do
      sleep 10
    done
    sleep 5
    echo "[$(date)] 生成完成，开始数据验证..."
    
    cd "$SSM"
    $PY -c "
import json
m = json.load(open('runs/stage_b_gen2k/manifest.json'))
print(f'游戏: {m[\"games\"]}, 步: {m[\"steps\"]}')
print(f'终止分布: {m[\"gen\"][\"termination_reason_counts\"]}')
print(f'封顶率: {m[\"gen\"][\"truncated_rate\"]*100:.1f}%')
" 2>&1
    
    # 验证完整性
    $PY -c "
import sys; sys.path.insert(0, '.')
import chess
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action
r = V3ShardReader('runs/stage_b_gen2k')
bad = 0
for i in range(min(500, len(r.meta_all))):
    g = r.game(i)
    board = chess.Board()
    for t, a in enumerate(g['actions']):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        if a not in legal:
            bad += 1; break
        board.push(legal[a])
print(f'验证: {len(r.meta_all)}局中{min(500, len(r.meta_all))}局, {bad}坏')
" 2>&1
    
    echo "[$(date)] 验证完成。准备启动训练..."
    echo "训练命令（手动执行）："
    echo "PYTHONUNBUFFERED=1 $PY train/stage_b2.py --data data/shards --selfplay $SSM/runs/stage_b_gen2k --out $SSM/runs/stage_b_training_round1 --ckpt runs/stage_a_20260915/best.pt --microbatch 8 --accum 64 --workers 12"
    break
  fi
  sleep 30
done

echo "[$(date)] 监控结束"