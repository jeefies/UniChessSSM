#!/usr/bin/env bash
set -u -o pipefail
SSM=/home/jeefy/UniChessSSM
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
cd "$SSM"

echo "[$(date)] 清理旧的损坏数据"
rm -rf "$SSM/runs/stage_b_smoke"
mkdir -p "$SSM/runs/stage_b_smoke"

echo "[$(date)] 启动 1000 局生成（4 进程 × concurrency=24, n_sims=32）"
$PY tools/ssm_gumbel_selfplay.py \
  --ckpt runs/stage_a_20260915/best.pt \
  --out "$SSM/runs/stage_b_smoke" --tag smoke \
  --games 1000 --concurrency 24 --workers 4 \
  --n_sims 32 --m0 8 --seed 42 \
  >> "$SSM/runs/stage_b_smoke/gen.log" 2>&1

GENRC=$?
echo "[$(date)] 生成退出码 $GENRC"

echo "[$(date)] 验证 1000 局回放"
cat > /tmp/val_smoke.py << 'PYEOF'
import sys
sys.path.insert(0, "/home/jeefy/UniChessSSM")
import chess
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action
r = V3ShardReader("/home/jeefy/UniChessSSM/runs/stage_b_smoke")
bad = 0
for i in range(len(r.meta_all)):
    g = r.game(i)
    board = chess.Board()
    for t, a in enumerate(g["actions"]):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        if a not in legal:
            bad += 1
            print(f"FAIL game {i} ply {t} action {a} fen {board.fen()}")
            break
        board.push(legal[a])
print(f"验证完毕：{len(r.meta_all)} 局，{bad} 坏")
PYEOF
$PY /tmp/val_smoke.py 2>&1 | tee -a "$SSM/runs/stage_b_smoke/gen.log"

echo "[$(date)] 全部完成"
cat "$SSM/runs/stage_b_smoke/manifest.json"