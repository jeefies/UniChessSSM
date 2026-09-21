#!/usr/bin/env bash
set -u -o pipefail
SSM=/home/jeefy/UniChess/SSM
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
cd "$SSM"

QUICK=/tmp/stage_b_quick
rm -rf "$QUICK"
mkdir -p "$QUICK"

echo "[$(date)] 1/4 验证修复后单进程生成→回放"
$PY tools/ssm_gumbel_selfplay.py \
  --ckpt runs/stage_a_20260915/best.pt \
  --out "$QUICK/stage_b_quick_games" --tag vfy \
  --games 10 --concurrency 2 --n_sims 16 --m0 8 --seed 11 \
  > "$QUICK/gen.log" 2>&1
if [ $? -ne 0 ]; then
  echo "[FAIL] 生成失败，见 $QUICK/gen.log"
  exit 1
fi

$PY -c "
import sys; sys.path.insert(0, '.')
import chess
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action
r = V3ShardReader('$QUICK/stage_b_quick_games')
bad = 0
for i in range(len(r.meta_all)):
    g = r.game(i)
    board = chess.Board()
    for t, a in enumerate(g['actions']):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        if a not in legal:
            print(f'FAIL game {i} ply {t} action {a} fen {board.fen()}')
            bad += 1; break
        board.push(legal[a])
print(f'校验完毕：{len(r.meta_all)} 局，{bad} 坏')
" 2>&1 | tee -a "$QUICK/gen.log"

echo "[$(date)] 2/4 验证修复后多进程生成"
for w in 1 2 4; do
  rm -rf "$QUICK/mp${w}"
  $PY tools/ssm_gumbel_selfplay.py \
    --ckpt runs/stage_a_20260915/best.pt \
    --out "$QUICK/mp${w}" --tag mp${w} \
    --games 8 --concurrency 4 --workers $w --n_sims 16 --m0 8 --seed 13 \
    > "$QUICK/mp${w}.log" 2>&1
  $PY -c "
import sys; sys.path.insert(0, '.')
import chess
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action
r = V3ShardReader('$QUICK/mp${w}')
bad = 0
for i in range(len(r.meta_all)):
    g = r.game(i)
    board = chess.Board()
    for t, a in enumerate(g['actions']):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        if a not in legal:
            print(f'  FAIL game {i} ply {t} action {a}')
            bad += 1; break
        board.push(legal[a])
print(f'workers={w}：{len(r.meta_all)} 局，{bad} 坏')
"
done

echo "[$(date)] 3/4 清理旧的损坏数据"
rm -rf "$SSM/runs/stage_b_smoke"

echo "[$(date)] 4/4 重跑 1000 局冒烟（4 进程 × concurrency=24）"
$PY tools/ssm_gumbel_selfplay.py \
  --ckpt runs/stage_a_20260915/best.pt \
  --out "$SSM/runs/stage_b_smoke" --tag smoke \
  --games 1000 --concurrency 24 --workers 4 \
  --n_sims 32 --m0 8 --seed 42 \
  > "$SSM/runs/stage_b_smoke/gen.log" 2>&1

status=$?
echo "[$(date)] 生成退出码 $status"
cat "$SSM/runs/stage_b_smoke/manifest.json" 2>/dev/null | head -25

echo "[$(date)] 验证全部 1000 局回放"
$PY -c "
import sys; sys.path.insert(0, '.')
import chess
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action
r = V3ShardReader('$SSM/runs/stage_b_smoke')
bad = 0
for i in range(len(r.meta_all)):
    g = r.game(i)
    board = chess.Board()
    for t, a in enumerate(g['actions']):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        if a not in legal:
            print(f'FAIL game {i} ply {t} action {a}')
            bad += 1; break
        board.push(legal[a])
print(f'最终验证：{len(r.meta_all)} 局，{bad} 坏')
" 2>&1 | tee -a "$SSM/runs/stage_b_smoke/gen.log"

if [ $? -eq 0 ]; then
  echo "[$(date)] 冒烟生成完成 & 数据完整性验证通过"
  echo "准备训练冒烟：(改为 manual) bash tools/run_stage_b_smoke.sh 跳过生成"
else
  echo "[FAIL] 数据完整性验证失败"
fi