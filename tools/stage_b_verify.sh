#!/usr/bin/env bash
# Step 1: 综合验证脚本（终止原因核对 + 重建四格对照 + π′ 熵/KL + ID 集合一致性）
set -u -o pipefail
SSM=/home/jeefy/UniChessSSM
PY=/home/jeefy/miniconda3/envs/unichess/bin/python
cd "$SSM"
mkdir -p /tmp/stage_b_verify

LOG=/tmp/stage_b_verify/verify.log
exec > >(tee "$LOG") 2>&1

echo "=== [1/4] 终止原因交叉核对 ==="
$PY -c '
import sys, json, chess, collections
sys.path.insert(0, "/home/jeefy/UniChessSSM")
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action

r = V3ShardReader("/home/jeefy/UniChessSSM/runs/stage_b_smoke")
n = len(r.meta_all)
assert n == 1000, f"expected 1000 games, got {n}"

term_counts = collections.Counter()
truncated_u300 = 0
exactly_300_checkmate = 0
term_mismatch = 0
longest = 0
last_ply_modes = collections.Counter()  # termination at last ply
fifty_move_found = 0
threefold_found = 0

for i in range(n):
    g = r.game(i)
    meta = g["meta"]
    actions = g["actions"]
    plies = int(meta["n_plies"])
    recorded_term = int(meta["termination_reason"])
    is_trunc = bool(meta["is_truncated"])
    result = int(meta["result"])
    longest = max(longest, plies)

    board = chess.Board()
    occ = {}
    for t, a in enumerate(actions):
        a = int(a)
        legal = {move_to_action(m): m for m in board.legal_moves}
        mv = legal.get(a)
        if mv is None:
            print(f"FAIL: game {i} ply {t} action {a} illegal!")
            break
        board.push(mv)
        key = board.fen().split(" ")[0]
        occ[key] = occ.get(key, 0) + 1

    # 判断实际终局
    actual_over = board.is_game_over(claim_draw=True)
    if plies == 300 and board.is_checkmate():
        exactly_300_checkmate += 1

    # 检查 300 ply 末步是否已经将杀（不应被封顶覆盖）
    if plies == 300:
        last_ply_modes["truncated_or_end"] += 1
        if actual_over:
            last_ply_modes["actually_over_at_300"] += 1

    if plies < 300 and not actual_over and not is_trunc:
        # Game ended before cap but board thinks not over - check termination_reason
        if not is_trunc:
            term_mismatch += 1
            print(f"NOTE: game {i} ended at {plies} ply, board not over, recorded term={recorded_term}")

    term_counts[recorded_term] += 1
    if recorded_term == 5:  # truncated
        truncated_u300 += 1 if plies < 300 else 0

    # 手动检查五十步和三次重复
    if board.is_fifty_moves():
        fifty_move_found += 1
    if board.is_repetition(3):
        threefold_found += 1

print(f"总局数: {n}")
print(f"终止原因分布: {dict(term_counts)}")
print(f"最长局长: {longest} ply")
print(f"300 ply 末尾已将杀: {exactly_300_checkmate}")
print(f"300 ply 末步自然终局: {last_ply_modes}")
print(f"实际五十步和棋: {fifty_move_found}")
print(f"实际三次重复: {threefold_found}")
print(f"封顶但不足300ply: {truncated_u300}")
print(f"记录与实际不匹配: {term_mismatch}")

# 封顶局面占比
total_plies = int(sum(int(r.meta_all[i]["n_plies"]) for i in range(n)))
cap_plies = sum(300 for i in range(n) if int(r.meta_all[i]["termination_reason"]) == 5)
print(f"总ply: {total_plies}, 封顶ply: {cap_plies}, 占比: {cap_plies/max(total_plies,1)*100:.1f}%")
'

echo "=== [2/4] 重建四格对照 ==="
$PY -c '
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/jeefy/UniChessSSM")
from stateseq.model import SeqModel
from stateseq import losses
from stateseq.data.dataset import SequenceDataset
from stateseq.data.dataset_selfplay import SelfPlayDataset

device = "cuda"
model = SeqModel(dropout=0.0).to(device)
ckpt = torch.load("runs/stage_a_20260915/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt.get("model", ckpt))
model.eval()

def measure_recon(ds, label, n_batches=4, microbatch=8):
    total_rec = []
    total_acc = []
    total_dyn = []
    count = 0
    for vb in ds.val_batch(n_batches, microbatch, device):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            weights = losses.LossWeights(w_v=1.0)
            _, m = model.forward_train(vb["batch"], weights, step=0, total_steps=100,
                                       valid_mask=vb["valid"])
        total_rec.append(m["loss_recon"])
        total_acc.append(m.get("recon_whole_board_acc", 0.0))
        total_dyn.append(m.get("dyn_rel_err", 0.0))
        count += 1
    print(f"  {label}: recon={np.mean(total_rec):.4f} acc={np.mean(total_acc)*100:.1f}% dyn={np.mean(total_dyn):.4f} ({count} batches)")

# Stage A 需要从 human_ds 加载
try:
    human_ds = SequenceDataset("data/shards", workers=4, t_max=300)
    print(f"  human val 局数: {len(human_ds.val_indices)}")
    measure_recon(human_ds, "Stage A @ human val")
    human_ds.close()
except Exception as e:
    print(f"  human ds FAIL: {e}")

try:
    sp_ds = SelfPlayDataset("runs/stage_b_smoke", workers=4, t_max=300)
    print(f"  selfplay val 局数: {len(sp_ds.val_indices)}")
    measure_recon(sp_ds, "Stage A @ selfplay val")
    sp_ds.close()
except Exception as e:
    print(f"  selfplay ds FAIL: {e}")
'

echo "=== [3/4] π′ 目标熵与 KL 分解 ==="
$PY -c '
import sys, numpy as np, math
sys.path.insert(0, "/home/jeefy/UniChessSSM")
from stateseq.data.gshards import V3ShardReader

r = V3ShardReader("/home/jeefy/UniChessSSM/runs/stage_b_smoke")
n = min(100, len(r.meta_all))
all_entropies = []
all_ce = []
all_kl = []
total_legal_counts = []
total_probs_sum = []
total_ids = []
n_bad = 0

for i in range(n):
    g = r.game(i)
    metas = [g["meta"]]
    actions = [g["actions"]]
    for acts, probs in zip(g["pipol_actions"], g["pipol_probs"]):
        if len(acts) == 0:
            continue
        probs = np.asarray(probs, dtype=np.float64)
        ps = probs / probs.sum()
        entropy = -np.sum(ps * np.log(ps + 1e-30))
        all_entropies.append(entropy)
        all_legal_counts.append(len(acts))
        all_probs_sum.append(float(probs.sum()))
        all_ids.append(acts)

        # Check for duplicates
        if len(set(int(a) for a in acts)) != len(acts):
            n_bad += 1
            print(f"  DUPLICATE ID in game {i}")
        
        # Check ID uniqueness and whether they differ from the global legal set
        # (the action IDs should cover all positions in this position)

print(f"检查 {n} 局共 {len(all_entropies)} ply π′ 目标")
print(f"平均目标熵 H(π′): {np.mean(all_entropies):.4f} (中位数 {np.median(all_entropies):.4f})")
print(f"H(π′) P5/P25/P75/P95: {np.percentile(all_entropies, [5,25,75,95])}")
print(f"平均合法着数: {np.mean(all_legal_counts):.1f}")
print(f"概率和最大偏差: {max(abs(s-1.0) for s in all_probs_sum):.2e}")
print(f"ID 重复批: {n_bad}/{len(all_ids)}")
'

echo "=== [4/4] 动作 ID 集合一致性 ==="
$PY -c '
import sys, chess, numpy as np
sys.path.insert(0, "/home/jeefy/UniChessSSM")
from stateseq.data.gshards import V3ShardReader
from stateseq.actions import move_to_action, NUM_ACTIONS

r = V3ShardReader("/home/jeefy/UniChessSSM/runs/stage_b_smoke")
n = min(200, len(r.meta_all))
bad = 0
total_ply = 0
legal_count_mismatch = 0
id_set_mismatch = 0
pushed_not_in_pipol = 0

for i in range(n):
    g = r.game(i)
    actions = g["actions"]
    pipol_acts = g["pipol_actions"]
    pipol_probs = g["pipol_probs"]
    board = chess.Board()
    for t, a in enumerate(actions):
        a = int(a)
        legal_moves = list(board.legal_moves)
        legal_ids = {move_to_action(m) for m in legal_moves}
        
        # 检查实战动作合法性
        if a not in legal_ids:
            bad += 1
            break
        
        # 检查 pipol 目标 ID 集合是否与合法着 ID 一致
        if t < len(pipol_acts):
            pipol_set = set(int(x) for x in pipol_acts[t])
            if pipol_set != legal_ids:
                id_set_mismatch += 1
            if len(pipol_acts[t]) != len(legal_ids):
                legal_count_mismatch += 1
        
        board.push([m for m in legal_moves if move_to_action(m) == a][0])
        total_ply += 1

print(f"检查 {n} 局 {total_ply} ply")
print(f"实战动作非法: {bad}")
print(f"pipol ID 集合不匹配: {id_set_mismatch}")
print(f"pipol legal_count 不匹配: {legal_count_mismatch}")
print(f"结论: {\"ALL PASS\" if bad==0 and id_set_mismatch==0 and legal_count_mismatch==0 else \"ISSUES FOUND\"}")
' 2>&1

echo "=== DONE ==="