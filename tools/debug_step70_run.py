import sys
sys.path.insert(0, "/home/jeefy/UniChess")
import json
import chess
import numpy as np
from Kit.api.types import SearchBudget, GameStart, EvalRequest
from Kit.registry import EngineSpec, build_player_factory
from Kit.search.gumbel import completed_q, qtransform_completed, improved_policy

with open("/home/jeefy/UniChess/SSM/configs/match_gen4_vs_champ_gen1.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

with open("/home/jeefy/UniChess/SSM/runs/match_gen4_vs_champ_gen1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        if g.get("game") == 41:
            game41 = g
            break

factory = build_player_factory(EngineSpec.from_dict(conf["a"]))
player = factory()

def drive(gen):
    val = None
    while True:
        try:
            req = gen.send(val)
            if isinstance(req, EvalRequest):
                ev = req.evaluator
                val = ev.evaluate(req.payloads)
            else:
                val = None
        except StopIteration as stop:
            return stop.value

# 初始化并步进到第 70 步
board = chess.Board()
for uci in game41["opening"]:
    board.push_uci(uci)

drive(player.new_game(GameStart(color=chess.BLACK, seed=game41.get("seed_a", 0),
                                opening=game41["opening"], game_id="g41")))

for uci in game41["moves"][:70]:
    board.push_uci(uci)

# 获取 root
root, ids = drive(player._root(board))
print(f"根节点合法着法数: {len(root.moves)}")

# 查找 h2f2 和 e5f4 的 action 索引
idx_h2f2 = [i for i, m in enumerate(root.moves) if m.uci() == "h2f2"][0]
idx_e5f4 = [i for i, m in enumerate(root.moves) if m.uci() == "e5f4"][0]

print(f"\n[先验对比]")
print(f"  h2f2: logit={root.logits[idx_h2f2]:.3f}, prior_p={np.exp(root.logits[idx_h2f2])/np.sum(np.exp(root.logits)):.4f}")
print(f"  e5f4 (绝杀步!): logit={root.logits[idx_e5f4]:.3f}, prior_p={np.exp(root.logits[idx_e5f4])/np.sum(np.exp(root.logits)):.4f}")

# 执行搜索
rng = np.random.default_rng(player.seed + len(board.move_stack) * 100003)
res = drive(player.search.search(board, root=root, rng=rng, simulations=64))

print(f"\n[搜索后统计 (sims=64)]")
print(f"最终选出着法: {res.move}")
r = res.root
visited = np.flatnonzero(r.n > 0)
for idx in visited:
    mv = r.moves[idx]
    n = r.n[idx]
    q = r.q_sum[idx] / n
    is_checkmate = " (一步绝杀!)" if mv.uci() == "e5f4" else ""
    is_rep = " (走入二次重复!)" if mv.uci() == "h2f2" else ""
    print(f"  着法 {mv.uci()}{is_checkmate}{is_rep}: N={n}, Q={q:.4f}, logit={r.logits[idx]:.3f}")
