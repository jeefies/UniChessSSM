import sys
sys.path.insert(0, "/home/jeefy/UniChess")
import json
import chess
from Kit.api.types import SearchBudget
from Kit.rules.fast import outcome as fast_outcome, copy_board
from Kit.search.gumbel import _material_score, GumbelConfig, Gumbel
from Kit.registry import EngineSpec, build_player_factory

with open("/home/jeefy/UniChess/SSM/configs/match_gen4_vs_champ_gen1.json", "r", encoding="utf-8") as f:
    conf = json.load(f)

# 重放到第 70 步
with open("/home/jeefy/UniChess/SSM/runs/match_gen4_vs_champ_gen1.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        g = json.loads(line)
        if g.get("game") == 41:
            game41 = g
            break

board = chess.Board()
for uci in game41["opening"]:
    board.push_uci(uci)
for uci in game41["moves"][:70]:
    board.push_uci(uci)

print(f"局面: {board.fen()}")

# 构建黑方 player (使用 fast engine)
factory = build_player_factory(EngineSpec.from_dict(conf["a"]))
player = factory()
from Kit.api.types import GameStart

from Kit.api import immediate

cfg = player.cfg
print(f"Player cfg:")
print(f"  simulations: {cfg.simulations}")
print(f"  contempt: {cfg.contempt}")
print(f"  stalemate_penalty: {cfg.stalemate_penalty}")
print(f"  insufficient_penalty: {cfg.insufficient_penalty}")
print(f"  twofold_penalty: {cfg.twofold_penalty}")
print(f"  claim_draw: {cfg.claim_draw}")
