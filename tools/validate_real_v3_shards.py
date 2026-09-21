"""Validation script for data/shards_real_v3 using V3ShardReader and chess.pgn."""

from __future__ import annotations

import sys
import chess
import chess.pgn
import numpy as np

from stateseq.actions import action_to_move, legal_mask, move_to_action
from stateseq.data.gshards import V3ShardReader, validate_v3_pipol


def validate():
    shard_dir = "data/shards_real_v3"
    pgn_path = "data/sample_real.pgn"

    print("Opening V3ShardReader...")
    reader = V3ShardReader(shard_dir)
    num_games = len(reader.meta_all)
    print(f"Reader loaded {num_games} games across {len(reader.metas)} shards.")

    with open(pgn_path, "r", encoding="utf-8") as f:
        pgn_games = []
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if g.headers.get("Variant", "Standard").lower() != "standard":
                continue
            if g.headers.get("Result", "*") not in ("1-0", "0-1", "1/2-1/2"):
                continue
            if not list(g.mainline_moves()):
                continue
            pgn_games.append(g)

    assert len(pgn_games) == num_games, f"PGN games ({len(pgn_games)}) != shard games ({num_games})"

    total_plies = 0
    total_branching = 0

    for idx in range(num_games):
        rec = reader.game(idx)
        meta = rec["meta"]
        actions = rec["actions"]
        pipol_acts = rec["pipol_actions"]
        pipol_probs = rec["pipol_probs"]

        g = pgn_games[idx]
        pgn_moves = list(g.mainline_moves())
        assert len(actions) == len(pgn_moves), f"Game {idx}: actions len {len(actions)} != pgn len {len(pgn_moves)}"
        assert int(meta["n_plies"]) == len(actions)

        board = g.board()
        legal_masks_game = []

        for t, (act_id, mv) in enumerate(zip(actions, pgn_moves)):
            # Verify legal moves in board
            curr_mask = legal_mask(board)
            legal_masks_game.append(curr_mask)

            # Check that played move matches action
            expected_act_id = move_to_action(mv)
            assert act_id == expected_act_id, f"Game {idx} ply {t}: act_id {act_id} != expected {expected_act_id}"

            # Check pipol actions and probabilities
            p_acts = pipol_acts[t]
            p_probs = pipol_probs[t]
            leg_moves_count = board.legal_moves.count()
            assert len(p_acts) == leg_moves_count, f"Game {idx} ply {t}: pipol len {len(p_acts)} != legal count {leg_moves_count}"
            assert len(p_probs) == leg_moves_count

            # Check played move is in pipol actions
            assert act_id in p_acts

            total_branching += leg_moves_count
            total_plies += 1
            board.push(mv)

        # Run validate_v3_pipol on the whole game
        validate_v3_pipol(
            pipol_acts,
            pipol_probs,
            n_plies=len(actions),
            legal_masks=np.array(legal_masks_game),
        )

    print("ALL VALIDATION CHECKS PASSED:")
    print(f"  - Verified {num_games} games")
    print(f"  - Total plies verified: {total_plies}")
    print(f"  - Mean branching factor: {total_branching / total_plies:.2f}")
    print(f"  - Exact move-by-move and legal-mask correspondence: 100% matched")


if __name__ == "__main__":
    validate()
