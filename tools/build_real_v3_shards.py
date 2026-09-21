"""Build a micro v3 shard dataset from data/sample_real.pgn into data/shards_real_v3."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import chess
import chess.pgn
import numpy as np

from stateseq.actions import move_to_action
from stateseq.conditions import time_control_bucket
from stateseq.data.gshards import (
    META_V3_DTYPE,
    RESULT_TO_LABEL,
    V3ShardWriter,
    encode_v3_pipol,
)


def build_real_v3_shards(
    pgn_path: str = "data/sample_real.pgn",
    out_dir: str = "data/shards_real_v3",
    tag: str = "real",
) -> dict:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    writer = V3ShardWriter(out_dir, tag=tag, shard_size=50_000)

    total_games = 0
    total_plies = 0
    total_branching = 0

    with open(pgn_path, "r", encoding="utf-8") as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            variant = game.headers.get("Variant", "Standard")
            if variant.lower() != "standard":
                continue

            result_str = game.headers.get("Result", "*")
            if result_str not in RESULT_TO_LABEL:
                continue

            w_elo = game.headers.get("WhiteElo")
            b_elo = game.headers.get("BlackElo")
            elo_vals = []
            if w_elo and w_elo.isdigit():
                elo_vals.append(float(w_elo))
            if b_elo and b_elo.isdigit():
                elo_vals.append(float(b_elo))
            elo_mean = float(np.mean(elo_vals)) if elo_vals else 2500.0
            elo_missing = 0 if len(elo_vals) == 2 else 1

            tc = game.headers.get("TimeControl", None)
            tc_b = int(time_control_bucket(tc))

            moves = list(game.mainline_moves())
            if not moves:
                continue

            board = game.board()
            actions: list[int] = []
            per_ply_actions: list[np.ndarray] = []
            per_ply_probs: list[np.ndarray] = []
            byte_offsets: list[int] = [0]
            curr_byte_offset = 0

            game_valid = True
            for move in moves:
                if move not in board.legal_moves:
                    game_valid = False
                    break

                # All legal moves at this position
                leg_moves = list(board.legal_moves)
                leg_count = len(leg_moves)
                total_branching += leg_count

                # Played action
                act_id = move_to_action(move)
                actions.append(act_id)

                # Legal actions
                leg_acts = np.array([move_to_action(m) for m in leg_moves], dtype=np.uint16)

                # Prior policy target with played move given high mass (e.g. 0.75 on played, rest uniform)
                # while strictly summing to 1.0
                if leg_count == 1:
                    probs = np.array([1.0], dtype=np.float32)
                else:
                    probs = np.full(leg_count, 0.25 / (leg_count - 1), dtype=np.float32)
                    played_idx = leg_moves.index(move)
                    probs[played_idx] = 0.75
                    # Renormalize to ensure exact 1.0 float sum
                    probs = probs / np.sum(probs)

                per_ply_actions.append(leg_acts)
                per_ply_probs.append(probs)

                # u16 legal_count (2 bytes) + leg_count * (u16 action + f16 prob = 4 bytes)
                ply_bytes = 2 + 4 * leg_count
                curr_byte_offset += ply_bytes
                byte_offsets.append(curr_byte_offset)

                board.push(move)

            if not game_valid or len(actions) == 0:
                continue

            n_plies = len(actions)
            meta = np.zeros((), dtype=META_V3_DTYPE)
            meta["n_plies"] = n_plies
            meta["tc_bucket"] = tc_b
            meta["result"] = RESULT_TO_LABEL[result_str]
            meta["elo_missing"] = elo_missing
            meta["elo_mean"] = elo_mean
            game_key = hashlib.sha256(f"real_lichess:{game_idx}".encode()).hexdigest()[:16]
            meta["game_key"] = game_key
            meta["gen_id"] = 0
            meta["ckpt_step"] = 0
            # Termination reason: 0=normal
            meta["termination_reason"] = 0
            meta["is_truncated"] = 0
            meta["start_type"] = 0
            meta["flags"] = 0

            pipol_blob = encode_v3_pipol(per_ply_actions, per_ply_probs)
            writer.add(
                meta=meta,
                actions=np.asarray(actions, dtype=np.uint16),
                pipol=pipol_blob,
                pipol_offset=np.asarray(byte_offsets, dtype=np.int32),
            )

            total_games += 1
            total_plies += n_plies
            game_idx += 1

    writer.flush()
    return {
        "games": total_games,
        "plies": total_plies,
        "avg_plies": total_plies / total_games if total_games else 0,
        "avg_branching": total_branching / total_plies if total_plies else 0,
        "out_dir": os.path.abspath(out_dir),
    }


if __name__ == "__main__":
    stats = build_real_v3_shards()
    print("Build finished successfully:", stats)
