"""Dual checkpoint evaluation on same 256-game validation set.

Evaluates Stage A best.pt and Stage B round1 best.pt on the same val set,
outputs metrics + sample IDs to a JSON file.

Usage:
  python tools/eval_dual_checkpoint.py runs/stage_b_val64 --out runs/dual_eval.json
"""

import argparse
import json
import os
import sys
import time

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.data.gshards import V3ShardReader
from stateseq.model import SeqModel
from stateseq.features import encode
from stateseq.actions import move_to_action
from stateseq.conditions import TimeControlBucket

DEVICE = "cuda"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model", ckpt)
    model = SeqModel(dropout=0.0)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model


def replay_board(actions_list, start_fen=None):
    """Replay sequence of action IDs on a chess board."""
    board = chess.Board()
    if start_fen:
        board = chess.Board(start_fen)
    for action_id in actions_list:
        found = False
        for m in board.legal_moves:
            if move_to_action(m) == int(action_id):
                board.push(m)
                found = True
                break
        if not found:
            return None
    return board


def evaluate(shard_dir, ckpt_path, label_str):
    model = load_model(ckpt_path)
    reader = V3ShardReader(shard_dir)
    n = len(reader.meta_all)

    total_positions = 0
    policy_ce_sum = 0.0
    value_ce_sum = 0.0
    wdl_prior_sum = np.zeros(3, dtype=np.float64)
    wdl_label_sum = np.zeros(3, dtype=np.float64)
    game_ids = []

    t0 = time.time()

    for gi in range(n):
        game = reader.game(gi)
        actions = game["actions"]
        n_plies = len(actions)
        result = int(game["meta"]["result"])
        game_ids.append((gi, n_plies, result))

        cache = model.initial_cache(1, device=DEVICE, dtype=torch.float32)
        board = chess.Board()
        label = np.zeros(3, dtype=np.float32)
        label[result] = 1.0
        wdl_label_sum += label

        for ply in range(n_plies):
            action_id = int(actions[ply])
            feats = encode(board, occurrence=0)

            f_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            tc_t = torch.tensor([int(TimeControlBucket.RAPID)], dtype=torch.long, device=DEVICE)
            elo_t = torch.tensor([2567.5], dtype=torch.float32, device=DEVICE)
            color_t = torch.tensor([1 if board.turn == chess.WHITE else 0], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                logits, wdl, mlh, x, cache_new = model.step(f_t, tc_t, elo_t, color_t, cache)
                cache = cache_new

            logits_np = logits.cpu().numpy()[0]
            wdl_np = wdl.cpu().numpy()[0]

            # Policy CE (BC on actual move)
            legal = [a for m in board.legal_moves if (a := move_to_action(m)) is not None]
            logits_masked = np.full(1936, -3e4, dtype=np.float32)
            logits_masked[legal] = logits_np[legal]
            logits_masked -= logits_masked.max()
            probs = np.exp(logits_masked, dtype=np.float64)
            probs /= probs.sum()
            action_prob = probs[action_id]
            policy_ce_sum += -np.log(max(action_prob, 1e-10))

            # Value CE
            wdl_probs = np.exp(wdl_np - wdl_np.max(), dtype=np.float64)
            wdl_probs /= wdl_probs.sum()
            value_ce_sum += -np.log(max(wdl_probs[result], 1e-10))
            wdl_prior_sum += wdl_probs

            total_positions += 1

            # Make actual move on board
            for m in board.legal_moves:
                if move_to_action(m) == action_id:
                    board.push(m)
                    break

    elapsed = time.time() - t0

    # Metrics
    policy_ce_avg = float(policy_ce_sum / max(total_positions, 1))
    value_ce_avg = float(value_ce_sum / max(total_positions, 1))
    wdl_prior_avg = wdl_prior_sum / max(total_positions, 1)
    wdl_label_dist = wdl_label_sum / max(total_positions, 1)

    # Constant baseline: predict label distribution for every position
    const_dist = np.array([wdl_label_dist[0], wdl_label_dist[1], wdl_label_dist[2]], dtype=np.float64)
    const_ce = float(-np.sum(wdl_label_dist * np.log(np.maximum(const_dist, 1e-10))))

    result = {
        "checkpoint_label": label_str,
        "n_games": n,
        "n_positions": total_positions,
        "policy_ce": round(policy_ce_avg, 6),
        "value_ce": round(value_ce_avg, 6),
        "value_ce_constant_baseline": round(const_ce, 6),
        "wdl_prior_mean": [round(float(wdl_prior_avg[0]), 6),
                           round(float(wdl_prior_avg[1]), 6),
                           round(float(wdl_prior_avg[2]), 6)],
        "wdl_label_distribution": [round(float(wdl_label_dist[0]), 6),
                                   round(float(wdl_label_dist[1]), 6),
                                   round(float(wdl_label_dist[2]), 6)],
        "elapsed_s": round(elapsed, 1),
        "positions_per_second": round(total_positions / elapsed, 1),
        "first_5_game_ids": [{"gi": gi, "n_plies": np, "result": r} for gi, np, r in game_ids[:5]],
        "last_5_game_ids": [{"gi": gi, "n_plies": np, "result": r} for gi, np, r in game_ids[-5:]],
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--out", default="runs/dual_eval.json")
    ap.add_argument("--ckpt-a", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--ckpt-b", default="runs/stage_b_training_round1/best.pt")
    args = ap.parse_args()

    ckpts = [
        (args.ckpt_a, "stage_a"),
        (args.ckpt_b, "stage_b_round1"),
    ]

    results = []
    for ckpt, label in ckpts:
        print("Evaluating %s on %s..." % (label, args.shard_dir))
        res = evaluate(args.shard_dir, ckpt, label)
        results.append(res)
        print("  policy_ce=%.4f value_ce=%.4f const_baseline=%.4f pos=%d games=%d" % (
            res["policy_ce"], res["value_ce"], res["value_ce_constant_baseline"],
            res["n_positions"], res["n_games"]))

    # Difference
    r0, r1 = results[0], results[1]
    diff = {
        "policy_ce_delta": round(r1["policy_ce"] - r0["policy_ce"], 6),
        "value_ce_delta": round(r1["value_ce"] - r0["value_ce"], 6),
    }
    print("Diff (round1 - stage_a): policy_ce=%.4f value_ce=%.4f" % (diff["policy_ce_delta"], diff["value_ce_delta"]))

    summary = {
        "shard_dir": args.shard_dir,
        "results": results,
        "diff": diff,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("Written to %s" % args.out)


if __name__ == "__main__":
    main()