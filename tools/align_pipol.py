"""Review Task B: search target -> training input end-to-end alignment.

From positions in a val shard:
1. Run Gumbel search (same code path as generator)
2. Capture in-memory pi' arrays (L, N, Q, completedQ, sigma, pi')
3. Encode through v3 pipol format (encode_v3_pipol)
4. Decode back (decode_v3_pipol / V3ShardReader)
5. Compare: in-memory pi' vs pipol-decoded pi' for each ply

This isolates: floating-point path, quantization (f16), encoding/decoding,
and checks whether training receives the same targets as search produced.

Usage:
  python tools/align_pipol.py runs/stage_b_val64 --out runs/pipol_alignment.json
"""

import argparse
import json
import os
import sys
import struct

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.data.gshards import V3ShardReader, encode_v3_pipol, decode_v3_pipol
from stateseq.data.sequences import _board_key
from stateseq.model import SeqModel
from stateseq.model_r import clone_cache
from stateseq.actions import move_to_action
from stateseq.adapter import encode_board, get_terminal_q, wdl_logits_to_q
from stateseq.gumbel import (
    completed_q, sigma, pi_prime, normalize_q, Node, C_VISIT, C_SCALE,
    order_halving, N_SIMS, M0,
)

DEVICE = "cuda"
N_POSITIONS = 5  # 5 positions with full search

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--out", default="runs/pipol_alignment.json")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model", ckpt)
    model = SeqModel(dropout=0.0)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    reader = V3ShardReader(args.shard_dir)

    per_ply_results = []
    n_done = 0

    for gi in range(min(len(reader.meta_all), 10)):
        if n_done >= N_POSITIONS:
            break
        game = reader.game(gi)
        actions = game["actions"]
        board = chess.Board()
        cache = model.initial_cache(1, device=DEVICE, dtype=torch.float32)
        occurrences: dict = {}

        for ply in range(min(len(actions), 30)):
            if n_done >= N_POSITIONS:
                break
            action_id = int(actions[ply])

            key = _board_key(board)
            prior = occurrences.get(key, 0)
            occurrences[key] = prior + 1
            feats, tc_val, elo_std, color = encode_board(board, occurrence=prior)
            f_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            tc_t = torch.tensor([int(tc_val)], dtype=torch.long, device=DEVICE)
            elo_t = torch.tensor([float(elo_std)], dtype=torch.float32, device=DEVICE)
            color_t = torch.tensor([color], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                logits, wdl, mlh, x, cache_new = model.step(f_t, tc_t, elo_t, color_t, cache)
                cache = cache_new

            logits_np = logits.cpu().numpy()[0].astype(np.float32)
            wdl_np = wdl.cpu().numpy()[0].astype(np.float32)
            q_root = wdl_logits_to_q(wdl_np)

            legal_actions = [a for m in board.legal_moves if (a := move_to_action(m)) is not None]
            legal_arr = np.array(legal_actions, dtype=np.int64)
            if len(legal_arr) < 2:
                for m in board.legal_moves:
                    if move_to_action(m) == action_id:
                        board.push(m)
                        break
                continue

            logits_masked = logits_np[legal_arr]

            # Define expand for search
            root_for_search = Node(legal=legal_arr.copy(), logits=logits_masked.copy(), q=q_root)

            def make_expand(root_board, mod, root_cache, root_occ):
                def _expand(node, action):
                    b = root_board.copy()
                    cache_e = clone_cache(root_cache)
                    occ_e = dict(root_occ)
                    for a in node.path:
                        mv = None
                        for m in b.legal_moves:
                            if move_to_action(m) == a:
                                mv = m
                                break
                        if mv is None:
                            raise RuntimeError(f"路径重放动作 {a} 在 {b.fen()} 上不合法")
                        b.push(mv)
                        key_e = _board_key(b)
                        fe_e, tc_e, elo_e, color_e = encode_board(b, occ_e.get(key_e, 0))
                        with torch.no_grad():
                            _, _, _, _, cache_e = mod.step(
                                torch.from_numpy(fe_e).float().unsqueeze(0).to(DEVICE),
                                torch.tensor([int(tc_e)], dtype=torch.long, device=DEVICE),
                                torch.tensor([float(elo_e)], dtype=torch.float32, device=DEVICE),
                                torch.tensor([color_e], dtype=torch.long, device=DEVICE),
                                cache_e)
                        occ_e[key_e] = occ_e.get(key_e, 0) + 1
                    mv = None
                    for m in b.legal_moves:
                        if move_to_action(m) == action:
                            mv = m
                            break
                    if mv is None:
                        raise RuntimeError(f"动作 {action} 在 {b.fen()} 上不合法")
                    b.push(mv)
                    new_path = node.path + (action,)
                    if b.is_game_over(claim_draw=True) or not list(b.legal_moves):
                        return Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32),
                                    q=get_terminal_q(b), depth=node.depth + 1, action=action,
                                    path=new_path, terminal=True)
                    key_e = _board_key(b)
                    fe_e, tc_e, elo_e, color_e = encode_board(b, occ_e.get(key_e, 0))
                    with torch.no_grad():
                        l_c, w_c, _, _, _ = mod.step(
                            torch.from_numpy(fe_e).float().unsqueeze(0).to(DEVICE),
                            torch.tensor([int(tc_e)], dtype=torch.long, device=DEVICE),
                            torch.tensor([float(elo_e)], dtype=torch.float32, device=DEVICE),
                            torch.tensor([color_e], dtype=torch.long, device=DEVICE),
                            cache_e)
                    q_c = wdl_logits_to_q(w_c[0].cpu().numpy())
                    legal_c = [a2 for m2 in b.legal_moves if (a2 := move_to_action(m2)) is not None]
                    l_np_c = l_c.cpu().numpy()[0].astype(np.float32)
                    l_mask = np.full(1936, -3e4, dtype=np.float32)
                    l_mask[legal_c] = l_np_c[legal_c]
                    return Node(np.array(legal_c, dtype=np.int64), l_mask[np.array(legal_c)], q_c,
                                depth=node.depth + 1, action=action, path=new_path)
                return _expand

            expand_fn = make_expand(board, model, cache, occurrences)
            result = order_halving(root_for_search, expand_fn,
                                   n_sims=N_SIMS, m0=M0, g=1.0, seed=n_done)

            if result["action"] is None or root_for_search.n_total == 0:
                for m in board.legal_moves:
                    if move_to_action(m) == action_id:
                        board.push(m)
                        break
                continue

            qmin = float(result["qmin"])
            qmax = float(result["qmax"])

            # π' from completedQ (in-memory)
            cq = completed_q(root_for_search, qmin, qmax).astype(np.float32)
            pi_mem = pi_prime(root_for_search, qmin, qmax, C_VISIT, C_SCALE).astype(np.float32)

            # Encode through v3 pipol format
            pipol_bytes = encode_v3_pipol([root_for_search.legal.astype(np.uint16)], [pi_mem.astype(np.float16)])
            pipol_actions, pipol_probs = decode_v3_pipol(pipol_bytes, 1)
            pi_pipol = pipol_probs[0].astype(np.float32)
            pipol_action_ids = pipol_actions[0].astype(np.int64)

            # Align by action ID
            id_map = {int(a): i for i, a in enumerate(root_for_search.legal)}
            pi_pipol_aligned = np.zeros(len(pi_mem), dtype=np.float32)
            for a, p in zip(pipol_action_ids, pi_pipol):
                if int(a) in id_map:
                    pi_pipol_aligned[id_map[int(a)]] = p

            # Metrics
            max_diff = float(np.max(np.abs(pi_mem - pi_pipol_aligned)))
            mean_abs_diff = float(np.mean(np.abs(pi_mem - pi_pipol_aligned)))
            pi_mem_entropy = -np.sum(pi_mem * np.log(np.maximum(pi_mem, 1e-10)))
            pi_pipol_entropy = -np.sum(pi_pipol_aligned * np.log(np.maximum(pi_pipol_aligned, 1e-10)))

            # Compare L (prior logits) distribution vs pi'
            pi_mem_stable = pi_mem - pi_mem.max()
            pi_mem_renorm = np.exp(pi_mem_stable) / np.exp(pi_mem_stable).sum()

            per_ply_results.append({
                "gi": gi, "ply": ply, "n_legal": len(legal_arr),
                "in_memory_entropy": round(float(pi_mem_entropy), 6),
                "pipol_entropy": round(float(pi_pipol_entropy), 6),
                "entropy_diff": round(float(pi_mem_entropy - pi_pipol_entropy), 8),
                "max_prob_diff": round(max_diff, 8),
                "mean_abs_diff": round(mean_abs_diff, 8),
                "n_nodes": result["n_nodes"],
                "n_terminal": result["n_terminal"],
                "qmin": round(qmin, 6),
                "qmax": round(qmax, 6),
                "survivors": result["survivors_per_round"],
            })

            n_done += 1

            for m in board.legal_moves:
                if move_to_action(m) == action_id:
                    board.push(m)
                    break

    # Aggregate
    diffs = [r["max_prob_diff"] for r in per_ply_results]
    ent_diffs = [r["entropy_diff"] for r in per_ply_results]

    summary = {
        "n_positions": len(per_ply_results),
        "config": {"n_sims": N_SIMS, "m0": M0, "g": 1.0},
        "pipol_encoding": {
            "format": "u16 legal_count + u16 action_id + f16 prob (per ply)",
            "max_pi_diff": round(max(diffs), 8),
            "mean_pi_diff": round(float(np.mean(diffs)), 8),
            "max_entropy_diff": round(max(ent_diffs), 6),
            "mean_entropy_diff": round(float(np.mean(ent_diffs)), 8),
        },
        "per_ply": per_ply_results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("n=%d" % len(per_ply_results))
    print("max_pi_diff=%.2e mean_pi_diff=%.2e" % (summary["pipol_encoding"]["max_pi_diff"], summary["pipol_encoding"]["mean_pi_diff"]))
    print("max_entropy_diff=%.2e mean_entropy_diff=%.2e" % (summary["pipol_encoding"]["max_entropy_diff"], summary["pipol_encoding"]["mean_entropy_diff"]))


if __name__ == "__main__":
    main()