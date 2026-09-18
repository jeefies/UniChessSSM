"""P1: 128-position Q→σ→π′ numerical trace.

Examines ~128 positions from generated self-play data to record:
- Raw Q range per position
- Normalized Q range
- σ(Q) vs policy logits magnitude
- π′ entropy / max prob (fp32)
- fp16 round-trip preservation
- Per-legal-action detailed breakdown for a few representative positions

Usage:
  python tools/trace_pipol.py runs/stage_b_val64
"""

import argparse
import json
import os
import sys
import struct
import traceback

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stateseq.data.gshards import V3ShardReader
from stateseq.model import SeqModel
from stateseq.features import encode
from stateseq.actions import move_to_action
from stateseq.conditions import TimeControlBucket
from stateseq.gumbel import (
    completed_q, sigma, pi_prime, normalize_q, Node, C_VISIT, C_SCALE,
    order_halving, N_SIMS, M0,
)
from stateseq.model_r import clone_cache

DEVICE = "cuda"
MAX_POSITIONS = 128
N_SEARCH_POSITIONS = 10  # positions to run full Gumbel search on

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--ckpt", default="runs/stage_a_20260915/best.pt")
    ap.add_argument("--out", default="runs/q_sigma_trace.json")
    args = ap.parse_args()

    # Load model
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model", ckpt)
    model = SeqModel(dropout=0.0)
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    reader = V3ShardReader(args.shard_dir)
    n_games = len(reader.meta_all)

    records = []
    detailed_positions = []
    positions_parsed = 0
    n_detailed = 0

    for gi in range(min(n_games, 100)):
        if positions_parsed >= MAX_POSITIONS:
            break
        game = reader.game(gi)
        actions = game["actions"]
        board = chess.Board()

        for ply in range(len(actions)):
            if positions_parsed >= MAX_POSITIONS:
                break
            action_id = int(actions[ply])

            # Encode and forward
            feats = encode(board, occurrence=0)
            f_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            tc_t = torch.tensor([int(TimeControlBucket.RAPID)], dtype=torch.long, device=DEVICE)
            elo_t = torch.tensor([2567.5], dtype=torch.float32, device=DEVICE)
            color_t = torch.tensor([1 if board.turn == chess.WHITE else 0], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                cache = model.initial_cache(1, device=DEVICE, dtype=torch.float32)
                logits, wdl, mlh, x, _ = model.step(f_t, tc_t, elo_t, color_t, cache)

            logits_np = logits.cpu().numpy()[0].astype(np.float32)
            wdl_np = wdl.cpu().numpy()[0].astype(np.float32)

            # Legal moves
            legal_actions = []
            for m in board.legal_moves:
                a = move_to_action(m)
                if a is not None:
                    legal_actions.append(a)
            legal_arr = np.array(legal_actions, dtype=np.int64)
            if len(legal_arr) == 0:
                # Make actual move and continue
                for m in board.legal_moves:
                    if move_to_action(m) == action_id:
                        board.push(m)
                        break
                continue

            # Masked logits
            logits_masked = logits_np[legal_arr]
            legal_logits_f32 = logits_masked.astype(np.float32)

            # Q for each legal action: wdl[0] - wdl[2] (same for all = root value)
            # For root, Q = wdl_np[0] - wdl_np[2] (same-same, the WDL is for the position, not per-move)
            q_root = float(wdl_np[0] - wdl_np[2])
            q_min = q_root
            q_max = q_root

            # Build node
            root = Node(legal=legal_arr.copy(), logits=legal_logits_f32.copy(), q=q_root)

            # Compute completedQ (no visits = v_mix = q_root)
            cq = completed_q(root, q_min, q_max).astype(np.float32)
            cq_norm = normalize_q(cq, q_min, q_max).astype(np.float32)
            sig = sigma(cq_norm, 0, C_VISIT, C_SCALE).astype(np.float32)

            # π′ = softmax(ℓ + σ(completedQ))
            pp = pi_prime(root, q_min, q_max, C_VISIT, C_SCALE).astype(np.float32)

            # π = softmax(ℓ)
            logits_stable = legal_logits_f32 - legal_logits_f32.max()
            pi_np = np.exp(logits_stable, dtype=np.float64)
            pi_np /= pi_np.sum()

            # Entropy of π′
            pp_eps = np.maximum(pp, 1e-10)
            entropy = -np.sum(pp * np.log(pp_eps).astype(np.float64))
            max_prob = float(pp.max())

            # σ magnitude vs logits magnitude
            sig_range = [float(sig.min()), float(sig.max())]
            logit_range = [float(legal_logits_f32.min()), float(legal_logits_f32.max())]
            q_values = np.array([q_root] * len(legal_arr))

            # fp16 round-trip test
            pp_fp32 = pp.astype(np.float32)
            pp_f16_bytes = b""
            for p in pp_fp32:
                pp_f16_bytes += struct.pack("<e", float(p))
            pp_recovered = np.array([struct.unpack("<e", pp_f16_bytes[i:i+2])[0] for i in range(0, len(pp_f16_bytes), 2)], dtype=np.float32)
            f16_max_diff = float(np.max(np.abs(pp_fp32 - pp_recovered)))
            f16_entropy = -np.sum(pp_recovered * np.log(np.maximum(pp_recovered, 1e-10)).astype(np.float64))
            f16_zeroed = int(np.sum(pp_recovered < 1e-10))
            f32_zeroed = int(np.sum(pp_fp32 < 1e-10))

            rec = {
                "gi": gi, "ply": ply, "n_legal": len(legal_arr),
                "fen": board.fen().split(" ")[0],
                "q_root": round(q_root, 6),
                "cq_range": [round(float(cq.min()), 6), round(float(cq.max()), 6)],
                "sigma_range": [round(sig_range[0], 6), round(sig_range[1], 6)],
                "logit_range": [round(logit_range[0], 6), round(logit_range[1], 6)],
                "sigma_vs_logit_ratio": abs(sig_range[1]) / max(abs(logit_range[1]), 1e-10),
                "pi_prime_entropy": round(float(entropy), 6),
                "pi_prime_max_prob": round(max_prob, 6),
                "f16_max_diff": round(f16_max_diff, 8),
                "f16_entropy": round(float(f16_entropy), 6),
                "f16_zeroed_count": f16_zeroed,
                "f32_zeroed_count": f32_zeroed,
            }
            records.append(rec)

            # Detailed record for first few positions
            if n_detailed < 3:
                per_action = []
                for i, aid in enumerate(legal_arr):
                    per_action.append({
                        "action_id": int(aid),
                        "logit": round(float(legal_logits_f32[i]), 6),
                        "completedQ": round(float(cq[i]), 6),
                        "sigma": round(float(sig[i]), 6),
                        "pi_prime": round(float(pp[i]), 6),
                    })
                detailed_positions.append({
                    "gi": gi, "ply": ply, "fen": board.fen().split(" ")[0],
                    "n_legal": len(legal_arr), "q_root": round(q_root, 6),
                    "per_action": per_action,
                })
                n_detailed += 1

            positions_parsed += 1

            # Make actual move
            for m in board.legal_moves:
                if move_to_action(m) == action_id:
                    board.push(m)
                    break

    # Aggregate stats
    entropies = [r["pi_prime_entropy"] for r in records]
    max_probs = [r["pi_prime_max_prob"] for r in records]
    n_legals = [r["n_legal"] for r in records]

    summary = {
        "n_positions": len(records),
        "entropy": {
            "mean": round(float(np.mean(entropies)), 6),
            "median": round(float(np.median(entropies)), 6),
            "min": round(float(np.min(entropies)), 6),
            "max": round(float(np.max(entropies)), 6),
            "p25": round(float(np.percentile(entropies, 25)), 6),
            "p75": round(float(np.percentile(entropies, 75)), 6),
        },
        "max_prob": {
            "mean": round(float(np.mean(max_probs)), 6),
            "median": round(float(np.median(max_probs)), 6),
            "p25": round(float(np.percentile(max_probs, 25)), 6),
            "p75": round(float(np.percentile(max_probs, 75)), 6),
            "p90": round(float(np.percentile(max_probs, 90)), 6),
            "p95": round(float(np.percentile(max_probs, 95)), 6),
        },
        "n_legal": {
            "mean": round(float(np.mean(n_legals)), 1),
            "min": int(np.min(n_legals)),
            "max": int(np.max(n_legals)),
            "median": int(np.median(n_legals)),
        },
        "f16_preservation": {
            "max_max_diff": round(max(r["f16_max_diff"] for r in records), 8),
            "mean_max_diff": round(float(np.mean([r["f16_max_diff"] for r in records])), 8),
            "positions_with_zeroed": sum(1 for r in records if r["f16_zeroed_count"] > r["f32_zeroed_count"]),
        },
        "sigma_vs_logit": {
            "mean_ratio": round(float(np.mean([r["sigma_vs_logit_ratio"] for r in records])), 4),
            "max_ratio": round(float(np.max([r["sigma_vs_logit_ratio"] for r in records])), 4),
            "sigma_always_dominant": sum(1 for r in records if r["sigma_vs_logit_ratio"] > 1.0),
        },
        "near_zero_entropy": sum(1 for e in entropies if e < 1e-6),
        "detailed_positions": detailed_positions,
    }

    # ---- 第 2 部分：在 N_SEARCH_POSITIONS 个局面上运行完整 Gumbel 搜索 ----
    # 验证搜索完成后的 π′ 锐化程度、Q 差异和 σ 主导性
    print("Running Gumbel search on %d positions..." % N_SEARCH_POSITIONS)
    search_records = []
    positions_searched = 0

    for gi in range(min(n_games, 50)):
        if positions_searched >= N_SEARCH_POSITIONS:
            break
        game = reader.game(gi)
        actions = game["actions"]
        board = chess.Board()
        cache = model.initial_cache(1, device=DEVICE, dtype=torch.float32)

        for ply in range(min(len(actions), 50)):
            if positions_searched >= N_SEARCH_POSITIONS:
                break
            action_id = int(actions[ply])

            feats = encode(board, occurrence=0)
            f_t = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
            tc_t = torch.tensor([int(TimeControlBucket.RAPID)], dtype=torch.long, device=DEVICE)
            elo_t = torch.tensor([2567.5], dtype=torch.float32, device=DEVICE)
            color_t = torch.tensor([1 if board.turn == chess.WHITE else 0], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                logits, wdl, mlh, x, cache_new = model.step(f_t, tc_t, elo_t, color_t, cache)
                cache = cache_new

            logits_np = logits.cpu().numpy()[0].astype(np.float32)
            wdl_np = wdl.cpu().numpy()[0].astype(np.float32)
            q_root = float(wdl_np[0] - wdl_np[2])
            x_np = x.cpu().numpy()[0]

            legal_actions = [a for m in board.legal_moves if (a := move_to_action(m)) is not None]
            legal_arr = np.array(legal_actions, dtype=np.int64)
            if len(legal_arr) == 0:
                for m in board.legal_moves:
                    if move_to_action(m) == action_id:
                        board.push(m)
                        break
                continue

            logits_masked = logits_np[legal_arr]

            # Define expand function for Gumbel search
            def make_expand(board_here, model_here, cache_here):
                def _expand(node, action):
                    mv = None
                    for m in board_here.legal_moves:
                        if move_to_action(m) == action:
                            mv = m
                            break
                    if mv is None:
                        return Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32), q=0.0, terminal=True)
                    board_here.push(mv)
                    feats_c = encode(board_here, occurrence=0)
                    f_c = torch.from_numpy(feats_c).float().unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        l_c, w_c, _, _, _ = model_here.step(f_c, tc_t, elo_t, color_t, cache_here)
                    board_here.pop()
                    q_c = float(w_c[0][0] - w_c[0][2])
                    legal_c = [a for m in board_here.legal_moves if (a := move_to_action(m)) is not None]
                    l_np = l_c.cpu().numpy()[0].astype(np.float32)
                    l_mask = np.full(1936, -3e4, dtype=np.float32)
                    l_mask[legal_c] = l_np[legal_c]
                    return Node(legal=np.array(legal_c, dtype=np.int64), logits=l_mask[np.array(legal_c)], q=q_c, depth=node.depth + 1, action=action)
                return _expand

            expand_fn = make_expand(board, model, clone_cache(cache))

            root_search = Node(legal=legal_arr.copy(), logits=logits_masked.copy(), q=q_root)
            result = order_halving(root_search, expand_fn, n_sims=N_SIMS, m0=M0, g=0.0,
                                    c_visit=C_VISIT, c_scale=C_SCALE)

            if result["action"] is None or root_search.n_total == 0:
                for m in board.legal_moves:
                    if move_to_action(m) == action_id:
                        board.push(m)
                        break
                continue

            qmin_search = float(result["qmin"])
            qmax_search = float(result["qmax"])

            # π′ from search (with completedQ from visit counts)
            pp_search = pi_prime(root_search, qmin_search, qmax_search, C_VISIT, C_SCALE).astype(np.float32)
            cq_search = completed_q(root_search, qmin_search, qmax_search).astype(np.float32)
            cq_norm_search = normalize_q(cq_search, qmin_search, qmax_search).astype(np.float32)
            sig_search = sigma(cq_norm_search, root_search.n_max, C_VISIT, C_SCALE).astype(np.float32)

            # π (raw policy)
            logits_stable_s = logits_masked - logits_masked.max()
            pi_raw = np.exp(logits_stable_s, dtype=np.float64)
            pi_raw /= pi_raw.sum()

            # π′ entropy
            pp_eps_s = np.maximum(pp_search, 1e-10)
            entropy_search = -np.sum(pp_search * np.log(pp_eps_s).astype(np.float64))
            max_prob_search = float(pp_search.max())

            # Compare π′ vs π
            kl_div = np.sum(pp_search * (np.log(np.maximum(pp_search, 1e-10)) - np.log(np.maximum(pi_raw, 1e-10))).astype(np.float64))

            search_records.append({
                "gi": gi, "ply": ply, "n_legal": len(legal_arr),
                "n_nodes": result["n_nodes"],
                "n_terminal": result["n_terminal"],
                "n_total_visits": root_search.n_total,
                "q_root": round(q_root, 6),
                "qmin": round(qmin_search, 6),
                "qmax": round(qmax_search, 6),
                "cq_range_search": [round(float(cq_search.min()), 6), round(float(cq_search.max()), 6)],
                "sigma_range_search": [round(float(sig_search.min()), 6), round(float(sig_search.max()), 6)],
                "logit_range": [round(float(logits_masked.min()), 6), round(float(logits_masked.max()), 6)],
                "sigma_vs_logit_ratio": abs(sig_search.max()) / max(abs(logits_masked.max()), 1e-10),
                "pi_prime_entropy": round(float(entropy_search), 6),
                "pi_prime_max_prob": round(max_prob_search, 6),
                "pi_entropy": round(float(-np.sum(pi_raw * np.log(np.maximum(pi_raw, 1e-10)))), 6),
                "kl_pi_prime_vs_pi": round(float(kl_div), 6),
                "selected_action": int(result["action"]),
                "n_sims_used": result["sims_used"],
                "budget_check": result["budget_check"],
                "survivors": result["survivors_per_round"],
            })

            positions_searched += 1

            # Make actual move
            for m in board.legal_moves:
                if move_to_action(m) == action_id:
                    board.push(m)
                    break

    # fp16 round-trip for search-based π′
    search_entropies = [r["pi_prime_entropy"] for r in search_records]
    search_max_probs = [r["pi_prime_max_prob"] for r in search_records]
    search_kls = [r["kl_pi_prime_vs_pi"] for r in search_records]

    summary["search_trace"] = {
        "n_positions": len(search_records),
        "entropy": {
            "mean": round(float(np.mean(search_entropies)), 6),
            "median": round(float(np.median(search_entropies)), 6),
            "min": round(float(np.min(search_entropies)), 6),
            "max": round(float(np.max(search_entropies)), 6),
            "p25": round(float(np.percentile(search_entropies, 25)), 6),
            "p75": round(float(np.percentile(search_entropies, 75)), 6),
        },
        "max_prob": {
            "mean": round(float(np.mean(search_max_probs)), 6),
            "median": round(float(np.median(search_max_probs)), 6),
            "p25": round(float(np.percentile(search_max_probs, 25)), 6),
            "p75": round(float(np.percentile(search_max_probs, 75)), 6),
            "p90": round(float(np.percentile(search_max_probs, 90)), 6),
            "p95": round(float(np.percentile(search_max_probs, 95)), 6),
        },
        "kl_pi_prime_vs_pi": {
            "mean": round(float(np.mean(search_kls)), 6),
            "median": round(float(np.median(search_kls)), 6),
            "p25": round(float(np.percentile(search_kls, 25)), 6),
            "p75": round(float(np.percentile(search_kls, 75)), 6),
        },
        "sigma_vs_logit_ratio": {
            "mean": round(float(np.mean([r["sigma_vs_logit_ratio"] for r in search_records])), 4),
            "max": round(float(np.max([r["sigma_vs_logit_ratio"] for r in search_records])), 4),
            "always_dominant": sum(1 for r in search_records if r["sigma_vs_logit_ratio"] > 1.0),
        },
        "near_zero_entropy": sum(1 for e in search_entropies if e < 1e-6),
        "records": search_records,
    }

    print("Search positions: %d" % len(search_records))
    print("Search entropy: mean=%.4f median=%.4f" % (
        summary["search_trace"]["entropy"]["mean"],
        summary["search_trace"]["entropy"]["median"]))
    print("Search max prob: mean=%.4f p90=%.4f p95=%.4f" % (
        summary["search_trace"]["max_prob"]["mean"],
        summary["search_trace"]["max_prob"]["p90"],
        summary["search_trace"]["max_prob"]["p95"]))
    print("Search sigma dominant ratio: mean=%.2f always_dominant=%d/%d" % (
        summary["search_trace"]["sigma_vs_logit_ratio"]["mean"],
        summary["search_trace"]["sigma_vs_logit_ratio"]["always_dominant"],
        len(search_records)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("Written to %s" % args.out)
    print("Positions: %d" % len(records))
    print("Entropy: mean=%.4f median=%.4f p25=%.4f p75=%.4f" % (
        summary["entropy"]["mean"], summary["entropy"]["median"],
        summary["entropy"]["p25"], summary["entropy"]["p75"]))
    print("Max prob: mean=%.4f median=%.4f p90=%.4f p95=%.4f" % (
        summary["max_prob"]["mean"], summary["max_prob"]["median"],
        summary["max_prob"]["p90"], summary["max_prob"]["p95"]))
    print("f16 max diff: %.2e mean diff: %.2e" % (
        summary["f16_preservation"]["max_max_diff"],
        summary["f16_preservation"]["mean_max_diff"]))
    print("Sigma dominant ratio: mean=%.2f max=%.2f always_dominant=%d/%d" % (
        summary["sigma_vs_logit"]["mean_ratio"],
        summary["sigma_vs_logit"]["max_ratio"],
        summary["sigma_vs_logit"]["sigma_always_dominant"],
        len(records)))


if __name__ == "__main__":
    main()