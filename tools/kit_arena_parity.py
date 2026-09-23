"""P3 验收：kit 驱动的 S 对弈与原版 ``ssm_gumbel_arena --batched --concurrency 1`` 逐局对照。

同一对权重、同一开局库与 seed、同一预算，两边各下 2×pairs 局，要求每局
``_pgn_fingerprint`` 完全相同，终局原因与扩展深度直方图逐局相同。kit 的前向数略少：
懒追赶每局省 1 次，终局叶子不再重放路径（kit 的 Gumbel 先判终局再调 Expander）。

只在并发 1 下成立：``SeqModel.step`` 的输出随批大小有 ~1e-5 的浮点差，并发 > 1 时两边
（原版驱动器与 kit）都不再逐位可复现，只有统计意义上的一致。

用法（远端，先 nvidia-smi 确认 GPU 空闲）::

    python tools/kit_arena_parity.py --ckpt-a runs/stage_b_training_2500_gen2/best.pt \\
        --ckpt-b runs/stage_b_training_1000_gen3/best.pt --pairs 4 --n_sims 32 --max_plies 40 \\
        --out /tmp/kit_parity/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import chess
import chess.pgn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.insert(0, KIT_ROOT)

import ssm_gumbel_arena as S  # noqa: E402
from stateseq import kit_adapter as ka  # noqa: E402
from stateseq.depth_hist import hist_merge, hist_summary  # noqa: E402
from unichess_kit.api import SearchBudget  # noqa: E402
from unichess_kit.pipelines.match import GameTask, play_game  # noqa: E402
from unichess_kit.rules.openings import OpeningBook, parse_line  # noqa: E402
from unichess_kit.rules.referee import StandardReferee  # noqa: E402
from unichess_kit.runtime import run_sync  # noqa: E402
from unichess_kit.search.gumbel import GumbelConfig  # noqa: E402


def _kit_fingerprint(rec: dict) -> str:
    board = chess.Board()
    for u in rec["opening"] + rec["moves"]:
        board.push_uci(u)
    return S._pgn_fingerprint({"pgn": str(chess.pgn.Game.from_board(board))})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--n_sims", type=int, default=32)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=40, help="开局之后的 ply 上限（S 口径）")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--openings-file", default=os.path.join(HERE, "data", "openings_200.txt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 开局：S 的 SAN 库 / 排列 与 kit 的 OpeningBook.plan 必须给出同一组开局
    library = S.load_openings_file(args.openings_file)
    plan_s = S.opening_plan(args.pairs, library, args.seed)
    book = OpeningBook.from_file(args.openings_file)
    plan_k = book.plan(args.pairs, args.seed)
    assert len(book) == len(library), (len(book), len(library))
    for (i_s, san), (i_k, uci) in zip(plan_s, plan_k):
        assert i_s == i_k and parse_line(san) == uci, (i_s, san, i_k, uci)

    models = [S.ArenaModel(args.ckpt_a), S.ArenaModel(args.ckpt_b)]

    # ---- 原版：批量驱动器，并发 1 ----
    s_args = argparse.Namespace(
        n_sims=args.n_sims, m0=args.m0, max_plies=args.max_plies, c_visit=S.C_VISIT,
        c_scale_a=S.C_SCALE, c_scale_b=S.C_SCALE, seed=args.seed, concurrency=1,
        sprt=False, games=2 * args.pairs, sprt_alpha=0.05, sprt_beta=0.05, sprt_min_games=64)
    t0 = time.time()
    s_games, _ = S.run_batched_arena(s_args, args.pairs, plan_s, t0, models=models)
    t_s = time.time() - t0

    # ---- kit：同一模型对象，逐局 run_sync（并发 1）----
    cfg = GumbelConfig(simulations=args.n_sims, m0=args.m0, g=0.0)
    ev_a = ka.SsmEvaluator(models[0].seq, "cuda", f"S:{os.path.abspath(args.ckpt_a)}")
    ev_b = ka.SsmEvaluator(models[1].seq, "cuda", f"S:{os.path.abspath(args.ckpt_b)}")
    fa, fb = ka.SsmPlayerFactory("A", ev_a, cfg), ka.SsmPlayerFactory("B", ev_b, cfg)
    t0 = time.time()
    k_games = []
    for p, (_, opening) in enumerate(plan_k):
        for k in range(2):
            g = 2 * p + k
            task = GameTask(game=g, pair=p, a_is_white=(k == 0), opening=tuple(opening),
                            seed_a=args.seed + g, seed_b=args.seed + g)
            referee = StandardReferee(max_plies=len(opening) + args.max_plies)
            n0 = ev_a.n_forwards + ev_b.n_forwards
            pa, pb = fa(), fb()
            rec = run_sync(play_game(task, pa, pb, referee, SearchBudget()))
            rec["n_forwards"] = ev_a.n_forwards + ev_b.n_forwards - n0
            rec["expand_depth_hist"] = hist_merge(pa.expand_hist, pb.expand_hist)
            k_games.append(rec)
    t_k = time.time() - t0

    rows, n_match = [], 0
    for gd, rec in zip(s_games, k_games):
        fp_s, fp_k = S._pgn_fingerprint(gd), _kit_fingerprint(rec)
        a_white = gd["white_ckpt_side"] == "A"
        same = (fp_s == fp_k and a_white == (rec["white"] == "A")
                and gd["termination_reason"] == rec["termination"]
                and gd["expand_depth_hist"] == rec["expand_depth_hist"])
        n_match += same
        rows.append({"game": rec["game"], "pair": rec["pair"], "a_white": a_white,
                     "match": same, "plies": rec["plies"],
                     "termination_s": gd["termination_reason"], "termination_k": rec["termination"],
                     "kit_forwards": rec["n_forwards"], "fp_s": fp_s, "fp_k": fp_k})
        if not same:
            print(f"[MISMATCH] game {rec['game']}\n  S  : {fp_s}\n  kit: {fp_k}")
    hist = [0]
    for rec in k_games:
        hist = hist_merge(hist, rec["expand_depth_hist"])
    summary = {"games": len(rows), "fingerprint_match": n_match,
               "distinct_games": len({r["fp_k"] for r in rows}),
               "s_elapsed_s": round(t_s, 1), "kit_elapsed_s": round(t_k, 1),
               "kit_forwards": ev_a.n_forwards + ev_b.n_forwards,
               "expand_depth": hist_summary(hist), "config": vars(args), "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"逐局指纹一致 {n_match}/{len(rows)}，不同对局 {summary['distinct_games']}，"
          f"S {t_s:.0f}s / kit {t_k:.0f}s")
    return 0 if n_match == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
