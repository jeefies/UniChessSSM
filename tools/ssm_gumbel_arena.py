"""Gumbel arena（换代评测）——对局由 UniChessKit 的 ``pipelines.match`` 驱动。

S 侧只提供 Player（``stateseq.kit_adapter.SsmPlayer``：R cache 懒追赶 + 路径重算展开 +
Gumbel 顺序减半，g=0 确定性）；成对开局、跨局拼批、多进程、断点续跑、裁决都在 kit。
切换前与原实现（``ArenaModel`` / ``play_one_game`` / ``BatchedArenaDriver``，见 git 历史）
在并发 1 下逐局 ``_pgn_fingerprint`` 一致（真实权重 gen2 vs gen3：8/8 与 16/16）。

用法：
  python tools/ssm_gumbel_arena.py --ckpt-a runs/champion.pt --ckpt-b runs/challenger.pt \\
      --out runs/arena_ab --games 64 --workers 4 --concurrency 24

口径（与原实现相同）：
- 第 p 对两局同一开局，A 先执白；开局 = 带种子的排列（kit ``OpeningBook.plan``，与原
  ``opening_plan`` 逐对相同），g=0 下开局是对局多样性的唯一来源。
- ``--max_plies`` 只计开局之后的 ply；裁决 claim_draw 口径（与 ``classify_final_board`` 一致）。
- 计分按模型（A 视角），``arena_result`` 0 = A 胜、1 = 和、2 = B 胜。

与原实现的差别：
- ``--sprt`` 改用 kit 的五项式（逐对）GSPRT，H0: Elo 0 vs H1: Elo 35（≈ 原 p 0.50 vs 0.55），
  接受或拒绝都会停；原实现是逐局伯努利 LLR、只在拒绝 H1 时停。
- 结果逐局写入 ``<out>/kit_results.jsonl``：同一命令重跑会核对配置哈希并续跑。
- 出错整批停止（kit 语义），不再逐局记 anomaly（字段保留，恒为 None）。

输出：
  arena.json        — 聚合统计（含终止分布、扩展深度直方图、kit 的 Elo/五项式统计）
  games.jsonl       — 逐局诊断（每行 JSON，字段同原实现）
  model_ids.json    — 双方检查点参数标识 + 初始局面前向比较
  kit_results.jsonl — kit 的原始逐局结果（断点续跑用）
  expand_hist.jsonl — 逐局扩展深度直方图（续跑时已完成局的直方图从这里读回）
  scoring_test.json — 计分正向测试（--test-scoring）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import chess
import chess.pgn
import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

from stateseq.depth_hist import hist_merge, hist_summary  # noqa: E402
from stateseq.gumbel import C_SCALE, C_VISIT  # noqa: E402
from unichess_kit.pipelines.match import MatchConfig, SprtConfig, run_match  # noqa: E402
from unichess_kit.registry import EngineSpec  # noqa: E402
from unichess_kit.rules.openings import OpeningBook  # noqa: E402

# ---- 内置开局库（ECO 经典变例；开局文件缺失时的后备）----
OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5",
    "d4 d5 c4 e6",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",
    "d4 Nf6 c4 g6 Nc3 Bg7",
    "e4 e6 d4 d5",
    "d4 Nf6 c4 e6 Nf3 Bb4+",
    "e4 c6 d4 d5",
    "c4 e5",
    "Nf3 Nf6 c4 g6",
    "d4 d5 c4 c6",
    "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "d4 Nf6 c4 c5",
    "e4 e5 Nf3 Nf6",
    "e4 g6",
    "d4 f5",
    "e4 e5 f4",
]

SPRT_ELO1 = 35.0  # ≈ 原实现 H1 的 p=0.55：-400·log10(1/0.55 − 1) = 34.9

# 选择 "a"/"b" 引擎时 kit 按 EngineSpec 加载（多进程时每个 worker 各自加载）
FACTORY = "stateseq.kit_adapter:make_player_factory"


def _pgn_fingerprint(gd: dict) -> str:
    """对局指纹：棋步序列（忽略 Event/Site 等元数据），用于统计"实际不同对局"数。"""
    body = str(gd.get("pgn", "")).split("\n\n")[-1]
    return " ".join(body.split())


def _result_str(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "*"
    if outcome.winner is None:
        return "½-½"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def _model_id(state_dict: dict) -> str:
    keys = sorted(state_dict.keys())
    buf = bytearray()
    for k in keys[:4]:
        t = state_dict[k]
        head = t.flatten()[:16].detach().cpu().numpy().astype(np.float32).view(np.uint8).tobytes()
        buf.extend(head)
    return hashlib.sha256(buf).hexdigest()[:16]


# ---- 计分正向测试 ----

SCORING_TEST_FENS = [
    ("k6R/8/1K6/8/8/8/8/8 b - - 0 1", chess.WHITE),
    ("K6r/8/1k6/8/8/8/8/8 w - - 0 1", chess.BLACK),
    ("k7/Q7/1K6/8/8/8/8/8 b - - 0 1", chess.WHITE),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", None),
]


def _run_scoring_test(out_dir: str) -> None:
    print("=== 计分正向测试 ===")
    os.makedirs(out_dir, exist_ok=True)
    games_log = []
    passed = 0
    for fen, expected in SCORING_TEST_FENS:
        board = chess.Board(fen)
        outcome = board.outcome(claim_draw=True)
        result_str = _result_str(board)
        winner = outcome.winner if outcome is not None else None
        ok = (winner == expected) or (winner is None and expected is None)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        games_log.append({"fen": fen, "expected": str(expected), "result": result_str,
                          "winner": str(winner), "status": status})
        print("  [%s] %s → %s" % (status, fen.split("/")[0], result_str))
    manifest = {"test": "scoring_positive_test", "total": len(SCORING_TEST_FENS),
                "passed": passed, "failed": len(SCORING_TEST_FENS) - passed, "games": games_log}
    with open(os.path.join(out_dir, "scoring_test.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("=== %d/%d PASS ===" % (passed, len(SCORING_TEST_FENS)))


# ---- 开局库 ----

def resolve_openings(path: str, out_dir: str) -> str:
    """开局文件路径；为空或不存在时把内置库写到 out_dir 并返回其路径。"""
    if path and os.path.exists(path):
        return path
    if path:
        print(f"[warn] 开局文件 {path} 不存在，退回内置 {len(OPENINGS)} 条", flush=True)
    builtin = os.path.join(out_dir, "openings_builtin.txt")
    with open(builtin, "w", encoding="utf-8") as fh:
        fh.write("\n".join(OPENINGS) + "\n")
    return builtin


# ---- kit 记录 → 原 games.jsonl 字段 ----

_ARENA_RESULT = {1.0: 0, 0.5: 1, 0.0: 2}  # A 得分 → arena_result（A 视角 0 胜 / 1 和 / 2 负）


def game_dict(rec: dict, ckpts: dict, opening_ids: dict, expand_hist: list | None) -> dict:
    board = chess.Board()
    for u in rec["opening"] + rec["moves"]:
        board.push_uci(u)
    white, black = rec["white"], ("B" if rec["white"] == "A" else "A")
    return {
        "opening_id": opening_ids.get(rec["pair"]),
        "seed": None,
        "ckpt_white": ckpts[white],
        "ckpt_black": ckpts[black],
        "n_plies": len(rec["moves"]),
        "termination_reason": rec["termination"],
        "is_truncated": rec["termination"] == "truncated",
        "board_result": _result_str(board),
        "arena_result": _ARENA_RESULT[rec["a_score"]],
        "anomaly": None,
        "pgn": str(chess.pgn.Game.from_board(board)),
        "expand_depth_hist": expand_hist or [],
        "game_idx": rec["game"],
        "pair_idx": rec["pair"],
        "white_ckpt_side": white,
        "black_ckpt_side": black,
        "elapsed_s": rec.get("elapsed_s"),
    }


def _aggregate_results(games_log, args, sprt_info=None) -> dict:
    wins_a = sum(1 for g in games_log if g["arena_result"] == 0)
    wins_b = sum(1 for g in games_log if g["arena_result"] == 2)
    draws = len(games_log) - wins_a - wins_b
    assert all(g["arena_result"] in (0, 1, 2) for g in games_log), \
        "Scoring invariant violated: W_A+W_B+D != N"
    score_a = wins_a + 0.5 * draws
    term_counts: dict = {}
    for gd in games_log:
        t = gd["termination_reason"]
        term_counts[t] = term_counts.get(t, 0) + 1
    truncated = term_counts.get("truncated", 0)
    anomalies = sum(1 for g in games_log if g["anomaly"])
    # 实际不同对局数（g=0 下同开局+同颜色的对局逐字节重复）
    distinct = len({_pgn_fingerprint(g) for g in games_log})
    res = {
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "c_visit": args.c_visit, "c_scale_a": args.c_scale_a, "c_scale_b": args.c_scale_b,
        "total_games": len(games_log),
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "score_a": score_a,
        "score_a_percent": score_a / max(len(games_log), 1) * 100,
        "n_sims": args.n_sims, "m0": args.m0,
        "elapsed_s": 0.0,
        "termination": term_counts,
        "truncated_rate": truncated / max(len(games_log), 1),
        "anomalies": anomalies,
        "workers": getattr(args, "workers", 1),
        # g=0 确定性对局：distinct_games 才是有效样本量，重复只会虚增权重
        "distinct_games": distinct,
        "duplicate_rate": 1.0 - distinct / max(len(games_log), 1),
    }
    hist: list[int] = []
    for g in games_log:
        hist = hist_merge(hist, g.get("expand_depth_hist"))
    res["expand_depth"] = hist_summary(hist)
    if sprt_info:
        res["sprt"] = sprt_info
    return res


def _read_kit_games(path: str) -> list:
    games = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("type") == "game":
                    games.append(rec)
    return sorted(games, key=lambda r: r["game"])


def _compare_models(ckpt_a: str, ckpt_b: str) -> dict:
    """双方参数标识 + 初始局面前向比较（模型是否真的不同）。比较完即释放显存。"""
    import torch

    from stateseq import kit_adapter as ka

    ids, outs = {}, {}
    for side, path in (("a", ckpt_a), ("b", ckpt_b)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ids[side] = _model_id(ck.get("model", ck))
        ev = ka.SsmEvaluator.from_checkpoint(path, "cuda")
        payload = ka.encode_payload(chess.Board(), 0, ev.initial_cache())
        (logits, wdl, _), = ev.evaluate([payload])
        p = np.exp(logits - logits.max())
        outs[side] = (p / p.sum(), wdl)
        del ev
    torch.cuda.empty_cache()
    policy_diff = float(np.max(np.abs(outs["a"][0] - outs["b"][0])))
    wdl_diff = float(np.max(np.abs(outs["a"][1] - outs["b"][1])))
    return {"a": {"hash": ids["a"]}, "b": {"hash": ids["b"]}, "same_hash": ids["a"] == ids["b"],
            "forward_comparison": {"max_policy_prob_diff": policy_diff,
                                   "max_wdl_diff": wdl_diff,
                                   "models_differ_functionally": policy_diff > 1e-6}}


def engine_spec(ckpt: str, label: str, args, c_scale: float,
                server_dir: str | None = None) -> EngineSpec:
    kwargs = {"checkpoint": os.path.abspath(ckpt), "name": label,
              "simulations": args.n_sims, "m0": args.m0, "g": 0.0,
              "c_visit": args.c_visit, "c_scale": c_scale,
              "engine": getattr(args, "engine", "server")}
    if kwargs["engine"] == "server":        # 块大小、精度决定数值 → 进配置哈希
        kwargs["server_chunk"] = server_chunk(args)
        if server_precision(args) != "fp32":    # fp32 不写：旧运行的配置哈希不变，可续跑
            kwargs["server_precision"] = server_precision(args)
    # 槽数（淘汰 + 重算逐位不变）与服务目录（每次运行不同）都不影响结果 → runtime：
    # 照样传给工厂，但不进配置哈希，续跑时可以不同
    runtime = {"pool_slots": resolve_pool_slots(args)}
    if server_dir:
        runtime["server_dir"] = server_dir
    return EngineSpec(factory=FACTORY, root=HERE, label=label, kwargs=kwargs, runtime=runtime)


def server_chunk(args) -> int:
    from stateseq.fast_eval import FIXED_CHUNK
    return int(getattr(args, "server_chunk", 0) or FIXED_CHUNK)


def server_precision(args) -> str:
    return getattr(args, "server_precision", None) or "fp32"


def resolve_pool_slots(args) -> int:
    """--pool-slots 0 = 按并发自动（每进程 A/B 共用一套槽，每局钉住两方当前局面）。"""
    slots = int(getattr(args, "pool_slots", 0) or 0)
    if slots > 0 or getattr(args, "engine", "server") == "reference":
        return slots or 2048
    from stateseq.fast_eval import auto_pool_slots
    return auto_pool_slots(args.concurrency, args.m0, holds_per_game=2)


def run_arena(args) -> tuple[list, dict | None, dict]:
    """跑评测 → (games_log, sprt_info, kit 汇总)。"""
    os.makedirs(args.out, exist_ok=True)
    openings = resolve_openings(args.openings_file, args.out)
    num_pairs = args.games // 2
    book = OpeningBook.from_file(openings)
    plan = book.plan(num_pairs, args.seed)
    opening_ids = {p: int(i) for p, (i, _) in enumerate(plan)}
    if num_pairs > len(book):
        print(f"[warn] 对数 {num_pairs} > 开局数 {len(book)}：开局必然重复，"
              f"g=0 下同开局 pair 的对局逐字节相同，统计功效被稀释", flush=True)
    print(f"[openings] 库规模 {len(book)}，对数 {num_pairs}，"
          f"实际不同对局数 {len(set(opening_ids.values())) * 2}", flush=True)

    sprt = (SprtConfig(elo0=0.0, elo1=SPRT_ELO1, alpha=args.sprt_alpha, beta=args.sprt_beta,
                       min_pairs=max(1, args.sprt_min_games // 2)) if args.sprt else None)
    cfg = MatchConfig(pairs=num_pairs, seed=args.seed, max_plies=args.max_plies,
                      concurrency=args.concurrency, openings=openings, sprt=sprt,
                      workers=max(1, args.workers), max_plies_after_opening=True)
    server = None
    if getattr(args, "engine", "server") == "server":
        from stateseq.gpu_server import GpuServer

        server = GpuServer([os.path.abspath(args.ckpt_a), os.path.abspath(args.ckpt_b)],
                           n_clients=max(1, args.workers),
                           slots_per_client=resolve_pool_slots(args),
                           chunk=server_chunk(args),
                           precision=server_precision(args)).start()
        print(f"[gpu_server] {server.dir} 每客户端槽 {server.meta['slots_per_client']}", flush=True)
    server_dir = server.dir if server is not None else None
    spec_a = engine_spec(args.ckpt_a, "A", args, args.c_scale_a, server_dir)
    spec_b = engine_spec(args.ckpt_b, "B", args, args.c_scale_b, server_dir)

    kit_path = os.path.join(args.out, "kit_results.jsonl")
    hist_path = os.path.join(args.out, "expand_hist.jsonl")
    done_hists: dict = {}
    if os.path.exists(kit_path) and os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done_hists[row["game"]] = row["hist"]
    hist_fh = open(hist_path, "a" if os.path.exists(kit_path) else "w", encoding="utf-8")
    hists: dict = {}

    def observer(ev: dict) -> None:
        if ev.get("type") == "move":
            h = (ev.get("info") or {}).get("expand_hist")
            if h:
                hists[ev["game"]] = hist_merge(hists.get(ev["game"], []), h)

    t0 = time.time()

    def progress(rec: dict, records: list) -> None:
        # 多进程时 worker 的逐步事件先于该局结果到达（同一队列），此时该局直方图已完整
        h = hists.pop(rec["game"], [])
        done_hists[rec["game"]] = h
        hist_fh.write(json.dumps({"game": rec["game"], "hist": h}) + "\n")
        hist_fh.flush()
        n = len(records)
        if n % 8 == 0 or n == 2 * num_pairs:
            w = sum(1 for r in records if r["a_score"] == 1.0)
            lo = sum(1 for r in records if r["a_score"] == 0.0)
            print("  [%.0fs] games %d/%d  A +%d =%d -%d" % (time.time() - t0, n, 2 * num_pairs,
                                                            w, n - w - lo, lo), flush=True)

    try:
        summary = run_match(cfg, spec_a=spec_a, spec_b=spec_b, out_path=kit_path,
                            progress=progress, observer=observer)
    finally:
        hist_fh.close()
        if server is not None:
            server.stop()
    if server is not None:
        summary["gpu_server"] = server.final_stats
    ckpts = {"A": args.ckpt_a, "B": args.ckpt_b}
    games_log = [game_dict(r, ckpts, opening_ids, done_hists.get(r["game"]))
                 for r in _read_kit_games(kit_path)]
    sprt_info = None
    if "sprt" in summary:
        s = summary["sprt"]
        played = len(games_log)
        sprt_info = {**s, "early_stopped": bool(summary.get("stopped_by_sprt")),
                     "stop_reason": s.get("verdict"), "games_played": played,
                     "games_planned": 2 * num_pairs,
                     "compute_saved_games": 2 * num_pairs - played,
                     "compute_saved_percent": (2 * num_pairs - played) / (2 * num_pairs) * 100.0}
    return games_log, sprt_info, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a")
    ap.add_argument("--ckpt-b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=8,
                    help="已废弃：曾误用于裁剪开局数（64 局实际只跑出 16 个不同对局）；"
                         "开局多样性由 --openings-file 决定")
    ap.add_argument("--openings-file", default=os.path.join(HERE, "data", "openings_200.txt"),
                    help="开局库文件（每行一条 SAN/UCI 序列）；留空或缺失则用内置 16 条。"
                         "g=0 的 arena 无随机性，开局多样性是对局多样性唯一来源")
    ap.add_argument("--n_sims", type=int, default=256)
    ap.add_argument("--m0", type=int, default=16)
    ap.add_argument("--max_plies", type=int, default=300, help="开局之后的 ply 上限")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--workers", type=int, default=1, help="并行工作进程数（每进程各自攒批）")
    ap.add_argument("--batched", action="store_true",
                    help="已废弃（恒为跨局攒批，保留以兼容旧脚本）")
    ap.add_argument("--concurrency", type=int, default=24,
                    help="每进程的并发局数（批大小≈该值×模型数，默认 24）")
    ap.add_argument("--test-scoring", action="store_true")
    ap.add_argument("--engine", choices=("server", "fast", "reference"), default="server",
                    help="server：共享 GPU 服务跨进程拼批（P4 默认；批不变，结果与 workers/concurrency "
                         "无关、逐位可复现）；fast：每进程各自的 GPU 槽池 + CUDA graph；"
                         "reference：原 cat/split + 重放 + 串行（与 P3 逐位对照）")
    ap.add_argument("--pool-slots", type=int, default=0,
                    help="server/fast：每进程（每客户端）状态槽数（A/B 共用；每槽约 1 MB，不足时 LRU 淘汰"
                         " + 重算）。0 = 按 concurrency×(2+m0) 自动")
    ap.add_argument("--server-chunk", type=int, default=0,
                    help="server：批不变块大小（行，取 fast_eval.BUCKETS 之一，如 32/64/128）。块越小低负载时浪费越少、满载时吞吐越低；"
                         "块大小决定数值（进配置哈希）。0 = 默认 FIXED_CHUNK")
    ap.add_argument("--server-precision", choices=("fp32", "tf32"), default="fp32",
                    help="server：前向精度。tf32 矩阵乘走 TF32 张量核（GPU 约快 1.4 倍，与 fp32 有小偏差，"
                         "见 design-deviations §2.6）；决定数值（非 fp32 时进配置哈希）")
    ap.add_argument("--c_visit", type=float, default=C_VISIT, help="双方共用的 c_visit")
    ap.add_argument("--c_scale_a", type=float, default=C_SCALE, help="A 侧 c_scale")
    ap.add_argument("--c_scale_b", type=float, default=C_SCALE, help="B 侧 c_scale")
    ap.add_argument("--sprt", action="store_true",
                    help=f"五项式 GSPRT 早停（H0 Elo 0 vs H1 Elo {SPRT_ELO1:g}）")
    ap.add_argument("--sprt-min-games", type=int, default=64, help="SPRT 判决最少局数")
    ap.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT Type I error bound")
    ap.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT Type II error bound")
    args = ap.parse_args()

    if args.test_scoring:
        _run_scoring_test(args.out)
        return
    if not args.ckpt_a or not args.ckpt_b:
        ap.error("需要 --ckpt-a 与 --ckpt-b")
    if args.games < 2 or args.games % 2:
        ap.error("--games 必须是 >= 2 的偶数（成对开局）")

    os.makedirs(args.out, exist_ok=True)
    model_ids = _compare_models(args.ckpt_a, args.ckpt_b)
    with open(os.path.join(args.out, "model_ids.json"), "w") as fh:
        json.dump(model_ids, fh, indent=1)
    print("A hash=%s B hash=%s same=%s policy_diff=%.2e" % (
        model_ids["a"]["hash"], model_ids["b"]["hash"], model_ids["same_hash"],
        model_ids["forward_comparison"]["max_policy_prob_diff"]))
    if args.pairs != 8:
        print("[info] --pairs 已废弃（开局多样性由 --openings-file 决定）", flush=True)

    t0 = time.time()
    games_log, sprt_info, summary = run_arena(args)
    elapsed = time.time() - t0
    manifest = _aggregate_results(games_log, args, sprt_info=sprt_info)
    manifest["elapsed_s"] = elapsed
    manifest["kit"] = {k: summary.get(k) for k in (
        "elo", "elo_ci95", "los", "pentanomial", "elo_pentanomial", "elo_pentanomial_ci95",
        "by_color", "mean_plies", "batch", "config_hash", "provenance")}
    if "gpu_server" in summary:
        manifest["gpu_server"] = summary["gpu_server"]

    with open(os.path.join(args.out, "arena.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(args.out, "games.jsonl"), "w") as fh:
        for gd in games_log:
            fh.write(json.dumps(gd) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "kit"}, indent=1))
    print("用时 %.0fs" % elapsed)


if __name__ == "__main__":
    main()
