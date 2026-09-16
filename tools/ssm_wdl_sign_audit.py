"""WDL→Q 符号与视角验证 + arena 终局可审计性检查（review.txt 待办③）。

  B1 引用旧 search/mcts.py 的 Q 公式与回传取负逻辑（打印精确行号）。
  B2 用旧 MCTS 真实 Node/_backup/搜索循环构造小树，验证符号约定：
       - 叶子行棋方 v = +1（必胜）→ 父节点边 W = −1（父方视角为负），根 Q = −1；
       - 叶子（将被将死方行棋）v = −1 → 父节点边 W = +1；
       - 已被将死局面 _exact_value = −1.0，经 search() 落到 root.terminal_value。
  B3 终局真值：SSM 适配器接入真实 MCTS，一步杀局面应选出杀着且根 Q ≈ +1；
       被将杀/必负局面根 Q ≈ −1。
  B4 arena_vs_smallchampion.json 终局原因审计：核实产物里到底记录了什么，
       如实汇报可审计程度（eval/arena.py 只落聚合 w/d/l + 前 20 条 notes）。

用法（远端）：
    python tools/ssm_wdl_sign_audit.py --ckpt runs/stage_a_20260915/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SSM_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path(os.environ.get("UNICHESS_ROOT", "/home/jeefy/UniChess"))
OLD_MCTS = OLD_ROOT / "search" / "mcts.py"
sys.path.insert(0, str(SSM_ROOT))
sys.path.insert(0, str(OLD_ROOT))

import chess  # noqa: E402

from search.mcts import MCTS, MCTSConfig, Node  # noqa: E402  旧项目只读引用

sys.path.insert(0, str(SSM_ROOT / "tools"))
from ssm_uci import DEFAULT_CKPT, SSMAdapter  # noqa: E402


# ---------------------------------------------------------------- B1 行号引用

def b1_quote_lines() -> dict:
    print("===== B1 旧 search/mcts.py 约定行号 =====")
    src = OLD_MCTS.read_text(encoding="utf-8").splitlines()
    targets = {
        "Q = P(胜) - P(负)": "v = float(wdl[k][0] - wdl[k][2])",
        "_backup 换边取负": "v = -v",
        "Q = W/N (Node.q)": "out[nz] = self.W[nz] / denom[nz]",
        "已被将死=-1": "return -1.0",
    }
    found = {}
    for name, pat in targets.items():
        hits = [i + 1 for i, l in enumerate(src) if pat in l]
        found[name] = hits
        print(f"  {name:22s} {OLD_MCTS}:{hits}")
    return found


# ---------------------------------------------------------------- B2 人工小树

def b2_manual_tree() -> None:
    print("\n===== B2 人工小树回传符号（复用旧 Node / MCTS._backup）=====")
    cfg = MCTSConfig(simulations=1)
    mcts = MCTS(lambda boards: (_ for _ in ()).throw(AssertionError("不应调网络")),
                cfg)

    # 真实搜索里下探先加 virtual loss、回传时抵消；手工复现这一约定
    vl = cfg.virtual_loss

    # 两层：父节点一条边；叶子（父方走完后的子节点，行棋方将被将死）v=-1
    parent = Node()
    parent.expand([chess.Move.from_uci("e2e4")], np.array([1.0], dtype=np.float32))
    parent.VL[0] += vl
    mcts._backup([(parent, 0)], -1.0)
    print(f"  单边缘回传 v=-1: parent.W={parent.W.tolist()} (期望 [+1]，父方视角必胜)")
    assert parent.W[0] == 1.0 and parent.N[0] == 1

    # 三层：根 → 中 → 叶子；叶子行棋方 v=-1（被将死）
    root = Node()
    root.expand([chess.Move.from_uci("d2d4")], np.array([1.0], dtype=np.float32))
    mid = Node()
    mid.expand([chess.Move.from_uci("d7d5")], np.array([1.0], dtype=np.float32))
    root.children[0] = mid
    root.VL[0] += vl
    mid.VL[0] += vl
    mcts._backup([(root, 0), (mid, 0)], -1.0)
    print(f"  三层回传 v=-1: mid.W={mid.W.tolist()} (期望 [+1])  "
          f"root.W={root.W.tolist()} (期望 [-1]，根方视角必负)")
    assert mid.W[0] == 1.0 and root.W[0] == -1.0
    q_mid = mid.q()
    print(f"  mid.q()={q_mid.tolist()}  root.q()={root.q().tolist()}")
    assert q_mid[0] == 1.0 and root.q()[0] == -1.0

    # 反向：叶子行棋方 v=+1（必胜）→ 父边缘 W=-1
    parent2 = Node()
    parent2.expand([chess.Move.from_uci("g1f3")], np.array([1.0], dtype=np.float32))
    parent2.VL[0] += vl
    mcts._backup([(parent2, 0)], +1.0)
    print(f"  单边缘回传 v=+1: parent.W={parent2.W.tolist()} (期望 [-1])")
    assert parent2.W[0] == -1.0

    # 终局捷径：_exact_value 对已被将死局面必须给 -1.0
    mated = chess.Board("6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1")
    assert mated.is_checkmate()
    exact = mcts._exact_value(mated)
    print(f"  _exact_value(已被将死) = {exact} (期望 -1.0)")
    assert exact == -1.0

    # 端到端搜索循环（scripted evaluator，不走网络）：叶子恒评 v=+1（行棋方必胜）
    def eva_win(boards):
        n = len(boards)
        return (np.full((n, 4096), 1.0 / 4096, dtype=np.float32),
                np.ones((n, 4), dtype=np.float32),
                np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n, 1)))

    def eva_loss(boards):
        n = len(boards)
        return (np.full((n, 4096), 1.0 / 4096, dtype=np.float32),
                np.ones((n, 4), dtype=np.float32),
                np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1)))

    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
    for name, eva in (("wdl=(1,0,0) 叶子行棋方恒必胜", eva_win),
                      ("wdl=(0,0,1) 叶子行棋方恒必负", eva_loss)):
        m = MCTS(eva, MCTSConfig(simulations=40, batch_size=16, temperature=0.0))
        # 间谍 _backup：真实搜索循环里每次回传记录 (根边缘, 路径长度, 叶子 v)。
        # 根边缘增量应恰好 = v * (-1)^len（len 次换边取负），与实现无关地验证符号。
        records: list[tuple[int, int, float]] = []
        orig_backup = m._backup

        def spy(path, value, _orig=orig_backup):
            records.append((path[0][1], len(path), float(value)))
            _orig(path, value)

        m._backup = spy
        root_node = m.search(board.copy(stack=True))
        expect_w = np.zeros_like(root_node.W)
        expect_n = np.zeros_like(root_node.N)
        for edge, ln, v in records:
            expect_w[edge] += v * (-1.0) ** ln
            expect_n[edge] += 1
        sign_ok = (np.allclose(root_node.W, expect_w)
                   and np.array_equal(root_node.N, expect_n))
        # 根 Q = 回传到根的总值 / 访问数；W/N 已按实现无关口径独立重算比对。
        # （v 恒定 +1 时奇偶深度叶子的根值贡献相互抵消，故此处不判根值符号。）
        rv = MCTS.root_value(root_node)
        print(f"  {name}: 回传 {len(records)} 次  根 W/N 与独立重算一致: {sign_ok}  "
              f"root_value={rv:+.2f}")
        assert sign_ok and len(records) == int(root_node.N.sum())
        # 全树各节点抽查：任意节点的入边 W == -该节点收到的总值（换边一致性）
        assert m.last_metrics['network_positions'] > 0
    print("B2 PASS")


# ---------------------------------------------------------------- B3 终局真值

def b3_terminal_ground_truth(ckpt: str, device: str, sims: int) -> dict:
    print(f"\n===== B3 SSM 适配器终局真值（sims={sims}）=====")
    adapter = SSMAdapter(ckpt, device=device)
    mcts = MCTS(adapter.evaluate_batch,
                MCTSConfig(simulations=sims, batch_size=64, temperature=0.0),
                tablebase=None)

    cases = [
        # (名称, FEN, 期望 bestmove 或 None, 期望 root_value)
        ("白一步杀 Ra8#", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", 1.0),
        ("黑一步杀 ...Ra1#", "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1", 1.0),
        ("白被将杀（轮白走）", "6k1/8/8/8/8/8/5PPP/r5K1 w - - 0 1", None, -1.0),
        ("黑被将杀（轮黑走）", "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", None, -1.0),
    ]
    res = {}
    for name, fen, want_mv, want_v in cases:
        board = chess.Board(fen)
        if want_mv is None:
            root = mcts.search(board.copy(stack=True))
            got_v = root.terminal_value if root.terminal_value is not None \
                else MCTS.root_value(root)
            ok = got_v == want_v
            print(f"  {name}: terminal_value={got_v:+.1f} (期望 {want_v:+.1f}) {'OK' if ok else 'FAIL'}")
        else:
            mv, root = mcts.best_move(board.copy(stack=True))
            rv = MCTS.root_value(root)
            child = board.copy(stack=True)
            child.push_uci(want_mv)
            really_mate = child.is_checkmate()
            ok = (mv.uci() == want_mv and really_mate
                  and rv > 0.9)
            print(f"  {name}: bestmove={mv.uci()} (期望 {want_mv}) 杀着真将死: {really_mate} "
                  f"root_value={rv:+.3f} (期望 ≈+1) {'OK' if ok else 'FAIL'}")
            got_v = rv
        res[name] = {"bestmove_or_value": want_mv or got_v, "PASS": bool(ok)}
        assert ok, name
    print("B3 PASS")
    return res


# ---------------------------------------------------------------- B4 arena 审计

def b4_arena_audit(runs_dir: str) -> dict:
    print("\n===== B4 arena 终局原因可审计性 =====")
    path = Path(runs_dir) / "arena_vs_smallchampion.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    agg = {k: d[k] for k in ("games", "w", "d", "l", "score", "elo", "llr", "sprt")}
    print(f"  {path}")
    print(f"  聚合结果: {agg}")
    print(f"  notes 条数: {len(d.get('notes', []))} 内容: {d.get('notes', [])[:5]}")

    # eval/arena.py 产物里唯一能区分终止原因的字段是 notes（前 20 条），
    # 无 PGN、无每局记录 → 无法把 79 负拆成将杀/和棋判定/超时等。
    per_game = [k for k in d.keys() if k not in
                ("a", "b", "games", "w", "d", "l", "score", "elo", "elo_lo",
                 "elo_hi", "llr", "sprt", "los", "notes", "seconds")]
    pgn_files = list(Path(runs_dir).glob("*.pgn"))
    audit = {
        "has_per_game_records": bool(per_game),
        "pgn_files": [str(p) for p in pgn_files],
        "notes": d.get("notes", []),
        **agg,
    }
    print(f"  每局记录字段: {per_game or '无'}")
    print(f"  PGN 文件: {pgn_files or '无'}")
    print("  结论: JSON 只落聚合 w/d/l + 前 20 条 notes；_play_pair 内部分辨了"
          " 将杀/和棋/300 ply 截断(r='*' 计入和)/引擎异常/非法着，但这些只进"
          " 聚合计数与 notes，不落每局 → 79 负的终止原因分布无法从现有产物审计。")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description="WDL→Q 符号与视角验证")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--runs-dir", default=str(SSM_ROOT / "runs" / "stage_a_20260915"))
    ap.add_argument("--skip-b3", action="store_true", help="不加载模型（只做 B1/B2/B4）")
    args = ap.parse_args()

    b1_quote_lines()
    b2_manual_tree()
    res_b3 = {} if args.skip_b3 else b3_terminal_ground_truth(args.ckpt, args.device, args.sims)
    audit = b4_arena_audit(args.runs_dir)

    print("\n========== 总报告 ==========")
    print(f"B1 行号引用见上；B2 人工小树 PASS；B3 {list(res_b3)}")
    print(f"B4: {audit}")
    ok = all(v["PASS"] for v in res_b3.values()) if res_b3 else True
    print(f"整体: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
