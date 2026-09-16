"""Stage A SeqModel 的 UCI 引擎适配器（与旧 AlphaZero 引擎同口径对打用）。

口径要点（与旧 search/mcts.py 的 evaluator 约定完全对齐）：
  - evaluate_batch(list[chess.Board]) -> (policy[N,4096], promo[N,4], wdl[N,3])，
    softmax 后、行棋方视角；4096 = from*64+to，索引在「已 orient（mirror 到行棋方
    视角）」的坐标系下；promo 4 维按旧 core/moves.PROMO_PIECES=(Q,R,B,N)。
  - SSM 特征 stateseq/features.encode 是**白方绝对坐标、不随走子方翻转**，
    所以特征永远编码原始棋盘；mirror 只发生在「把策略概率放进 4096 数组」时
    （对每个合法着 mv：idx = orient_move(mv, turn) 的 from*64+to，
    值 = SSM 对原始 mv 的概率）。
  - SSM policy 动作空间 1936（stateseq/actions.py），升变 R/B/N 是独立动作、
    升后走后走法动作；旧 mcts 以 policy[idx]*promo[pi] 合成升变先验，
    这里对每个升变 (from,to) 把 Q/R/B/N 四个动作概率归一放进 4 维 promo 头、
    policy[idx] 放四者之和，乘积恰好还原各升变动作概率。

推理条件输入（训练同分布，参考 stateseq/conditions.py 与 data/shards/manifest.json）：
  - tc_bucket 固定一桶（默认 RAPID=2，可用 --tc-bucket 改）；
  - elo_std 用训练集 elo_stats 常量标准化（默认 elo=2567.5 = P99 上限，求最强；
    可用 --elo 改）；
  - color 取特征侧：白走=1 / 黑走=0，逐步取值。

重复局面 occurrence 沿 move_stack 从根重放统计（stateseq/data/sequences.py::_board_key
同口径）。同一 evaluate_batch 内按 move_stack 元组去重，整序列 trunk 扫描取末位 h。

搜索直接复用旧项目 search/mcts.py（只读 sys.path 引用），sims 从环境变量
UNICHESS_MCTS 读取（默认 400），与旧引擎同口径。

用法：
    UNICHESS_MCTS=400 python tools/ssm_uci.py --ckpt runs/stage_a_20260915/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SSM_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path(os.environ.get("UNICHESS_ROOT", "/home/jeefy/UniChess"))  # 旧项目：绝对只读引用
sys.path.insert(0, str(SSM_ROOT))
sys.path.insert(0, str(OLD_ROOT))

import chess  # noqa: E402

from stateseq.actions import move_to_action  # noqa: E402
from stateseq.data.sequences import _board_key  # noqa: E402
from stateseq.features import FEATURE_DIM, encode as ssm_encode  # noqa: E402
from stateseq.model import SeqModel  # noqa: E402

from core.encoding import orient_move  # noqa: E402  旧项目只读引用
from core.moves import POLICY_SIZE, PROMO_TO_IDX  # noqa: E402
from search.mcts import MCTS, MCTSConfig  # noqa: E402

NAME = "UniChessSSM-StageA"
AUTHOR = "jeefy"

# data/shards/manifest.json 的 elo_stats（Stage A 训练标准化常量，P1/P99 截断后统计）
ELO_STATS_MEAN = 1656.1096813511026
ELO_STATS_STD = 390.9460086323725

DEFAULT_CKPT = str(SSM_ROOT / "runs" / "stage_a_20260915" / "best.pt")


class SSMAdapter:
    """SeqModel -> 旧 MCTS evaluator 三元组的适配器。"""

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 tc_bucket: int = 2, elo: float = 2567.5, debug: bool = False):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = SeqModel().to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.step = int(ckpt.get("step", -1))
        self.tc_bucket = int(tc_bucket)
        self.elo_std = float((elo - ELO_STATS_MEAN) / ELO_STATS_STD)
        self.debug = debug
        self.last_eval_seconds = 0.0
        self.last_replayed_positions = 0

    # ---- 序列重放：move_stack -> (features, colors)，末位即当前局面 ----

    def _replay(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray, str]:
        b = board.copy()
        moves = list(b.move_stack)
        for mv in reversed(moves):
            b.pop()
        root_fen = b.fen()  # FEN 根（arena 开局库）是序列重放的真实起点
        occ: dict = {}
        feats = np.zeros((len(moves) + 1, FEATURE_DIM), dtype=np.float32)
        colors = np.zeros(len(moves) + 1, dtype=np.int64)
        for t in range(len(moves) + 1):
            key = _board_key(b)
            prior = occ.get(key, 0)
            occ[key] = prior + 1
            feats[t] = ssm_encode(b, occurrence=prior)
            colors[t] = 1 if b.turn == chess.WHITE else 0
            if t < len(moves):
                b.push(moves[t])
        return feats, colors, root_fen

    def _forward_last(self, feats_pad: torch.Tensor, colors_pad: torch.Tensor,
                      lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """整序列 trunk 扫描取各序列末位 h -> f 头。返回 (policy_logits, wdl_logits)。"""
        device = self.device
        x = self.model.encode(feats_pad.to(device))                    # (N, T, 512)
        n, t_max, d = x.shape
        tc = torch.full((n, t_max), self.tc_bucket, dtype=torch.long, device=device)
        elo = torch.full((n, t_max), self.elo_std, dtype=torch.float32, device=device)
        cond = self.model.cond(tc, elo, colors_pad.to(device))
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            h = self.model.trunk(x + cond)                             # (N, T, 512)
        idx = torch.tensor([n_ - 1 for n_ in lengths], device=device)
        h_last = h[torch.arange(n, device=device), idx]                # (N, 512)
        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            policy_logits, wdl_logits, _ = self.model.f(h_last)
        return policy_logits.float(), wdl_logits.float()

    # ---- 旧 evaluator 接口 ----

    def evaluate_batch(self, boards: list[chess.Board]):
        t0 = time.time()
        # 同批内按 (根 FEN, move_stack) 去重（MCTS 叶子大量共享前缀；
        # FEN 根的 move_stack 为空，只看 move_stack 会把不同开局塌缩成一条）
        uniq: dict[tuple, list[int]] = {}
        order: list[tuple] = []
        root_fens: dict[tuple, str] = {}
        for i, b in enumerate(boards):
            key = (b.root().fen(), tuple(b.move_stack))
            if key not in uniq:
                uniq[key] = []
                order.append(key)
            uniq[key].append(i)

        feats_list, colors_list, lengths = [], [], []
        for key in order:
            f, c, root_fen = self._replay(boards[uniq[key][0]])
            feats_list.append(f)
            colors_list.append(c)
            lengths.append(len(f))
            root_fens[key] = root_fen

        n, t_max = len(order), max(lengths)
        feats_pad = torch.zeros((n, t_max, FEATURE_DIM), dtype=torch.float32)
        colors_pad = torch.zeros((n, t_max), dtype=torch.long)
        for i, (f, c) in enumerate(zip(feats_list, colors_list)):
            feats_pad[i, :len(f)] = torch.from_numpy(f)
            colors_pad[i, :len(c)] = torch.from_numpy(c)

        with torch.no_grad():
            policy_logits, wdl_logits = self._forward_last(feats_pad, colors_pad, lengths)
            # masked softmax：只保留合法着对应动作，天然零泄漏
            mask = torch.zeros_like(policy_logits, dtype=torch.bool)
            for i, key in enumerate(order):
                b = boards[uniq[key][0]]
                m = np.zeros(policy_logits.shape[1], dtype=bool)
                for mv in b.legal_moves:
                    m[move_to_action(mv)] = True
                mask[i] = torch.from_numpy(m)
            neg = torch.finfo(policy_logits.dtype).min
            probs = torch.softmax(policy_logits.masked_fill(~mask, neg), dim=-1)
            wdl = torch.softmax(wdl_logits, dim=-1)

        probs_np = probs.cpu().numpy().astype(np.float32)
        wdl_np = wdl.cpu().numpy().astype(np.float32)

        policy_out = np.zeros((len(boards), POLICY_SIZE), dtype=np.float32)
        promo_out = np.ones((len(boards), 4), dtype=np.float32)
        row_of = {key: i for i, key in enumerate(order)}
        for i, b in enumerate(boards):
            key = (b.root().fen(), tuple(b.move_stack))
            row = row_of[key]
            self._fill_4096(b, probs_np[row], policy_out[i], promo_out[i])

        self.last_eval_seconds = time.time() - t0
        self.last_replayed_positions = sum(lengths)
        if self.debug:
            print(f"info string eval_batch: {len(boards)} boards ({len(order)} uniq), "
                  f"{sum(lengths)} replayed positions, {self.last_eval_seconds:.3f}s",
                  file=sys.stderr, flush=True)
        return policy_out, promo_out, wdl_np

    @staticmethod
    def _fill_4096(board: chess.Board, probs: np.ndarray,
                   policy: np.ndarray, promo: np.ndarray) -> None:
        """SSM 1936 动作概率 -> 旧 4096 + 4 维 promo 头（orient 坐标系）。"""
        turn = board.turn
        promo_groups: dict[tuple[int, int], dict[int, float]] = {}
        for mv in board.legal_moves:
            p = float(probs[move_to_action(mv)])
            om = orient_move(mv, turn)
            if mv.promotion is None:
                policy[om.from_square * 64 + om.to_square] = p
            else:
                promo_groups.setdefault((om.from_square, om.to_square), {})[mv.promotion] = p
        for (frm, to), pieces in promo_groups.items():
            total = float(sum(pieces.values()))
            policy[frm * 64 + to] = total
            if total > 0:
                for piece, p in pieces.items():
                    promo[PROMO_TO_IDX[piece]] = p / total


# ---------------------------------------------------------------- UCI 循环

class UciLoop:
    def __init__(self, args):
        self.args = args
        self.adapter: SSMAdapter | None = None
        self.mcts: MCTS | None = None
        self.board = chess.Board()
        self.move_time = 0.0

    def _ensure(self) -> None:
        if self.adapter is None:
            sims = int(os.environ.get("UNICHESS_MCTS", "400"))
            batch = int(os.environ.get("UNICHESS_MCTS_BATCH", "64"))
            self.adapter = SSMAdapter(
                self.args.ckpt, device=self.args.device,
                tc_bucket=self.args.tc_bucket, elo=self.args.elo,
                debug=self.args.debug)
            self.mcts = MCTS(
                self.adapter.evaluate_batch,
                MCTSConfig(simulations=sims, batch_size=batch, temperature=0.0),
                tablebase=None)  # SSM 侧无 Syzygy，与旧引擎 UNICHESS_SYZYGY="" 同口径
            print(f"info string loaded step={self.adapter.step} "
                  f"sims={sims} device={self.args.device}", file=sys.stderr, flush=True)

    def _position(self, parts: list[str]) -> None:
        if len(parts) < 2:
            return
        if parts[1] == "startpos":
            self.board = chess.Board()
            rest = parts[2:]
        elif parts[1] == "fen":
            idx = parts.index("moves") if "moves" in parts else len(parts)
            self.board = chess.Board(" ".join(parts[2:idx]))
            rest = parts[idx:]
        else:
            return
        if rest and rest[0] == "moves":
            for u in rest[1:]:
                try:
                    self.board.push_uci(u)
                except ValueError:
                    print(f"info string 忽略非法走法 {u}", flush=True)

    def _go(self) -> None:
        self._ensure()
        if self.board.is_game_over(claim_draw=False):
            print("bestmove 0000", flush=True)
            return
        t0 = time.time()
        mv, _ = self.mcts.best_move(self.board.copy(stack=True))
        dt = time.time() - t0
        self.move_time += dt
        # UCI info：用一次轻量根评估输出可读分值（复用 adapter 缓存无，直接省略显式 cp 也行）
        print(f"info depth {self.mcts.cfg.simulations} time {int(dt * 1000)}", flush=True)
        if self.args.debug:
            print(f"info string move {mv.uci()} {dt:.2f}s "
                  f"(eval_batch 累计特征重放 {self.adapter.last_replayed_positions} 位置)",
                  file=sys.stderr, flush=True)
        print(f"bestmove {mv.uci()}", flush=True)

    def run(self) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == "uci":
                print(f"id name {NAME}")
                print(f"id author {AUTHOR}")
                print("uciok", flush=True)
            elif cmd == "isready":
                self._ensure()
                print("readyok", flush=True)
            elif cmd == "ucinewgame":
                self.board = chess.Board()
            elif cmd == "position":
                self._position(parts)
            elif cmd == "go":
                self._go()
            elif cmd == "quit":
                return 0
            elif cmd == "stop":
                continue  # 同步搜索，stop 无操作（与旧 uci.py 同行为）
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="UniChessSSM Stage A UCI 适配器")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tc-bucket", type=int, default=2, help="推理固定时间控制桶（默认 RAPID）")
    ap.add_argument("--elo", type=float, default=2567.5, help="推理固定 Elo（默认 P99 上限，标准化后约 +2.33σ）")
    ap.add_argument("--debug", action="store_true")
    return UciLoop(ap.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
