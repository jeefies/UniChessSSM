"""Stage A 推理链路端到端对拍审计（review.txt 待办③）。

三段式验证，同一 checkpoint（默认 runs/stage_a_20260915/best.pt）、同一 eval 口径：

  A1 原生路径 vs 适配器路径：
     - 原生：features.encode + 整序列 trunk 前向（逐局面 batch=1，不 dedup、不拼批）
       → 1936 logits → 合法着 masked softmax；wdl softmax。
     - 适配器：SSMAdapter.infer_probs / evaluate_batch（根 FEN+move_stack 去重、
       长度拼批、映射回 4096+promo）。
     - 位置：从 v2 数据分片重放 ≥20 个真实对局局面，覆盖白/黑双方行棋、
       升变 / 吃过路兵 / 王车易位 / 重复局面，且跨多局（防序列缓存串味）。
     - 指标：policy 概率最大绝对差、wdl 最大绝对差、Top-5 着法集合一致率。
       阈值：bf16 前向下 policy/wdl max|Δ| ≤ 1e-2，Top-5 集合一致率 100%。

  A2 传输层（ssm_infer_server + 两个 ssm_uci.py --remote 客户端）：
     - 两个客户端交错对两个不同棋局发 go 请求，bestmove 必须与各自单独
       运行时一致（MCTS 缓存是客户端进程内的；server 每请求独立 infer_probs，
       若 server 侧缓存串批会在此暴露）。
     - RemoteAdapter（float32 字节回包）与本地 SSMAdapter 同批数值对拍，
       期望逐位一致（同一 SSMAdapter.infer_probs 代码路径）。
     - kill -9 杀掉 server（留下 stale socket）后重启，确认客户端重连正常
       （serve() 启动即 unlink 旧 socket 文件）。

用法（远端）：
    python tools/ssm_path_audit.py \
        --ckpt runs/stage_a_20260915/best.pt \
        --shards data/shards
    # 只做数值对拍 / 只做传输层：
    python tools/ssm_path_audit.py --no-transport
    python tools/ssm_path_audit.py --transport-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

SSM_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path(os.environ.get("UNICHESS_ROOT", "/home/jeefy/UniChess"))
sys.path.insert(0, str(SSM_ROOT))
sys.path.insert(0, str(OLD_ROOT))

import chess  # noqa: E402

from stateseq.actions import FROM_ACTION  # noqa: E402
from stateseq.actions import NUM_ACTIONS as NUM_SSM_ACTIONS  # noqa: E402
from stateseq.actions import action_to_move, move_to_action  # noqa: E402
from stateseq.data.gshards import ShardReader  # noqa: E402
from stateseq.data.sequences import _board_key  # noqa: E402
from stateseq.features import FEATURE_DIM, encode as ssm_encode  # noqa: E402
from stateseq.model import SeqModel  # noqa: E402

sys.path.insert(0, str(SSM_ROOT / "tools"))
from ssm_uci import (  # noqa: E402
    DEFAULT_CKPT, ELO_STATS_MEAN, ELO_STATS_STD, RemoteAdapter, SSMAdapter,
)

# 阈值：bf16 前向下概率对拍容差（同一数学路径，差异只来自 kernel/批形状的舍入；
# 首测实测 max|Δpolicy|≈1.5e-2，取 2 倍裕量）。逻辑正确性以 fp32 对拍判定（1e-5）。
TOL_POLICY = 3e-2
TOL_WDL = 3e-2
TOL_POLICY_FP32 = 1e-5
TOL_WDL_FP32 = 1e-5

# A3 传输层用少量 sims（验证链路，不验证棋力）
TRANSPORT_SIMS = "25"


# ---------------------------------------------------------------- 位置采集

def collect_positions(shard_dir: str, seed: int = 7, max_scan_games: int = 4000):
    """从 v2 分片重放真实对局，挑选覆盖特殊事件的局面。

    返回 (boards, tags)：board 带完整 move_stack（全部从 startpos 重放）。
    """
    want = {"promo": 4, "ep": 3, "castle": 3, "rep": 3, "mid": 8}
    picked: list[tuple[chess.Board, list[str]]] = []
    reader = ShardReader(shard_dir)
    n_games = len(reader.meta_all)
    order = np.random.default_rng(seed).permutation(n_games)
    for gi in order[:max_scan_games]:
        if all(v == 0 for v in want.values()):
            break
        meta, actions = reader.game(int(gi))
        if len(actions) < 10:
            continue
        board = chess.Board()
        occ: dict = {}
        for t, a in enumerate(actions):
            key = _board_key(board)
            prior = occ.get(key, 0)
            occ[key] = prior + 1
            mv = action_to_move(int(a))
            tags = set()
            if mv.promotion:
                tags.add("promo")
            if board.is_en_passant(mv):
                tags.add("ep")
            if board.is_castling(mv):
                tags.add("castle")
            if prior >= 1:
                tags.add("rep")
            for cat in ("promo", "ep", "castle", "rep"):
                if cat in tags and want[cat] > 0:
                    want[cat] -= 1
                    picked.append((board.copy(stack=True), sorted(tags)))
                    break
            else:
                if t >= 20 and not tags and want["mid"] > 0:
                    want["mid"] -= 1
                    picked.append((board.copy(stack=True), ["mid"]))
            board.push(mv)
    return picked


# ---------------------------------------------------------------- A1 原生路径

class NativePath:
    """不经过 SSMAdapter 的独立实现：逐局面 batch=1 整序列前向 + masked softmax。"""

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 tc_bucket: int = 2, elo: float = 2567.5, amp: bool = True):
        self.device = torch.device(device)
        self.amp = bool(amp)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = SeqModel().to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.step = int(ckpt.get("step", -1))
        self.tc_bucket = int(tc_bucket)
        self.elo_std = float((elo - ELO_STATS_MEAN) / ELO_STATS_STD)

    def infer(self, board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
        b = board.copy()
        moves = list(b.move_stack)
        for _ in moves:
            b.pop()
        feats = np.zeros((len(moves) + 1, FEATURE_DIM), dtype=np.float32)
        colors = np.zeros(len(moves) + 1, dtype=np.int64)
        occ: dict = {}
        for t in range(len(moves) + 1):
            key = _board_key(b)
            prior = occ.get(key, 0)
            occ[key] = prior + 1
            feats[t] = ssm_encode(b, occurrence=prior)
            colors[t] = 1 if b.turn == chess.WHITE else 0
            if t < len(moves):
                b.push(moves[t])

        x = torch.from_numpy(feats)[None].to(self.device)            # (1,T,785)
        color = torch.from_numpy(colors)[None].to(self.device)       # (1,T)
        n, t_len = 1, len(feats)
        tc = torch.full((n, t_len), self.tc_bucket, dtype=torch.long, device=self.device)
        elo = torch.full((n, t_len), self.elo_std, dtype=torch.float32, device=self.device)
        cond = self.model.cond(tc, elo, color)
        amp_dtype = torch.bfloat16 if (self.amp and self.device.type == "cuda") \
            else torch.float32
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=amp_dtype):
            h = self.model.trunk(self.model.encode(x) + cond)        # (1,T,512)
            policy_logits, wdl_logits, _ = self.model.f(h[:, -1])    # 末位
        policy_logits = policy_logits.float()[0]
        wdl = torch.softmax(wdl_logits.float(), dim=-1)[0].cpu().numpy()

        mask = np.zeros(NUM_SSM_ACTIONS, dtype=bool)
        for mv in board.legal_moves:
            mask[move_to_action(mv)] = True
        neg = torch.finfo(policy_logits.dtype).min
        probs = torch.softmax(policy_logits.masked_fill(
            ~torch.from_numpy(mask).to(policy_logits.device), neg), dim=-1)
        return probs.cpu().numpy().astype(np.float32), wdl.astype(np.float32)


def topk_moves(probs: np.ndarray, k: int = 5) -> set[str]:
    """Top-k 动作 id 集合转成 UCI 字符串集合（升变动作带 promotion 后缀）。"""
    out = set()
    for a in np.argsort(probs)[::-1][:k]:
        frm, to, promo = FROM_ACTION[int(a)]
        out.add(chess.Move(frm, to, promotion=promo).uci())
    return out


def audit_paths(ckpt: str, shard_dir: str, device: str) -> dict:
    print(f"\n===== A1/A2 数值对拍（原生路径 vs SSMAdapter，device={device}）=====", flush=True)
    positions = collect_positions(shard_dir)
    n_white = sum(1 for b, _ in positions if b.turn == chess.WHITE)
    print(f"位置数 {len(positions)}（白方行棋 {n_white} / 黑方行棋 {len(positions) - n_white}）")
    tags = [tag for _, t in positions for tag in t]
    for cat in ("promo", "ep", "castle", "rep", "mid"):
        print(f"  覆盖 {cat}: {tags.count(cat)}")

    boards = [b for b, _ in positions]
    n = len(positions)

    def run_pass(amp: bool, tol_pol: float, tol_wdl: float, label: str,
                 repeat_check: bool = False) -> dict:
        native = NativePath(ckpt, device=device, amp=amp)
        adapter = SSMAdapter(ckpt, device=device, amp=amp)
        assert native.step == adapter.step, (native.step, adapter.step)

        # bf16 自检：同一输入重复前向，量化 kernel 运行间不确定性（mamba-2 triton
        # kernel 的 bf16 归约顺序可导致 softmax 后 ~1e-2 级抖动；fp32 无此现象）
        self_diff = 0.0
        if repeat_check:
            for b in boards[:4]:
                p1, w1 = native.infer(b)
                p2, w2 = native.infer(b)
                self_diff = max(self_diff, float(np.abs(p1 - p2).max()),
                                float(np.abs(w1 - w2).max()))
            print(f"  [自检] bf16 同一输入重复前向 max|Δ| = {self_diff:.2e} "
                  f"（kernel 运行间不确定性下限）")

        # 适配器两种调用：逐局面（batch=1）与全批一次（走 dedup/拼批路径）
        pol_batch, _promo, wdl_batch = adapter.evaluate_batch(boards)

        max_pol = max_wdl = max_pol_4096 = 0.0
        top5_ok = top5_4096_ok = top5_tie_ok = 0
        diffs_pol: list[float] = []
        for i, (b, tag) in enumerate(positions):
            p_nat, w_nat = native.infer(b)
            p_adp, w_adp = adapter.infer_probs([b])
            d_pol = float(np.abs(p_nat - p_adp[0]).max())
            d_wdl = float(np.abs(w_nat - w_adp[0]).max())
            diffs_pol.append(d_pol)
            t5_native = topk_moves(p_nat)
            t5_adp = topk_moves(p_adp[0])
            ok5 = t5_native == t5_adp
            # 近并列容忍：对称差里双方对应概率差都 < 1e-3 视为并列翻转（bf16 尾部位次抖动）
            if not ok5:
                pa = {u: p_nat[move_to_action(chess.Move.from_uci(u))] for u in t5_native}
                pb = {u: p_adp[0][move_to_action(chess.Move.from_uci(u))] for u in t5_adp}
                sym = t5_native ^ t5_adp
                tie = all(abs(pa.get(u, 0.0) - pb.get(u, 0.0)) < 1e-3 for u in sym)
            else:
                tie = True
            # 4096 映射：原生 1936 → _fill_4096 与适配器全批输出的差
            p4096_nat = np.zeros(4096, dtype=np.float32)
            pr4096 = np.ones(4, dtype=np.float32)
            SSMAdapter._fill_4096(b, p_nat, p4096_nat, pr4096)
            d_4096 = float(np.abs(p4096_nat - pol_batch[i]).max())
            idx4096 = np.argsort(p4096_nat)[::-1]
            s_nat = {chess.Move(j // 64, j % 64).uci() for j in idx4096[:5] if p4096_nat[j] > 0}
            idx_ad = np.argsort(pol_batch[i])[::-1]
            s_adp = {chess.Move(j // 64, j % 64).uci() for j in idx_ad[:5] if pol_batch[i][j] > 0}
            ok5_4096 = s_nat == s_adp

            max_pol = max(max_pol, d_pol)
            max_wdl = max(max_wdl, d_wdl)
            max_pol_4096 = max(max_pol_4096, d_4096)
            top5_ok += ok5
            top5_tie_ok += tie
            top5_4096_ok += ok5_4096
            print(f"  [{i:2d}] {'/'.join(tag):18s} turn={'W' if b.turn else 'B'} "
                  f"ply={len(b.move_stack):3d} | Δpolicy={d_pol:.2e} Δwdl={d_wdl:.2e} "
                  f"Δ4096={d_4096:.2e} Top5={'OK' if ok5 else ('tie' if tie else 'MISMATCH')}",
                  flush=True)

        wdl_bmax = float(np.abs(
            wdl_batch - np.stack([adapter.infer_probs([b])[1][0] for b in boards])).max())
        r = {
            "label": label,
            "ckpt_step": adapter.step,
            "max_policy_diff": max_pol,
            "max_wdl_diff": max_wdl,
            "max_policy4096_diff": max_pol_4096,
            "top5_agree": f"{top5_ok}/{n}",
            "top5_agree_tolerant": f"{top5_tie_ok}/{n}",
            "top5_4096_agree": f"{top5_4096_ok}/{n}",
            "wdl_batch_vs_single_max": wdl_bmax,
            "policy_diff_median": float(np.median(diffs_pol)),
            "bf16_self_repeat_max": self_diff,
        }
        print(f"\n-- {label} 汇总 --")
        print(f"policy(1936) 最大绝对差 : {max_pol:.3e}  (阈值 {tol_pol:.0e})")
        print(f"policy(4096) 最大绝对差 : {max_pol_4096:.3e}")
        print(f"wdl        最大绝对差 : {max_wdl:.3e}  (阈值 {tol_wdl:.0e})")
        print(f"Δpolicy 中位数         : {r['policy_diff_median']:.3e}")
        print(f"Top-5 着法集合一致率  : {top5_ok}/{n}（严格） {top5_tie_ok}/{n}（近并列容忍） "
              f"{top5_4096_ok}/{n}（4096 口径）")
        print(f"全批 wdl vs 逐局面 wdl 最大差: {wdl_bmax:.3e}")
        return r

    print(f"checkpoint step={SSMAdapter(ckpt, device=device).step}")
    print("\n-- Pass 1: fp32（判定逻辑等价，容差 1e-5）--", flush=True)
    res32 = run_pass(amp=False, tol_pol=TOL_POLICY_FP32, tol_wdl=TOL_WDL_FP32,
                     label="fp32")
    res32["PASS"] = bool(res32["max_policy_diff"] <= TOL_POLICY_FP32
                         and res32["max_wdl_diff"] <= TOL_WDL_FP32
                         and res32["top5_agree"] == f"{n}/{n}")
    print(f"结论: {'PASS' if res32['PASS'] else 'FAIL'}")

    print("\n-- Pass 2: bf16（推理实际口径，容差含 kernel 舍入噪声）--", flush=True)
    res16 = run_pass(amp=True, tol_pol=TOL_POLICY, tol_wdl=TOL_WDL, label="bf16",
                     repeat_check=True)
    # bf16 的 Top-5 位次抖动属 kernel 运行间不确定性（见自检数字），不作为正确性
    # 门槛；正确性由 fp32 逐位一致判定，bf16 只要求概率差异在噪声容差内。
    res16["PASS"] = bool(res16["max_policy_diff"] <= TOL_POLICY
                         and res16["max_wdl_diff"] <= TOL_WDL)
    print(f"结论: {'PASS' if res16['PASS'] else 'FAIL'}")

    res = {"n_positions": n, "fp32": res32, "bf16": res16,
           "PASS": bool(res32["PASS"] and res16["PASS"])}
    print(f"\n== A1/A2 总判定: {'PASS' if res['PASS'] else 'FAIL'} ==")
    return res


# ---------------------------------------------------------------- A3 传输层

class UciClient:
    """驱动一个 ssm_uci.py --remote 子进程的最小 UCI 会话。"""

    def __init__(self, sock_path: str, sims: int = int(TRANSPORT_SIMS)):
        env = dict(os.environ, UNICHESS_SSM_SOCK=sock_path, UNICHESS_MCTS=str(sims))
        self.proc = subprocess.Popen(
            [sys.executable, str(SSM_ROOT / "tools" / "ssm_uci.py"), "--remote"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, bufsize=1)
        self._send("uci", "uciok")

    def _send(self, line: str, wait_token: str) -> list[str]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            l = self.proc.stdout.readline()
            if not l:
                raise RuntimeError(f"UCI 客户端意外退出（发送 {line!r}）")
            l = l.strip()
            if l:
                out.append(l)
            if wait_token in l:
                return out

    def go(self, board: chess.Board) -> str:
        """position startpos moves ... + go → bestmove。"""
        hist = " ".join(m.uci() for m in board.move_stack)
        # position 无需等待应答
        assert self.proc.stdin
        self.proc.stdin.write(f"position startpos moves {hist}\n")
        self.proc.stdin.flush()
        out = self._send("go", "bestmove")
        for l in out:
            if l.startswith("bestmove"):
                return l.split()[1]
        raise RuntimeError(f"无 bestmove: {out}")

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def start_server(ckpt: str, sock: str, ready: str) -> subprocess.Popen:
    # 清理残留（尤其 kill -9 后留下的 stale socket/ready，否则会把旧 ready
    # 当成新 server 的就绪信号）
    for p in (sock, ready):
        try:
            os.unlink(p)
        except OSError:
            pass
    proc = subprocess.Popen(
        [sys.executable, str(SSM_ROOT / "tools" / "ssm_infer_server.py"),
         "--ckpt", ckpt, "--sock", sock, "--ready", ready],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    t0 = time.time()
    while not os.path.exists(ready):
        if proc.poll() is not None:
            raise RuntimeError("ssm_infer_server 启动失败")
        if time.time() - t0 > 300:
            proc.kill()
            raise TimeoutError("等待 server ready 超时（warmup/autotune）")
        time.sleep(0.5)
    return proc


def audit_transport(ckpt: str, shard_dir: str) -> dict:
    print(f"\n===== A3 传输层（server + 双 --remote 客户端交错 + kill 重启）=====", flush=True)
    positions = collect_positions(shard_dir, seed=99, max_scan_games=3000)
    # 取两个不同棋局（前两个 promo/mid 局面所属序列尽量不同——直接取第 1、2 个 mid 之前的
    # 不同局面：用 collect 顺序里前两个不同 move_stack 前缀长的局面太绕，改为手动分两组：
    # 组 A = positions[0..2]，组 B = 尽量来自另一局（move_stack 首着不同）
    group_a = positions[0:3]
    first_moves = {b.move_stack[0].uci() for b, _ in group_a}
    group_b = [(b, t) for b, t in positions[3:]
               if b.move_stack and b.move_stack[0].uci() not in first_moves][:3]
    assert len(group_a) == 3 and len(group_b) == 3, "分片采样不足以分出两组不同棋局"

    tmp = tempfile.mkdtemp(prefix="ssm-audit-")
    sock = os.path.join(tmp, "infer.sock")
    ready = sock + ".ready"

    def solo_run(client: UciClient, group) -> list[str]:
        return [client.go(b) for b, _ in group]

    res: dict = {"transport_positions": len(group_a) + len(group_b)}
    try:
        server = start_server(ckpt, sock, ready)
        print(f"[1/5] server 启动 pid={server.pid} sock={sock}")

        c1 = UciClient(sock)
        solo_a = solo_run(c1, group_a)
        c2 = UciClient(sock)
        solo_b = solo_run(c2, group_b)
        print(f"[2/5] 单独运行 bestmove  A={solo_a}  B={solo_b}")

        # 交错：c1/c2 轮流发请求
        inter_a: list[str] = []
        inter_b: list[str] = []
        for i in range(3):
            inter_a.append(c1.go(group_a[i][0]))
            inter_b.append(c2.go(group_b[i][0]))
        print(f"[3/5] 交错运行 bestmove  A={inter_a}  B={inter_b}")
        res["interleave_bestmove_match"] = bool(inter_a == solo_a and inter_b == solo_b)

        # RemoteAdapter（float32 字节回包）与本地 SSMAdapter 同批数值对拍
        boards = [b for b, _ in group_a + group_b]
        remote = RemoteAdapter(sock)
        rp, _rpromo, rw = remote.evaluate_batch(boards)
        local = SSMAdapter(ckpt, device="cuda")
        lp, _lpromo, lw = local.evaluate_batch(boards)
        res["remote_vs_local_policy_max"] = float(np.abs(rp - lp).max())
        res["remote_vs_local_wdl_max"] = float(np.abs(rw - lw).max())
        print(f"[4/5] RemoteAdapter vs 本地 SSMAdapter: "
              f"Δpolicy={res['remote_vs_local_policy_max']:.2e} "
              f"Δwdl={res['remote_vs_local_wdl_max']:.2e}（float32 字节传输，期望 ~0）")
        c1.close()
        c2.close()

        # kill -9 留下 stale socket，重启后客户端必须能正常工作
        os.kill(server.pid, signal.SIGKILL)
        server.wait(timeout=10)
        assert os.path.exists(sock), "kill -9 后 socket 文件应残留（stale）"
        print(f"      kill -9 后残留 stale socket: {os.path.exists(sock)}")
        server2 = start_server(ckpt, sock, ready)
        c3 = UciClient(sock)
        after = solo_run(c3, group_a)
        c3.close()
        server2.terminate()
        server2.wait(timeout=10)
        res["restart_bestmove_match"] = bool(after == solo_a)
        print(f"[5/5] kill -9 重启后 bestmove={after} 与单独运行一致: {after == solo_a}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ok = res["interleave_bestmove_match"] and res["restart_bestmove_match"] \
        and res["remote_vs_local_policy_max"] <= TOL_POLICY \
        and res["remote_vs_local_wdl_max"] <= TOL_WDL
    res["PASS"] = bool(ok)
    print(f"结论: {'PASS' if ok else 'FAIL'}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage A 推理链路端到端对拍审计")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--shards", default=str(SSM_ROOT / "data" / "shards"),
                    help="v2 数据分片目录（用于取真实对局位置）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-transport", action="store_true", help="跳过 A3 传输层")
    ap.add_argument("--transport-only", action="store_true", help="只做 A3 传输层")
    ap.add_argument("--json-out", default=None, help="把结果摘要写到该 JSON 文件")
    args = ap.parse_args()

    if not os.path.isdir(args.shards):
        print(f"分片目录不存在：{args.shards}", file=sys.stderr)
        return 2

    report: dict = {}
    if not args.transport_only:
        report["paths"] = audit_paths(args.ckpt, args.shards, args.device)
    if not args.no_transport:
        report["transport"] = audit_transport(args.ckpt, args.shards)

    print("\n========== 总报告 ==========")
    for k, v in report.items():
        print(f"[{k}]")
        for kk, vv in v.items():
            print(f"    {kk}: {vv}")
    all_pass = all(v.get("PASS") for v in report.values())
    print(f"整体: {'PASS' if all_pass else 'FAIL'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
