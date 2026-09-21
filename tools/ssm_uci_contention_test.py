"""SSM UCI 引擎多进程共享单卡时的每步耗时诊断。

模拟 arena 口径：N 个 ssm_uci.sh 进程并发，各自从开局库 FEN 连走 M 步，
统计每步 go 耗时。用于定位 python-chess 11s（timeout=10 + movetime=1）超时的原因。

用法：
    python tools/ssm_uci_contention_test.py --procs 4 --moves 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import chess


def drive_one(proc_id: int, cmd: list[str], fen: str, moves_to_play: int,
              results: dict) -> None:
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(s: str) -> None:
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def recv_until(token: str) -> list[str]:
        lines = []
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"proc{proc_id}: engine died")
            lines.append(line.strip())
            if token in line:
                return lines

    send("uci")
    recv_until("uciok")
    send("isready")
    recv_until("readyok")

    board = chess.Board(fen)
    times = []
    for _ in range(moves_to_play):
        if board.is_game_over(claim_draw=False):
            break
        mv_s = " ".join(m.uci() for m in board.move_stack)
        send(f"position fen {fen} moves {mv_s}".rstrip())
        t0 = time.time()
        send("go movetime 1000")
        lines = recv_until("bestmove")
        dt = time.time() - t0
        times.append(dt)
        bm = [l for l in lines if l.startswith("bestmove")][0].split()[1]
        board.push_uci(bm)
    send("quit")
    p.wait(timeout=30)
    results[proc_id] = times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--moves", type=int, default=12)
    ap.add_argument("--engine", default="/home/jeefy/UniChess/SSM/tools/ssm_uci.sh")
    ap.add_argument("--fen", default="r1bqkbnr/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    args = ap.parse_args()

    results: dict[int, list[float]] = {}
    threads = []
    t0 = time.time()
    for i in range(args.procs):
        th = threading.Thread(target=drive_one,
                              args=(i, [args.engine], args.fen, args.moves, results))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    total = time.time() - t0

    all_times = [t for ts in results.values() for t in ts]
    all_times.sort()
    print(f"procs={args.procs} moves={args.moves} wall={total:.1f}s")
    for i in sorted(results):
        ts = results[i]
        print(f"  proc{i}: n={len(ts)} min={min(ts):.2f} med={sorted(ts)[len(ts)//2]:.2f} "
              f"max={max(ts):.2f} 全部={[f'{t:.1f}' for t in ts]}")
    if all_times:
        print(f"  全体: p50={all_times[len(all_times)//2]:.2f} "
              f"p95={all_times[int(len(all_times)*0.95)]:.2f} max={all_times[-1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
