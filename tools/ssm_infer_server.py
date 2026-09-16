"""SSM 单进程 GPU 推理服务器：为多个 UCI 客户端（tools/ssm_uci.py --remote）服务。

为什么需要它（见 ssm_uci.py 模块 docstring）：
  Mamba-2 trunk 的 triton kernel 首调用要做 autotune（~4.2s/进程），多进程 arena 下
  N 个引擎并发 autotune 互相踩踏（实测 4 进程各 ~18s），超出 python-chess 的
  play 超时（10s + movetime）。改成单 GPU 进程后只 autotune 一次，UCI 客户端为
  纯 CPU 进程（秒级启动、零显存占用、无并发热身）。

协议（Unix socket，SOCK_STREAM）：
  请求: 4B 小端长度 + JSON {"items": [{"fen": 根FEN, "moves": [uci...]}, ...]}
  响应: n*(1936+3) float32 原样字节（每行 1936 维 masked softmax 概率 + 3 维 wdl）

单连接逐请求处理；GPU 前向全局串行（模型很小，排队开销可忽略）。

用法：
    python tools/ssm_infer_server.py --ckpt runs/stage_a_20260915/best.pt \
        --sock /tmp/unichess-ssm-infer.sock --ready /tmp/unichess-ssm-infer.ready
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))

import chess  # noqa: E402

from ssm_uci import DEFAULT_SOCK, SSMAdapter  # noqa: E402


def recv_all(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("客户端断开")
        buf += chunk
    return buf


class InferServer:
    def __init__(self, adapter: SSMAdapter, sock_path: str):
        self.adapter = adapter
        self.sock_path = sock_path
        self.lock = threading.Lock()          # GPU 前向串行
        self.n_req = 0
        self.n_board = 0

    def warmup(self) -> None:
        """触发 Mamba-2 kernel autotune / CUDA 热身，必须在 ready 前完成。"""
        b = chess.Board()
        for u in ("e2e4", "e7e5", "g1f3"):
            b.push_uci(u)
        t0 = time.time()
        with self.lock:
            self.adapter.infer_probs([b, b.copy(stack=True)])
        print(f"[server] warmup done in {time.time() - t0:.1f}s", flush=True)

    def handle(self, conn: socket.socket) -> None:
        while True:
            hdr = recv_all(conn, 4)
            (ln,) = struct.unpack("<I", hdr)
            req = json.loads(recv_all(conn, ln).decode())
            boards = []
            for it in req["items"]:
                b = chess.Board(it["fen"])
                for u in it["moves"]:
                    b.push_uci(u)
                boards.append(b)
            # 重放/拼批/前向在锁内串行（模型很小，排队开销可忽略；保证 batch 语义）
            with self.lock:
                # 复用 infer_probs 的去重/拼批/前向；直接喂 board 即可
                probs, wdl = self.adapter.infer_probs(boards)
            out = np.concatenate([probs, wdl], axis=1).astype(np.float32).tobytes()
            conn.sendall(out)
            self.n_req += 1
            self.n_board += len(boards)

    def serve(self) -> None:
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        os.chmod(self.sock_path, 0o600)
        srv.listen(16)
        print(f"[server] listening {self.sock_path} step={self.adapter.step}", flush=True)
        while True:
            conn, _ = srv.accept()
            th = threading.Thread(target=self._client, args=(conn,), daemon=True)
            th.start()

    def _client(self, conn: socket.socket) -> None:
        try:
            self.handle(conn)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "stage_a_20260915" / "best.pt"))
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--ready", default=DEFAULT_SOCK + ".ready")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tc-bucket", type=int, default=2)
    ap.add_argument("--elo", type=float, default=2567.5)
    args = ap.parse_args()

    adapter = SSMAdapter(args.ckpt, device=args.device,
                         tc_bucket=args.tc_bucket, elo=args.elo)
    server = InferServer(adapter, args.sock)
    server.warmup()
    Path(args.ready).write_text(f"pid={os.getpid()} step={adapter.step}\n")
    try:
        server.serve()
    finally:
        for p in (args.sock, args.ready):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
