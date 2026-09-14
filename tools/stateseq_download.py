#!/usr/bin/env python3
"""Stage A 下载看门狗 v2：每路径单连接（直连 + Windows 代理各一），限速自动重启。

实测结论：lichess 对单 IP 按固定管道限速（直连 ~110-140 KB/s，代理经隧道 ~170 KB/s），
多连接不增反触发 429；两条独立路径并行 ≈ 300 KB/s。
策略：每月 4 GB 前缀切成两段（各 2 GB），直连/代理各下一段；
看门狗每分钟检查速率，低于阈值或停滞即杀掉用断点续传（Range 从已下大小处重启）重启，
每 20 分钟全局重启换新鲜连接。临时空间：单月 4 GB。

用法：setsid nohup python tools/stateseq_download.py > data/download.log 2>&1 &
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

PROXY = "http://172.16.1.55:7897"
MONTHS = ["2026-08", "2026-07", "2026-06"]
PREFIX_BYTES = 4 * 1024**3          # 每月取前 4 GB
SEGMENTS = [                          # (路径, 起点, 终点) —— 两段并行
    ("direct", 0, PREFIX_BYTES // 2),
    ("proxy", PREFIX_BYTES // 2, PREFIX_BYTES),
]
STALL_BPS = 30 * 1024                 # 单段低于此速率即重启
GLOBAL_RESTART_SEC = 20 * 60
CHECK_SEC = 60
MAX_SEG_SEC = 6 * 3600                # 单段超时保护


def url(month: str) -> str:
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def http_size(month: str) -> int:
    req = urllib.request.Request(url(month), method="HEAD")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return int(resp.headers["Content-Length"])
        except Exception as exc:  # noqa: BLE001
            print(f"HEAD fail {month}: {exc}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"无法获取 {month} 大小")


def start_seg(month: str, seg_idx: int, path: str, start: int, end: int, out: str) -> subprocess.Popen:
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if have > 0:
        print(f"  resume seg{seg_idx} {path} at +{have/1e6:.1f}MB", flush=True)
    # 追加写入（shell >>）：curl -o 默认截断，重启会丢已下字节（v2 教训）
    cmd = f"curl -s -f --max-time {MAX_SEG_SEC} -r {start + have}-{end} "
    if path == "proxy":
        cmd += f"-x {PROXY} "
    cmd += f"\"{url(month)}\" >> {out}"
    return subprocess.Popen(["bash", "-c", cmd])


def seg_complete(out: str, expect: int) -> bool:
    return os.path.exists(out) and os.path.getsize(out) >= expect


def download_month(month: str) -> None:
    final = os.path.join(RAW, f"lichess_standard_{month}.pgn.zst")
    if os.path.exists(final + ".done"):
        print(f"SKIP {month}", flush=True)
        return
    total = http_size(month)
    if total < PREFIX_BYTES:
        raise RuntimeError(f"{month} 文件异常小: {total}")
    outs = [os.path.join(RAW, f"{month}.seg{i}") for i in range(len(SEGMENTS))]
    expects = [e - s + 1 for _, s, e in SEGMENTS]

    procs: list[subprocess.Popen | None] = [None] * len(SEGMENTS)
    seg_path = [SEGMENTS[i][0] for i in range(len(SEGMENTS))]
    last_size = [0] * len(SEGMENTS)
    last_time = [0.0] * len(SEGMENTS)
    consec_restart = [0] * len(SEGMENTS)
    cooldown_until = [0.0] * len(SEGMENTS)
    t_start = time.time()

    def ensure(i: int) -> None:
        if procs[i] is not None and procs[i].poll() is None:
            return
        if seg_complete(outs[i], expects[i]):
            procs[i] = None
            return
        if time.time() < cooldown_until[i]:
            procs[i] = None
            return
        _tag, s, e = SEGMENTS[i]
        procs[i] = start_seg(month, i, seg_path[i], s, e, outs[i])
        last_size[i] = os.path.getsize(outs[i]) if os.path.exists(outs[i]) else 0
        last_time[i] = time.time()

    def kill(i: int) -> None:
        if procs[i] is not None and procs[i].poll() is None:
            procs[i].send_signal(signal.SIGKILL)
            procs[i].wait()
        procs[i] = None

    print(f"== {month} 开始 {time.strftime('%F %T')}", flush=True)
    for i in range(len(SEGMENTS)):
        ensure(i)

    while True:
        time.sleep(CHECK_SEC)
        done = all(seg_complete(outs[i], expects[i]) for i in range(len(SEGMENTS)))
        if done:
            break
        for i in range(len(SEGMENTS)):
            if seg_complete(outs[i], expects[i]):
                kill(i)
                continue
            now = time.time()
            sz = os.path.getsize(outs[i]) if os.path.exists(outs[i]) else 0
            rate = (sz - last_size[i]) / max(now - last_time[i], 1)
            stalled = rate < STALL_BPS
            if stalled:
                consec_restart[i] += 1
                if consec_restart[i] >= 3:
                    seg_path[i] = "proxy" if seg_path[i] == "direct" else "direct"  # 翻转路径
                    consec_restart[i] = 0
                    print(f"  seg{i} 连续低速，切换路径→{seg_path[i]}", flush=True)
                else:
                    print(f"  seg{i} 低速 {rate/1024:.0f}KB/s，重启", flush=True)
                kill(i)
                ensure(i)
            else:
                consec_restart[i] = 0
                last_size[i], last_time[i] = sz, now
        if time.time() - t_start > GLOBAL_RESTART_SEC:
            print("  定时全局重启", flush=True)
            for i in range(len(SEGMENTS)):
                kill(i)
            for i in range(len(SEGMENTS)):
                ensure(i)
            t_start = time.time()
        done_bytes = sum(min(os.path.getsize(o), e) if os.path.exists(o) else 0 for o, e in zip(outs, expects))
        print(f"  {month} 进度 {done_bytes/1e9:.2f}/4.00 GB", flush=True)

    for i in range(len(SEGMENTS)):
        kill(i)
    with open(final, "wb") as fh:
        for out in outs:
            with open(out, "rb") as seg:
                while True:
                    buf = seg.read(16 * 1024**2)
                    if not buf:
                        break
                    fh.write(buf)
    for out in outs:
        os.remove(out)
    open(final + ".done", "w").close()
    print(f"== {month} 完成 {time.strftime('%F %T')} size={os.path.getsize(final)}", flush=True)


def main() -> None:
    for month in MONTHS:
        download_month(month)
    print("ALL_MONTHS_DONE", flush=True)


if __name__ == "__main__":
    main()
