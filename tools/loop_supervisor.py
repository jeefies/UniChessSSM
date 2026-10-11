#!/usr/bin/env python3
"""
UniChess SSM Stage B - 8小时持续演化监督看门狗 (Loop Supervisor)
确保在 8 小时晋升之路上连续接力，进程异常自愈，直到达成 8 小时目标。
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", "/home/jeefy/UniChess")
RUNS_DIR = Path(IMPORT_ROOT) / "SSM" / "runs" / "loop_stage_b"
LOG_FILE = RUNS_DIR / "supervisor.log"

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] [Supervisor] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def is_pipeline_running() -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", "loop_pipeline.py"],
            capture_output=True, text=True, check=False
        )
        pids = [p.strip() for p in res.stdout.strip().split() if p.strip()]
        return len(pids) > 0
    except Exception as e:
        log(f"检查进程异常: {e}")
        return False

def start_pipeline():
    log("正在拉起新一批自对弈大循环 (tools/start_loop.sh)...")
    cmd = ["bash", str(Path(IMPORT_ROOT) / "SSM" / "tools" / "start_loop.sh")]
    # 在后台独立进程组拉起
    subprocess.Popen(
        cmd,
        cwd=IMPORT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    time.sleep(5)
    if is_pipeline_running():
        log("新批次大循环拉起成功！")
    else:
        log("警告：拉起后未检测到 loop_pipeline.py 存活，将在下一轮重试")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage B 8-Hour Loop Supervisor")
    parser.add_argument("--hours", type=float, default=8.0, help="运行持续时长 (小时)")
    parser.add_argument("--check-interval", type=int, default=30, help="轮询心跳间隔 (秒)")
    args = parser.parse_args()

    start_time = time.time()
    end_time = start_time + args.hours * 3600
    target_dt = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

    log(f"8小时晋升之路看门狗已启动！目标终止时间: {target_dt} (运行时长: {args.hours}h)")

    while True:
        now = time.time()
        if now >= end_time:
            log("🎉【目标达成】8 小时持续晋升之路已满额完成！看门狗平稳退出。")
            break

        remaining_h = (end_time - now) / 3600
        running = is_pipeline_running()

        if not running:
            log(f"检测到 loop_pipeline.py 已结束/未运行，距离 8h 目标还剩 {remaining_h:.2f} 小时，自动接力启动下一批！")
            start_pipeline()
        
        time.sleep(args.check_interval)

if __name__ == "__main__":
    main()
