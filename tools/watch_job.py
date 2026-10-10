"""通用后台任务监视器 (UniChess Background Job Watcher)

用于监控后台自对弈、训练或竞技场长任务。当任务完成或异常时及时退出，
配合 CLI 任务引擎实现无人工介入的主动事件唤醒与流水线自动流转。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="UniChess Job Watcher")
    parser.add_argument("--log", required=True, help="要监控的日志文件路径")
    parser.add_argument("--pid", type=int, default=None, help="目标主进程 PID（可选）")
    parser.add_argument("--done-pattern", default="聚合完成|SPRT|VERDICT|训练完成|Training complete",
                        help="表示完成的正则关键字（默认匹配自对弈聚合、Arena裁决、训练结束）")
    parser.add_argument("--error-pattern", default="Traceback|CUDA out of memory|RuntimeError: 以下 Worker",
                        help="表示异常终止的正则关键字")
    parser.add_argument("--interval", type=float, default=10.0, help="轮询检查间隔秒数 (默认 10s)")
    parser.add_argument("--heartbeat", type=float, default=60.0, help="输出心跳摘要间隔秒数 (默认 60s)")
    parser.add_argument("--timeout", type=float, default=28800.0, help="最长超时等待秒数 (默认 8h)")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log)
    print(f"[Watcher] 启动任务监视器")
    print(f"[Watcher] 目标日志: {log_path}")
    if args.pid:
        print(f"[Watcher] 目标 PID: {args.pid} (状态: {'运行中' if is_pid_running(args.pid) else '已不存在'})")
    print(f"[Watcher] 完成判定模式: /{args.done_pattern}/")
    print(f"[Watcher] 异常判定模式: /{args.error_pattern}/")

    done_re = re.compile(args.done_pattern)
    err_re = re.compile(args.error_pattern)

    t0 = time.time()
    last_heartbeat = t0
    last_size = 0
    file_found = False

    # 等待日志文件生成（最多 60 秒）
    wait_file_t0 = time.time()
    while not os.path.exists(log_path):
        if time.time() - wait_file_t0 > 60.0:
            print(f"[Watcher 错误] 超过 60 秒未发现日志文件：{log_path}", file=sys.stderr)
            sys.exit(1)
        time.sleep(1.0)

    print(f"[Watcher] 已捕获目标日志文件，开始持续追踪流...")

    with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
        while True:
            elapsed = time.time() - t0
            if elapsed > args.timeout:
                print(f"[Watcher 超时] 任务运行时间超过设定上限 {args.timeout} 秒，监视器退出", file=sys.stderr)
                sys.exit(124)

            # 读取新增内容
            lines = fp.readlines()
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                if done_re.search(line_str):
                    print(f"\n[Watcher 完成] 命中完成判定：{line_str}")
                    print(f"[Watcher 汇总] 任务已确认完成，累计耗时 {elapsed:.1f} 秒！")
                    sys.exit(0)

                if err_re.search(line_str):
                    print(f"\n[Watcher 异常] 命中断言错误：{line_str}", file=sys.stderr)
                    # 再读几行以便捕获后续 traceback
                    time.sleep(1.0)
                    extra_lines = fp.readlines()
                    for el in extra_lines[:15]:
                        print(f"  {el.strip()}", file=sys.stderr)
                    sys.exit(1)

            # 进程存活检查
            if args.pid is not None and not is_pid_running(args.pid):
                # 进程已经退出，做最后一次日志读取
                time.sleep(1.0)
                final_lines = fp.readlines()
                for fl in final_lines:
                    if done_re.search(fl):
                        print(f"\n[Watcher 完成] 进程正常退出并命中完成判定：{fl.strip()}")
                        sys.exit(0)
                print(f"\n[Watcher 警告] 目标进程 PID={args.pid} 已退出，且未命中完成关键字！", file=sys.stderr)
                sys.exit(2)

            now = time.time()
            if now - last_heartbeat >= args.heartbeat:
                last_heartbeat = now
                file_size = os.path.getsize(log_path)
                print(f"[Watcher 心跳] 运行时长 {elapsed/60:.1f}m | 日志大小 {file_size/1024:.1f}KB")

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
