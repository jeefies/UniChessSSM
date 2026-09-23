"""P4 共享 GPU 前向服务：N 个纯 CPU 搜索进程 → 1 个 GPU 进程跨进程拼大批。

为什么：各进程自己拼批时单拍每模型只有 ~12–24 行，CUDA graph 在小批上吃不满 GPU
（实测 16 行 ~18k 局面/秒、128 行 ~47k），多进程又只能分时轮流占用 GPU。服务进程把同一时刻
所有进程的请求按模型合成一批，GPU 越忙、排队的请求越多、批就越大（自然攒批，无需等待超时）。
搜索进程不建 CUDA context、不加载权重，显存只剩服务进程一份（权重 + 全体状态槽 + graph）。

协议（``/dev/shm`` 下一个目录，仅 Linux）：

- ``meta.json``：模型列表、客户端数、每客户端槽数 / 行数、服务 pid；``ready`` 出现后可连接；
- ``hdr``：int64[n_clients, 2] = (行数, 模型号)；``c{i}.in`` / ``c{i}.out``：客户端 i 的输入 /
  输出行（np.memmap，float32，行格式见 ``fast_eval.IN_DIM`` / ``OUT_DIM``）；
- ``req``：共享 FIFO，客户端写 1 字节（自己的编号），服务端一次读出全部就绪客户端；
- ``resp{i}``：客户端 i 的应答 FIFO，1 字节：0 完成 / 1 服务端出错（``error.txt``）；
- ``claim{i}``：客户端编号认领（O_EXCL 创建，写入 pid）。

每个客户端同一时刻至多一个在途请求。槽号：客户端 i 拥有 [2 + i·S, 2 + (i+1)·S)，
簿记（分配 / 钉住 / 淘汰 / 重算）都在客户端（``fast_eval.SlotPool``），服务端只按行里的
父槽 / 子槽号执行。

数值：服务端恒用批不变模式（块大小 ``chunk``，默认 ``fast_eval.FIXED_CHUNK``）——每行结果与同批其他行、拼批时机、
进程数、淘汰重算都无关，整个搜索逐位可复现（``tests/test_gpu_server.py`` 验证与同进程批不变
求值器逐位相同）。
"""
from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Optional, Sequence

import numpy as np

from .fast_eval import (FIXED_CHUNK, IN_DIM, OUT_DIM, RESERVED_SLOTS, SlotPool,
                        SsmFastEvaluator, StateTensors)

MAX_CLIENTS = 255           # 请求 FIFO 里客户端编号占 1 字节
DEFAULT_ROWS = 512          # 每客户端单次请求行数上限（24 局 × m0=16 = 384）
GPU_RESERVE_BYTES = 3 << 30  # 状态槽之外留给权重 / graph 池 / cuBLAS 工作区的显存
# 攒批最长等待（毫秒）：只影响吞吐，不影响结果。P4 实测闭环里等待反而更慢（客户端被拖住），默认 0
MAX_WAIT_MS = float(os.environ.get("UNICHESS_GPU_MAX_WAIT_MS", "0"))


def _paths(d: str) -> dict:
    return {"meta": os.path.join(d, "meta.json"), "ready": os.path.join(d, "ready"),
            "error": os.path.join(d, "error.txt"), "stats": os.path.join(d, "stats.json"),
            "hdr": os.path.join(d, "hdr"), "req": os.path.join(d, "req")}


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------ 服务端

# 服务进程用独立解释器启动（而非 multiprocessing spawn）：spawn 会在子进程里重新执行调用方的
# __main__，没有 main 保护的脚本（如性能剖析脚本）会因此启动失败。sys.path 照搬启动方。
_BOOT = ("import json, sys; spec = json.load(open(sys.argv[1], encoding='utf-8')); "
         "sys.path[:] = spec['sys_path']; "
         "from stateseq.gpu_server import _server_entry; _server_entry(spec)")


def _server_entry(spec: dict) -> None:
    d = spec["dir"]
    try:
        _serve(spec)
    except BaseException:  # noqa: 一切异常都写 error.txt，客户端与启动方据此报错
        with open(_paths(d)["error"], "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


def _serve(spec: dict) -> None:
    import torch

    from .fast_eval import StepEngine, model_dims
    from .kit_adapter import load_seq_model

    d, device = spec["dir"], spec["device"]
    P = _paths(d)
    parent = os.getppid()
    seqs, paths = [], []
    for ck in spec["checkpoints"]:
        seq, path = load_seq_model(ck, device)
        seqs.append(seq)
        paths.append(str(path))
    dims = model_dims(seqs[0])
    if any(model_dims(s) != dims for s in seqs):
        raise ValueError("共享 GPU 服务的各模型状态形状必须相同")
    n, R = int(spec["n_clients"]), int(spec["rows_per_client"])
    S = int(spec["slots_per_client"])
    bps = StateTensors.bytes_per_slot(dims)
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    cap = (free - GPU_RESERVE_BYTES) // (n * bps)
    if cap < S:
        if cap < 64:
            raise RuntimeError(f"显存不足：空闲 {free / 2**30:.1f} GiB，放不下 {n} 个客户端的状态槽")
        print(f"[gpu_server] 每客户端槽数 {S} 超出显存预算，降为 {cap}（淘汰 / 重算增多，结果不变）",
              file=sys.stderr, flush=True)
        S = int(cap)
    tensors = StateTensors(*dims, RESERVED_SLOTS + n * S, device)
    engines = [StepEngine(seq, tensors, device, cuda_graphs=spec["cuda_graphs"],
                          batch_invariant=True, chunk=int(spec["chunk"]), capacity=n * R)
               for seq in seqs]
    for eng in engines:
        eng.warmup()

    hdr = np.memmap(P["hdr"], dtype=np.int64, mode="w+", shape=(n, 2))
    cin = [np.memmap(os.path.join(d, f"c{i}.in"), dtype=np.float32, mode="w+", shape=(R, IN_DIM))
           for i in range(n)]
    cout = [np.memmap(os.path.join(d, f"c{i}.out"), dtype=np.float32, mode="w+",
                      shape=(R, OUT_DIM)) for i in range(n)]
    os.mkfifo(P["req"])
    for i in range(n):
        os.mkfifo(os.path.join(d, f"resp{i}"))
    # 服务端以读写方式打开全部 FIFO：打开不阻塞、客户端来去不会产生 EOF / EPIPE
    req_fd = os.open(P["req"], os.O_RDWR)
    resp_fds = [os.open(os.path.join(d, f"resp{i}"), os.O_RDWR) for i in range(n)]
    meta = {"pid": os.getpid(), "models": paths, "n_clients": n, "slots_per_client": S,
            "rows_per_client": R, "bytes_per_slot": bps, "chunk": int(spec["chunk"])}
    with open(P["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(P["ready"] + ".tmp", "w") as f:
        f.write("ok")
    os.replace(P["ready"] + ".tmp", P["ready"])

    stats = {"ticks": 0, "requests": 0, "rows": 0, "chunks": 0, "busy_s": 0.0,
             "max_tick_rows": 0, "slots_per_client": S, "chunk": int(spec["chunk"]),
             "max_wait_ms": float(spec["max_wait_ms"]), "held_ticks": 0, "held_s": 0.0}
    t_start = last_dump = time.perf_counter()
    K = int(spec["chunk"])
    max_wait = float(spec["max_wait_ms"]) / 1000.0
    live: set = set()               # 认领了编号且进程还在的客户端（每秒刷新）
    last_live = -1.0

    def dump():
        stats["wall_s"] = time.perf_counter() - t_start
        with open(P["stats"] + ".tmp", "w", encoding="utf-8") as f:
            json.dump(stats, f)
        os.replace(P["stats"] + ".tmp", P["stats"])

    def on_term(*_):                    # 启动方 stop()：写出最终统计后退出
        dump()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)

    def refresh_live():
        live.clear()
        for i in range(n):
            try:
                with open(os.path.join(d, f"claim{i}"), encoding="utf-8") as f:
                    pid = int(f.read() or 0)
            except (OSError, ValueError):
                continue
            if pid and _alive(pid):
                live.add(i)

    # 攒批：块大小固定（批不变），不满一块也按整块算。GPU 是瓶颈时，与其立刻算半空的块，
    # 不如在还有客户端正在算（尚未提交）时稍等，把块填满。触发条件任一：
    # 填充浪费 ≤ 1/4；存活客户端全都在等；最早的请求已等满 max_wait。只影响拼批时机，
    # 批不变 ⇒ 结果不变。
    pending: list = []
    first_t = 0.0
    while True:
        timeout = 1.0 if not pending else max(0.0, first_t + max_wait - time.perf_counter())
        ready_fds, _, _ = select.select([req_fd], [], [], timeout)
        now = time.perf_counter()
        if now - last_dump > 2.0:
            dump()
            last_dump = now
        if ready_fds:
            if not pending:
                first_t = now
            pending.extend(os.read(req_fd, 4096))
        elif not pending:
            if os.getppid() != parent:      # 启动方已退出（被杀）：不留孤儿进程
                return
            continue
        if now - last_live > 1.0:
            refresh_live()
            last_live = now
        by_model: dict = {}
        for cid in pending:
            rows, mid = int(hdr[cid, 0]), int(hdr[cid, 1])
            by_model.setdefault(mid, []).append((cid, rows))
        padded = sum(-(-sum(k for _, k in it) // K) * K for it in by_model.values())
        real = sum(k for it in by_model.values() for _, k in it)
        if (4 * (padded - real) > padded and not live.issubset(pending)
                and now - first_t < max_wait):
            continue
        if now - first_t > 1e-4:
            stats["held_ticks"] += 1
            stats["held_s"] += now - first_t
        ready, pending = pending, []
        t0 = time.perf_counter()
        status = b"\x00"
        try:
            tick_rows = 0
            for mid in sorted(by_model):
                items = by_model[mid]
                eng = engines[mid]
                total = sum(k for _, k in items)
                buf = eng.in_rows(total)
                off = 0
                for cid, k in items:
                    buf[off:off + k] = cin[cid][:k]
                    off += k
                out = eng.execute(total)
                off = 0
                for cid, k in items:
                    cout[cid][:k] = out[off:off + k]
                    off += k
                tick_rows += total
                stats["chunks"] += -(-total // eng.chunk)
            stats["ticks"] += 1
            stats["requests"] += len(ready)
            stats["rows"] += tick_rows
            stats["max_tick_rows"] = max(stats["max_tick_rows"], tick_rows)
        except BaseException:
            with open(P["error"], "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            status = b"\x01"
        stats["busy_s"] += time.perf_counter() - t0
        for cid in ready:
            os.write(resp_fds[cid], status)
        if status != b"\x00":
            dump()
            return


class GpuServer:
    """启动方（arena / 自对弈主进程）持有的服务句柄：``with GpuServer(...) as srv: ... srv.dir``。"""

    def __init__(self, checkpoints: Sequence, n_clients: int, slots_per_client: int, *,
                 rows_per_client: int = DEFAULT_ROWS, device: str = "cuda",
                 cuda_graphs: bool = True, chunk: int = FIXED_CHUNK,
                 max_wait_ms: float = MAX_WAIT_MS, root: Optional[str] = None):
        if not 1 <= int(n_clients) <= MAX_CLIENTS:
            raise ValueError(f"客户端数须在 1..{MAX_CLIENTS}")
        self.spec = {"checkpoints": [str(c) for c in checkpoints], "n_clients": int(n_clients),
                     "slots_per_client": int(slots_per_client),
                     "rows_per_client": int(rows_per_client), "device": device,
                     "cuda_graphs": bool(cuda_graphs), "chunk": int(chunk),
                     "max_wait_ms": float(max_wait_ms)}
        self.root = root if root is not None else ("/dev/shm" if os.path.isdir("/dev/shm")
                                                   else None)
        self.dir: Optional[str] = None
        self.proc = None
        self.meta: dict = {}
        self.final_stats: dict = {}

    def start(self, timeout: float = 600.0) -> "GpuServer":
        self.dir = tempfile.mkdtemp(prefix="unichess_gpu_", dir=self.root)
        spec = dict(self.spec, dir=self.dir, sys_path=[os.path.abspath(p) if p else os.getcwd()
                                                       for p in sys.path])
        spec_path = os.path.join(self.dir, "spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        # -I：不把 cwd / 脚本目录放进 sys.path（sys.path 由 spec 给定），也不读 PYTHON* 环境变量
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", _BOOT, spec_path],
                                     stdin=subprocess.DEVNULL)
        P = _paths(self.dir)
        t0 = time.time()
        while not os.path.exists(P["ready"]):
            if os.path.exists(P["error"]) or self.proc.poll() is not None:
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                msg = _read(P["error"]) if os.path.exists(P["error"]) \
                    else f"exitcode={self.proc.returncode}"
                self.stop()
                raise RuntimeError(f"GPU 服务启动失败：\n{msg}")
            if time.time() - t0 > timeout:
                self.stop()
                raise RuntimeError("GPU 服务启动超时")
            time.sleep(0.05)
        with open(P["meta"], encoding="utf-8") as f:
            self.meta = json.load(f)
        return self

    def stats(self) -> dict:
        try:
            with open(_paths(self.dir)["stats"], encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def stop(self) -> None:
        if self.proc is not None:
            alive = self.proc.poll() is None
            if alive:
                self.proc.terminate()       # SIGTERM：服务端写出最终统计后退出
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            if alive:
                self.final_stats = self.stats()
            self.proc = None
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

    def __enter__(self) -> "GpuServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ------------------------------------------------------------------ 客户端

class RemoteBackend:
    """搜索进程侧：认领一个客户端编号，持有该编号的槽簿记（同进程的 A/B 求值器共用）。"""

    _by_dir: dict = {}

    @classmethod
    def connect(cls, server_dir: str) -> "RemoteBackend":
        key = os.path.realpath(server_dir)
        be = cls._by_dir.get(key)
        if be is None:
            be = cls._by_dir[key] = cls(key)
        return be

    def __init__(self, d: str):
        P = _paths(d)
        if not os.path.exists(P["ready"]):
            raise RuntimeError(f"GPU 服务未就绪：{d}")
        with open(P["meta"], encoding="utf-8") as f:
            meta = json.load(f)
        self.dir = d
        self.meta = meta
        self.server_pid = int(meta["pid"])
        n, S, R = meta["n_clients"], meta["slots_per_client"], meta["rows_per_client"]
        cid = None
        for i in range(n):
            try:
                fd = os.open(os.path.join(d, f"claim{i}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            cid = i
            break
        if cid is None:
            raise RuntimeError(f"GPU 服务的 {n} 个客户端名额已满（workers 数须 ≤ 服务端 n_clients）")
        self.cid = cid
        self.max_rows = int(R)
        self.hdr = np.memmap(P["hdr"], dtype=np.int64, mode="r+", shape=(n, 2))
        self.cin = np.memmap(os.path.join(d, f"c{cid}.in"), dtype=np.float32, mode="r+",
                             shape=(R, IN_DIM))
        self.cout = np.memmap(os.path.join(d, f"c{cid}.out"), dtype=np.float32, mode="r",
                              shape=(R, OUT_DIM))
        self.req_fd = os.open(P["req"], os.O_WRONLY)
        self.resp_fd = os.open(os.path.join(d, f"resp{cid}"), os.O_RDONLY)
        self.pool = SlotPool(S, base=RESERVED_SLOTS + cid * S)
        self._model_ids = {os.path.realpath(p): k for k, p in enumerate(meta["models"])}

    def model_id(self, checkpoint_path: str) -> int:
        k = self._model_ids.get(os.path.realpath(str(checkpoint_path)))
        if k is None:
            raise KeyError(f"GPU 服务未加载该模型：{checkpoint_path}（已加载 {list(self._model_ids)}）")
        return k

    def in_rows(self, m: int) -> np.ndarray:
        return self.cin[:m]

    def execute(self, m: int, model_id: int) -> np.ndarray:
        self.hdr[self.cid, 0] = m
        self.hdr[self.cid, 1] = model_id
        os.write(self.req_fd, bytes((self.cid,)))
        while not select.select([self.resp_fd], [], [], 10.0)[0]:
            if not _alive(self.server_pid):
                raise RuntimeError("GPU 服务进程已退出")
        status = os.read(self.resp_fd, 1)
        if status != b"\x00":
            err = _paths(self.dir)["error"]
            msg = _read(err) if os.path.exists(err) else repr(status)
            raise RuntimeError(f"GPU 服务出错：\n{msg}")
        return self.cout[:m]


def remote_evaluator(server_dir: str, checkpoint, chunk: int = 0) -> SsmFastEvaluator:
    """chunk>0 时核对服务端块大小（块大小决定数值，配置里写的和实际跑的必须一致）。"""
    from .kit_adapter import _resolve
    path = _resolve(checkpoint).resolve()
    be = RemoteBackend.connect(server_dir)
    if chunk and int(chunk) != int(be.meta["chunk"]):
        raise ValueError(f"GPU 服务块大小 {be.meta['chunk']} 与配置 {chunk} 不符")
    return SsmFastEvaluator(be, be.pool, f"S-srv:{path}", model_id=be.model_id(str(path)))
