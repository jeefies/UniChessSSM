"""P4 共享 GPU 服务（``stateseq.gpu_server``）。需要 CUDA、Linux（/dev/shm + FIFO）与兄弟仓库 Kit。

1. 服务端求值器与同进程批不变求值器整次搜索逐位相同（含小槽池淘汰 + 重算）；
2. 两个模型按模型号分派，互不串台；
3. 多个搜索进程同时连一个服务：各自结果与单进程逐位相同（与拼批时机 / 进程数无关）；
4. 出错传播：坏请求 → 客户端 RuntimeError（带服务端 traceback）；坏权重 → 启动即报错。
"""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
KIT_ROOT = os.environ.get("UNICHESS_KIT_ROOT", os.path.join(os.path.dirname(HERE), "Kit"))
if os.path.isdir(KIT_ROOT) and KIT_ROOT not in sys.path:
    sys.path.append(KIT_ROOT)  # 追加而非前插：Kit 的 tests 包不得遮蔽本仓库的 tests

try:
    import torch
    import chess
    import numpy as np

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - 本机（Windows）无 torch
    _HAS_CUDA = False

try:
    import unichess_kit  # noqa: F401

    _HAS_KIT = True
except ImportError:  # pragma: no cover
    _HAS_KIT = False

FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
]


def _search(evaluator, fen: str, sims: int = 64, parallel: bool = True) -> tuple:
    """一次完整 Gumbel 搜索 → (着法, 根访问数 bytes, Q 累加 bytes, π′ bytes, 节点数)。"""
    from stateseq import kit_adapter as ka
    from unichess_kit.api import GameStart
    from unichess_kit.runtime import Batcher, run_sync
    from unichess_kit.search.gumbel import GumbelConfig

    board = chess.Board(fen)
    cfg = GumbelConfig(simulations=sims, m0=8, g=0.0, parallel=parallel)
    player = ka.SsmPlayer("S", evaluator, cfg)
    run_sync(player.new_game(GameStart(color=board.turn, fen=fen, seed=11)))
    batcher = Batcher()
    root, _ = run_sync(player._root(board), batcher)
    res = run_sync(player._search(board, root, np.random.default_rng(5), sims), batcher)
    out = (str(res.move), res.root.n.tobytes(), res.root.q_sum.tobytes(),
           res.pi_prime(cfg)[1].tobytes(), res.stats["n_nodes"])
    player.close()
    return out


def _worker_main(server_dir: str, ckpt: str, out: str) -> None:
    """子进程入口（``python test_gpu_server.py --worker DIR CKPT OUT``）：连服务、依次搜索，结果 pickle 到 OUT。"""
    from stateseq.gpu_server import remote_evaluator

    ev = remote_evaluator(server_dir, ckpt)
    res = [_search(ev, f) for f in FENS]
    with open(out, "wb") as f:
        pickle.dump(res, f)


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT and os.path.isdir("/dev/shm"),
                     "需要 CUDA、Linux /dev/shm 与兄弟仓库 Kit")
class TestGpuServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stateseq import fast_eval as fe
        from stateseq.gpu_server import GpuServer
        from stateseq.model import SeqModel

        cls.fe = fe
        cls.tmp = tempfile.mkdtemp(prefix="p4_gpusrv_test_")
        cls.ckpts = []
        for seed in (3, 4):
            torch.manual_seed(seed)
            seq = SeqModel(dropout=0.0)
            path = os.path.join(cls.tmp, f"m{seed}.pt")
            torch.save({"model": seq.state_dict()}, path)
            cls.ckpts.append(path)
        cls.server = GpuServer(cls.ckpts, n_clients=4, slots_per_client=256).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _local(self, k: int, slots: int = 1024):
        from stateseq.kit_adapter import load_seq_model

        fe = self.fe
        seq, _ = load_seq_model(self.ckpts[k], "cuda")
        store = (fe.StateTensors(*fe.model_dims(seq), fe.RESERVED_SLOTS + slots, "cuda"),
                 fe.SlotPool(slots))
        return fe.SsmFastEvaluator.local(seq, "cuda", f"S:inv{k}", batch_invariant=True,
                                         store=store)

    def _remote(self, k: int):
        from stateseq.gpu_server import remote_evaluator

        return remote_evaluator(self.server.dir, self.ckpts[k])

    def test_search_bitwise_vs_local_invariant(self):
        remote = self._remote(0)
        for fen in FENS:
            with self.subTest(fen=fen):
                self.assertEqual(_search(self._local(0), fen), _search(remote, fen))

    def test_tiny_server_pool_eviction_bitwise(self):
        """每客户端仅 12 槽（串行模拟）：大量淘汰 + 重算，结果仍与同进程大池逐位相同。"""
        from stateseq.gpu_server import GpuServer, remote_evaluator

        with GpuServer(self.ckpts[:1], n_clients=1, slots_per_client=12) as srv:
            remote = remote_evaluator(srv.dir, self.ckpts[0])
            for fen in FENS[:2]:
                with self.subTest(fen=fen):
                    self.assertEqual(_search(self._local(0), fen, parallel=False),
                                     _search(remote, fen, parallel=False))
            self.assertGreater(remote.pool.evictions, 10)
            self.assertGreater(remote.pool.rematerialized, 0)

    def test_models_dispatched_by_id(self):
        a, b = self._remote(0), self._remote(1)
        self.assertIs(a.pool, b.pool, "同进程的 A/B 求值器共用本客户端的槽簿记")
        self.assertNotEqual(a.model_id, b.model_id)
        ra, rb = _search(a, FENS[0], sims=32), _search(b, FENS[0], sims=32)
        self.assertNotEqual(ra[3], rb[3])
        self.assertEqual(rb, _search(self._local(1), FENS[0], sims=32))

    def test_worker_processes_bitwise(self):
        """两个搜索进程同时打同一服务（请求交错拼批）：与本进程结果逐位相同。"""
        expect = [_search(self._local(0), f) for f in FENS]
        outs = [os.path.join(self.tmp, f"w{i}.pkl") for i in range(2)]
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker",
                                   self.server.dir, self.ckpts[0], o],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                 for o in outs]
        for p in procs:
            log = p.communicate(timeout=600)[0].decode(errors="replace")
            self.assertEqual(p.returncode, 0, log[-3000:])
        for o in outs:
            with open(o, "rb") as f:
                self.assertEqual(pickle.load(f), expect)

    def test_unknown_checkpoint_rejected(self):
        from stateseq.gpu_server import remote_evaluator

        with self.assertRaises(KeyError):
            remote_evaluator(self.server.dir, os.path.join(self.tmp, "nope.pt"))


@unittest.skipUnless(_HAS_CUDA and _HAS_KIT and os.path.isdir("/dev/shm"),
                     "需要 CUDA、Linux /dev/shm 与兄弟仓库 Kit")
class TestGpuServerErrors(unittest.TestCase):
    def test_bad_checkpoint_fails_at_start(self):
        from stateseq.gpu_server import GpuServer

        with self.assertRaises(RuntimeError) as cm:
            GpuServer(["/nonexistent/model.pt"], n_clients=1, slots_per_client=64).start()
        self.assertIn("启动失败", str(cm.exception))

    def test_server_error_propagates_to_client(self):
        from stateseq.gpu_server import GpuServer, RemoteBackend
        from stateseq.model import SeqModel

        tmp = tempfile.mkdtemp(prefix="p4_gpusrv_err_")
        try:
            path = os.path.join(tmp, "m.pt")
            torch.save({"model": SeqModel(dropout=0.0).state_dict()}, path)
            with GpuServer([path], n_clients=1, slots_per_client=64) as srv:
                be = RemoteBackend.connect(srv.dir)
                be.in_rows(1)[:] = 0
                with self.assertRaises(RuntimeError) as cm:
                    be.execute(1, 7)          # 不存在的模型号 → 服务端 IndexError
                self.assertIn("IndexError", str(cm.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker_main(*sys.argv[2:])
    else:
        unittest.main()
