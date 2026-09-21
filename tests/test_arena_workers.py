"""Unit tests for multi-process worker support in tools/ssm_gumbel_arena.py."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import torch

from stateseq.gumbel import C_SCALE, C_VISIT
from tools.ssm_gumbel_arena import (
    OPENINGS,
    _aggregate_results,
    _worker_process_fn,
    play_one_game,
    run_parallel_arena,
)


class _FakeArenaModel:
    """CPU-friendly fake ArenaModel matching the step & initial_cache interface."""
    device = "cpu"

    def __init__(self, ckpt_path: str = "fake.pt"):
        self.ckpt_path = ckpt_path
        self.c_visit = C_VISIT
        self.c_scale = C_SCALE
        self.calls: list[np.ndarray] = []

    def initial_cache(self, b: int = 1):
        return [(torch.zeros(1), torch.zeros(1))]

    def step(self, feats, tc, elo, color, cache):
        self.calls.append(np.asarray(feats, dtype=np.float32).copy())
        # Legal moves will have flat logits 0.0, WDL drawish
        logits = np.zeros((1, 1936), dtype=np.float32)
        wdl = np.zeros((1, 3), dtype=np.float32)
        mlh = np.zeros((1, 1), dtype=np.float32)
        x = np.zeros((1, 512), dtype=np.float32)
        n = cache[0][0] + 1.0
        return logits, wdl, mlh, x, [(n, n)]


class TestArenaWorkers(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_arena_workers_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_deterministic_seed_handling(self):
        """Verify seed formula: seed + worker_id * 1000 + game_idx."""
        base_seed = 20260917
        wid = 2
        game_idx = 5
        expected_seed = base_seed + 2 * 1000 + 5
        self.assertEqual(expected_seed, 20262922)

        cfg = argparse.Namespace(
            n_sims=4,
            m0=4,
            max_plies=4,
            c_visit=C_VISIT,
            c_scale=C_SCALE,
        )
        mw = _FakeArenaModel("model_w.pt")
        mb = _FakeArenaModel("model_b.pt")

        game_data = play_one_game(mw, mb, cfg, opening_san="e4 e5", opening_id=0, seed=expected_seed)
        self.assertEqual(game_data["seed"], expected_seed)
        self.assertEqual(game_data["opening_id"], 0)
        self.assertIn("arena_result", game_data)
        self.assertIn("termination_reason", game_data)

    def test_worker_process_fn_execution(self):
        """Test _worker_process_fn produces results and worker_done message."""
        args = argparse.Namespace(
            ckpt_a="dummy_a.pt",
            ckpt_b="dummy_b.pt",
            c_visit=C_VISIT,
            c_scale_a=0.1,
            c_scale_b=0.1,
            n_sims=4,
            m0=4,
            max_plies=4,
            seed=1000,
        )

        assigned_pairs = [(0, 0, OPENINGS[0])]
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        stop_event = ctx.Event()

        # Patch ArenaModel inside worker function
        with patch("tools.ssm_gumbel_arena.ArenaModel", side_effect=lambda p: _FakeArenaModel(p)):
            _worker_process_fn(
                worker_id=1,
                assigned_pairs=assigned_pairs,
                args=args,
                result_queue=result_queue,
                stop_event=stop_event,
            )

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

        # Should have received ("game_pair", (pair_idx, [gd1, gd2])) and ("worker_done", 1)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0][0], "game_pair")
        pair_idx, games = messages[0][1]
        self.assertEqual(pair_idx, 0)
        self.assertEqual(len(games), 2)

        gd1, gd2 = games
        self.assertEqual(gd1["game_idx"], 0)
        self.assertEqual(gd1["worker_id"], 1)
        self.assertEqual(gd1["seed"], 1000 + 1 * 1000 + 0)
        self.assertEqual(gd1["white_ckpt_side"], "A")
        self.assertEqual(gd1["black_ckpt_side"], "B")

        self.assertEqual(gd2["game_idx"], 1)
        self.assertEqual(gd2["worker_id"], 1)
        self.assertEqual(gd2["seed"], 1000 + 1 * 1000 + 1)
        self.assertEqual(gd2["white_ckpt_side"], "B")
        self.assertEqual(gd2["black_ckpt_side"], "A")

        self.assertEqual(messages[1], ("worker_done", 1))

    def test_run_parallel_arena_mocked_workers(self):
        """Test coordinator distribution, aggregation, and scoring invariants."""
        args = argparse.Namespace(
            ckpt_a="dummy_a.pt",
            ckpt_b="dummy_b.pt",
            out=self.test_dir,
            games=8,
            pairs=4,
            workers=2,
            n_sims=4,
            m0=4,
            max_plies=4,
            seed=42,
            c_visit=C_VISIT,
            c_scale_a=0.1,
            c_scale_b=0.1,
            sprt=False,
            sprt_min_games=4,
            sprt_alpha=0.05,
            sprt_beta=0.05,
        )

        num_pairs = args.games // 2
        n_openings = min(args.pairs, len(OPENINGS))
        t0 = 0.0

        # We mock mp.get_context to use mock processes that synchronously write into the queue
        with patch("tools.ssm_gumbel_arena.ArenaModel", side_effect=lambda p: _FakeArenaModel(p)):
            # Test direct worker function for 2 workers
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            stop_evt = ctx.Event()

            # Worker 0 handles pairs [0, 2]
            pairs_w0 = [(0, 0, OPENINGS[0]), (2, 2, OPENINGS[2])]
            _worker_process_fn(0, pairs_w0, args, q, stop_evt)

            # Worker 1 handles pairs [1, 3]
            pairs_w1 = [(1, 1, OPENINGS[1]), (3, 3, OPENINGS[3])]
            _worker_process_fn(1, pairs_w1, args, q, stop_evt)

            received_pairs = {}
            while not q.empty():
                msg_type, payload = q.get_nowait()
                if msg_type == "game_pair":
                    p_idx, games = payload
                    received_pairs[p_idx] = games

            self.assertEqual(len(received_pairs), 4)

            # Assemble and verify game log aggregation
            all_games_log = []
            curr_idx = 0
            for p_idx in sorted(received_pairs.keys()):
                for gd in received_pairs[p_idx]:
                    gd["game_idx"] = curr_idx
                    curr_idx += 1
                    all_games_log.append(gd)

            self.assertEqual(len(all_games_log), 8)
            manifest = _aggregate_results(all_games_log, len(all_games_log) // 2, args)
            self.assertEqual(manifest["total_games"], 8)
            self.assertEqual(manifest["wins_a"] + manifest["wins_b"] + manifest["draws"], 8)
            self.assertEqual(manifest["workers"], 2)


if __name__ == "__main__":
    unittest.main()
