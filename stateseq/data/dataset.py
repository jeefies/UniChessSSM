"""重放式数据集：从动作列表分片重建逐步张量（特征/合法mask/标签），多进程预取。

单 worker 实测 ~16.6k 步/s（python-chess 合法着生成为主），16 worker 充足喂 GPU。
Elo 加权：w = (e−e_min)/(e_max−e_min)·(r−1)+1（r=20，e_min/e_max 为构建期 P1/P99）；
缺失 Elo 的局权重 1.0、elo_std 0（conditions.UNKNOWN 桶，D2）。
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import torch

from ..actions import NUM_ACTIONS, move_to_action
from ..features import FEATURE_DIM, encode
from ..losses import elo_weights
from .gshards import ShardReader
from .sequences import T_MAX, _board_key


def replay_game(actions: np.ndarray, meta: np.ndarray) -> dict[str, np.ndarray]:
    """动作 id 序列 → 逐步张量（含断言：动作必须在合法着集合内，规则引擎权威）。
    序列截断到 T_MAX=200（设计文档 T≤200，§7.3）。"""
    import chess  # 延迟导入：worker 进程内初始化

    if len(actions) > T_MAX:
        actions = actions[:T_MAX]
    board = chess.Board()
    n = len(actions)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    masks = np.zeros((n, NUM_ACTIONS), dtype=bool)
    results = np.zeros(n, dtype=np.int64)
    moves_left = np.zeros(n, dtype=np.float32)
    color = np.zeros(n, dtype=np.int64)

    occurrences: dict = {}
    result = int(meta["result"])
    for t in range(n):
        key = _board_key(board)
        prior = occurrences.get(key, 0)
        occurrences[key] = prior + 1

        legal = list(board.legal_moves)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        played = None
        for mv in legal:
            a = move_to_action(mv)
            mask[a] = True
            if a == actions[t]:
                played = mv
        if played is None:  # pragma: no cover - 数据完整性防线
            raise ValueError(f"动作 {actions[t]} 不在合法着集合: {board.fen()}")

        feats[t] = encode(board, occurrence=prior)
        masks[t] = mask
        # result 为白视角（0 胜/1 和/2 负）；走子方归一：白走步取原值，黑走步翻转
        results[t] = result if board.turn == chess.WHITE else (2 - result)
        moves_left[t] = min(n - t, T_MAX)
        color[t] = 1 if board.turn == chess.WHITE else 0
        board.push(played)
    return {
        "features": feats,
        "legal_mask": masks,
        "actions": actions.astype(np.int64),
        "results": results,
        "moves_left": moves_left,
        "color": color,
    }


_reader: ShardReader | None = None
_elo_stats: dict | None = None


def _worker_init(shard_dir: str) -> None:
    global _reader, _elo_stats
    _reader = ShardReader(shard_dir)
    _elo_stats = _reader.manifest.get("elo_stats", {})


def _worker_build(index: int) -> tuple[int, dict[str, np.ndarray], float, float, int]:
    assert _reader is not None and _elo_stats is not None
    meta, actions = _reader.game(index)
    data = replay_game(actions, meta)
    elo = float(meta["elo_mean"])
    if int(meta["elo_missing"]):
        w, elo_std = 1.0, 0.0
    else:
        w = float(elo_weights(torch.tensor(elo), _elo_stats["e_min"], _elo_stats["e_max"], r=20.0))
        elo_std = (elo - _elo_stats["mean"]) / _elo_stats["std"]
    tc = int(meta["tc_bucket"])
    return index, data, w, elo_std, tc


class SequenceDataset:
    """整序列数据集：采样局 → 多进程重放 → 拼 TrainBatch（按长度排序减少填充）。"""

    def __init__(self, shard_dir: str, workers: int = 12, seed: int = 20260915):
        self.reader = ShardReader(shard_dir)
        self.n_games = len(self.reader.meta_all)
        self.lengths = self.reader.meta_all["n_plies"].astype(np.int64)
        self.workers = workers
        self.seed = seed
        self.pool = mp.Pool(workers, initializer=_worker_init, initargs=(shard_dir,))
        self.val_indices = [i for i in range(self.n_games) if bool(self.reader.is_val_arr[i])]
        self.train_indices = [i for i in range(self.n_games) if not bool(self.reader.is_val_arr[i])]
        if not self.val_indices:  # 小样本兜底：尾部 1% 作 val
            cut = max(self.n_games // 100, 1)
            self.val_indices = list(range(self.n_games - cut, self.n_games))
            self.train_indices = list(range(self.n_games - cut))

    def _collate(self, items: list[tuple[int, dict, float, float, int]]) -> dict[str, torch.Tensor]:
        items = sorted(items, key=lambda it: -len(it[1]["actions"]))
        b = len(items)
        t = max(len(it[1]["actions"]) for it in items)
        from ..model import TrainBatch  # 延迟导入避免环

        # numpy 侧 pad+stack（逐行 np 赋值是 memcpy），再一次 from_numpy；避免 torch 逐行 setitem（实测慢 100×）
        def pad_stack(key: str, dtype, shape_tail: tuple = ()) -> torch.Tensor:
            out = np.zeros((b, t) + shape_tail, dtype=dtype)
            for i, (_, d, *_rest) in enumerate(items):
                n = len(d["actions"])
                out[i, :n] = d[key]
            return torch.from_numpy(out)

        features = pad_stack("features", np.float32, (FEATURE_DIM,))
        legal = pad_stack("legal_mask", np.bool_, (NUM_ACTIONS,))
        actions = pad_stack("actions", np.int64)
        results = pad_stack("results", np.int64)          # 填充位保持 0（masked 不贡献）
        moves_left = pad_stack("moves_left", np.float32)
        color = pad_stack("color", np.int64)
        valid = torch.from_numpy(
            np.arange(t)[None, :] < np.asarray([len(it[1]["actions"]) for it in items])[:, None]
        )
        elo_w = torch.tensor([it[2] for it in items], dtype=torch.float32)
        elo_std = torch.tensor([it[3] for it in items], dtype=torch.float32)
        tc = torch.tensor([it[4] for it in items], dtype=torch.long)
        return {
            "batch": TrainBatch(features, actions, legal, results, moves_left,
                                elo_w, tc, elo_std, color),
            "valid": valid,
        }

    def epoch_batches(self, microbatch: int, device: str, shuffle: bool = True, prefetch: int = 3):
        """生成一整 epoch 的 microbatch；长度分桶（块内等长减少填充）+ map_async 预取重叠 GPU。"""
        if shuffle:
            order = np.argsort(self.lengths[self.train_indices], kind="stable")
            chunks = [self.train_indices[i] for i in order]
            chunks = [chunks[j:j + microbatch] for j in range(0, len(chunks), microbatch)]
            rng = np.random.default_rng(self.seed)
            rng.shuffle(chunks)  # 只乱桶序，桶内保持长度有序
        else:
            chunks = [self.train_indices[j:j + microbatch]
                      for j in range(0, len(self.train_indices), microbatch)]
        pending: list = []
        for ch in chunks[:prefetch]:
            pending.append(self.pool.map_async(_worker_build, ch))
        for i, ch in enumerate(chunks):
            items = pending.pop(0).get()
            if i + prefetch < len(chunks):
                pending.append(self.pool.map_async(_worker_build, chunks[i + prefetch]))
            yield self._build_batch(items, device)

    def _build_batch(self, items: list[tuple[int, dict, float, float, int]], device: str):
        out = self._collate(items)
        out["batch"] = _to_device(out["batch"], device)
        out["valid"] = out["valid"].to(device, non_blocking=True)
        return out

    def val_batch(self, n_batches: int, microbatch: int, device: str, seed: int = 777):
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self.val_indices, size=min(n_batches * microbatch, len(self.val_indices)),
                          replace=False)
        out = []
        for j in range(0, len(idxs), microbatch):
            items = self.pool.map(_worker_build, list(idxs[j:j + microbatch]))
            out.append(self._build_batch(items, device))
        return out

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()


def _to_device(batch, device: str):
    from ..model import TrainBatch

    return TrainBatch(*[getattr(batch, f).to(device, non_blocking=True)
                        for f in batch.__dataclass_fields__])
