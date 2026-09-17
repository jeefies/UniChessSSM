"""自对弈（v3 分片）训练数据集：重放 + π′ 软目标（§2.5 / §2.6）。

结构上照搬 `dataset.SequenceDataset`（多进程重放、长度分桶预取），差异仅在于：
- 用 V3ShardReader 读取变长 π′ 目标（每 ply 全部合法着 + 概率）；
- 训练序列长度默认 300（不静默继承 Stage A 的 T_MAX=200，§2.6）；
- 额外产出稠密 (B,T,NUM_ACTIONS) 软目标张量，供 policy_soft_loss 使用；
- 额外产出 mlh_valid：封顶截断局（is_truncated=1）整局剔除 mlh（§2.5）。
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from ..actions import NUM_ACTIONS
from ..features import FEATURE_DIM
from .dataset import replay_game
from .gshards import V3ShardReader
from .sequences import T_MAX as STAGE_A_T_MAX

B2_T_MAX = 300


_reader: V3ShardReader | None = None


def _worker_init(shard_dir: str) -> None:
    global _reader
    _reader = V3ShardReader(shard_dir)


def _worker_build(index: int, t_max: int = B2_T_MAX):
    assert _reader is not None
    g = _reader.game(index)
    meta = g["meta"]
    data = replay_game(g["actions"], meta, t_max=t_max,
                       pipol_actions=g["pipol_actions"], pipol_probs=g["pipol_probs"])
    tc = int(meta["tc_bucket"])
    is_truncated = bool(meta["is_truncated"])
    return index, data, tc, is_truncated


class SelfPlayDataset:
    """v3 自对弈整序列数据集：采样局 → 多进程重放（含 π′）→ 拼批。"""

    def __init__(self, shard_dir: str, workers: int = 12, seed: int = 20260916, t_max: int = B2_T_MAX):
        self.reader = V3ShardReader(shard_dir)
        self.n_games = len(self.reader.meta_all)
        self.lengths = self.reader.meta_all["n_plies"].astype(np.int64)
        self.workers = workers
        self.seed = seed
        self.t_max = t_max
        self.pool = mp.Pool(workers, initializer=_worker_init, initargs=(shard_dir,))
        self.val_indices = [i for i in range(self.n_games) if bool(self.reader.is_val_arr[i])]
        self.train_indices = [i for i in range(self.n_games) if not bool(self.reader.is_val_arr[i])]
        if not self.val_indices:
            cut = max(self.n_games // 100, 1)
            self.val_indices = list(range(self.n_games - cut, self.n_games))
            self.train_indices = list(range(self.n_games - cut))

    def _collate(self, items: list[tuple[int, dict, int, bool]]) -> dict[str, torch.Tensor]:
        items = sorted(items, key=lambda it: -len(it[1]["actions"]))
        b = len(items)
        t = max(len(it[1]["actions"]) for it in items)
        from ..model import TrainBatch

        def pad_stack(key: str, dtype, shape_tail: tuple = ()) -> torch.Tensor:
            out = np.zeros((b, t) + shape_tail, dtype=dtype)
            for i, (_, d, *_rest) in enumerate(items):
                n = len(d["actions"])
                out[i, :n] = d[key]
            return torch.from_numpy(out)

        features = pad_stack("features", np.float32, (FEATURE_DIM,))
        legal = pad_stack("legal_mask", np.bool_, (NUM_ACTIONS,))
        actions = pad_stack("actions", np.int64)
        results = pad_stack("results", np.int64)
        moves_left = pad_stack("moves_left", np.float32)
        color = pad_stack("color", np.int64)
        valid = torch.from_numpy(
            np.arange(t)[None, :] < np.asarray([len(it[1]["actions"]) for it in items])[:, None]
        )
        elo_w = torch.ones(b, dtype=torch.float32)  # 自对弈条件固定（Elo 2567.5），无需 D9 加权
        elo_std = torch.zeros(b, dtype=torch.float32)
        tc = torch.tensor([it[2] for it in items], dtype=torch.long)

        # π′ 软目标：稠密 (B, T, NUM_ACTIONS)，非法/填充位置为 0
        soft_target = np.zeros((b, t, NUM_ACTIONS), dtype=np.float32)
        for i, (_, d, *_rest) in enumerate(items):
            for j, (acts, probs) in enumerate(zip(d["pipol_actions"], d["pipol_probs"])):
                if len(acts):
                    soft_target[i, j, acts] = probs

        is_trunc = torch.tensor([it[3] for it in items], dtype=torch.bool)
        mlh_valid = valid & ~is_trunc.unsqueeze(1)

        return {
            "batch": TrainBatch(features, actions, legal, results, moves_left,
                                elo_w, tc, elo_std, color),
            "valid": valid,
            "mlh_valid": mlh_valid,
            "policy_soft_target": torch.from_numpy(soft_target),
        }

    def epoch_batches(self, microbatch: int, device: str, shuffle: bool = True, prefetch: int = 3):
        if shuffle:
            order = np.argsort(self.lengths[self.train_indices], kind="stable")
            chunks = [self.train_indices[i] for i in order]
            chunks = [chunks[j:j + microbatch] for j in range(0, len(chunks), microbatch)]
            rng = np.random.default_rng(self.seed)
            rng.shuffle(chunks)
        else:
            chunks = [self.train_indices[j:j + microbatch]
                      for j in range(0, len(self.train_indices), microbatch)]
        pending: list = []
        for ch in chunks[:prefetch]:
            pending.append(self.pool.starmap_async(_worker_build, [(idx, self.t_max) for idx in ch]))
        for i, ch in enumerate(chunks):
            items = pending.pop(0).get()
            if i + prefetch < len(chunks):
                nxt = chunks[i + prefetch]
                pending.append(self.pool.starmap_async(_worker_build, [(idx, self.t_max) for idx in nxt]))
            yield self._build_batch(items, device)

    def _build_batch(self, items, device: str):
        out = self._collate(items)
        from .dataset import _to_device
        out["batch"] = _to_device(out["batch"], device)
        out["valid"] = out["valid"].to(device, non_blocking=True)
        out["mlh_valid"] = out["mlh_valid"].to(device, non_blocking=True)
        out["policy_soft_target"] = out["policy_soft_target"].to(device, non_blocking=True)
        return out

    def val_batch(self, n_batches: int, microbatch: int, device: str, seed: int = 778):
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self.val_indices, size=min(n_batches * microbatch, len(self.val_indices)),
                          replace=False)
        out = []
        for j in range(0, len(idxs), microbatch):
            batch_idx = list(idxs[j:j + microbatch])
            items = self.pool.starmap(_worker_build, [(idx, self.t_max) for idx in batch_idx])
            out.append(self._build_batch(items, device))
        return out

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()
