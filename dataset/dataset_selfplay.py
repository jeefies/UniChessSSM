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
from .gshards import V3ShardReader, validate_v3_pipol
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
    # §2.5 读取端完整性校验：概率和 ∈ [0.99,1.01]、支持集与规则引擎合法着一致
    if data.get("pipol_actions") is not None:
        validate_v3_pipol(data["pipol_actions"], data["pipol_probs"], len(data["actions"]),
                          legal_masks=data["legal_mask"])
    tc = int(meta["tc_bucket"])
    is_truncated = bool(meta["is_truncated"])
    # np.void structured scalar：只能按字段名取值（无 .get）
    elo_mean = float(meta["elo_mean"])
    # flags = 本局开局注入 ply 数（旧分片为 0）；供拼批构造 book_mask（§P1-3 降权）
    book_plies = int(meta["flags"]) if "flags" in meta.dtype.names else 0
    return index, data, tc, is_truncated, elo_mean, book_plies


def build_book_mask(lengths: "list[int] | np.ndarray", book_counts: "list[int] | np.ndarray",
                    t: int) -> np.ndarray:
    """(B,T) bool：第 i 局的前 book_counts[i] 个**有效** ply 为 True（开局注入段）。

    同时受该局实际长度约束：超出 n_plies 或 t_max 的填充位一律 False。
    旧分片 flags=0 ⇒ 全 False。
    """
    idx = np.arange(t)[None, :]
    return (idx < np.asarray(book_counts, dtype=np.int64)[:, None]) & \
           (idx < np.asarray(lengths, dtype=np.int64)[:, None])


class SelfPlayDataset:
    """v3 自对弈整序列数据集：采样局 → 多进程重放（含 π′）→ 拼批。"""

    def __init__(self, shard_dir: str, workers: int = 12, seed: int = 20260916, t_max: int = B2_T_MAX):
        self.reader = V3ShardReader(shard_dir)
        self.n_games = len(self.reader.meta_all)
        self.lengths = self.reader.meta_all["n_plies"].astype(np.int64)
        self.workers = workers
        self.seed = seed
        self.t_max = t_max
        # spawn 而非 Linux 默认的 fork：调用方进程往往已初始化 CUDA（模型已上卡 / 同进程的 GPU 测试），
        # fork 出的子进程继承 CUDA 对象，回收时 "CUDA error: initialization error" 崩溃、池挂死。
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(shard_dir,))
        self.val_indices = [i for i in range(self.n_games) if bool(self.reader.is_val_arr[i])]
        self.train_indices = [i for i in range(self.n_games) if not bool(self.reader.is_val_arr[i])]
        if not self.val_indices:
            cut = max(self.n_games // 100, 1)
            self.val_indices = list(range(self.n_games - cut, self.n_games))
            self.train_indices = list(range(self.n_games - cut))

    def _collate(self, items: list[tuple]) -> dict[str, torch.Tensor]:
        """items: (index, data, tc, is_truncated, elo_mean, book_plies) 元组列表。"""
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
        from ..conditions import standardize_elo
        elo_arr = np.array([it[4] for it in items], dtype=np.float64)
        elo_std = torch.tensor(standardize_elo(elo_arr).astype(np.float32))
        tc = torch.tensor([it[2] for it in items], dtype=torch.long)

        # π′ 软目标：稠密 (B, T, NUM_ACTIONS)，非法/填充位置为 0
        soft_target = np.zeros((b, t, NUM_ACTIONS), dtype=np.float32)
        for i, (_, d, *_rest) in enumerate(items):
            for j, (acts, probs) in enumerate(zip(d["pipol_actions"], d["pipol_probs"])):
                if len(acts):
                    soft_target[i, j, acts] = probs

        is_trunc = torch.tensor([it[3] for it in items], dtype=torch.bool)
        mlh_valid = valid & ~is_trunc.unsqueeze(1)
        # 开局注入段掩码（B,T）：训练侧对 book ply 的 policy 软 CE 降权（§P1-3）。
        # book 着法是分布外强制目标，且审计显示开局并非主要败因，降权让训练聚焦
        # 模型自身搜索产生的 π′。旧分片 flags=0 ⇒ 全 False ⇒ 权重全 1，行为不变。
        book_mask = torch.from_numpy(build_book_mask(
            [len(it[1]["actions"]) for it in items], [it[5] for it in items], t))

        return {
            "batch": TrainBatch(features, actions, legal, results, moves_left,
                                elo_w, tc, elo_std, color),
            "valid": valid,
            "mlh_valid": mlh_valid,
            "policy_soft_target": torch.from_numpy(soft_target),
            "book_mask": book_mask,
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
        out["book_mask"] = out["book_mask"].to(device, non_blocking=True)
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
