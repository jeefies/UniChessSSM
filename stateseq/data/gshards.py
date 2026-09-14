"""动作列表分片：每局一条变长记录（~150 B/局），训练时重放重建。

布局（每片两个文件，原子保存）：
- shard-XXXXX.meta.npz：offsets (N+1, int64) 指向动作池；meta 结构化数组：
    n_plies u8 / tc_bucket u8 / result u8 / elo_missing u8 / elo_mean f16 / game_key U16
- shard-XXXXX.actions.bin：uint16 动作 id 池（n_plies 个/局）

game_key：SHA256(month:局序号) 前 8 字节十六进制 → train/val 划分（val 0.5%，同局不跨集）。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np

from ..conditions import time_control_bucket

RESULT_TO_LABEL = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
GAMES_PER_SHARD = 50_000


def make_game_key(month: str, game_index: int) -> str:
    return hashlib.sha256(f"{month}:{game_index}".encode()).hexdigest()[:16]


def is_val_key(game_key: str, val_frac: float = 0.005) -> bool:
    return int(game_key[:2], 16) < val_frac * 256


def encode_game_record(actions: list[int], tc: str | None, result: str,
                       elo_mean: float | None, game_key: str) -> tuple[np.ndarray, np.ndarray]:
    """→ (meta_structured_scalar, actions uint16 数组)。"""
    meta = np.zeros((), dtype=META_DTYPE)
    meta["n_plies"] = len(actions)
    meta["tc_bucket"] = int(time_control_bucket(tc))
    meta["result"] = RESULT_TO_LABEL[result]
    meta["elo_missing"] = 0 if elo_mean is not None else 1
    meta["elo_mean"] = 1500.0 if elo_mean is None else float(elo_mean)
    meta["game_key"] = game_key
    return meta, np.asarray(actions, dtype=np.uint16)


META_DTYPE = np.dtype([
    ("n_plies", np.uint16),
    ("tc_bucket", np.uint8),
    ("result", np.uint8),
    ("elo_missing", np.uint8),
    ("elo_mean", np.float16),
    ("game_key", "U16"),
])


class ShardBuilder:
    """worker 内顺序写入；每 GAMES_PER_SHARD 局flush一片。"""

    def __init__(self, out_dir: str, tag: str):
        self.out_dir = out_dir
        self.tag = tag
        os.makedirs(out_dir, exist_ok=True)
        self._metas: list[np.ndarray] = []
        self._action_chunks: list[np.ndarray] = []
        self.shard_files: list[str] = []
        self.games = 0
        self.steps = 0

    def add(self, meta: np.ndarray, actions: np.ndarray) -> None:
        self._metas.append(meta)
        self._action_chunks.append(actions)
        self.games += 1
        self.steps += int(meta["n_plies"])
        if len(self._metas) >= GAMES_PER_SHARD:
            self.flush()

    def flush(self) -> None:
        if not self._metas:
            return
        idx = len(self.shard_files)
        metas = np.stack(self._metas)
        pool = np.concatenate(self._action_chunks)
        offsets = np.zeros(len(metas) + 1, dtype=np.int64)
        np.cumsum(metas["n_plies"], out=offsets[1:])
        base = os.path.join(self.out_dir, f"shard-{self.tag}-{idx:05d}")
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        pool.tofile(tmp)
        os.replace(tmp, base + ".actions.bin")
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        np.savez(tmp, metas=metas, offsets=offsets)
        os.replace(tmp, base + ".meta.npz")
        self.shard_files.append(base)
        self._metas, self._action_chunks = [], []


def write_manifest(shard_dir: str, shards: list[str], months: list[str],
                   stats: dict) -> None:
    manifest = {
        "shards": [os.path.basename(s) for s in shards],
        "months": months,
        "games": stats.get("games", 0),
        "steps": stats.get("steps", 0),
        "skipped": stats.get("skipped", 0),
    }
    fd, tmp = tempfile.mkstemp(dir=shard_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(shard_dir, "manifest.json"))


class ShardReader:
    """训练加载器侧：mmap 读取全部片的动作池与 meta。"""

    def __init__(self, shard_dir: str):
        with open(os.path.join(shard_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.metas: list[np.ndarray] = []
        self.offsets: list[np.ndarray] = []
        self.pools: list[np.memmap] = []
        for name in self.manifest["shards"]:
            base = os.path.join(shard_dir, name)
            npz = np.load(base + ".meta.npz")
            self.metas.append(npz["metas"])
            self.offsets.append(npz["offsets"])
            self.pools.append(np.memmap(base + ".actions.bin", dtype=np.uint16, mode="r"))
        self.meta_all = np.concatenate(self.metas)
        counts = [len(m) for m in self.metas]
        self.shard_of = np.repeat(np.arange(len(counts)), counts)
        self._cumcounts = np.concatenate([[0], np.cumsum(counts)])

    def game(self, global_index: int) -> tuple[np.ndarray, np.ndarray]:
        """→ (meta 标量, 动作 uint16 数组)。"""
        shard = int(self.shard_of[global_index])
        local = global_index - int(self._cumcounts[shard])
        npz_meta = self.metas[shard][local]
        start = int(self.offsets[shard][local])
        n = int(npz_meta["n_plies"])
        return npz_meta, np.asarray(self.pools[shard][start:start + n])
