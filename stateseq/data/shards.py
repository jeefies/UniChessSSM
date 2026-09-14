"""分片存储（设计文档 §4.2）：二进制分片 + 按 game_id 哈希划分 train/val（同一局不得跨集，val 0.5%）。

每片固定局数；逐步记录为固定大小（STEP_DTYPE），索引文件记录 (game_key, start, length)。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

import numpy as np

from ..actions import NUM_ACTIONS
from ..features import FEATURE_DIM
from .sequences import StepRecord, T_MAX

STEP_DTYPE = np.dtype([
    ("features", np.float32, FEATURE_DIM),
    ("legal_mask", np.uint8, NUM_ACTIONS),
    ("action", np.int32),
    ("result", np.int8),
    ("moves_left", np.int16),
    ("tc_bucket", np.int8),
    ("elo_mean", np.float32),
    ("elo_missing", np.uint8),
    ("color", np.uint8),
])

VAL_FRAC = 0.005  # val 取 0.5%


def record_to_array(rec: StepRecord) -> np.ndarray:
    arr = np.zeros((), dtype=STEP_DTYPE)
    arr["features"] = rec.features
    arr["legal_mask"] = rec.legal_mask.astype(np.uint8)
    arr["action"] = rec.action
    arr["result"] = rec.result
    arr["moves_left"] = rec.moves_left
    arr["tc_bucket"] = rec.tc_bucket
    arr["elo_mean"] = rec.elo_mean
    arr["elo_missing"] = int(rec.elo_missing)
    arr["color"] = rec.color
    return arr


def game_key(game_index: int, source_tag: str = "") -> str:
    """game_id 哈希键（§4.2）：用源标签 + 局序号构造稳定 id。"""
    return hashlib.sha1(f"{source_tag}:{game_index}".encode()).hexdigest()


def is_val_split(game_key_hex: str, val_frac: float = VAL_FRAC) -> bool:
    """按 game_id 哈希划分 train/val：前导字节 < val_frac*256 → val。"""
    return int(game_key_hex[:2], 16) < val_frac * 256


class ShardWriter:
    """顺序写入分片：records.bin（memmap 追加）+ index.json。"""

    def __init__(self, shard_dir: str, games_per_shard: int = 1000):
        self.shard_dir = shard_dir
        self.games_per_shard = games_per_shard
        os.makedirs(shard_dir, exist_ok=True)
        self._buf: list[np.ndarray] = []
        self._index: list[dict] = []
        self._shard_idx = 0
        self._games_in_shard = 0
        self.games_written = 0

    def add_game(self, key: str, records: list[StepRecord]) -> None:
        if not records:
            return
        self._buf.append(np.stack([record_to_array(r) for r in records]))
        self._index.append({"key": key, "start": sum(len(b) for b in self._buf[:-1]), "length": len(records)})
        self._games_in_shard += 1
        if self._games_in_shard >= self.games_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        records = np.concatenate(self._buf)
        base = os.path.join(self.shard_dir, f"shard-{self._shard_idx:05d}")
        fd, tmp = tempfile.mkstemp(dir=self.shard_dir, suffix=".tmp")
        os.close(fd)
        records.tofile(tmp)
        os.replace(tmp, base + ".bin")  # 原子保存纪律
        with open(base + ".index.json", "w", encoding="utf-8") as fh:
            json.dump(self._index, fh)
        self._shard_idx += 1
        self.games_written += self._games_in_shard
        self._buf, self._index, self._games_in_shard = [], [], 0


def read_shard(shard_dir: str, shard_idx: int) -> tuple[np.ndarray, list[dict]]:
    base = os.path.join(shard_dir, f"shard-{shard_idx:05d}")
    records = np.fromfile(base + ".bin", dtype=STEP_DTYPE)
    with open(base + ".index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    return records, index
