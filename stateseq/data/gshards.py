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
import struct
import tempfile
import zlib

import numpy as np

from ..conditions import time_control_bucket

RESULT_TO_LABEL = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
GAMES_PER_SHARD = 50_000

_U64 = (1 << 64) - 1


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """向量化 splitmix64（与 C++ 常用实现同参数；仅 Python 侧使用，无需跨语言一致）。"""
    x = (x + 0x9E3779B97F4A7C15) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


META_V2_DTYPE = np.dtype([
    ("n_plies", np.uint16),
    ("tc_bucket", np.uint8),
    ("result", np.uint8),
    ("elo_missing", np.uint8),
    ("elo_mean", np.float32),
    ("local_idx", np.uint32),
])

# v3 扩展 meta：v2 的 16B + 16B 扩展
# 扩展部分：u32 gen_id / u32 ckpt_step / u8 termination_reason / u8 is_truncated
#          / u8 start_type / u8 flags / u32 pad
META_V3_EXT_DTYPE = np.dtype([
    ("gen_id", np.uint32),
    ("ckpt_step", np.uint32),
    ("termination_reason", np.uint8),
    ("is_truncated", np.uint8),
    ("start_type", np.uint8),
    ("flags", np.uint8),
    ("pad", np.uint32),
])

META_V3_DTYPE = np.dtype([
    ("n_plies", np.uint16),
    ("tc_bucket", np.uint8),
    ("result", np.uint8),
    ("elo_missing", np.uint8),
    ("elo_mean", np.float16),
    ("game_key", "U16"),
    ("gen_id", np.uint32),
    ("ckpt_step", np.uint32),
    ("termination_reason", np.uint8),
    ("is_truncated", np.uint8),
    ("start_type", np.uint8),
    ("flags", np.uint8),
    ("pad", np.uint32),
])


def make_game_key(month: str, game_index: int) -> str:
    return hashlib.sha256(f"{month}:{game_index}".encode()).hexdigest()[:16]


def is_val_key(game_key: str, val_frac: float = 0.005) -> bool:
    gk = game_key
    if isinstance(gk, bytes):
        gk = gk.decode("utf-8", errors="ignore")
    gk = str(gk).strip()
    if not gk:
        return False
    return int(gk[:2], 16) < val_frac * 256


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
        with open(tmp, "wb") as fh:
            np.savez(fh, metas=metas, offsets=offsets)
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
    """训练加载器侧：mmap 读取全部片的动作池与 meta。支持 v1(npz) 与 v2(C++ 二进制)。

    v2 布局：shard-<month>-w<k>.meta.bin（16B/局：u16 n_plies, u8 tc, u8 result,
    u8 elo_missing, u8 pad, f32 elo_mean, u32 local_idx, u32 pad）+ .actions.bin。
    val 划分：splitmix64(crc32(month), worker, local_idx)（0.5%，同局不跨集）。
    """

    def __init__(self, shard_dir: str):
        with open(os.path.join(shard_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.metas: list[np.ndarray] = []
        self.offsets: list[np.ndarray] = []
        self.pools: list[np.memmap] = []
        is_val_parts: list[np.ndarray] = []
        for name in self.manifest["shards"]:
            base = os.path.join(shard_dir, name)
            if os.path.exists(base + ".meta.bin"):  # v2（C++ 二进制）
                raw = np.fromfile(base + ".meta.bin", dtype=np.uint8).reshape(-1, 16)
                meta = np.zeros(len(raw), dtype=META_V2_DTYPE)
                meta["n_plies"] = raw[:, 0:2].copy().view(np.uint16).reshape(-1)
                meta["tc_bucket"] = raw[:, 2]
                meta["result"] = raw[:, 3]
                meta["elo_missing"] = raw[:, 4]
                meta["elo_mean"] = raw[:, 6:10].copy().view(np.float32).reshape(-1)
                meta["local_idx"] = raw[:, 10:14].copy().view(np.uint32).reshape(-1)
                toks = os.path.basename(name).split("-")  # shard,YYYY,MM,wK
                month = "-".join(toks[1:-1])
                month_crc = zlib.crc32(month.encode())
                worker = int(toks[-1][1:])
                h = _splitmix64(np.full(len(meta), month_crc, dtype=np.uint64) ^ np.uint64(worker) << 32)
                h = _splitmix64(h ^ meta["local_idx"].astype(np.uint64))
                is_val_parts.append((h % 100000) < 500)
                off = np.zeros(len(meta) + 1, dtype=np.int64)
                np.cumsum(meta["n_plies"].astype(np.int64), out=off[1:])
                self.offsets.append(off)
            else:  # v1 npz
                npz = np.load(base + ".meta.npz")
                meta = npz["metas"]
                self.offsets.append(npz["offsets"])
                keys = meta["game_key"]
                is_val_parts.append(np.array([is_val_key(str(k)) for k in keys]))
            self.metas.append(meta)
            self.pools.append(np.memmap(base + ".actions.bin", dtype=np.uint16, mode="r"))
        self.meta_all = np.concatenate(self.metas)
        self.is_val_arr = np.concatenate(is_val_parts)
        counts = [len(m) for m in self.metas]
        self.shard_of = np.repeat(np.arange(len(counts)), counts)
        self._cumcounts = np.concatenate([[0], np.cumsum(counts)])
        self.meta_all = np.concatenate(self.metas)
        self.is_val_arr = np.concatenate(is_val_parts)
        counts = [len(m) for m in self.metas]
        self.shard_of = np.repeat(np.arange(len(counts)), counts)
        self._cumcounts = np.concatenate([[0], np.cumsum(counts)])

    def game(self, global_index: int) -> tuple[np.ndarray, np.ndarray]:
        """→ (meta 标量, 动作 uint16 数组)。"""
        shard = int(self.shard_of[global_index])
        local = global_index - int(self._cumcounts[shard])
        rec = self.metas[shard][local]
        start = int(self.offsets[shard][local])
        n = int(rec["n_plies"])
        return rec, np.asarray(self.pools[shard][start:start + n])


# ------------------------- v3 分片（Stage B 自对弈） -------------------------

class V3ShardWriter:
    """v3 分片写入器：支持变长 pipol 目标 + 终局元数据扩展。"""

    def __init__(self, out_dir: str, tag: str, shard_size: int = GAMES_PER_SHARD):
        self.out_dir = out_dir
        self.tag = tag
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self._metas: list[np.ndarray] = []
        self._action_chunks: list[np.ndarray] = []
        self._pipol_chunks: list[bytes] = []
        self._pipol_offsets: list[np.ndarray] = []
        self.shard_files: list[str] = []
        self.games = 0
        self.steps = 0

    def add(self, meta: np.ndarray, actions: np.ndarray, pipol: bytes, pipol_offset: np.ndarray) -> None:
        self._metas.append(meta)
        self._action_chunks.append(actions)
        self._pipol_chunks.append(pipol)
        self._pipol_offsets.append(pipol_offset)
        self.games += 1
        self.steps += int(meta["n_plies"])
        if len(self._metas) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._metas:
            return
        idx = len(self.shard_files)
        metas = np.stack(self._metas)
        pool = np.concatenate(self._action_chunks)
        offsets = np.zeros(len(metas) + 1, dtype=np.int64)
        np.cumsum(metas["n_plies"], out=offsets[1:])
        # pipol 偏移：每局一个 (n_plies+1,) 的 int32 段
        pipol_offsets = np.zeros(len(metas) + 1, dtype=np.int64)
        for i, poff in enumerate(self._pipol_offsets):
            pipol_offsets[i + 1] = pipol_offsets[i] + poff[-1]
        pipol_blob = b"".join(self._pipol_chunks)

        base = os.path.join(self.out_dir, f"shard-{self.tag}-{idx:05d}")
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        pool.tofile(tmp)
        os.replace(tmp, base + ".actions.bin")
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        with open(tmp, "wb") as fh:
            np.savez(fh, metas=metas, offsets=offsets)
        os.replace(tmp, base + ".meta.npz")
        # pipol 变长目标
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        pipol_offsets.tofile(tmp)
        os.replace(tmp, base + ".pipol.offsets.bin")
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        os.close(fd)
        with open(tmp, "wb") as fh:
            fh.write(pipol_blob)
        os.replace(tmp, base + ".pipol.bin")
        self.shard_files.append(base)
        write_manifest(self.out_dir, self.shard_files, [],
                       {"games": self.games, "steps": self.steps})
        self._metas, self._action_chunks, self._pipol_chunks, self._pipol_offsets = [], [], [], []


class V3ShardReader:
    """v3 分片读取器：读取动作序列 + 变长 pipol 目标。"""

    def __init__(self, shard_dir: str):
        with open(os.path.join(shard_dir, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.metas: list[np.ndarray] = []
        self.offsets: list[np.ndarray] = []
        self.pools: list[np.memmap] = []
        self.pipol_offsets: list[np.memmap] = []
        self.pipol_blobs: list[bytes] = []
        is_val_parts: list[np.ndarray] = []
        for name in self.manifest["shards"]:
            base = os.path.join(shard_dir, name)
            npz = np.load(base + ".meta.npz")
            meta = npz["metas"]
            self.offsets.append(npz["offsets"])
            keys = meta["game_key"] if "game_key" in meta.dtype.names else []
            is_val_parts.append(np.array([is_val_key(str(k)) for k in keys]))
            self.metas.append(meta)
            self.pools.append(np.memmap(base + ".actions.bin", dtype=np.uint16, mode="r"))
            if os.path.exists(base + ".pipol.bin"):
                self.pipol_offsets.append(np.fromfile(base + ".pipol.offsets.bin", dtype=np.int64))
                with open(base + ".pipol.bin", "rb") as fh:
                    self.pipol_blobs.append(fh.read())
            else:
                self.pipol_offsets.append(None)
                self.pipol_blobs.append(None)
        self.meta_all = np.concatenate(self.metas)
        self.is_val_arr = np.concatenate(is_val_parts)
        counts = [len(m) for m in self.metas]
        self.shard_of = np.repeat(np.arange(len(counts)), counts)
        self._cumcounts = np.concatenate([[0], np.cumsum(counts)])

    def game(self, global_index: int) -> dict:
        """→ {meta, actions, pipol_actions, pipol_probs}。"""
        shard = int(self.shard_of[global_index])
        local = global_index - int(self._cumcounts[shard])
        rec = self.metas[shard][local]
        start = int(self.offsets[shard][local])
        n = int(rec["n_plies"])
        actions = np.asarray(self.pools[shard][start:start + n])
        pipol_actions = None
        pipol_probs = None
        if self.pipol_blobs[shard] is not None:
            poff = self.pipol_offsets[shard]
            blob = self.pipol_blobs[shard]
            s = int(poff[local])
            e = int(poff[local + 1])
            chunk = blob[s:e]
            # 解析变长记录：每 ply = u16 legal_count + legal_count × (u16 action_id + f16 prob)
            pipol_actions = []
            pipol_probs = []
            offset = 0
            for _ in range(n):
                if offset + 2 > len(chunk):
                    break
                legal_count = struct.unpack_from("<H", chunk, offset)[0]
                offset += 2
                ply_actions = []
                ply_probs = []
                for _ in range(legal_count):
                    if offset + 4 > len(chunk):
                        break
                    action_id = struct.unpack_from("<H", chunk, offset)[0]
                    prob = struct.unpack_from("<e", chunk, offset + 2)[0]
                    ply_actions.append(action_id)
                    ply_probs.append(float(prob))
                    offset += 4
                pipol_actions.append(np.array(ply_actions, dtype=np.uint16))
                pipol_probs.append(np.array(ply_probs, dtype=np.float32))
        return {
            "meta": rec,
            "actions": actions,
            "pipol_actions": pipol_actions,
            "pipol_probs": pipol_probs,
        }


def validate_v3_pipol(pipol_actions, pipol_probs, n_plies: int,
                      legal_masks: np.ndarray | None = None,
                      prob_sum_lo: float = 0.99, prob_sum_hi: float = 1.01) -> None:
    """v3 π′ 读取端完整性校验（规格 §2.5）：概率和 ∈ [lo, hi]；支持集与规则引擎一致。

    legal_masks 提供时（(n_plies, NUM_ACTIONS) bool，训练加载器重放可得）额外校验：
    支持集大小 == 当步合法着数、且每个目标动作都在合法掩码内。
    任何不一致 raise ValueError——调用方据此拒绝该批（数据完整性错误）。
    """
    if pipol_actions is None or pipol_probs is None:
        raise ValueError("v3 分片缺少 π′ 目标（pipol）")
    if len(pipol_actions) < n_plies or len(pipol_probs) < n_plies:
        raise ValueError(f"pipol 记录数不足：{len(pipol_actions)}/{len(pipol_probs)} < {n_plies}")
    for t in range(n_plies):
        acts = np.asarray(pipol_actions[t])
        probs = np.asarray(pipol_probs[t], dtype=np.float64)
        if len(acts) == 0 or len(acts) != len(probs):
            raise ValueError(f"ply {t}: 非法 π′ 支持集大小 acts={len(acts)} probs={len(probs)}")
        s = float(probs.sum())
        if not (prob_sum_lo <= s <= prob_sum_hi):
            raise ValueError(f"ply {t}: π′ 概率和 {s:.4f} 超出 [{prob_sum_lo}, {prob_sum_hi}]")
        if legal_masks is not None:
            mask = np.asarray(legal_masks[t])
            if int(mask.sum()) != len(acts):
                raise ValueError(f"ply {t}: π′ 支持集 {len(acts)} != 规则引擎合法着 {int(mask.sum())}")
            if not bool(np.all(mask[acts.astype(np.int64)])):
                raise ValueError(f"ply {t}: π′ 支持集含非法着法")


def encode_v3_pipol(per_ply_actions: list[np.ndarray], per_ply_probs: list[np.ndarray]) -> bytes:
    """把每 ply 的 (legal_actions, probs) 编码为 v3 pipol 二进制。"""
    buf = bytearray()
    for acts, probs in zip(per_ply_actions, per_ply_probs):
        buf += struct.pack("<H", len(acts))
        for a, p in zip(acts, probs):
            buf += struct.pack("<He", int(a), float(p))
    return bytes(buf)


def decode_v3_pipol(blob: bytes, n_plies: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """从 v3 pipol 二进制解码。"""
    actions_list = []
    probs_list = []
    offset = 0
    for _ in range(n_plies):
        if offset + 2 > len(blob):
            break
        legal_count = struct.unpack_from("<H", blob, offset)[0]
        offset += 2
        acts = []
        probs = []
        for _ in range(legal_count):
            if offset + 4 > len(blob):
                break
            a = struct.unpack_from("<H", blob, offset)[0]
            p = struct.unpack_from("<e", blob, offset + 2)[0]
            acts.append(a)
            probs.append(p)
            offset += 4
        actions_list.append(np.array(acts, dtype=np.uint16))
        probs_list.append(np.array(probs, dtype=np.float32))
    return actions_list, probs_list

