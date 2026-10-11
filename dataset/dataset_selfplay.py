"""自对弈（v3 分片）训练数据集：重放 + π′ 软目标（§2.5 / §2.6）。

结构上照搬 `dataset.SequenceDataset`（多进程重放、长度分桶预取），差异仅在于：
- 用 V3ShardReader 读取变长 π′ 目标（每 ply 全部合法着 + 概率）；
- 训练序列长度默认 300（不静默继承 Stage A 的 T_MAX=200，§2.6）；
- 额外产出稠密 (B,T,NUM_ACTIONS) 软目标张量，供 policy_soft_loss 使用；
- 额外产出 mlh_valid：封顶截断局（is_truncated=1）整局剔除 mlh（§2.5）；
- 额外产出逐 ply 搜索价值 q_search/q_valid（D2/D3，老分片为 None）与价值逐位置权重
  value_weight（PCR 低预算 ply 降权，D7）。
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


_readers: list[V3ShardReader] | None = None


def _worker_init(shard_dirs: str | list[str]) -> None:
    global _readers
    dirs = [shard_dirs] if isinstance(shard_dirs, str) else list(shard_dirs)
    _readers = [V3ShardReader(d) for d in dirs]


def _worker_build(target: int | tuple[int, int], t_max: int = B2_T_MAX):
    assert _readers is not None
    if isinstance(target, (tuple, list)):
        gen_idx, local_index = target
        reader = _readers[gen_idx]
    else:
        reader = _readers[0]
        local_index = target
    g = reader.game(local_index)
    meta = g["meta"]
    data = replay_game(g["actions"], meta, t_max=t_max,
                       pipol_actions=g["pipol_actions"], pipol_probs=g["pipol_probs"])
    # §2.5 读取端完整性校验：概率和 ∈ [0.99,1.01]、支持集与规则引擎合法着一致
    if data.get("pipol_actions") is not None:
        validate_v3_pipol(data["pipol_actions"], data["pipol_probs"], len(data["actions"]),
                          legal_masks=data["legal_mask"])
    # 逐 ply 搜索价值（白方绝对视角）：与 actions 同步截断到 t_max；NaN/inf 一律按无记录处理
    q = g.get("qsearch")
    if q is not None:
        n = len(data["actions"])
        qs, qsims = q
        data["q_search"] = np.nan_to_num(np.asarray(qs, dtype=np.float32)[:n],
                                         nan=0.0, posinf=0.0, neginf=0.0)
        data["q_sims"] = np.asarray(qsims, dtype=np.int64)[:n]
    tc = int(meta["tc_bucket"])
    is_truncated = bool(meta["is_truncated"])
    # np.void structured scalar：只能按字段名取值（无 .get）
    elo_mean = float(meta["elo_mean"])
    # flags = 本局开局注入 ply 数（旧分片为 0）；供拼批构造 book_mask（§P1-3 降权）
    book_plies = int(meta["flags"]) if "flags" in meta.dtype.names else 0
    # 真实终局是否落在加载窗口内：封顶截断局 / 被 t_max 截断 ⇒ λ-return 锚点改用 q̂[last]
    # （截断局的 z=0 是约定值，不是真标签）
    truncated_by_tmax = int(meta["n_plies"]) > len(data["actions"])
    return target, data, tc, is_truncated, elo_mean, book_plies, truncated_by_tmax


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
    """v3 自对弈整序列数据集：采样局 → 多进程重放（含 π′）→ 拼批。

    支持单目录（向后兼容，行为与逐位结果不变）与多代目录列表（按代等权均匀采样）。
    """

    def __init__(self, shard_dirs: str | list[str], workers: int = 12, seed: int = 20260916,
                 t_max: int = B2_T_MAX, value_lowsim_weight: float = 0.5,
                 value_lowsim_sims: int = 32):
        if isinstance(shard_dirs, str):
            self.shard_dirs = [shard_dirs]
            self.is_multigen = False
        else:
            self.shard_dirs = [str(d) for d in shard_dirs]
            self.is_multigen = len(self.shard_dirs) > 1

        self.readers = [V3ShardReader(d) for d in self.shard_dirs]
        self.reader = self.readers[0]
        self.workers = workers
        self.seed = seed
        self.t_max = t_max
        # D7：搜索预算低于该阈值的 ply，其 q̂ 噪声大 ⇒ 价值损失降权
        self.value_lowsim_weight = float(value_lowsim_weight)
        self.value_lowsim_sims = int(value_lowsim_sims)

        self.per_gen_lengths: list[np.ndarray] = []
        self.per_gen_train: list[list[int]] = []
        self.per_gen_val: list[list[int]] = []
        self.per_gen_weights: list[np.ndarray] = []

        for r in self.readers:
            n_g = len(r.meta_all)
            lens = r.meta_all["n_plies"].astype(np.int64)
            self.per_gen_lengths.append(lens)
            val_idx = [i for i in range(n_g) if bool(r.is_val_arr[i])]
            train_idx = [i for i in range(n_g) if not bool(r.is_val_arr[i])]
            if not val_idx and n_g > 0:
                cut = max(n_g // 100, 1)
                val_idx = list(range(n_g - cut, n_g))
                train_idx = list(range(n_g - cut))
            self.per_gen_val.append(val_idx)
            self.per_gen_train.append(train_idx)

            # 基于信息论自信息的惊讶度重要性采样 (Surprise-based PER):
            # 衡量单局结果相对该对手/分片预期胜率的偏离程度 (Information Surprise)。
            # 若对某对手胜率 p=100%，则该对手的赢局信息量严格为 0；
            # 遭遇爆冷翻车 (弱对手翻车) 或奇迹逆袭/守和 (强敌抗压) 时，信息量达到峰值！
            w = np.ones(n_g, dtype=np.float64)
            if "result" in r.meta_all.dtype.names and n_g > 0:
                res = r.meta_all["result"].astype(np.int64)  # 0=白胜, 1=和, 2=黑胜
                # 统计该数据源的真实基准胜率/得分率
                n_win = (res == 0).sum()
                n_draw = (res == 1).sum()
                score_rate = float((n_win + 0.5 * n_draw) / max(n_g, 1))
                p = np.clip(score_rate, 0.02, 0.98)

                # 惊讶度信息量计算 (Negative Log-Likelihood Surprise):
                # 胜局信息量: -log(p)
                # 负局信息量: -log(1 - p)
                # 和棋信息量: -0.5 * log(p * (1 - p))
                info = np.zeros(n_g, dtype=np.float64)
                info[res == 0] = -np.log(p)
                info[res == 2] = -np.log(1.0 - p)
                info[res == 1] = -0.5 * np.log(p * (1.0 - p))

                # 平滑映射并设置安全截断门限 [0.05, 5.0]
                w = np.clip(info, 0.05, 5.0)
            self.per_gen_weights.append(w)

        if not self.is_multigen:
            self.n_games = len(self.readers[0].meta_all)
            self.lengths = self.per_gen_lengths[0]
            self.train_indices = self.per_gen_train[0]
            self.val_indices = self.per_gen_val[0]
        else:
            self.n_games = sum(len(r.meta_all) for r in self.readers)
            self.lengths = (np.concatenate(self.per_gen_lengths)
                            if self.per_gen_lengths else np.array([], dtype=np.int64))
            self.train_indices = [(g, idx) for g, tr in enumerate(self.per_gen_train) for idx in tr]
            self.val_indices = [(g, idx) for g, va in enumerate(self.per_gen_val) for idx in va]

        # 便捷属性：聚合所有 generations 的惊讶度采样权重
        if not self.is_multigen:
            self.weights = self.per_gen_weights[0] if self.per_gen_weights else np.array([], dtype=np.float64)
        else:
            self.weights = np.concatenate(self.per_gen_weights) if self.per_gen_weights else np.array([], dtype=np.float64)

        # spawn 进程池
        init_arg = self.shard_dirs[0] if not self.is_multigen else self.shard_dirs
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init,
                                                 initargs=(init_arg,))

    def _collate(self, items: list[tuple]) -> dict[str, torch.Tensor]:
        """items: (index, data, tc, is_truncated, elo_mean, book_plies, truncated_by_tmax) 列表。"""
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

        # ---- 逐 ply 搜索价值（D2/D3/D7）----
        # 任一局有记录才产出；无记录的局整行 q_valid=False（λ-return 对其自动退化为 MC）。
        q_search = q_valid = value_weight = None
        if any("q_search" in d for _, d, *_ in items):
            q_arr = np.zeros((b, t), dtype=np.float32)
            v_arr = np.zeros((b, t), dtype=np.bool_)
            w_arr = np.ones((b, t), dtype=np.float32)
            for i, (_, d, *_rest) in enumerate(items):
                if "q_search" not in d:
                    continue
                n = len(d["actions"])
                # 防御性对齐到本局长度（正常情况二者都已在 _worker_build 截到 n）
                m = min(n, len(d["q_search"]), len(d["q_sims"]))
                q_arr[i, :m] = np.asarray(d["q_search"], dtype=np.float32)[:m]
                sims = np.zeros(n, dtype=np.int64)
                sims[:m] = np.asarray(d["q_sims"], dtype=np.int64)[:m]
                ok = sims > 0
                v_arr[i, :n] = ok
                w_arr[i, :n] = np.where(ok & (sims < self.value_lowsim_sims),
                                        self.value_lowsim_weight, 1.0)
            q_search = torch.from_numpy(q_arr)
            q_valid = torch.from_numpy(v_arr)
            value_weight = torch.from_numpy(w_arr)
        # 真实终局不在加载窗口内 ⇒ λ-return 锚点用 q̂[last] 而非 z
        anchor_last = torch.tensor([bool(it[3]) or bool(it[6]) for it in items], dtype=torch.bool)

        return {
            "batch": TrainBatch(features, actions, legal, results, moves_left,
                                elo_w, tc, elo_std, color, q_search, q_valid),
            "valid": valid,
            "mlh_valid": mlh_valid,
            "policy_soft_target": torch.from_numpy(soft_target),
            "book_mask": book_mask,
            "value_weight": value_weight,
            "anchor_last": anchor_last,
        }

    def epoch_batches(self, microbatch: int, device: str, shuffle: bool = True, prefetch: int = 3):
        if not self.is_multigen:
            if shuffle:
                rng = np.random.default_rng(self.seed)
                w_arr = self.per_gen_weights[0][self.train_indices]
                p_arr = w_arr / max(float(w_arr.sum()), 1e-9)
                sampled_tr = rng.choice(self.train_indices, size=len(self.train_indices), replace=True, p=p_arr)
                order = np.argsort(self.lengths[sampled_tr], kind="stable")
                chunks = [sampled_tr[i] for i in order]
                chunks = [chunks[j:j + microbatch] for j in range(0, len(chunks), microbatch)]
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
            return

        # 多代模式：每代分别按长度分桶打包，各代等权采样 chunk (各代内部按胜负重要性加权抽样)
        G = len(self.shard_dirs)
        rng = np.random.default_rng(self.seed)
        gen_chunk_lists: list[list[list[tuple[int, int]]]] = []
        for g in range(G):
            tr_idx = self.per_gen_train[g]
            lens = self.per_gen_lengths[g]
            if not tr_idx:
                gen_chunk_lists.append([])
                continue
            if shuffle:
                w_g = self.per_gen_weights[g][tr_idx]
                p_g = w_g / max(float(w_g.sum()), 1e-9)
                sampled_tr = rng.choice(tr_idx, size=len(tr_idx), replace=True, p=p_g)
                order = np.argsort(lens[sampled_tr], kind="stable")
                sorted_indices = [sampled_tr[i] for i in order]
                g_chunks = [sorted_indices[j:j + microbatch]
                            for j in range(0, len(sorted_indices), microbatch)]
                rng.shuffle(g_chunks)
            else:
                g_chunks = [tr_idx[j:j + microbatch]
                            for j in range(0, len(tr_idx), microbatch)]
            gen_chunk_lists.append([[(g, idx) for idx in ch] for ch in g_chunks])

        active_gens = [g for g in range(G) if gen_chunk_lists[g]]
        if not active_gens:
            return

        total_chunks = sum(len(gen_chunk_lists[g]) for g in active_gens)
        gen_ptrs = {g: 0 for g in active_gens}
        all_scheduled_chunks = []
        ptr_active = 0
        while len(all_scheduled_chunks) < total_chunks:
            g = active_gens[ptr_active % len(active_gens)]
            ptr_active += 1
            ch_list = gen_chunk_lists[g]
            cur_p = gen_ptrs[g]
            if cur_p >= len(ch_list):
                if shuffle:
                    rng.shuffle(ch_list)
                cur_p = 0
                gen_ptrs[g] = 0
            all_scheduled_chunks.append(ch_list[cur_p])
            gen_ptrs[g] = cur_p + 1

        pending: list = []
        for ch in all_scheduled_chunks[:prefetch]:
            pending.append(self.pool.starmap_async(_worker_build, [(item, self.t_max) for item in ch]))
        for i, ch in enumerate(all_scheduled_chunks):
            items = pending.pop(0).get()
            if i + prefetch < len(all_scheduled_chunks):
                nxt = all_scheduled_chunks[i + prefetch]
                pending.append(self.pool.starmap_async(_worker_build, [(item, self.t_max) for item in nxt]))
            yield self._build_batch(items, device)

    def _build_batch(self, items, device: str):
        out = self._collate(items)
        from .dataset import _to_device
        out["batch"] = _to_device(out["batch"], device)
        out["valid"] = out["valid"].to(device, non_blocking=True)
        out["mlh_valid"] = out["mlh_valid"].to(device, non_blocking=True)
        out["policy_soft_target"] = out["policy_soft_target"].to(device, non_blocking=True)
        out["book_mask"] = out["book_mask"].to(device, non_blocking=True)
        if out["value_weight"] is not None:
            out["value_weight"] = out["value_weight"].to(device, non_blocking=True)
        out["anchor_last"] = out["anchor_last"].to(device, non_blocking=True)
        return out

    def val_batch(self, n_batches: int, microbatch: int, device: str, seed: int = 778):
        if not self.is_multigen:
            rng = np.random.default_rng(seed)
            idxs = rng.choice(self.val_indices, size=min(n_batches * microbatch, len(self.val_indices)),
                              replace=False)
            out = []
            for j in range(0, len(idxs), microbatch):
                batch_idx = list(idxs[j:j + microbatch])
                items = self.pool.starmap(_worker_build, [(idx, self.t_max) for idx in batch_idx])
                out.append(self._build_batch(items, device))
            return out

        rng = np.random.default_rng(seed)
        active_val_gens = [g for g in range(len(self.shard_dirs)) if self.per_gen_val[g]]
        if not active_val_gens:
            return []
        out = []
        for b_i in range(n_batches):
            g = active_val_gens[b_i % len(active_val_gens)]
            val_idx = self.per_gen_val[g]
            chosen = rng.choice(val_idx, size=min(microbatch, len(val_idx)),
                                replace=(len(val_idx) < microbatch))
            items = self.pool.starmap(_worker_build, [((g, idx), self.t_max) for idx in chosen])
            out.append(self._build_batch(items, device))
        return out

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()
