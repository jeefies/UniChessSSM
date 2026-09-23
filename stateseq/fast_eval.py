"""P4 快速前向：GPU 状态槽 + 显式索引的 Mamba 单步 + 分桶 CUDA graph。

与 ``kit_adapter.SsmEvaluator``（参考实现，逐位对照用）的区别：

- **状态常驻 GPU 槽**（``StateTensors``）：每个已评估局面（搜索树节点 / 对局当前局面）占一个槽，
  存 12 层 (conv, ssm) 状态。子节点前向 = 把父槽复制到新槽再**原地**单步
  （``causal_conv1d_update`` / ``selective_state_update`` 的 ``*_indices`` 参数），
  不再 torch.cat / 切片 / clone，也不再从根重放路径——每次模拟恰好 1 次前向。
- **槽簿记与张量分离**：``SlotPool`` 只管分配 / 钉住 / LRU 淘汰（纯 CPU），``StepEngine`` 管 GPU。
  同进程时两者在一起；共享 GPU 服务（``gpu_server``）时簿记留在各搜索进程（各占一段槽号），
  张量与前向在服务进程。
- **LRU 淘汰 + 重算兜底**：槽满时淘汰最久未用的树节点（``NodeState.slot = None``）；
  之后若要以它为父展开，从最近的在池祖先按原输入逐步重算（``SsmFastEvaluator.ensure``）。
- **CUDA graph**：批大小向上取整到桶，整步（槽复制 + E + 条件 + 12 层 + 头）一张图，
  消除 ~4 ms 的逐 kernel 启动开销。填充行读零槽、写垃圾槽。

数值：单步的算子序列与 ``SeqModel.step`` / ``Mamba2.step`` 逐条相同（A/dt_bias/D 的 repeat 视图
同为零步长，selective_state_update 走同一 tie_hdim 分支），同批大小下逐位一致
（``tests/test_fast_eval.py`` 验证）；批大小（桶）不同时差 ~1e-5，与原实现相同性质。

**批不变模式**（``batch_invariant=True``，共享 GPU 服务恒用）：每块一律填充到固定的 ``chunk`` 行
（默认 ``FIXED_CHUNK``）。
前向里没有跨行的归约（GEMM 只沿特征维归约，注意力 / 归一化 / 状态更新都是逐样本），
同形状下每行结果只取决于该行输入——与同批其他行、拼批时机、并发与进程数、是否被淘汰重算都无关，
整个搜索因此逐位可复现（块大小不同则结果不同 → 块大小属于配置，进结果哈希）。
代价是不满一块也按整块算（低负载时浪费）。
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
from einops import rearrange, repeat

from unichess_kit.api import EvalRequest

from .actions import NUM_ACTIONS

try:
    from causal_conv1d import causal_conv1d_update
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:  # pragma: no cover - 仅 GPU 环境可用
    causal_conv1d_update = None
    selective_state_update = None

FEAT_DIM = 785
# 一行输入（float32）：特征 785 | elo | tc | color | 父槽 | 子槽（整数以 float32 精确表示，< 2^24）
IN_DIM = FEAT_DIM + 5
COL_ELO, COL_TC, COL_COLOR, COL_PAR, COL_CHI = range(FEAT_DIM, IN_DIM)
OUT_DIM = NUM_ACTIONS + 3      # logits | wdl
ZERO_SLOT = 0       # 恒为零的初始状态（从不写）
SCRATCH_SLOT = 1    # 填充行 / 预热写入的垃圾槽
RESERVED_SLOTS = 2
BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256)
FIXED_CHUNK = 64    # 批不变模式的块大小（P4 扫参：服务模式下 64 比 128 快约 5%——闭环里填充浪费少、单拍延迟低）


def auto_pool_slots(concurrency: int, m0: int, holds_per_game: int) -> int:
    """按并发估的槽数：同时钉住的上界 ≈ 每局 holds（对局当前局面）+ m0（轮内并发叶子的父节点），
    再加一整批新分配（最大桶）；乘 1.25 留出 LRU 余量。少于此数时淘汰 / 重算增多但结果不变，
    钉住数超过槽数才会报"状态池耗尽"。"""
    floor = int(concurrency) * (int(holds_per_game) + int(m0)) + BUCKETS[-1] + 8
    return max(256, int(floor * 1.25))


def model_dims(seq) -> tuple:
    """(层数, (conv 宽, conv 通道), (头数, 头维, 状态维))；只支持 d_mlp=0、ngroups=1、rmsnorm 的 Mamba2。"""
    blocks = list(seq.r.blocks)
    b0 = blocks[0]
    d_mlp = (b0.in_proj.out_features - 2 * b0.d_ssm - 2 * b0.ngroups * b0.d_state
             - b0.nheads) // 2
    if d_mlp != 0 or b0.ngroups != 1 or not b0.rmsnorm:
        raise ValueError("快速单步只实现 d_mlp=0、ngroups=1、rmsnorm=True 的 Mamba2")
    return (len(blocks), (b0.d_conv, b0.conv1d.weight.shape[0]),
            (b0.nheads, b0.headdim, b0.d_state))


# ------------------------------------------------------------------ 槽簿记（纯 CPU）

class NodeState:
    """一个局面的 R 状态句柄。

    slot：所在槽（None = 已被淘汰或尚未计算）；inputs：本步前向输入 (feats, tc, elo, color)，
    与 parent 一起用于淘汰后重算；root_occ / path_keys：展开子节点时算 occurrence
    （= 根处计数 + 根到本节点路径上的出现次数，与原重放的 encode-before-increment 同口径）。
    被垃圾回收时把槽还给池（经 _pending，由池在下次分配前统一回收）。
    """

    __slots__ = ("pool", "slot", "gen", "parent", "inputs", "root_occ", "path_keys", "pins",
                 "__weakref__")

    def __init__(self, pool, parent, inputs, root_occ=None, path_keys=()):
        self.pool = pool
        self.slot = None
        self.gen = 0
        self.parent = parent
        self.inputs = inputs
        self.root_occ = root_occ
        self.path_keys = path_keys
        self.pins = 0

    @property
    def evictable(self) -> bool:
        return self.inputs is not None and self.parent is not None

    def __del__(self):
        pool, slot = self.pool, self.slot
        if pool is not None and slot is not None:
            pool._pending.append((slot, self.gen))


class SlotPool:
    """槽号簿记：本池拥有全局槽号 [base, base+slots)；槽 0（零状态）/ 1（垃圾槽）全局共用。

    空闲表 + 未钉住节点的 LRU（弱引用）。钉住（pins>0）的节点不会被淘汰：对局当前局面（hold）、
    正作为父节点等待前向的节点、ensure 中的重算链。
    """

    def __init__(self, slots: int, base: int = RESERVED_SLOTS):
        if slots < 6:
            raise ValueError("状态池至少 6 个槽")
        self.slots = int(slots)
        self.base = int(base)
        self.free = list(range(self.base + self.slots - 1, self.base - 1, -1))
        self.gen = [0] * (self.base + self.slots)
        self.lru: OrderedDict = OrderedDict()
        self._pending: list = []
        self.zero = NodeState(None, None, None)     # pool=None：永不归还
        self.zero.slot = ZERO_SLOT
        self.evictions = 0
        self.rematerialized = 0
        self.peak_used = 0

    @property
    def used(self) -> int:
        return self.slots - len(self.free)

    def collect(self) -> None:
        pending = self._pending
        while pending:
            slot, gen = pending.pop()
            if self.gen[slot] != gen:
                continue            # 该槽已被淘汰并重新分配过
            self.gen[slot] += 1
            self.lru.pop(slot, None)
            self.free.append(slot)

    def alloc(self, k: int) -> list:
        self.collect()
        out = []
        for _ in range(k):
            if self.free:
                s = self.free.pop()
            else:
                if not self.lru:
                    raise RuntimeError(
                        f"状态池耗尽（{self.slots} 槽全部钉住）：调大 pool_slots 或降低并发")
                s, ref = self.lru.popitem(last=False)
                node = ref()
                if node is not None:
                    node.slot = None
                    self.evictions += 1
                self.gen[s] += 1
            out.append(s)
        used = self.used
        if used > self.peak_used:
            self.peak_used = used
        return out

    def assign(self, node: NodeState, slot: int) -> None:
        node.slot = slot
        node.gen = self.gen[slot]
        if node.pins == 0 and node.evictable:
            self.lru[slot] = weakref.ref(node)

    def pin(self, node: NodeState) -> None:
        """钉住：可以钉尚未计算 / 已被淘汰的节点（ensure 重算前先钉住），算出后不进 LRU。"""
        node.pins += 1
        if node.pool is not None and node.slot is not None:
            self.lru.pop(node.slot, None)

    def unpin(self, node: NodeState) -> None:
        node.pins -= 1
        if node.pins == 0 and node.slot is not None and node.evictable and node.pool is not None:
            self.lru[node.slot] = weakref.ref(node)
            self.lru.move_to_end(node.slot)

    def stats(self) -> dict:
        return {"pool_slots": self.slots, "pool_peak_used": self.peak_used,
                "evictions": self.evictions, "rematerialized": self.rematerialized}


# ------------------------------------------------------------------ GPU 侧

class StateTensors:
    """conv (L, S, W, D)、ssm (L, S, H, P, N)，fp32；S 含 2 个保留槽。

    层在外：causal_conv1d / selective_state_update 的核按 int32 算「槽号 × 槽步长」，槽在外时
    槽步长是全部层之和（~196k 元素），1.1 万槽即越过 2^31 → 非法访存（W=16 实测）；层在外时
    槽步长只是一层（~16k 元素），上限 13 万槽。
    """

    def __init__(self, n_layers: int, conv_wd: tuple, ssm_hpn: tuple, total_slots: int, device):
        w, d = conv_wd
        self.dims = (n_layers, tuple(conv_wd), tuple(ssm_hpn))
        self.total_slots = int(total_slots)
        per_slot = max(w * d, int(np.prod(ssm_hpn)))
        if self.total_slots * per_slot >= 2 ** 31:
            raise ValueError(f"状态槽 {self.total_slots} 过多：单层偏移超出 int32")
        self.conv = torch.zeros((n_layers, total_slots, w, d), dtype=torch.float32, device=device)
        self.ssm = torch.zeros((n_layers, total_slots) + tuple(ssm_hpn), dtype=torch.float32,
                               device=device)
        # 每层视图：conv 与 Mamba2.allocate_inference_cache 同为通道在后（stride(1)==1）
        self.conv_views = [self.conv[li].transpose(1, 2) for li in range(n_layers)]
        self.ssm_views = [self.ssm[li] for li in range(n_layers)]

    @staticmethod
    def bytes_per_slot(dims: tuple) -> int:
        n_layers, (w, d), hpn = dims
        return n_layers * (w * d + int(np.prod(hpn))) * 4


# 进程内所有 StepEngine 共用一个预热流与一个 graph 内存池：每桶各开新流会让缓存分配器按流各留一份
# 预热显存，每个引擎各一池会让 A/B 两份中间量互不复用（实测每模型各多占 ~1.2 GiB）。
# 共池安全的前提：各 graph 只在当前流上串行重放，且输出都拷进池外的持久缓冲区。
# 首张 graph 常驻（引擎被回收后池句柄仍有效；它本身不再重放）。
_GRAPH_SHARED: dict = {}


def _graph_shared(device) -> dict:
    dev = torch.device(device)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _GRAPH_SHARED.get(key)
    if st is None:
        st = _GRAPH_SHARED[key] = {"stream": torch.cuda.Stream(device=device), "pool": None}
    return st


class StepEngine:
    """一个模型的批量单步前向：输入行 (m, IN_DIM) → 输出行 (m, OUT_DIM)，状态在 ``tensors`` 里原地更新。

    ``in_rows(m)`` 给出可直接填写的 pinned 输入视图，``execute(m)`` 分块（每块 ≤ 最大桶）
    异步 H2D → 重放 → D2H，最后同步一次，返回 pinned 输出视图（下次 execute 前有效）。
    """

    def __init__(self, seq, tensors: StateTensors, device: str, *, cuda_graphs: bool = True,
                 batch_invariant: bool = False, chunk: int = FIXED_CHUNK,
                 capacity: int = BUCKETS[-1]):
        if causal_conv1d_update is None or selective_state_update is None:
            raise RuntimeError("快速单步需要 causal_conv1d 与 mamba_ssm 的 CUDA kernel")
        if model_dims(seq) != tensors.dims:
            raise ValueError("状态张量形状与模型不符")
        if seq.f.w_p.out_features != NUM_ACTIONS:
            raise ValueError("策略头维度与动作空间不符")
        self.seq = seq
        self.tensors = tensors
        self.device = device
        self.cuda_graphs = bool(cuda_graphs)
        self.batch_invariant = bool(batch_invariant)
        if batch_invariant and chunk not in BUCKETS:
            raise ValueError(f"批不变块大小须取 {BUCKETS} 之一")
        self.buckets = (int(chunk),) if batch_invariant else BUCKETS
        self.chunk = self.buckets[-1]
        self.capacity = int(capacity)
        # 与 Mamba2.step 相同的派生张量（参数不变，预先算好；repeat 视图同为零步长）
        self._layers = []
        for blk in seq.r.blocks:
            a = -torch.exp(blk.A_log.float())
            self._layers.append({
                "blk": blk,
                "conv_w": rearrange(blk.conv1d.weight, "d 1 w -> d w"),
                "A": repeat(a, "h -> h p n", p=blk.headdim, n=blk.d_state).to(dtype=torch.float32),
                "dt_bias": repeat(blk.dt_bias, "h -> h p", p=blk.headdim),
                "D": repeat(blk.D, "h -> h p", p=blk.headdim),
            })
        dev = torch.device(device)
        self.d_in = torch.zeros((self.chunk, IN_DIM), dtype=torch.float32, device=dev)
        self.d_out = torch.zeros((self.chunk, OUT_DIM), dtype=torch.float32, device=dev)
        # 输入多留一块：末块填充行落在 [m, m+填充) 上
        self.h_in = torch.zeros((self.capacity + self.chunk, IN_DIM), dtype=torch.float32,
                                pin_memory=True)
        self.h_out = torch.zeros((self.capacity, OUT_DIM), dtype=torch.float32, pin_memory=True)
        self.h_in_np = self.h_in.numpy()
        self.h_out_np = self.h_out.numpy()
        self._graphs: dict = {}

    def _bucket(self, n: int) -> int:
        if not self.cuda_graphs and not self.batch_invariant:
            return n
        for b in self.buckets:
            if b >= n:
                return b
        return self.buckets[-1]

    # ---- 前向 ----

    def _mamba_step(self, li: int, hs: torch.Tensor, idx32: torch.Tensor) -> torch.Tensor:
        """``Mamba2.step`` 的逐条复刻（rmsnorm=True, d_mlp=0, ngroups=1），状态按槽原地更新。"""
        lay = self._layers[li]
        blk = lay["blk"]
        zxbcdt = blk.in_proj(hs.squeeze(1))
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt, [0, 0, blk.d_ssm, blk.d_ssm + 2 * blk.ngroups * blk.d_state, blk.nheads],
            dim=-1)
        xBC = causal_conv1d_update(xBC, self.tensors.conv_views[li], lay["conv_w"],
                                   blk.conv1d.bias, blk.activation, conv_state_indices=idx32)
        x, B, C = torch.split(xBC, [blk.d_ssm, blk.ngroups * blk.d_state,
                                    blk.ngroups * blk.d_state], dim=-1)
        dt = repeat(dt, "b h -> b h p", p=blk.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=blk.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=blk.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=blk.headdim)
        y = selective_state_update(self.tensors.ssm_views[li], x_reshaped, dt, lay["A"], B, C,
                                   lay["D"], z=None, dt_bias=lay["dt_bias"], dt_softplus=True,
                                   state_batch_indices=idx32)
        y = rearrange(y, "b h p -> b (h p)")
        y = blk.norm(y, z)
        return blk.out_proj(y).unsqueeze(1)

    def _forward(self, b: int) -> None:
        """整步（静态缓冲区的前 b 行）：父槽 → 子槽复制，然后与 SeqModel.step 同序的前向。"""
        seq = self.seq
        rows = self.d_in[:b]
        feats = rows[:, :FEAT_DIM].contiguous()
        elo = rows[:, COL_ELO].contiguous()
        ints = rows[:, COL_TC:].to(torch.long)
        tc, color = ints[:, 0].contiguous(), ints[:, 1].contiguous()
        par, chi = ints[:, 2].contiguous(), ints[:, 3].contiguous()
        st = self.tensors
        st.conv.index_copy_(1, chi, st.conv.index_select(1, par))
        st.ssm.index_copy_(1, chi, st.ssm.index_select(1, par))
        # 新分配再拷入：b=1 时列切片的 .contiguous()/.to() 都是空操作（单元素视为连续），
        # 步长仍是 IN_DIM，而 causal_conv1d 逐字检查索引 stride(0)==1
        idx32 = torch.empty((b,), dtype=torch.int32, device=chi.device).copy_(chi)
        x = seq.encode(feats).unsqueeze(1)
        cond = seq.cond(tc, elo, color).unsqueeze(1)
        h = seq.in_norm(x + cond)
        for li, norm in enumerate(seq.r.norms):
            h = h + self._mamba_step(li, norm(h), idx32)
        logits, wdl, _mlh = seq.f(h)
        self.d_out[:b, :NUM_ACTIONS].copy_(logits.squeeze(1))
        self.d_out[:b, NUM_ACTIONS:].copy_(wdl.squeeze(1))

    def _graph(self, b: int):
        g = self._graphs.get(b)
        if g is not None:
            return g
        self.d_in[:b, COL_PAR].fill_(ZERO_SLOT)
        self.d_in[:b, COL_CHI].fill_(SCRATCH_SLOT)
        shared = _graph_shared(self.device)
        side = shared["stream"]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):          # 预热（triton 编译 / cuBLAS 选算法），只写垃圾槽
                self._forward(b)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=shared["pool"]):
            self._forward(b)
        if shared["pool"] is None:
            shared["pool"] = g.pool()
            shared["keep"] = g      # 池随最后一张引用它的 graph 释放；留一张保证句柄一直有效
        self._graphs[b] = g
        return g

    def warmup(self, buckets: Optional[Sequence[int]] = None) -> None:
        if self.cuda_graphs:
            with torch.no_grad():
                for b in (buckets or self.buckets):
                    self._graph(b)
            torch.cuda.synchronize()

    def in_rows(self, m: int) -> np.ndarray:
        if m > self.capacity:
            raise ValueError(f"一次前向 {m} 行超过容量 {self.capacity}")
        return self.h_in_np[:m]

    @torch.no_grad()
    def execute(self, m: int) -> np.ndarray:
        chunks = []
        for s in range(0, m, self.chunk):
            e = min(m, s + self.chunk)
            chunks.append((s, e, self._bucket(e - s)))
        if self.cuda_graphs:            # 首次捕获会改写静态缓冲区，须在排入任何拷贝之前
            for _, _, b in chunks:
                self._graph(b)
        hin, hin_np = self.h_in, self.h_in_np
        for s, e, b in chunks:
            if b > e - s:               # 填充行（只可能是末块）：读零槽、写垃圾槽
                hin_np[e:s + b, COL_PAR] = ZERO_SLOT
                hin_np[e:s + b, COL_CHI] = SCRATCH_SLOT
            self.d_in[:b].copy_(hin[s:s + b], non_blocking=True)
            if self.cuda_graphs:
                self._graphs[b].replay()
            else:
                self._forward(b)
            self.h_out[s:e].copy_(self.d_out[:e - s], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.h_out_np[:m]


class LocalBackend:
    """同进程后端：一个模型一个 StepEngine。"""

    def __init__(self, engine: StepEngine):
        self.engine = engine
        self.max_rows = engine.capacity

    def in_rows(self, m: int) -> np.ndarray:
        return self.engine.in_rows(m)

    def execute(self, m: int, model_id: int) -> np.ndarray:
        return self.engine.execute(m)


# 同进程同形状的模型（arena 的 A/B）共用一套状态张量与槽簿记
_LOCAL_STORES: dict = {}


def local_store(dims: tuple, slots: int, device) -> tuple:
    key = (dims, str(device))
    st = _LOCAL_STORES.get(key)
    if st is None:
        st = _LOCAL_STORES[key] = (StateTensors(*dims, RESERVED_SLOTS + slots, device),
                                   SlotPool(slots))
    return st


# ------------------------------------------------------------------ 求值器

class SsmFastEvaluator:
    """kit BatchEvaluator：负载 = 待计算的 ``NodeState``（父节点已钉住）；
    结果 = (logits[1936] fp32, wdl[3], 该 NodeState)。重算负载（节点已被淘汰）也走这里，
    同一拍重复出现的节点只算一次，返回 (None, None, node)。

    backend：``LocalBackend``（同进程 GPU）或 ``gpu_server.RemoteBackend``（共享 GPU 服务）。
    """

    fast = True

    def __init__(self, backend, pool: SlotPool, model_key: str, model_id: int = 0):
        self.backend = backend
        self.pool = pool
        self.model_key = model_key
        self.model_id = int(model_id)
        self.n_forwards = 0
        self.n_calls = 0

    @classmethod
    def local(cls, seq, device: str, model_key: str, *, pool_slots: int = 2048,
              cuda_graphs: bool = True, batch_invariant: bool = False,
              chunk: int = FIXED_CHUNK, store: Optional[tuple] = None) -> "SsmFastEvaluator":
        """同进程快速求值器。store=(StateTensors, SlotPool)；缺省按形状取进程内共享的一套。"""
        tensors, pool = store if store is not None else local_store(model_dims(seq),
                                                                     pool_slots, device)
        engine = StepEngine(seq, tensors, device, cuda_graphs=cuda_graphs,
                            batch_invariant=batch_invariant, chunk=chunk)
        return cls(LocalBackend(engine), pool, model_key)

    @classmethod
    def from_checkpoint(cls, checkpoint, device: str = "cuda", **kw) -> "SsmFastEvaluator":
        from .kit_adapter import load_seq_model
        seq, path = load_seq_model(checkpoint, device)
        return cls.local(seq, device, f"S-fast:{path}:{device}", **kw)

    # ---- 状态句柄 API（与 SsmPlayer 交接）----

    def root_state(self) -> NodeState:
        return self.pool.zero

    def child(self, parent: NodeState, feats, tc, elo, color, root_occ=None,
              path_keys=()) -> NodeState:
        """新建待计算节点并钉住父节点（父必须在池中或已被钉住；由 evaluate 解钉）。"""
        node = NodeState(self.pool, parent, (feats, int(tc), float(elo), int(color)),
                         root_occ, path_keys)
        self.pool.pin(parent)
        return node

    def hold(self, node: NodeState) -> None:
        """对局当前局面：永久钉住，丢弃重算信息（不会被淘汰，也不需要）。"""
        self.pool.pin(node)
        node.parent = None
        node.inputs = None

    def release(self, node: Optional[NodeState]) -> None:
        if node is not None and node is not self.pool.zero:
            self.pool.unpin(node)

    def make_root(self, ply: int, state: NodeState, occurrence: dict) -> NodeState:
        """搜索根 = 对局当前局面（已 hold）；记下根处 occurrence 供子节点编码。"""
        state.root_occ = occurrence
        state.path_keys = ()
        return state

    def make_expander(self):
        from .kit_adapter import SsmFastExpander
        return SsmFastExpander(self)

    def ensure(self, nodes: Sequence[NodeState]):
        """协程：保证 nodes 都在池中，返回时 nodes 各被钉住一次（调用方用完须各 ``pool.unpin``）。

        被淘汰的从最近在池祖先逐层重算，共享祖先只算一次。目标与整条重算链从一开始就钉住：
        同一拍里别的请求（arena 另一方的求值器共用本池）的分配不得淘汰刚重算出的中间祖先，
        也不得淘汰已在池中的目标。
        """
        pool = self.pool
        for node in nodes:
            pool.pin(node)
        depth_of: dict = {}
        levels: list = []
        chained: list = []
        for node in nodes:
            chain = []
            p = node
            while p.slot is None and id(p) not in depth_of:
                if not p.evictable:
                    raise RuntimeError("状态已丢失且无法重算（缺父节点或输入）")
                chain.append(p)
                p = p.parent
            base = depth_of[id(p)] + 1 if p.slot is None else 0
            for k, q in enumerate(reversed(chain)):
                depth_of[id(q)] = base + k
                if base + k >= len(levels):
                    levels.append([])
                levels[base + k].append(q)
                pool.pin(q)
                chained.append(q)
        try:
            for level in levels:
                for q in level:
                    pool.pin(q.parent)      # 与 evaluate 里逐负载的解钉配对
                pool.rematerialized += len(level)
                yield EvalRequest(self, level)
        finally:
            for q in chained:
                pool.unpin(q)

    # ---- 前向 ----

    def evaluate(self, payloads: Sequence) -> list:
        self.n_calls += 1
        results: dict = {}
        todo: list = []
        seen: set = set()
        for node in payloads:
            if node.slot is None and id(node) not in seen:
                seen.add(id(node))
                todo.append(node)
        step = self.backend.max_rows
        for start in range(0, len(todo), step):
            chunk = todo[start:start + step]
            lg, wd = self._run(chunk)
            for i, node in enumerate(chunk):
                results[id(node)] = (lg[i], wd[i])
        for node in payloads:       # 每个负载各解钉一次父节点（重复负载也各钉过一次）
            self.pool.unpin(node.parent)
        out = []
        for node in payloads:
            r = results.pop(id(node), None)
            out.append((r[0], r[1], node) if r is not None else (None, None, node))
        return out

    def _run(self, chunk: list) -> tuple:
        m = len(chunk)
        self.n_forwards += m
        slots = self.pool.alloc(m)
        rows = self.backend.in_rows(m)
        for node in chunk:
            if node.parent.slot is None:
                raise RuntimeError("父状态不在池中（未经 ensure 钉住就展开）")
        rows[:, :FEAT_DIM] = [node.inputs[0] for node in chunk]
        rows[:, FEAT_DIM:] = [(node.inputs[2], node.inputs[1], node.inputs[3], node.parent.slot, s)
                              for node, s in zip(chunk, slots)]
        out = self.backend.execute(m, self.model_id)
        lg = out[:, :NUM_ACTIONS].copy()
        wd = out[:, NUM_ACTIONS:].copy()
        for node, s in zip(chunk, slots):
            self.pool.assign(node, s)
        return lg, wd

    def stats(self) -> dict:
        return {"forwards": self.n_forwards, "calls": self.n_calls, **self.pool.stats()}
