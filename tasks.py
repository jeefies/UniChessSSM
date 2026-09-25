"""S 的训练任务：Stage A（人类棋谱预热）与 Stage B2（人类 + 自对弈混合）。

这两个任务都是 ``Kit.planes19.task.TrainTask`` 协议的自定义实现——S 是**有状态**模型，
数据是变长局面序列（不是固定形状的记录），所以不能用 ``Planes19Task``：

* ``batches`` 复用 ``SSM.dataset`` 的 ``epoch_batches``（无限流，一个迭代器走完一整代就重开，
  与旧 ``train/stage_a.py`` 的 StopIteration 重启同口径）；
* ``loss`` 就是一次 ``SeqModel.forward_train``，``weights`` / ``log_target`` 等口径由配置给；
* ``validate`` 返回 ``{"score": ..., ...}``，Trainer 按 score 创新低导出 best.pt，
  选表口径与旧脚本一致（Stage A 看人类 val 的 policy CE，Stage B2 看自对弈 val 的 policy CE）。
* ``loss`` 多收一个 ``total_steps``（Trainer 只在函数声明了这个形参时才传），
  因为 recon 权重按训练进度退火：``w_r_start + (w_r_end - w_r_start) * min(step/total, 1)``。

``make_task`` 按 ``kind`` 分发。配置示例见 ``SSM/configs/stage_a.json`` 与 ``stage_b2.json``；
``steps`` / ``accum`` / ``precision`` / ``optimizer`` / ``schedule`` 等通用字段在 Trainer 顶层，
不在本模块——本模块只认数据与损失口径。
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

from .dataset.dataset import SequenceDataset
from .model import SeqModel
from .model.losses import LossWeights

#: 与旧脚本一致：训练一律 bf16 autocast（scan 部分保持 fp32）
def _autocast(device):
    return (torch.autocast(device.type, dtype=torch.bfloat16)
            if device.type == "cuda" else contextlib.nullcontext())


class SsmTask:
    """两个 Stage 共用的骨架：建模型、参数组、数据集生命周期。

    子类只需实现 ``batches`` / ``loss`` / ``validate``。
    """

    #: Stage A / B 都用 dropout 0.1，训练脚本里写死过
    dropout = 0.1

    def __init__(self, *, data, workers=12, microbatch=32, dropout: Optional[float] = None,
                 weights: Optional[dict] = None, ckpt=None, val_batches=8, log_every=50,
                 runtime=None):
        self.data = dict(data or {})
        self.workers = int(workers)
        self.microbatch = int(microbatch)
        self.dropout = float(self.dropout if dropout is None else dropout)
        self.weights = LossWeights(**(weights or {}))
        self.ckpt = ckpt
        self.val_batches = int(val_batches)
        self.log_every = int(log_every)
        self.runtime = dict(runtime or {})
        self._open = False

    # ------------------------------------------------------------ TrainTask
    def build_model(self):
        return SeqModel(dropout=self.dropout)

    def param_groups(self, model):
        return [{"params": list(model.parameters())}]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, st: dict) -> None:
        pass

    def export(self, model, step: int) -> dict:
        """与检查点格式一致：model / cfg / step，``count_parameters`` 便于核对结构。"""
        from .model import count_parameters
        return {"model": model.state_dict(), "step": int(step),
                "cfg": count_parameters(model)}

    def on_train_start(self, model, ctx):
        """模型建好后加载底座权重（Stage B2 从 Stage A 的 best.pt 续训）。"""
        if self.ckpt:
            ckpt = torch.load(str(self.ckpt), map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt)
            model.load_state_dict(sd, strict=False)

    def close(self) -> None:
        for name in ("human_ds", "sp_ds"):
            ds = getattr(self, name, None)
            if ds is not None:
                ds.close()
                setattr(self, name, None)

    # ------------------------------------------------------------ 工具
    def _to(self, item, device):
        """序列的 cache/张量搬设备（与原 ``SequenceDataset`` 的行为保持一致）。"""
        from .dataset.dataset import _to_device
        return _to_device(item, device)


class StageATask(SsmTask):
    """Stage A：人类棋谱（v2 分片）单来源预热，recon 权重 1.0→0.1 退火。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.human_ds = None

    def auto_steps(self, accum=1):
        """"跑一个 epoch"：训练局数 ÷ 有效 batch（microbatch × accum）。

        只读 manifest 统计局数，不起 worker 池（池交给 ``batches``）。人类语料固定，
        所以这个数也是固定的；数据换了它跟着变，比写死一个步数安全。
        """
        from .dataset.gshards import ShardReader
        reader = ShardReader(self.data["dir"])
        is_val = reader.is_val_arr
        n = int(len(is_val) - int(is_val.sum()))
        eff = self.microbatch * int(accum)
        return -(-n // eff)

    def batches(self, ctx):
        if self.human_ds is None:
            # seq_len 必须与 T_MAX 一致；SSM 吞吐关键就在"一次前向覆盖整局"，
            # 静默截短历史会让 cache 与旧模型完全对不上
            self.human_ds = SequenceDataset(self.data["dir"], workers=self.workers,
                                            t_max=int(self.data.get("t_max", 200)))
        it = None
        while True:
            if it is None:
                it = iter(self.human_ds.epoch_batches(self.microbatch, ctx.device, shuffle=True))
            try:
                item = next(it)
            except StopIteration:
                it = None
                continue
            yield item

    def loss(self, model, batch, step, total_steps=None):
        item = batch
        with _autocast(_dev(model)):
            total, metrics = model.forward_train(
                item["batch"], self.weights, step, total_steps or 0,
                valid_mask=item["valid"])
        return total, metrics

    def validate(self, model, ctx):
        metrics: dict[str, list] = {}
        with torch.no_grad():
            for item in self.human_ds.val_batch(self.val_batches, self.microbatch, ctx.device):
                with _autocast(ctx.device):
                    _, m = model.forward_train(item["batch"], self.weights, 0,
                                               ctx.cfg.steps, valid_mask=item["valid"])
                for k, v in m.items():
                    if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                        metrics.setdefault(k, []).append(float(v))
        out = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}
        # 旧脚本：best.pt 按人类 val 的 policy CE 选
        out["score"] = out.get("loss_policy", float("inf"))
        return out


class StageB2Task(SsmTask):
    """Stage B2：人类（10%）+ 自对弈 v3（85%）混合，来源权重显式写在 loss 里。"""

    def __init__(self, *, selfplay=None, w_selfplay=0.85, w_human=0.10, mlh_log=False,
                 opening_loss_weight=0.25, t_max=300, limit_games=500000, **kw):
        super().__init__(**kw)
        self.selfplay = dict(selfplay or {})
        self.w_selfplay = float(w_selfplay)
        self.w_human = float(w_human)
        self.mlh_log = bool(mlh_log)
        self.opening_w = float(opening_loss_weight)
        # §2.6：自对弈是完整 300 ply，不得静默继承 Stage A 的 T_MAX=200
        self.t_max = int(t_max)
        # 人类库是千万级（19.3M 局），只取其前 limit_games 局参与混批——先截断再 shuffle，
        # 顺序与旧脚本 --limit-games 一致；不截断等于每步都在 19M 局里抽，另是一个分布。
        self.limit_games = int(limit_games)
        self.human_ds = self.sp_ds = None

    def _sp_policy_weights(self, item):
        """自对弈 policy 逐位置权重：book ply × opening_w，其余 ×1，填充位 ×0。"""
        w = torch.where(item["book_mask"],
                        torch.full_like(item["valid"], self.opening_w, dtype=torch.float32),
                        torch.ones_like(item["valid"], dtype=torch.float32))
        return w * item["valid"].float()

    def auto_steps(self, accum=1):
        """每代遍历预算：``min(3×buffer 局数 ÷ 有效 batch, 2000)``（§2.6）。

        只看自对弈 replay buffer 的局数——人类库固定且远大于 buffer，混入会让预算恒被
        2000 步上限吃满，失去"控制对陈旧搜索目标过拟合遍数"的本意。
        """
        from .dataset.gshards import V3ShardReader
        reader = V3ShardReader(self.selfplay["dir"])
        is_val = reader.is_val_arr
        buffer_games = int(len(is_val) - int(is_val.sum()))
        if buffer_games <= 0:
            raise RuntimeError(f"自对弈分片 {self.selfplay['dir']} 无可训练局——Stage B2 的核心"
                               f"监督来源缺失，请检查生成产物")
        eff = self.microbatch * int(accum)
        steps = min(3 * buffer_games // eff, 2000)
        if steps < 1:
            raise RuntimeError(f"遍历预算不足 1 步（buffer {buffer_games} 局 / 有效 batch {eff}）"
                               f"——请增大 replay buffer 或减小 microbatch×accum")
        return steps

    def batches(self, ctx):
        from .dataset.dataset_selfplay import SelfPlayDataset
        if self.human_ds is None:
            self.human_ds = SequenceDataset(self.data["dir"], workers=self.workers,
                                            t_max=self.t_max)
            self.sp_ds = SelfPlayDataset(self.selfplay["dir"], workers=self.workers,
                                         t_max=self.t_max)
        if self.limit_games > 0:
            self.human_ds.train_indices = self.human_ds.train_indices[:self.limit_games]
        if not self.sp_ds.train_indices:
            raise RuntimeError(
                f"自对弈分片 {self.selfplay['dir']} 无可训练局——Stage B2 的核心监督来源缺失，"
                f"不应静默退化为纯人类数据训练，请检查生成产物")
        human_it = sp_it = None
        while True:
            # 迭代器耗尽时**当场重建并重取**，不丢已取到的另一来源批次——旧脚本
            # （train/stage_b2.py）就是这个口径。若改成"标记 None + continue"，自对弈
            # buffer 每走完一代（1000 局 ÷ 4 ≈ 250 批）就会顺手丢掉一个人类批，
            # 人类流相对旧脚本偏移一批：损失在 step≈15 起飘、之后逐步放大。
            # 顺序也必须 human 优先（旧脚本先建 human 迭代器）。
            if human_it is None:
                human_it = iter(self.human_ds.epoch_batches(self.microbatch, ctx.device,
                                                            shuffle=True))
            if sp_it is None:
                sp_it = iter(self.sp_ds.epoch_batches(self.microbatch, ctx.device, shuffle=True))
            try:
                h = next(human_it)
            except StopIteration:
                human_it = iter(self.human_ds.epoch_batches(self.microbatch, ctx.device,
                                                            shuffle=True))
                h = next(human_it)
            try:
                sp = next(sp_it)
            except StopIteration:
                sp_it = iter(self.sp_ds.epoch_batches(self.microbatch, ctx.device, shuffle=True))
                sp = next(sp_it)
            yield {"human": h, "selfplay": sp}

    def loss(self, model, batch, step, total_steps=None):
        total_steps = total_steps or 0
        h, sp = batch["human"], batch["selfplay"]
        dev = _dev(model)
        with _autocast(dev):
            total_h, m_h = model.forward_train(h["batch"], self.weights, step, total_steps,
                                               valid_mask=h["valid"], log_target=self.mlh_log)
            total_sp, m_sp = model.forward_train(
                sp["batch"], self.weights, step, total_steps, valid_mask=sp["valid"],
                policy_soft_target=sp["policy_soft_target"], mlh_valid_mask=sp["mlh_valid"],
                policy_weights=self._sp_policy_weights(sp), log_target=self.mlh_log)
        total = self.w_selfplay * total_sp + self.w_human * total_h
        parts = {f"selfplay_{k}": v for k, v in m_sp.items()}
        parts.update({f"human_{k}": v for k, v in m_h.items()})
        parts["w_selfplay"], parts["w_human"] = self.w_selfplay, self.w_human
        return total, parts

    def validate(self, model, ctx):
        agg: dict[str, list] = {}
        with torch.no_grad():
            for item in self.human_ds.val_batch(self.val_batches, self.microbatch, ctx.device):
                with _autocast(ctx.device):
                    _, m = model.forward_train(item["batch"], self.weights, 0, ctx.cfg.steps,
                                               valid_mask=item["valid"], log_target=self.mlh_log)
                for k, v in m.items():
                    if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                        agg.setdefault(f"human_{k}", []).append(float(v))
            for item in self.sp_ds.val_batch(self.val_batches, self.microbatch, ctx.device):
                with _autocast(ctx.device):
                    _, m = model.forward_train(
                        item["batch"], self.weights, 0, ctx.cfg.steps,
                        valid_mask=item["valid"],
                        policy_soft_target=item["policy_soft_target"],
                        mlh_valid_mask=item["mlh_valid"],
                        policy_weights=self._sp_policy_weights(item), log_target=self.mlh_log)
                for k, v in m.items():
                    if k.startswith("loss_") or k in ("recon_whole_board_acc", "dyn_rel_err"):
                        agg.setdefault(f"selfplay_{k}", []).append(float(v))
        out = {f"val_{k}": sum(v) / max(len(v), 1) for k, v in agg.items()}
        # §2.6：每代以自对弈 held-out policy CE 选当代表
        out["score"] = out.get("val_selfplay_loss_policy", float("inf"))
        return out


def _dev(model):
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


TASKS = {"stage_a": StageATask, "stage_b2": StageB2Task}


def make_task(runtime=None, *, kind: str = "stage_a", **kwargs) -> SsmTask:
    """``SSM.tasks:make_task`` → ``Kit.train.TrainTask``。

    * ``kind``：``stage_a`` / ``stage_b2``；
    * ``data``：人类分片目录（``{"dir": ..., "t_max": 200}``）；
    * ``selfplay``：Stage B2 的自对弈 v3 分片目录；
    * 其余：``workers`` / ``microbatch`` / ``dropout`` / ``weights`` / ``ckpt`` /
      ``w_selfplay`` / ``w_human`` / ``mlh_log`` / ``opening_loss_weight`` / ``t_max``。
    """
    if kind not in TASKS:
        raise KeyError(f"没有训练任务 {kind!r}（可选：{', '.join(TASKS)}）")
    known = {"dir", "t_max"}
    unknown = sorted(set(kwargs.get("data") or {}) - known)
    if unknown:
        raise KeyError(f"data 有未知字段 {unknown}（可选：{sorted(known)}）")
    cls = TASKS[kind]
    if cls is StageB2Task:
        return StageB2Task(runtime=runtime, **kwargs)
    allowed = {"data", "workers", "microbatch", "dropout", "weights", "ckpt", "val_batches",
               "log_every"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"Stage A 不接受参数 {unknown}")
    return StageATask(runtime=runtime, **kwargs)


__all__ = ["SsmTask", "StageATask", "StageB2Task", "make_task"]
