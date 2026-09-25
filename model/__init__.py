"""S 的模型包：结构与损失定义（与旧 ``unichess_stateseq`` 的键名逐字节兼容）。

- ``model.py``：``SeqModel``（E/R/f + cond），``forward_train`` 里含损失组装
- ``losses.py``：``LossWeights`` 与各损失项
- ``layers.py`` / ``heads.py``：RMSNorm、MLP、预测头
- ``d.py`` / ``e.py`` / ``g.py`` / ``r.py``：重建解码器、棋盘编码器、动作学习圭表、Mamba 塔
"""
from .layers import RMSNorm, ResidualMLP
from .losses import LossWeights, elo_weights
from .model import SeqModel, TrainBatch, count_parameters

__all__ = ["SeqModel", "TrainBatch", "count_parameters", "LossWeights", "elo_weights",
           "RMSNorm", "ResidualMLP"]
