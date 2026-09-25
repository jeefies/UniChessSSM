"""UniChessSSM：状态序列模型（设计文档 v2.0）。

E（格子级 Transformer，权重共享走南北两条）、R（2 层基础 Mamba）、f（policy/WDL/moves-left）。
训练链助手：D（MLP 重建解码器）、G（确定性动作学习圭表），均不参与推理。

仓库根即包（import 根是 `~/UniChess`，`import SSM.model` / `SSM.kit`）：

===============  ====================================================
``model/``       模型结构与损失定义（state_dict 键名冻结，权重靠它加载）
``infer/``       GPU 状态槽 / GPU 服务进程 / 原地单步（arena 与自对弈的吞吐路径）
``dataset/``     分片读写、序列构造、自对弈数据构建（``build/``）+ ``cpp/pgn2shards.cpp``
``kit.py``       kit 接入：BatchEvaluator / Expander / Player（有状态模型，句柄交接）
``tasks.py``     Stage A / Stage B 的 ``Kit.train.TrainTask``
``configs/``     训练与换代配置
``engine.py``    Server 模型插件（六方法 GameEngine）
===============  ====================================================

搜索、裁决、统计不在本仓库：Gumbel 与 PUCT 在 ``Kit/search/``，规则与残局表在
``Kit/rules/``。历史坑与自洽检查见 ``AGENTS.md``，设计文档在 ``docs/``。
"""
from .model import SeqModel, count_parameters

__all__ = ["SeqModel", "count_parameters"]

__version__ = "0.0.1"
