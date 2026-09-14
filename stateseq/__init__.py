"""UniChessSSM：状态序列模型（设计文档 v2.0）。

E（格子级 Transformer，权重共享走两遍）→ R（12 层基础 Mamba）→ f（policy/WDL/moves-left）。
训练期辅助：D（MLP 重建解码器）、g（残差动力学侧枝），均不参与推理。
"""

__version__ = "0.0.1"
