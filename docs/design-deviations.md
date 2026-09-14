# 与权威设计文档的偏差记录

> 权威设计文档：`../UniChess/docs/state-sequence-model-design.md`（v2.0）。本文件只记录实现时经文档作者确认的偏差，原目录文档不动。

## 1. §10.1 验收 #3 的断言口径（2026-09-14，作者确认：方案 A）

- **原文**：整序列 vs 逐步递推的 logits 差 < 1e-4（fp32）。
- **实测**：官方 `mamba_ssm` kernel 并行 scan 与单步 decode 两条数值路径存在固有浮点噪声（每层 ~1e-5，12 层 + policy 头放大后 raw logits 差 ~2e-4，且 T=4 即达 1.1e-4，非递推累积、非实现错误——逐步路径重跑 diff=0.0，同输入喂两条 R 路径 diff~3e-4）。
- **现口径**：policy 断 **softmax 概率差 < 1e-4**（实测 ~1e-5，推理实际消费的是 masked softmax 概率）；WDL / moves-left 维持 logits < 1e-4；policy raw logits 差（~2e-4）作诊断指标随冒烟持续记录。真实现法 bug 的差在 2e0 量级（曾抓出 color 条件 bug，diff=2.22），任何 1e-3 门槛均可捕获。

## 2. 实现层记录（非文档偏差）

- `aux.py` 更名为 `model_d.py` / `model_g.py`（`aux` 是 Windows 保留设备名，任何扩展名均不可创建）。
- §5.6 g 侧枝"MLP 1024→1024→512，残差"按同维首层残差实现：`h = z + GELU(W₁z)`（1024→1024），再 `W₂`（1024→512）；predictor 为 512→512 两层。
- 参数实测 26.40M（预算表 ~27M）：E 1.14M / R 19.25M / f 1.52M / D 1.39M / g+predictor 3.09M（g 超预算 1.6M 主因 emb_a 1936×512≈0.99M，总量仍在 D4 的 ~28M 内）。
- §3.2 特征实际 785 维（文档称"约 790"）：12×64 棋子 + 走子方 1 + 易位 4 + 过路兵 8 + 半回合 1 + 全回合 1 + 重复 2 = 785。
