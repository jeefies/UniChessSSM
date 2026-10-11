# 价值头与学习目标改造：实验记录

> 本文件是 `docs/elo-degradation-analysis.md` §19（0-sim 58.5% vs 64-sim 45% 的搜索-蒸馏脱节）
> 所提改造的**逐轮实验日志**。每轮记录：配置、门禁结果、诊断、结论。
> 实现细节见 `docs/stage-b-implementation.md`；设计依据见 `docs/state-sequence-model-design.md` D1–D10。
> 最后更新：2026-10-11。

## 基线（改造前）

Champion gen4（`runs/champion.pt`），现役循环近 10 代：`steps_per_s` 0.815、峰值显存 3.4–4.1 GB、
Arena SPRT α=β=0.05 / H₁=35 Elo。价值头为 `nn.Linear(512,3)` 单层线性，价值目标为终局 z 硬标签。

## 改造内容（代码已入 main）

| 项 | 内容 | 提交 |
|---|---|---|
| D1 价值塔 | `head_norm` 与 `w_v` 之间插 2×`ResidualMLP(512,768)` + `w_mat` 子力差辅助头，均零初始化 ⇒ 旧 ckpt 加载后 step 0 逐位等价；参数 26.40M → 27.981M（D4 内） | `bc3cae1` 起 |
| D3 qsearch 数据 | v3 分片新增 `shard-*.qsearch.bin`（每 ply 3 B：f16 白方绝对视角 q̂ + u8 sims），`root_search_value = Σπ′·completedQ`，Kit 零改动 | 同上 |
| D2 λ-return | `g[t]=(1−λ)q̂[t+1]+λg[t+1]`，闭式一次 bmm（递推版 9.6ms → 0.125ms，训练每步调 32 次，吞吐恢复至与循环持平 0.816） | `91a749b` |
| D4 WDL 软目标 | 矩匹配投影（**第一版**：和棋质量 d=clamp(pD.detach(),0.05,0.5)） | `bc3cae1` |
| D5 不需要重预训练 | 零初始化 + λ 退火；`load_seq_model` 改 strict=False 并只允许缺零初始化新模块 | `c6c4492` |
| 门禁工具 | `tools/sims_sweep.py`（0/8/32/64 sims 曲线）、`configs/match_arena_vtower_vs_champ.json` | `c7254b4` |
| 路径根治 | `_resolve` 改 import 根优先/cwd 兜底；跨进程边界全部绝对化 | `07626b0`/`a9d0448`/Kit `0f0df9f` |

---

## Round 1（gen_0002）：λ=0.5，第一版软目标投影

**配置**：`configs/stage_b8_value_tower.json`；数据 = 3000 局带 q̂ 自对弈（口径与现役循环一致：
64 sims / m0 16 / c_scale 0.02 / PCR 0.5@16 / adaptive 256 / 四重防和铁壁 / 联盟 gen3,2,1）；
λ 退火 1.0→0.5（前 25% 步）；`value_lowsim_weight=0.5`；`material_weight=0.1`；139 步。

**落盘数据校验**：317,223 ply **100% 有记录**，q∈[−1,1]，sims 分布 16/64/255 = 120818/142020/54385
（对应 PCR 50% / 深算 15%，与口径相符）。

**训练侧诊断（新增指标）**：
- `val_selfplay_value_q_corr` = **0.940**、`val_human_value_q_corr` = **0.836**（网络 q 与 λ-return
  目标的相关性，即搜索-蒸馏对齐度；mc 模式下没有这个数，无法直接对比）
- `val_selfplay_loss_value` 0.775 / `val_human_loss_value` 0.839（与循环 gen_0029 的 0.515/1.040
  **不可比**：目标定义不同）
- `target_g_absmean` 0.442（自对弈）/ 0.235（人类）

**门禁结果**：

| 门禁 | 局数 | 得分率 | Elo | LOS |
|---|---|---|---|---|
| 0-sim 纯策略 | 100 | 46.5% | −24.4 | 0.148 |
| sims-sweep 8 | 100 | 56.0% | +41.9 | 0.897 |
| sims-sweep 32 | 100 | 55.5% | +38.4 | 0.876 |
| sims-sweep 64 | 100 | 51.0% | +7.0 | 0.585 |
| **Arena GSPRT（权威）** | 426 | 49.06% | **−6.53** | 0.341 |

SPRT llr = −3.015，第 367 局穿下界 −2.944 ⇒ **H0，不晋级**。

**诊断**：不是"没效果"，而是"效果用错了地方"。

1. **优势局转换能力退化（主因）**：`draw_attribution` = 候选走丢优势局 **41** : 霸主走丢 **5**（8:1）。
   候选在优势下把胜势走成和棋的比例是霸主的 8 倍。
2. **曲线形状自洽**：8/32 sims 有 +40 Elo（浅搜索时估值权重大，价值头确实变好了），
   到 64 sims 衰减到 +7（真实算力占主导后，"不擅转换"把估值优势吃掉了）。
3. **机理**：λ-return 是平滑估计，`target_g_absmean` 0.442 vs 终局 z 的 ±1/0 极端分布 ⇒ 优势局的
   价值被系统性地往 0 压缩，搜索缺乏驱动力去避开三次重复/子力不足。
4. **第二个嫌疑（本轮才发现）**：第一版软目标投影把**和棋类稀释**了——和棋质量 d 取自模型自身
   预测的 clamp 值，于是 g=0 的目标变成 `((1−d)/2, d, (1−d)/2)` 而不是 one-hot `(0,1,0)`。
   这直接削弱和棋信号，与观测到的走丢和棋完全同向。
5. 0-sim −24 Elo：共享主干被不同的 value 目标 reshaping，策略先验略降（CI ±46，噪声级但方向一致）。

**结论**：门禁正常工作，挡下了一次真实退化。两个待修点都有了明确嫌疑和对应的小改动：
λ 取高端（收缩更轻）+ 投影改成确定性最大熵形式（和棋质量 = 1−|g|，g=0 时精确 one-hot）。

**门禁缺口**：本轮 41:5 的走丢和棋率在 100 局 cheap match 上就能看到，却要等 426 局的 SPRT 才确认。
`Kit match` 已输出 `draw_attribution`，应升为一等门禁判据。

---

## Round 2 及以后

（待填）
