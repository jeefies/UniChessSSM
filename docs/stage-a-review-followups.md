# Stage A 评审待办进度（2026-09-16 停机前快照）

背景：review.txt（仓库根目录，未入库）两份评审 + 用户转发的第三份评价，共 5 项待办 + arena 归因问题。
核心问题：arena −880 Elo（vs 旧 small champion，双方 UNICHESS_MCTS=400）归因于
"预训练不足"还是"适配链路 bug"。

## 状态

| 项 | 内容 | 状态 |
|---|---|---|
| ① | stage-a-experiment.md 五处勘误（value_gain 口径/sg 表述/分母句/Mish/口径声明） | ✅ 完成（a2f6520） |
| ② | held-out 4,096 局独立评估 | ✅ 完成（35e8d91/d08ddb1）：value_gain 0.100、Top-1 44.1%，主结论复现。产物 runs/stage_a_20260915/heldout4096_step37758.{json,md} |
| ③ | 链路对拍 + WDL→Q 符号验证 + 80 局终止审计 | 🔵 代码完成（8 个 commit 至 2ab6bc0），**审计脚本未跑完** |
| ④ | 纯 policy vs MCTS-400 vs 屏蔽 value 三路对照（16 组开局换色） | ⬜ 未开始。需给 tools/ssm_uci.py 加纯 policy / 屏蔽非终局 value 两个模式开关 |
| ⑤ | g 动作对齐断言测试 tests/test_g_alignment.py | ✅ 5/5 通过（远端用 unittest runner，无 pytest） |

## ③ 的未竟事项（恢复后先跑这些）

已就绪的脚本（均已入库）：
- `tools/ssm_path_audit.py`：原生 vs 适配器 vs 双客户端 server 全链路对拍（fp32 1e-5 逻辑等价 / bf16 3e-2 容差口径已改好）；
- `tools/ssm_wdl_sign_audit.py`：WDL→Q 符号（手工树 VL 约定已修正，间谍 _backup 实现无关口径已改好）+ 终局真值（杀着边缘 Q≈+1）；
- 80 局终止原因审计：arena_vs_smallchampion.json（runs/ 远端）统计终局类型。

恢复方式：Agent(resume="ac4949f61")（③⑤ 的 coder 实例，有自己的上下文），或直接手动跑上述脚本。

## 已确认事实（不必重查）

- 旧 mcts.py 合约：(W,D,L) 行棋方视角，Q = wdl[0]−wdl[2]，回传取负——与 SSM 一致（WDL 顺序张冠李戴假设已排除）；
- uci.py 不解析 go 参数，双方固定 UNICHESS_MCTS=400，超时风险靠 --remote（0.63s/步）缓释，待终止审计证实；
- 评审假设 1（value 是全模型最弱环 → MCTS 退化为 policy-only）是主嫌疑，④ 是判决实验。

## 其他

- review.txt 留在仓库根目录（未入库，用户文件）；
- tools/pgn2shards 是远端 C++ 构建产物（未入库），.gitignore 未覆盖；
- 本文件完成使命后应删除或并入 stage-a-experiment.md。
