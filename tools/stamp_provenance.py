"""给 v3 分片补生成版本记录与教师状态标记（manifest["provenance"]）。

动机：既有 manifest 没有记录生成时的 git commit，无法区分"只受 meta 分类错误影响"
与"生成时搜索路径本身就有错"。后者修 meta 不会重算当时的 π′。

事实（由分片 mtime 与 git log 比对确定）：四个数据集**全部**生成于 f1146ac
（2026-09-19 00:04，adapter 修复）之前，因此生成时的搜索使用：
  - q = wdl[0] − wdl[2] 直接作用于**原始 logits**（未过 softmax），量纲不是 [−1,1]；
  - 条件输入用未标准化的原始 Elo；
  - 终局叶子由网络前向估值，而非规则真值 get_terminal_q。
这三项都进入 completedQ → σ → π′，故这些分片的 π′ 一律标为 legacy_teacher。
"""

from __future__ import annotations

import argparse
import json
import os

# 数据集 → (生成完成时间, 生成时 HEAD, 教师状态, 说明)
PROVENANCE = {
    "stage_b_smoke": ("2026-09-17 19:26", "pre-f1146ac", "legacy_teacher",
                      "n_sims=32 冒烟；另含 504/1000 动作解析错误，已隔离，不得用于训练"),
    "stage_b_val64": ("2026-09-17 21:23", "pre-f1146ac", "legacy_teacher",
                      "验证集；z/动作可用，π′ 为旧教师"),
    "stage_b_gen2k": ("2026-09-18 02:59", "pre-f1146ac", "legacy_teacher",
                      "Round 1 训练集；z/动作可用，π′ 为旧教师"),
    "stage_b_gen_round2": ("2026-09-18 22:28", "pre-f1146ac", "legacy_teacher",
                           "Round 2 训练集；z/动作可用，π′ 为旧教师"),
}

SEARCH_DEFECTS = [
    "q 由原始 WDL logits 直接相减（未过 softmax），量纲非 [−1,1]",
    "条件输入使用未标准化 Elo",
    "终局叶子用网络估值，未用规则真值 get_terminal_q",
]

META_REPAIR = {
    "commit": "784dc64",
    "tool": "tools/repair_v3_meta.py",
    "scope": "termination_reason / is_truncated / result（按 claim_draw=True 重放重算）",
    "result_changed": 0,
    "backup": ("未保留原始 .meta.npz 副本（疏漏）；旧值可由重放 + 旧分类逻辑"
               "（is_repetition(3)/is_fifty_moves() 链）确定性重建"),
}

QNORM_CHANGE = {
    "commit": "154d673",
    "change": "Q 归一化由全树 qbox 改为逐节点 completed-Q 量程（qtransform_completed）",
    "affects": "本分片生成时未生效；后续新生成数据与 arena 均已生效",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", default="runs")
    args = ap.parse_args()

    for name, (when, commit, status, note) in PROVENANCE.items():
        path = os.path.join(args.runs_dir, name, "manifest.json")
        if not os.path.exists(path):
            print(f"  跳过（不存在）：{path}")
            continue
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["provenance"] = {
            "generated_at": when,
            "generated_at_commit": commit,
            "teacher_status": status,
            "note": note,
            "search_defects_at_generation": SEARCH_DEFECTS,
            "meta_repair": META_REPAIR,
            "qnorm_change_after_generation": QNORM_CHANGE,
            "usable": {
                "actions": True,
                "result_z": True,
                "termination_meta": "已修复",
                "pi_prime": False,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)
        print(f"  {name}: {status}（生成于 {when}, {commit}）")


if __name__ == "__main__":
    main()
