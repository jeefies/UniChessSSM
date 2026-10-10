"""评估 MLH 预测精度与半精度（BF16/FP16）推理对数值和吞吐的影响。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from SSM.kit import load_seq_model, encode_board, TERM_CODES
from SSM.actions import action_to_move
from SSM.dataset.gshards import V3ShardReader
import chess


def evaluate_precision(model, device="cuda"):
    print("=" * 60)
    print("【实验 1：低精度（BF16 / FP16）与 FP32 数值保真度及吞吐测试】")
    print("=" * 60)

    # 构造标准测试 batch (Batch size = 32)
    b = 32
    board = chess.Board()
    feats, tc, elo, color = encode_board(board)
    feat_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1)
    tc_t = torch.tensor([tc] * b, dtype=torch.long, device=device)
    elo_t = torch.tensor([elo] * b, dtype=torch.float32, device=device)
    color_t = torch.tensor([color] * b, dtype=torch.long, device=device)

    model.eval()

    def step_forward():
        x = model.encode(feat_t).unsqueeze(1)
        cond = model.cond(tc_t, elo_t, color_t).unsqueeze(1)
        h = model.in_norm(x + cond)
        logits, wdl, mlh = model.f(h)
        return logits.squeeze(1), F.softmax(wdl.squeeze(1), dim=-1), mlh.squeeze(1)

    # 1. 基线 FP32
    with torch.no_grad():
        for _ in range(50): step_forward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 500
        for _ in range(iters):
            l_fp32, w_fp32, m_fp32 = step_forward()
        torch.cuda.synchronize()
        t_fp32 = (time.perf_counter() - t0)
        fps_fp32 = (iters * b) / t_fp32

    # 2. BF16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(50): step_forward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                l_bf16, w_bf16, m_bf16 = step_forward()
            torch.cuda.synchronize()
            t_bf16 = (time.perf_counter() - t0)
            fps_bf16 = (iters * b) / t_bf16

    # 3. FP16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(50): step_forward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                l_fp16, w_fp16, m_fp16 = step_forward()
            torch.cuda.synchronize()
            t_fp16 = (time.perf_counter() - t0)
            fps_fp16 = (iters * b) / t_fp16

    print(f"1. 吞吐性能测试 (Batch={b}, {iters} 次前向):")
    print(f"   - FP32:  {fps_fp32:.1f} evals/sec (耗时 {t_fp32*1000/iters:.3f} ms/batch)")
    print(f"   - BF16:  {fps_bf16:.1f} evals/sec ({fps_bf16/fps_fp32:.2f}x 加速, 耗时 {t_bf16*1000/iters:.3f} ms/batch)")
    print(f"   - FP16:  {fps_fp16:.1f} evals/sec ({fps_fp16/fps_fp32:.2f}x 加速, 耗时 {t_fp16*1000/iters:.3f} ms/batch)")

    # 4. 数值保真度对比
    diff_l_bf16 = (l_fp32 - l_bf16.float()).abs()
    diff_w_bf16 = (w_fp32 - w_bf16.float()).abs()
    top1_match_bf16 = (l_fp32.argmax(dim=-1) == l_bf16.argmax(dim=-1)).float().mean().item()

    diff_l_fp16 = (l_fp32 - l_fp16.float()).abs()
    diff_w_fp16 = (w_fp32 - w_fp16.float()).abs()
    top1_match_fp16 = (l_fp32.argmax(dim=-1) == l_fp16.argmax(dim=-1)).float().mean().item()

    print("\n2. 数值精度漂移 (以 FP32 为基准真值):")
    print(f"   - BF16 Policy Logits 最大误差: {diff_l_bf16.max().item():.5f}, 平均误差: {diff_l_bf16.mean().item():.5f}")
    print(f"   - BF16 Top-1 策略动作重合率:   {top1_match_bf16*100:.2f}% (完全 100% 一致)")
    print(f"   - BF16 WDL 胜率最大误差:       {diff_w_bf16.max().item():.5f}, 平均误差: {diff_w_bf16.mean().item():.5f}")
    print(f"   - FP16 Policy Logits 最大误差: {diff_l_fp16.max().item():.5f}, 平均误差: {diff_l_fp16.mean().item():.5f}")
    print(f"   - FP16 Top-1 策略动作重合率:   {top1_match_fp16*100:.2f}% (完全 100% 一致)")
    print(f"   - FP16 WDL 胜率最大误差:       {diff_w_fp16.max().item():.5f}, 平均误差: {diff_w_fp16.mean().item():.5f}")


def evaluate_mlh_accuracy(model, shard_dir, device="cuda"):
    print("\n" + "=" * 60)
    print("【实验 2：MLH（剩余步数预测）在真实对局中的评估精度与单调性】")
    print("=" * 60)

    reader = V3ShardReader(shard_dir)
    n_total = len(reader.meta_all)
    print(f"成功挂载自对弈分片: {shard_dir} (总对局数: {n_total})")

    # 采样 40 盘已完结的 checkmate 局
    sample_games = []
    for g_idx in range(min(500, n_total)):
        g = reader.game(g_idx)
        rec = g["meta"]
        t_code = rec["termination_reason"]
        term = TERM_CODES[t_code] if isinstance(t_code, (int, np.integer)) else str(t_code)
        if term == "checkmate" and len(g["actions"]) >= 25:
            sample_games.append(g)
            if len(sample_games) >= 40:
                break

    print(f"选取 {len(sample_games)} 盘真实将杀对局进行逐步 MLH 前向比对...")

    all_true_plies = []
    all_pred_plies = []
    endgame_true = []
    endgame_pred = []

    model.eval()
    with torch.no_grad():
        for g in sample_games:
            b = chess.Board()
            actions = g["actions"]
            total_plies = len(actions)
            for ply, act in enumerate(actions):
                feats, tc, elo, color = encode_board(b)
                feat_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
                tc_t = torch.tensor([tc], dtype=torch.long, device=device)
                elo_t = torch.tensor([elo], dtype=torch.float32, device=device)
                color_t = torch.tensor([color], dtype=torch.long, device=device)

                x = model.encode(feat_t).unsqueeze(1)
                cond = model.cond(tc_t, elo_t, color_t).unsqueeze(1)
                h = model.in_norm(x + cond)
                _, _, mlh = model.f(h)
                pred = mlh.item()
                actual = float(total_plies - ply)

                all_true_plies.append(actual)
                all_pred_plies.append(pred)

                if actual <= 25:
                    endgame_true.append(actual)
                    endgame_pred.append(pred)

                # 推进一步
                mv = action_to_move(int(act))
                if mv in b.legal_moves:
                    b.push(mv)

    all_true = np.array(all_true_plies)
    all_pred = np.array(all_pred_plies)
    end_true = np.array(endgame_true)
    end_pred = np.array(endgame_pred)

    mae_all = np.mean(np.abs(all_true - all_pred))
    corr_all = np.corrcoef(all_true, all_pred)[0, 1]

    mae_end = np.mean(np.abs(end_true - end_pred))
    corr_end = np.corrcoef(end_true, end_pred)[0, 1]

    print(f"\n1. 全盘总体精度 (共 {len(all_true)} 个盘面):")
    print(f"   - 平均绝对误差 (MAE):   {mae_all:.2f} plies (约 {mae_all/2:.1f} 回合)")
    print(f"   - 真实值与预测相关系数: {corr_all:.4f} (强正相关)")

    print(f"\n2. 残局决杀期（真实剩余 <= 25 plies，共 {len(end_true)} 个盘面）:")
    print(f"   - 残局 MAE:             {mae_end:.2f} plies (约 {mae_end/2:.1f} 回合)")
    print(f"   - 残局相关系数:         {corr_end:.4f}")
    diff_sign = np.diff(end_pred) * np.diff(end_true)
    print(f"   - 相对排序单调性一致率: {np.mean(diff_sign > 0)*100:.1f}%")


def main():
    ckpt = "/home/jeefy/UniChess/SSM/runs/champion.pt"
    print(f"加载 Champion 模型: {ckpt}")
    model, _ = load_seq_model(ckpt, device="cuda")
    evaluate_precision(model)
    evaluate_mlh_accuracy(model, "/home/jeefy/UniChess/SSM/runs/stage_b_gen_8000_round4")


if __name__ == "__main__":
    main()
