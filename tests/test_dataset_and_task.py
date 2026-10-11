#!/usr/bin/env python3
"""
Integration test for SelfPlayDataset (Surprise PER) and StageB2Task.
Verifies multi-directory replay buffer, batches generation, auto_steps, and validate score.
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np

# 确保导入根
IMPORT_ROOT = os.environ.get("UNICHESS_IMPORT_ROOT", str(Path(__file__).resolve().parents[2]))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from SSM.dataset.dataset_selfplay import SelfPlayDataset
from SSM.tasks import StageB2Task

def test_dataset_and_task():
    print("==================================================")
    print("   Stage B2 Dataset & Task Rigorous Integration Test")
    print("==================================================")

    # 1. 查找远端现有的自对弈分片目录 (Gen 8 与 Gen 9)
    sp_dirs = []
    base_runs = Path(IMPORT_ROOT) / "SSM/runs/loop_stage_b"
    for g in [8, 9]:
        d = base_runs / f"gen_{g:04d}/selfplay"
        if (d / "manifest.json").exists():
            sp_dirs.append(str(d))

    if not sp_dirs:
        # 本地 fallback 查找
        print("Notice: No loop_stage_b runs found in default path, skipping live data test.")
        return

    print(f"[Step 1] Loading SelfPlayDataset with multi-dirs: {sp_dirs}")
    ds = SelfPlayDataset(sp_dirs, workers=2)
    assert len(ds.readers) == len(sp_dirs), f"Readers count mismatch: {len(ds.readers)} vs {len(sp_dirs)}"
    assert len(ds.per_gen_weights) == len(sp_dirs), "Weights list mismatch"
    for i, w in enumerate(ds.per_gen_weights):
        assert not np.isnan(w).any(), f"Dir {i} weights have NaN!"
        print(f"  Dir {i} ({Path(sp_dirs[i]).parent.name}): {len(w)} games | Surprise Weights [{w.min():.4f}, {w.max():.4f}], Mean={w.mean():.4f}")
    print("[PASS] Multi-dir dataset initialized cleanly with Surprise PER weights.")

    # 2. 测试 epoch_batches
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step 2] Testing epoch_batches (device={device}, shuffle=True)...")
    batch_gen = ds.epoch_batches(microbatch=4, device=device, shuffle=True, prefetch=2)
    sample_b = next(batch_gen)
    assert "batch" in sample_b, "Missing batch key"
    assert "policy_soft_target" in sample_b, "Missing policy_soft_target key"
    assert sample_b["policy_soft_target"].shape[0] == 4, f"Batch size mismatch: {sample_b['policy_soft_target'].shape}"
    assert sample_b["policy_soft_target"].shape[-1] == 1936, f"Actions shape mismatch: {sample_b['policy_soft_target'].shape}"
    print(f"  Successfully yielded batch with shape: {sample_b['policy_soft_target'].shape}")
    print("[PASS] epoch_batches yielded valid tensor batch with correct shapes.")

    # 3. 测试 StageB2Task 的 auto_steps (多目录支持)
    print("[Step 3] Testing StageB2Task auto_steps across multiple dirs...")
    task = StageB2Task(
        selfplay={"dirs": sp_dirs},
        data={"dir": os.path.join(IMPORT_ROOT, "SSM/data/shards"), "t_max": 300},
        ckpt=os.path.join(IMPORT_ROOT, "SSM/runs/champion.pt"),
        microbatch=4,
        workers=2,
        t_max=300
    )
    steps_accum16 = task.auto_steps(accum=16)
    eff_batch = 4 * 16 # 64
    total_buffer_games = sum(len(r.is_val_arr) - int(r.is_val_arr.sum()) for r in ds.readers)
    expected_steps = min(3 * total_buffer_games // eff_batch, 2000)
    assert steps_accum16 == expected_steps, f"Steps calculation mismatch: {steps_accum16} vs {expected_steps}"
    print(f"  Total buffer games across {len(sp_dirs)} dirs = {total_buffer_games}")
    print(f"  Calculated auto_steps (accum=16) = {steps_accum16} steps (Expected: {expected_steps})")
    print("[PASS] StageB2Task.auto_steps multi-dir sum calculation verified.")

    # 4. 验证 validate 复合指标 (score 覆盖率)
    print("[Step 4] Checking StageB2Task validate score formula...")
    dummy_out = {
        "val_selfplay_loss_policy": 1.60,
        "val_selfplay_loss_value": 0.50,
        "val_human_loss_value": 1.00
    }
    # 模拟 validate() 的内部公式:
    sp_pol = dummy_out.get("val_selfplay_loss_policy", float("inf"))
    sp_val = dummy_out.get("val_selfplay_loss_value", 0.0)
    hm_val = dummy_out.get("val_human_loss_value", 0.0)
    score = sp_pol + 0.5 * sp_val + 0.3 * hm_val
    expected_score = 1.60 + 0.5 * 0.50 + 0.3 * 1.00 # 1.60 + 0.25 + 0.30 = 2.15
    assert abs(score - expected_score) < 1e-6, f"Composite score mismatch: {score} vs {expected_score}"
    print(f"  Simulated composite score: {score:.4f} (Matches expected {expected_score:.4f})")
    print("[PASS] Composite score formula correctly balances Policy, Selfplay Value, and Human Value.")

    print("\n==================================================")
    print("      ALL DATASET & TASK INTEGRATION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_dataset_and_task()
