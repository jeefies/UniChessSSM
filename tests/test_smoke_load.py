"""验证冒烟测试生成数据的加载与惊讶度采样权重"""
import os
import sys

IMPORT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if IMPORT_ROOT not in sys.path:
    sys.path.insert(0, IMPORT_ROOT)

from SSM.dataset.dataset_selfplay import SelfPlayDataset

def test_load():
    smoke_dir = "/tmp/smoke_sp"
    ds = SelfPlayDataset([smoke_dir])
    print(f"[Smoke Verify] Successfully loaded {ds.n_games} games from {smoke_dir}!")
    print(f"[Smoke Verify] Weights shape: {ds.weights.shape}")
    print(f"[Smoke Verify] Min weight: {ds.weights.min().item():.4f}, Max: {ds.weights.max().item():.4f}, Mean: {ds.weights.mean().item():.4f}")
    assert ds.n_games == 16, f"Expected 16 games, got {ds.n_games}"
    assert ds.weights.shape[0] == 16
    print("[PASS] Smoke selfplay dataset loaded and verified completely!")

if __name__ == "__main__":
    test_load()
