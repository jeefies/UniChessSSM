#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, "/home/jeefy/UniChess")
from SSM.dataset.dataset_selfplay import SelfPlayDataset

def test():
    sp_path = "/home/jeefy/UniChess/SSM/runs/loop_stage_b/gen_0009/selfplay"
    if not os.path.exists(sp_path):
        print(f"Path not found: {sp_path}")
        return
    ds = SelfPlayDataset(sp_path, workers=2)
    w = ds.per_gen_weights[0]
    print(f"OK! SelfPlayDataset loaded successfully.")
    print(f"Total games: {len(w)}")
    print(f"Surprise Weights - Min: {w.min():.4f}, Max: {w.max():.4f}, Mean: {w.mean():.4f}")
    unique_weights = sorted(list(set(np.round(w, 4)))) if 'np' in globals() else sorted(list(set([round(x, 4) for x in w])))
    print(f"Unique weight values: {unique_weights}")

if __name__ == "__main__":
    test()
