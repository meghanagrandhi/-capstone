"""Write the forecaster grid run list -> configs/grid_runs.csv (one row per training run; read by scripts/run_grid_queue.sh).
Per model (47 runs): text share {10,30,50,70,90} x lr {1e-4, 2e-4} x LoRA rank {8, 16} x seed {0, 1}  (40)
                     + seed 2 at the pilot settings (lr 2e-4, r 16) for every share                  (5)
                     + short runs (steps = half) at share 10 and 90, seed 0, pilot settings          (2)
Rows are ordered so a partly finished queue still covers every share and setting (seed 0 first, then seed 1, ...).
LOCKED TEST SET: every run with text share 50 (column locked_test=1). Nobody fits or tunes on them until the very end.
Usage: python -m scripts.make_grid
"""
import itertools
import os

import pandas as pd

MODELS = {"qwen2vl": 120, "smolvlm": 125, "internvl": 125}  # default steps per model (pilot)
SHARES, LRS, RANKS = (10, 30, 50, 70, 90), (2e-4, 1e-4), (16, 8)


def main():
    rows = []
    for model, steps in MODELS.items():
        for seed in (0, 1):
            for lr, r, share in itertools.product(LRS, RANKS, SHARES):
                rows.append(dict(model=model, share=share, lr=lr, rank=r, steps=steps, seed=seed))
        for share in SHARES:
            rows.append(dict(model=model, share=share, lr=2e-4, rank=16, steps=steps, seed=2))
        for share in (10, 90):
            rows.append(dict(model=model, share=share, lr=2e-4, rank=16, steps=steps // 2, seed=0))
    g = pd.DataFrame(rows)
    g["locked_test"] = (g.share == 50).astype(int)
    os.makedirs("configs", exist_ok=True)
    g.to_csv("configs/grid_runs.csv", index=False)
    print(g.groupby("model").size().to_dict(), "runs;", int(g.locked_test.sum()), "locked (share 50)")


if __name__ == "__main__":
    main()
