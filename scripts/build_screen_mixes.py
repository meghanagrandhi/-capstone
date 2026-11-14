"""Screen mixtures (CPU only): subset / size variants of the reference mix, for the dataset-combination screen.

Uses the same within-subset labels, yes/no balancing, grid probe exclusion and MMStar overlap check as run_tier2.
Each variant writes results/tier2/screen/<name>/mixture_<name>_text{10,90}.csv and a summary json.
Usage: python -m scripts.build_screen_mixes [--only name ...]
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_tier2 as t2mod
from src.ioutil import safe_to_csv

FULL = "results/tier2/full"
OUT = "results/tier2/screen"
# name: (subsets, total mixture size)
SPECS = {
    "aokvqa_only": (["aokvqa"], 300),
    "cocoqa_only": (["cocoqa"], 300),
    "tallyqa_only": (["tallyqa"], 300),
    "vsr_only": (["vsr"], 900),
    "size450": (None, 450),
    "size1200": (None, 1200),
}


def build(name, subsets, size, cfg, eval_fp, probe_ids):
    t2 = dict(cfg["tier2"])
    t2["mixture_size"] = size
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(FULL, "pool_labeled_matched.csv"))
    if subsets:
        df = df[df.subset.isin(subsets)]
    df.to_csv(os.path.join(out, "pool_labeled_matched.csv"), index=False)
    t2mod.stage_mix_balanced(cfg, t2, out, eval_fp, False, fracs=[0.1, 0.9], prefix=name,
                             summary="summary.json", exclude_ids=probe_ids, rng_per_frac=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    eval_fp = t2mod.merge_fps([json.load(open(os.path.join(FULL, "..", "mmstar_fingerprints.json")))])
    probe_ids = set(pd.read_csv(os.path.join(FULL, "grid_probe_items.csv")).id)
    for name, (subsets, size) in SPECS.items():
        if a.only and name not in a.only:
            continue
        try:
            build(name, subsets, size, cfg, eval_fp, probe_ids)
            print(f"OK {name}")
        except ValueError as e:
            print(f"NOT FEASIBLE {name}: {e}")


if __name__ == "__main__":
    main()
