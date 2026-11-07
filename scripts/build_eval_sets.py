"""Tier 3 eval sets (fixed, saved to disk):
  A = n_A items sampled at random (seeded) from the 500-item calibration sample          (general MC set)
  G = n_G items from the SEPARATE G candidate pool with blind k <= G_max_k of 6 orders   (image-necessary set)
The 'image-necessary' property is then TESTED on fresh option orders that were not used to select G:
blind accuracy on G should be near/below chance, while A's blind accuracy is the unselected reference.
Usage: python -m scripts.build_eval_sets [--smoke]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_mmstar_by_ids
from src.eval import evaluate
from src.vlm import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all", action="store_true", help="v2: A = all calibration items, G = all candidates (k<=G_max_k)")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t1, t3 = cfg["tier1"], cfg["tier3"]
    nA = t3["smoke_n_A"] if a.smoke else t3["n_A"]
    nG = t3["smoke_n_G"] if a.smoke else t3["n_G"]
    out = os.path.join(t3["out_dir"], "eval_sets_smoke" if a.smoke else "eval_sets_v2" if a.all else "eval_sets")
    os.makedirs(out, exist_ok=True)
    rng = np.random.RandomState(t3["seed"])

    calib_ids = sorted(pd.read_csv(os.path.join(t1["out_dir"], "full", "per_item.csv")).id.unique())
    gdir = os.path.join(t1["out_dir"], "gpool_smoke" if a.smoke else "gpool")
    gl = pd.read_csv(os.path.join(gdir, "gpool_labels.csv"))
    if not a.smoke:
        assert len(gl) >= 990 and gl.n_orders.iloc[0] == 6, f"G pool labeling incomplete: {len(gl)} items"
    cand = gl[gl.k_blind <= t3["G_max_k"]].id.tolist()
    if a.all:
        nA, nG = len(calib_ids), len(cand)
    assert len(cand) >= nG, f"only {len(cand)} G candidates with k<={t3['G_max_k']}"
    A = sorted(rng.choice(calib_ids, nA, replace=False).tolist())
    G = sorted(rng.choice(cand, nG, replace=False).tolist())
    assert not (set(A) & set(G)), "A and G overlap"
    assert set(G) <= set(gl.id) and not (set(G) & set(calib_ids)), "G must come from the separate pool"
    json.dump(dict(A=A, G=G, G_max_k=t3["G_max_k"], n_candidates=len(cand), seed=t3["seed"]),
              open(os.path.join(out, "eval_sets.json"), "w"), indent=1)
    print(f"A: {len(A)} items from calibration sample; G: {len(G)} items from {len(cand)} candidates with k<={t3['G_max_k']}")

    # test the property on fresh orders (not used for selection) + real-image accuracy at the base model
    items = {"A": load_mmstar_by_ids(t1, A), "G": load_mmstar_by_ids(t1, G)}
    model = make_model(cfg["model"]).load()
    res = {}
    for name, its in items.items():
        accs = {}
        for s in t3["fresh_blind_seeds"]:
            accs[s] = float(evaluate(model, its, s, "blank", os.path.join(out, f"fresh_blind_{name}.csv"), t1["blank_gray"]).correct.mean())
        real = float(evaluate(model, its, 0, "real", os.path.join(out, f"base_real_{name}.csv"), t1["blank_gray"]).correct.mean())
        res[name] = dict(n=len(its), blind_acc_fresh_orders=accs, blind_acc_fresh_mean=float(np.mean(list(accs.values()))),
                         real_acc_base_canonical=real, analytic_se_real=float(np.sqrt(real * (1 - real) / len(its))))
        print(name, res[name])
    json.dump(res, open(os.path.join(out, "property_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
