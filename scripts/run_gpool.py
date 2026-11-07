"""G candidate pool: label the MMStar items NOT in the 500-item calibration sample by blind k-of-6.
k = number of the 6 option orders (canonical + 5 shuffled seeds) in which the base model answers correctly
with a blank gray image. Kept in its own folder, separate from the calibration sample (results/tier1/full).
Usage: python -m scripts.run_gpool [--smoke]
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_mmstar_rest, load_mmstar
from src.eval import evaluate
from src.vlm import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t1 = cfg["tier1"]
    out = os.path.join(t1["out_dir"], "gpool_smoke" if a.smoke else "gpool")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "per_item.csv")
    t0 = time.time()

    items = load_mmstar_rest(t1, t1["n_items"], t1["sample_seed"])
    if a.smoke:
        items = items[:10]
    # hard separation from the calibration sample
    calib_ids = set(pd.read_csv(os.path.join(t1["out_dir"], "full", "per_item.csv")).id)
    assert not (calib_ids & {it["id"] for it in items}), "G candidate pool overlaps calibration sample"
    seeds = [0] + (t1["smoke_shuffle_seeds"] if a.smoke else t1["shuffle_seeds"])

    model = make_model(cfg["model"]).load()
    runs = {s: evaluate(model, items, s, "blank", cache, t1["blank_gray"]) for s in seeds}
    wide = pd.DataFrame({f"blank_correct_seed{s}": r.set_index("id").correct for s, r in runs.items()})
    wide["k_blind"] = wide.sum(axis=1).astype(int)
    wide["n_orders"] = len(seeds)
    cat = pd.concat(runs.values()).drop_duplicates("id").set_index("id").category
    wide["category"] = cat
    wide.index.name = "id"
    wide.to_csv(os.path.join(out, "gpool_labels.csv"))
    dist = wide.k_blind.value_counts().reindex(range(len(seeds) + 1), fill_value=0)
    summ = dict(n_items=len(wide), n_orders=len(seeds), k_hist=dist.tolist(),
                blind_acc_mean=float(wide[[c for c in wide if c.startswith("blank_correct")]].mean().mean()),
                model_revision=cfg["model"].get("revision"), runtime_min=(time.time() - t0) / 60,
                note="k alone is not proof of image-necessity; see README. Separate from the 500-item calibration sample.")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("k histogram (k=0..%d):" % len(seeds), dist.tolist(), " mean blind acc %.3f" % summ["blind_acc_mean"])
    print(f"saved {out}/gpool_labels.csv; runtime {summ['runtime_min']:.1f} min")


if __name__ == "__main__":
    main()
