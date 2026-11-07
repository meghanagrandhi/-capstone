"""Tier 3 eval v2 (bigger sets + swapped-image test + 100-item attention probe), from saved adapters.
  A: all 500 calibration items, real image
  A: all 500 calibration items; G: all 581 G candidates
  each with real image, blank (gray) image and swapped image (another item's picture from the same set),
  so image benefit = real - blank and real - swap can be computed per item at every checkpoint
  probe: 50 text-answerable + 50 image-necessary held-out training-pool items (in none of the --probe-mixes mixtures):
         answer NLL with real/blank image and mean attention mass on image tokens
Evaluated at the checkpoints closest to --pcts of training. Everything is cached per (run, step, set, condition).
Usage: python -m scripts.run_tier3_eval_v2 --runs matched_text10 matched_text90 matched_text10_s1 ...
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_mmstar_by_ids
from src.eval import evaluate, swap_map
from src.ioutil import safe_to_csv
from src.train import probe_metrics
from src.vlm import make_model, select_model

OUT = "results/tier3/eval_v2"


def pick_steps(steps, pcts):
    total = max(steps)
    return sorted({min(steps, key=lambda s: abs(100 * s / total - p)) for p in pcts})


def build_probe(per_class, mixes=("matched",), seed=123):
    pl = pd.read_csv("results/tier2/full/pool_labeled_matched.csv")
    used = set()
    for mix in mixes:
        for t in ("text10", "text90"):
            used |= set(pd.read_csv(f"results/tier2/full/mixture_{mix}_{t}.csv").id)
    rng = np.random.RandomState(seed)
    parts = []
    for lab in ("text_answerable", "image_necessary"):
        pool = pl[(pl.label == lab) & ~pl.id.isin(used)]
        assert len(pool) >= per_class, f"only {len(pool)} held-out {lab} items"
        parts.append(pool.iloc[rng.permutation(len(pool))[:per_class]])
    return pd.concat(parts).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pcts", nargs="+", type=float, default=[0, 20, 40, 60, 100])
    ap.add_argument("--probe-per-class", type=int, default=50)
    ap.add_argument("--probe-mixes", nargs="+", default=["matched"], choices=["matched", "balanced"],
                    help="mixtures the probe items must stay out of; use both for balanced runs")
    ap.add_argument("--sets-dir", default="results/tier3/eval_sets_v2")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N items per set")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default="qwen2vl", choices=["qwen2vl", "smolvlm", "internvl"])
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    select_model(cfg, a.model)
    t1 = cfg["tier1"]
    out = a.out
    os.makedirs(out, exist_ok=True)
    sets = json.load(open(os.path.join(a.sets_dir, "eval_sets.json")))
    items = {k: load_mmstar_by_ids(t1, sets[k][: a.limit or None]) for k in ("A", "G")}
    swaps = {k: swap_map(items[k], seed=0) for k in ("A", "G")}  # swapped images stay within the same set
    probe = build_probe(a.probe_per_class, a.probe_mixes)
    probe[["id", "subset", "label", "gap"]].to_csv(os.path.join(out, "probe_items.csv"), index=False)
    print(f"A={len(items['A'])} G={len(items['G'])} probe={len(probe)} ({probe.label.value_counts().to_dict()})", flush=True)

    from peft import PeftModel, load_peft_weights, set_peft_model_state_dict
    vlm = make_model(cfg["model"]).load()
    base = vlm.model
    pm = None
    t0 = time.time()
    for run in a.runs:
        cks = {int(re.search(r"step_(\d+)", c).group(1)): c
               for c in glob.glob(os.path.join("results/tier3", run, "adapters", "step_*"))}
        assert cks, f"no adapters for {run}"
        total = max(cks)
        for step in pick_steps(list(cks), a.pcts):
            if pm is None:
                pm = PeftModel.from_pretrained(base, cks[step], is_trainable=False)
            else:
                set_peft_model_state_dict(pm, load_peft_weights(cks[step], device="cuda"))
            pm.eval()
            vlm.model = pm
            tag = f"{run}_step{step:04d}"
            accs = {}
            # image benefit = acc(real) - acc(blank) and acc(real) - acc(swap), paired on the same items, for A and G
            for k, cond in (("A", "real"), ("G", "real"), ("G", "swap"), ("G", "blank"), ("A", "blank"), ("A", "swap")):
                d = evaluate(vlm, items[k], 0, cond, os.path.join(out, f"{tag}_{k}_{cond}.csv"), t1["blank_gray"],
                             log_every=1000, swap=swaps[k] if cond == "swap" else None)
                accs[f"{k}_{cond}"] = float(d.correct.mean())
            pf = os.path.join(out, f"{tag}_probe.json")
            if not os.path.exists(pf):
                m = {k: float(v) for k, v in probe_metrics(vlm, probe).items()}
                json.dump(m, open(pf, "w"), indent=1)
            print(f"  {run} step {step}/{total}: " + " ".join(f"{k}={v:.3f}" for k, v in accs.items()) +
                  f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    # gather everything that exists into one summary table
    rows = []
    for f in glob.glob(os.path.join(out, "*_step*_*_*.csv")):
        m = re.match(r"(.+)_step(\d+)_([AG])_(real|swap|blank)\.csv", os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f)
        run, step = m.group(1), int(m.group(2))
        sm = re.match(r"(?:(smolvlm|internvl)_)?((?:matched|balanced)_text\d+)(?:_s(\d+))?$", run)
        acc = float(d.correct.mean())
        rows.append(dict(run=run, model=(sm.group(1) or "qwen2vl") if sm else "?", mixture=sm.group(2) if sm else run,
                         seed=int(sm.group(3) or 0) if sm else -1,
                         step=step, set=m.group(3), cond=m.group(4), n=len(d), acc=acc,
                         se=float(np.sqrt(acc * (1 - acc) / len(d)))))
    safe_to_csv(pd.DataFrame(rows).sort_values(["model", "mixture", "seed", "step", "set", "cond"]), os.path.join(out, "checkpoint_eval_v2.csv"))
    print(f"saved {out}/checkpoint_eval_v2.csv ({len(rows)} rows); {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
