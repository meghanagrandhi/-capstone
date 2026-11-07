"""Tier 3b: evaluate A and G at every saved checkpoint with the Tier 1 evaluator (real image, canonical option
order, option-letter log-likelihood), paired on the same items. Per-item results cached per (run, step, set).
Usage: python -m scripts.run_tier3_eval --runs text10 text90 [--smoke] [--only-last]
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
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_mmstar_by_ids
from src.eval import evaluate
from src.vlm import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-last", action="store_true", help="evaluate only the final checkpoint (timing run)")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t1, t3 = cfg["tier1"], cfg["tier3"]
    sets = json.load(open(os.path.join(t3["out_dir"], "eval_sets_smoke" if a.smoke else "eval_sets", "eval_sets.json")))
    items = {k: load_mmstar_by_ids(t1, sets[k]) for k in ("A", "G")}
    eval_dir = os.path.join(t3["out_dir"], "eval_smoke" if a.smoke else "eval")
    os.makedirs(eval_dir, exist_ok=True)

    from peft import PeftModel, load_peft_weights, set_peft_model_state_dict
    vlm = make_model(cfg["model"]).load()
    base = vlm.model
    pm = None
    t0 = time.time()
    rows = []
    for run in a.runs:
        rname = run + ("_smoke" if a.smoke else "")
        cks = sorted(glob.glob(os.path.join(t3["out_dir"], rname, "adapters", "step_*")))
        assert cks, f"no adapters for run {rname}"
        total = int(re.search(r"step_(\d+)", cks[-1]).group(1))
        if a.only_last:
            cks = cks[-1:]
        for ck in cks:
            step = int(re.search(r"step_(\d+)", ck).group(1))
            if pm is None:
                pm = PeftModel.from_pretrained(base, ck, is_trainable=False)
            else:
                set_peft_model_state_dict(pm, load_peft_weights(ck, device="cuda"))
            pm.eval()
            vlm.model = pm
            for k in ("A", "G"):
                cache = os.path.join(eval_dir, f"{rname}_step{step:04d}_{k}.csv")
                d = evaluate(vlm, items[k], 0, "real", cache, t1["blank_gray"], log_every=1000)
                acc = float(d.correct.mean())
                rows.append(dict(run=rname, step=step, pct=100 * step / total if total else 0, set=k, n=len(d), acc=acc,
                                 se=float(np.sqrt(acc * (1 - acc) / len(d)))))
            print(f"  {rname} step {step}: " + " ".join(f"{r['set']}={r['acc']:.3f}" for r in rows[-2:]) +
                  f"  [{(time.time()-t0)/60:.1f} min]", flush=True)
    out = os.path.join(t3["out_dir"], "checkpoint_eval_smoke.csv" if a.smoke else
                       "checkpoint_eval_timing.csv" if a.only_last else "checkpoint_eval.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(out) and not a.only_last:
        old = pd.read_csv(out)
        new = pd.concat([old[~old.run.isin(new.run.unique())], new])
    new.to_csv(out, index=False)
    print(f"saved {out}; eval pass took {(time.time()-t0)/60:.1f} min for {len(rows)//2} checkpoints; "
          f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
