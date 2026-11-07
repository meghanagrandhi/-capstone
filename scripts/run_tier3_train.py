"""Tier 3a: QLoRA run on one mixture. Usage:
  python -m scripts.run_tier3_train --mixture text10|text90 [--smoke] [--steps N] [--tag _timing]
Saves adapters at 0/5/10/20/40/60/80/100% of steps + train_log.csv + probe_log.csv under results/tier3/<run>/."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.overlap import assert_no_overlap, mmstar_fingerprints
from src.train import attach_lora, train
from src.vlm import make_model, select_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True, choices=["text10", "text90", "matched_text10", "matched_text90",
                                                             "balanced_text10", "balanced_text90"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the run directory (e.g. _timing)")
    ap.add_argument("--seed", type=int, default=None, help="training seed (default: config); non-default seeds get a _s<seed> suffix")
    ap.add_argument("--model", default="qwen2vl", choices=["qwen2vl", "smolvlm", "internvl"])
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    prefix = select_model(cfg, a.model)
    t3, t2 = cfg["tier3"], cfg["tier2"]
    steps = a.steps or (t3["smoke_steps"] if a.smoke else t3["steps"])
    bs = t3["smoke_batch_size"] if a.smoke else t3["batch_size"]
    src_dir = os.path.join(t2["out_dir"], "smoke" if a.smoke else "full")
    seed = t3["seed"] if a.seed is None else a.seed
    run = f"{prefix}{a.mixture}{a.tag}" + (f"_s{seed}" if seed != t3["seed"] else "") + ("_smoke" if a.smoke else "")
    out = os.path.join(t3["out_dir"], run)
    os.makedirs(out, exist_ok=True)

    mix = pd.read_csv(os.path.join(src_dir, f"mixture_{a.mixture}.csv"))
    pair = {"text10": "text90", "text90": "text10", "matched_text10": "matched_text90", "matched_text90": "matched_text10",
            "balanced_text10": "balanced_text90", "balanced_text90": "balanced_text10"}
    other = pd.read_csv(os.path.join(src_dir, f"mixture_{pair[a.mixture]}.csv"))
    # HARD RULE re-checked at training time against ALL MMStar items (A and G sets are subsets of MMStar)
    fp = mmstar_fingerprints(cfg["tier1"], os.path.join(t2["out_dir"], "mmstar_fingerprints.json"))
    assert_no_overlap(mix.id.tolist(), mix.dhash.map(lambda x: int(x, 16)).tolist(), mix.qnorm.tolist(), fp,
                      t2["overlap_hamming"], f"mixture_{a.mixture} (at training time)")

    # held-out probe: fixed items from each class that are in NEITHER mixture
    pl = pd.read_csv(os.path.join(src_dir, "pool_labeled_matched.csv" if a.mixture.startswith(("matched", "balanced")) else "pool_labeled.csv"))
    used = set(mix.id) | set(other.id)
    rng = np.random.RandomState(t3["seed"])
    probe = []
    for lab in ("text_answerable", "image_necessary"):
        pool = pl[(pl.label == lab) & ~pl.id.isin(used)]
        if len(pool) < t3["probe_per_class"]:
            print(f"  warning: only {len(pool)} unused {lab} items for the probe; topping up from used items")
            pool = pl[pl.label == lab]
        probe.append(pool.iloc[rng.permutation(len(pool))[: t3["probe_per_class"]]])
    probe = pd.concat(probe).reset_index(drop=True)

    mc = dict(cfg["model"])
    vlm = make_model(mc).load()
    torch.manual_seed(seed)  # seeds the LoRA initialization too (seed-0 runs before this fix had unseeded LoRA init)
    attach_lora(vlm, t3["lora_r"], t3["lora_alpha"], t3["lora_dropout"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tlog, plog = train(vlm, mix, probe, steps, bs, t3["lr"], seed, out, t3["max_answer_tokens"])
    meta = dict(run=run, mixture=a.mixture, n_mix=len(mix), text_share=float((mix.label == "text_answerable").mean()),
                steps=steps, batch_size=bs, lr=t3["lr"], seed=seed, lora_r=t3["lora_r"], lora_alpha=t3["lora_alpha"],
                model=cfg["model"]["hf_id"], model_revision=cfg["model"].get("revision"), peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                train_minutes=(time.time() - t0) / 60, sec_per_step=float(tlog.elapsed_s.iloc[-1] / steps))
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    print("run meta:", meta)


if __name__ == "__main__":
    main()
