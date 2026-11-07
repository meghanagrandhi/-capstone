"""Unattended pipeline for everything after Tier 2 labeling. Each stage is a separate process (separate log in
results/pipeline_logs/); stops loudly on the first failure. Stages are resumable because every script caches.
Order (Tier 1, Tier 2, G pool, subset-matched mixtures and the Tier 3 smoke test are already done):
  eval sets -> timing run (50 steps + 1 eval pass) -> size steps -> train matched_text10 -> train matched_text90
  -> eval all checkpoints -> plots.
Usage: python -m scripts.run_pipeline
"""
import json
import math
import os
import re
import subprocess
import sys
import time

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = [sys.executable, "-W", "ignore", "-u", "-m"]
LOGDIR = "results/pipeline_logs"
os.makedirs(LOGDIR, exist_ok=True)
ENV = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf-8")
T0 = time.time()


def say(msg):
    print(f"[{(time.time()-T0)/60:6.1f} min] {msg}", flush=True)


def run(name, args):
    say(f"START {name}")
    t = time.time()
    with open(os.path.join(LOGDIR, f"{name}.log"), "w", encoding="utf-8") as f:
        rc = subprocess.call(PY + args, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    say(f"END   {name} rc={rc} ({(time.time()-t)/60:.1f} min)")
    if rc != 0:
        say(f"FAILED at {name}; see {LOGDIR}/{name}.log")
        sys.exit(rc)


def main():
    run("build_eval_sets", ["scripts.build_eval_sets"])
    # timing run: ~50 steps + one eval pass of A and G at the last checkpoint
    run("tier3_timing_train", ["scripts.run_tier3_train", "--mixture", "matched_text90", "--steps", "50", "--tag", "_timing"])
    run("tier3_timing_eval", ["scripts.run_tier3_eval", "--runs", "matched_text90_timing", "--only-last"])
    meta = json.load(open("results/tier3/matched_text90_timing/run_meta.json"))
    sps = meta["sec_per_step"]
    log = open(os.path.join(LOGDIR, "tier3_timing_eval.log"), errors="ignore").read()
    eval_min = float(log.split("eval pass took ")[1].split(" min")[0]) if "eval pass took" in log else float("nan")
    S = int(max(60, min(300, math.floor(1200 / sps / 10) * 10)))   # target ~20 min of training per run
    sizing = dict(sec_per_step=sps, steps_chosen=S, est_train_min_per_run=S * sps / 60,
                  eval_min_one_checkpoint_A_and_G=eval_min, est_eval_min_total=16 * eval_min,
                  est_remaining_min=2 * S * sps / 60 + 16 * eval_min, peak_vram_gib=meta["peak_vram_gib"])
    json.dump(sizing, open("results/tier3/step_sizing.json", "w"), indent=2)
    say(f"SIZING (estimate before launching the real runs): {sizing}")
    for mix in ("matched_text10", "matched_text90"):
        run(f"tier3_train_{mix}", ["scripts.run_tier3_train", "--mixture", mix, "--steps", str(S)])
    run("tier3_eval", ["scripts.run_tier3_eval", "--runs", "matched_text10", "matched_text90"])
    run("tier3_plot", ["scripts.plot_tier3"])
    say("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
