"""Tier 1: benchmark-blindness demo (inference only).
Usage (from repo root):  python -m scripts.run_tier1 [--smoke]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_mmstar
from src.eval import evaluate
from src.vlm import make_model


def vram_nvsmi():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t1 = cfg["tier1"]
    n = t1["smoke_n_items"] if a.smoke else t1["n_items"]
    seeds = t1["smoke_shuffle_seeds"] if a.smoke else t1["shuffle_seeds"]
    out = os.path.join(t1["out_dir"], "smoke" if a.smoke else "full")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "per_item.csv")
    t0 = time.time()
    rows_before = len(pd.read_csv(cache)) if os.path.exists(cache) else 0
    summ_path = os.path.join(out, "summary.json")

    items = load_mmstar(t1, n, t1["sample_seed"])
    print(f"{len(items)} items; categories:", pd.Series([i["category"] for i in items]).value_counts().to_dict())
    nopt = {len(i["options"]) for i in items}
    assert nopt == {4}, f"expected 4 options everywhere, got {nopt}"
    chance = 1 / 4

    model = make_model(cfg["model"]).load()
    print(f"image tokens (first item): {model.n_image_tokens(items[0]['image'])}  max_pixels={cfg['model']['max_pixels']}")
    print(f"VRAM after load: torch alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, nvidia-smi {vram_nvsmi()} MiB")

    runs = {}
    for seed in [0] + list(seeds):
        for cond in ("real", "blank"):
            runs[(seed, cond)] = evaluate(model, items, seed, cond, cache, t1["blank_gray"])
    peak = torch.cuda.max_memory_allocated() / 2**30

    acc = {k: v.correct.mean() for k, v in runs.items()}
    n_it = len(items)
    p_real, p_blank = acc[(0, "real")], acc[(0, "blank")]

    def se(p):
        return float(np.sqrt(p * (1 - p) / n_it))

    sh_real = np.array([acc[(s, "real")] for s in seeds])
    sh_blank = np.array([acc[(s, "blank")] for s in seeds])
    r0 = runs[(0, "real")].set_index("id").correct
    b0 = runs[(0, "blank")].set_index("id").correct
    both = float(((r0 == 1) & (b0 == 1)).sum() / max((r0 == 1).sum(), 1))
    img_nec = int(((r0 == 1) & (b0 == 0)).sum())
    blank_shuf = float(sh_blank.mean()) if len(seeds) else float("nan")
    bl = pd.concat([runs[(s, "blank")] for s in seeds]) if len(seeds) else runs[(0, "blank")]
    letter_share = bl.pred_letter_shown.value_counts(normalize=True).round(3).to_dict()

    res = dict(
        n_items=n_it, chance=chance,
        acc_real=float(p_real), acc_real_se_analytic=se(p_real),
        acc_blank=float(p_blank), acc_blank_se_analytic=se(p_blank),
        acc_blank_shuffled_mean=blank_shuf, acc_blank_shuffled_sd=sd(sh_blank),
        frac_real_correct_also_blind_correct=both, n_real_correct_blind_wrong=img_nec,
        noise_seeds=list(seeds), acc_real_shuffled=sh_real.tolist(),
        acc_real_shuffled_mean=float(sh_real.mean()) if len(seeds) else None,
        acc_real_shuffled_sd=sd(sh_real),
        analytic_se_at_shuffled_mean=se(float(sh_real.mean())) if len(seeds) else None,
        blind_letter_share_shuffled=letter_share,
        per_category_real=runs[(0, "real")].groupby("category").correct.mean().round(3).to_dict(),
        per_category_blind=runs[(0, "blank")].groupby("category").correct.mean().round(3).to_dict(),
        peak_vram_torch_gib=peak, vram_nvsmi_mib=vram_nvsmi(), runtime_min=(time.time() - t0) / 60,
        model=cfg["model"]["hf_id"], max_pixels=cfg["model"]["max_pixels"])
    res["model_revision"] = cfg["model"].get("revision")
    # Rerun from a complete cache: keep the original run's timing/VRAM instead of overwriting with ~0 / tiny numbers.
    rows_after = len(pd.read_csv(cache))
    if rows_after == rows_before and os.path.exists(summ_path):
        old = json.load(open(summ_path))
        for key in ("runtime_min", "peak_vram_torch_gib", "vram_nvsmi_mib"):
            if key in old:
                res[key] = old[key]
        res["timing_from_original_run"] = True
        peak = res["peak_vram_torch_gib"]
    json.dump(res, open(summ_path, "w"), indent=2)

    tbl = pd.DataFrame([
        ("acc, real image (canonical order)", p_real),
        ("acc, blank image (canonical order)", p_blank),
        ("chance (1/n_options)", chance),
        ("frac of real-correct items also correct blind", both),
        ("acc real, shuffled seeds: mean", res["acc_real_shuffled_mean"]),
        ("acc real, shuffled seeds: SD", res["acc_real_shuffled_sd"]),
        ("analytic SE sqrt(p(1-p)/n)", res["analytic_se_at_shuffled_mean"]),
        ("acc blank, shuffled seeds: mean", blank_shuf)], columns=["metric", "value"])
    tbl.to_csv(os.path.join(out, "results_table.csv"), index=False)
    print("\n" + tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("blind predicted-letter share (shuffled):", letter_share)
    print("per-category real :", res["per_category_real"])
    print("per-category blind:", res["per_category_blind"])
    print(f"peak VRAM (torch) {peak:.2f} GiB; runtime {res['runtime_min']:.1f} min")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, GRAY = "#1f2937", "#6b7280", "#2a6fdb", "#9ca3af"
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    vals, errs = [p_real, p_blank], [se(p_real), se(p_blank)]
    ax.bar([0, 1], vals, width=0.45, color=[BLUE, GRAY], yerr=errs,
           error_kw=dict(ecolor=INK, lw=1, capsize=4))
    for x, v, e in zip([0, 1], vals, errs):
        ax.text(x + 0.27, v, f"{v:.1%}", ha="left", va="center", color=INK, fontsize=10)
    if len(seeds):
        ax.errorbar([0], [res["acc_real_shuffled_mean"]], yerr=[res["acc_real_shuffled_sd"]], fmt="D",
                    color="white", mec=INK, ecolor=INK, ms=5, capsize=3,
                    label=f"{len(seeds)} shuffled-option seeds: mean ± SD")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.axhline(chance, color=MUTED, ls="--", lw=1)
    ax.text(1.69, chance - 0.012, "chance 25%", color=MUTED, fontsize=8, ha="right", va="top")
    ax.set_xticks([0, 1], ["real image", "blank (gray) image"], color=INK)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("accuracy", color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"{cfg['model']['hf_id'].split('/')[-1]} (4-bit) on MMStar, n={n_it}; bars ±1 analytic SE",
                 color=INK, fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tier1_bar.png"))
    print("saved", os.path.join(out, "tier1_bar.png"))

    print("\n=== 3-LINE SUMMARY ===")
    print(f"1) With the real image the model gets {p_real:.1%} on {n_it} MMStar items; with a blank gray image it gets {p_blank:.1%} (chance = {chance:.0%}).")
    print(f"2) Of the items it answers correctly with the image, {both:.0%} are also answered correctly blind.")
    if len(seeds):
        print(f"3) Reshuffling options over {len(seeds)} seeds moves accuracy by SD {sd(sh_real):.1%} (mean {sh_real.mean():.1%}); analytic SE is {res['analytic_se_at_shuffled_mean']:.1%}, so differences under ~2 SE are noise.")


if __name__ == "__main__":
    main()
