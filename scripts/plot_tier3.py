"""Tier 3 figures + paired item-level comparison vs checkpoint 0.
Usage: python -m scripts.plot_tier3 [--smoke]
Outputs: results/tier3/fig_A_G_vs_checkpoint.png, fig_training_diagnostics.png, paired_vs_step0.csv
Everything is labeled as a feasibility sketch (2 runs, 1 seed, n=300 per set) - NOT evidence."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK, MUTED = "#1f2937", "#6b7280"
RUN_COLORS = {"text10": "#2a6fdb", "text90": "#d9730d",   # fixed identity per mixture share
              "matched_text10": "#2a6fdb", "matched_text90": "#d9730d"}
RUN_LABELS = {"text10": "10% text-answerable mix", "text90": "90% text-answerable mix",
              "matched_text10": "10% text-answerable (subset-matched)", "matched_text90": "90% text-answerable (subset-matched)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t3 = cfg["tier3"]
    sfx = "_smoke" if a.smoke else ""
    out = t3["out_dir"]
    ce = pd.read_csv(os.path.join(out, f"checkpoint_eval{sfx}.csv"))
    ce["base_run"] = ce.run.str.replace("_smoke", "")
    t1sd = json.load(open("results/tier1/full/summary.json"))["acc_real_shuffled_sd"]  # Tier 1 option-shuffle SD at n=500

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- Figure 1: A and G vs checkpoint ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=150, sharex=True)
    for ax, k, title in ((axes[0], "A", "A: general MC (300 calibration items)"),
                         (axes[1], "G", "G: image-necessary (300 items, blind k<=1 of 6)")):
        d = ce[ce.set == k]
        p0 = d[d.pct == 0].acc.mean()
        n = int(d.n.iloc[0])
        sigma = max(np.sqrt(p0 * (1 - p0) / n), t1sd * np.sqrt(500 / n))  # analytic SE vs Tier-1 shuffle SD rescaled to n
        xs = [0, 100]
        ax.fill_between(xs, p0 - 2 * sigma, p0 + 2 * sigma, color="#e5e7eb", label="Tier-1 noise band ±2σ")
        ax.fill_between(xs, p0 - sigma, p0 + sigma, color="#cfd4dc", label="±1σ")
        ax.axhline(p0, color=MUTED, lw=0.8)
        for run, g in d.groupby("base_run"):
            g = g.sort_values("pct")
            ax.errorbar(g.pct, g.acc, yerr=g.se, marker="o", ms=4, lw=1.4, capsize=2, color=RUN_COLORS.get(run, "k"),
                        label=RUN_LABELS.get(run, run))
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("% of training steps (checkpoint)", color=INK)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax)
    axes[0].set_ylabel("accuracy (real image)", color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("FEASIBILITY SKETCH - NOT EVIDENCE   (2 runs, 1 seed, n=300 per set; "
                 "band: max(analytic SE, Tier-1 shuffle SD rescaled) around the step-0 value; bars: ±1 analytic SE)",
                 fontsize=8.5, color="#b42318", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, f"fig_A_G_vs_checkpoint{sfx}.png"))
    print("saved fig_A_G_vs_checkpoint")

    # ---------------- paired item-level comparison vs step 0 ----------------
    rows = []
    for run in ce.run.unique():
        for k in ("A", "G"):
            base_f = os.path.join(out, "eval" + sfx, f"{run}_step0000_{k}.csv")
            if not os.path.exists(base_f):
                continue
            b0 = pd.read_csv(base_f).set_index("id").correct
            for step in sorted(ce[(ce.run == run) & (ce.set == k)].step.unique()):
                c = pd.read_csv(os.path.join(out, "eval" + sfx, f"{run}_step{step:04d}_{k}.csv")).set_index("id").correct.reindex(b0.index)
                gained, lost = int(((b0 == 0) & (c == 1)).sum()), int(((b0 == 1) & (c == 0)).sum())
                p = float(stats.binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0
                rows.append(dict(run=run, set=k, step=step, acc=float(c.mean()), acc_step0=float(b0.mean()),
                                 gained=gained, lost=lost, net=gained - lost, mcnemar_exact_p=p))
    pd.DataFrame(rows).to_csv(os.path.join(out, f"paired_vs_step0{sfx}.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- Figure 2: training diagnostics ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), dpi=150)
    for run in sorted(ce.base_run.unique()):
        rn = run + sfx
        tl = pd.read_csv(os.path.join(out, rn, "train_log.csv"))
        pl = pd.read_csv(os.path.join(out, rn, "probe_log.csv"))
        w = max(3, len(tl) // 12)
        col = RUN_COLORS[run]
        for lab, ls, name in (("loss_text_answerable", "-", "text-answerable items"), ("loss_image_necessary", "--", "image-necessary items")):
            axes[0, 0].plot(tl.step, tl[lab].rolling(w, min_periods=1).mean(), ls=ls, color=col, lw=1.4,
                            label=f"{RUN_LABELS[run]} / {name}")
        axes[0, 1].plot(tl.step, tl.grad_norm.rolling(w, min_periods=1).mean(), color=col, lw=1.4, label=RUN_LABELS[run])
        for lab, ls, name in (("text_answerable", "-", "text-answerable probe"), ("image_necessary", "--", "image-necessary probe")):
            axes[1, 0].plot(pl.step, pl[f"probe_attn_img_mass_{lab}"], ls=ls, marker="o", ms=3, color=col, lw=1.4, label=f"{RUN_LABELS[run]} / {name}")
            axes[1, 1].plot(pl.step, pl[f"probe_gap_{lab}"], ls=ls, marker="o", ms=3, color=col, lw=1.4, label=f"{RUN_LABELS[run]} / {name}")
    axes[0, 0].set_title("train loss by item type (rolling mean; NaN when class absent in a batch)", loc="left", fontsize=8.5, color=INK)
    axes[0, 1].set_title("gradient norm (pre-clip, rolling mean)", loc="left", fontsize=8.5, color=INK)
    axes[1, 0].set_title("held-out probe: mean attention mass on image tokens (answer queries)", loc="left", fontsize=8.5, color=INK)
    axes[1, 1].set_title("held-out probe: loss gap = NLL(blank) - NLL(real)", loc="left", fontsize=8.5, color=INK)
    for ax in axes.flat:
        style(ax)
        ax.set_xlabel("optimizer step", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("FEASIBILITY SKETCH - NOT EVIDENCE (training diagnostics, 1 seed per run)", fontsize=8.5, color="#b42318", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, f"fig_training_diagnostics{sfx}.png"))
    print("saved fig_training_diagnostics")


if __name__ == "__main__":
    main()
