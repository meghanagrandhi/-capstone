"""Final figures for the report (all three models, 3 seeds). CPU only. Writes results/figures/*.png
  fig1_blindness.png      base models: accuracy with real / blank / swapped image (A set), chance line
  fig2_training_curves.png  A accuracy and G image benefit (real - swap) vs training step, mean over 3 seeds,
                            band = seed min..max; text10 vs text90; one row per model
  fig3_contrast_forest.png  text90 - text10 at the final step with 95% bootstrap CIs, every metric, all models
  fig4_A_split.png          A accuracy split by whether the base model can guess the item blind
All figures are labelled "feasibility sketch, not evidence".
Usage: python -m scripts.plot_final
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_tier3_v2 import cube, load

OUT = "results/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MIX_COLOR = {"matched_text10": "#2a78d6", "matched_text90": "#eb6834"}  # validated pair (categorical slots 1-2)
MIX_LABEL = {"matched_text10": "10% text-answerable", "matched_text90": "90% text-answerable"}
MODEL_COLOR = {"qwen2vl": "#6250d6", "smolvlm": "#199e70", "internvl": "#c98500"}  # validated (--pairs all)
MODEL_LABEL = {"qwen2vl": "Qwen2-VL-2B (4-bit QLoRA)", "smolvlm": "SmolVLM-2.2B (bf16 LoRA)",
               "internvl": "InternVL3-2B (bf16 LoRA)"}
MODELS = list(MODEL_COLOR)
ADIR = {"qwen2vl": "analysis_v2", "smolvlm": "analysis_v2_smolvlm", "internvl": "analysis_v2_internvl"}
NOTE = "Feasibility sketch, not evidence"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def note(fig):
    fig.text(0.995, 0.005, NOTE, ha="right", va="bottom", fontsize=8, color=INK2, style="italic")


def fig1(curves):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        c = curves[model]
        r = c[(c.mixture == "matched_text10") & (c.step == 0) & (c.set == "A")].iloc[0]
        vals = [r.real, r.real - r.ben_blank, r.real - r.ben_swap]
        x = np.arange(3)
        ax.bar(x, [100 * v for v in vals], width=0.6, color=MODEL_COLOR[model], edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):  # inside the bar top, so labels never collide with the chance line
            ax.text(xi, 100 * v - 1.5, f"{100 * v:.1f}%", ha="center", va="top", fontsize=9, color=SURFACE)
        ax.axhline(25, color=INK2, linestyle="--", linewidth=1)
        ax.text(2.45, 62, "dashed line = chance (25%)", ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xticks(x, ["real image", "blank image", "swapped image"])
        ax.set_title(MODEL_LABEL[model].split(" (")[0], fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 65)
    axes[0].set_ylabel("MMStar accuracy, A set (n=500)")
    fig.suptitle("Base models: how much does the image help?", fontsize=11)
    note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "fig1_blindness.png"), dpi=200)


def fig2(dfs, curves):
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    for row, model in enumerate(MODELS):
        df = dfs[model]
        ids = {s: sorted(df[df.set == s].id.unique()) for s in ("A", "G")}
        steps = sorted(df.step.unique())
        for col, (s, what, ylabel) in enumerate((("A", "real", "A accuracy (%)"),
                                                 ("G", "ben_swap", "G image benefit,\nreal - swapped (pts)"))):
            ax = axes[row, col]
            for mix in MIX_COLOR:
                per_seed = []
                for st in steps:
                    r = cube(df, "matched_text10" if st == 0 else mix, st, s, "real", ids[s])
                    v = r if what == "real" else r - cube(df, "matched_text10" if st == 0 else mix, st, s, "swap", ids[s])
                    per_seed.append(100 * v.mean(1))
                per_seed = np.array(per_seed)
                ax.fill_between(steps, per_seed.min(1), per_seed.max(1), color=MIX_COLOR[mix], alpha=0.15, linewidth=0)
                ax.plot(steps, per_seed.mean(1), color=MIX_COLOR[mix], linewidth=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=MIX_LABEL[mix])
                ax.text(steps[-1] + 2, per_seed.mean(1)[-1], MIX_LABEL[mix].split(" ")[0], color=INK2, fontsize=8,
                        va="center")
            ax.set_xlim(-3, steps[-1] + 14)
            ax.set_ylabel(ylabel)
            if row == len(MODELS) - 1:
                ax.set_xlabel("training step (batch 8)")
            if col == 0:
                ax.text(-0.3, 0.5, MODEL_LABEL[model].split(" (")[0], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Training on mostly text-answerable vs mostly image-necessary data (mean of 3 seeds; band = seed range)",
                 fontsize=10.5, y=0.995)
    note(fig)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig2_training_curves.png"), dpi=200)


def fig3():
    rows = []
    for model in MODELS:
        c = pd.read_csv(os.path.join("results/tier3", ADIR[model], "change_vs_step0.csv"))
        c = c[c.comparison == "contrast_90_minus_10"]
        names = {("A", "real"): "A accuracy", ("A", "ben_blank"): "A image benefit (real-blank)",
                 ("A", "ben_swap"): "A image benefit (real-swap)", ("G", "real"): "G accuracy",
                 ("G", "ben_blank"): "G image benefit (real-blank)", ("G", "ben_swap"): "G image benefit (real-swap)"}
        for r in c.itertuples():
            rows.append(dict(model=model, metric=names[(r.set, r.metric)], est=r.change_pts, lo=r.ci_lo, hi=r.ci_hi))
    p = pd.read_csv("results/tier3/paired_bench/paired_bench_stats.csv")
    p = p[p.comparison == "text90 - text10"]
    for r in p.itertuples():
        b = "MMVP" if r.bench == "mmvp" else "NaturalBench"
        if r.metric == "both_right":
            rows.append(dict(model=r.model, metric=f"{b} pair accuracy", est=r.change_pts, lo=r.ci_lo, hi=r.ci_hi))
        else:  # flip so that, like every other row, negative = less use of the image
            rows.append(dict(model=r.model, metric=f"{b} answer changes with image", est=-r.change_pts,
                             lo=-r.ci_hi, hi=-r.ci_lo))
    d = pd.DataFrame(rows)
    order = list(dict.fromkeys(d.metric))
    fig, ax = plt.subplots(figsize=(9, 7))
    for k, model in enumerate(MODELS):
        x = d[d.model == model].set_index("metric").loc[order]
        y = np.arange(len(order)) + (k - 1) * 0.24
        ax.errorbar(x.est, y, xerr=[x.est - x.lo, x.hi - x.est], fmt="o", color=MODEL_COLOR[model], markersize=5.5,
                    elinewidth=2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABEL[model])
    ax.axvline(0, color=INK2, linewidth=1)
    for i in np.arange(len(order) - 1) + 0.5:
        ax.axhline(i, color=GRID, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("90% minus 10% text-answerable run, final checkpoint\n(points; 95% bootstrap CI over items; 3 seeds)")
    ax.text(0.01, 1.01, "<- less image use / lower in the 90% run", transform=ax.transAxes, fontsize=8, color=INK2)
    ax.text(0.99, 1.01, "more in the 90% run ->", transform=ax.transAxes, fontsize=8, color=INK2, ha="right")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Hypothesis predicts values left of 0 on the image-use rows", fontsize=10.5)
    note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(OUT, "fig3_contrast_forest.png"), dpi=200)
    return d


def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        s = pd.read_csv(f"results/tier3/{ADIR[model]}/A_split_by_blind_k.csv")
        groups = list(dict.fromkeys(s.group))
        base = s[(s.step == 0) & (s.mixture == "matched_text10")].set_index("group").acc_real
        last = s.step.max()
        series = [("base model", base, "#b4b2a9")] + \
                 [(MIX_LABEL[m], s[(s.step == last) & (s.mixture == m)].set_index("group").acc_real, MIX_COLOR[m])
                  for m in MIX_COLOR]
        x = np.arange(len(groups))
        for j, (lab, v, col) in enumerate(series):
            xs = x + (j - 1) * 0.26
            ax.bar(xs, 100 * v.loc[groups], width=0.24, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
            for xi, val in zip(xs, v.loc[groups]):
                ax.text(xi, 100 * val + 0.8, f"{100 * val:.0f}", ha="center", fontsize=8, color=INK2)
        n = s.drop_duplicates("group").set_index("group").n_items
        ax.set_xticks(x, [f"{g.split(' (')[1][:-1]}\n({g.split(' (')[0].replace('blind_', 'blind ')}, n={n[g]})"
                          for g in groups], fontsize=9)
        ax.set_title(MODEL_LABEL[model].split(" (")[0], fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 75)
    axes[0].set_ylabel("A accuracy, real image (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("A set split by whether the base model can answer without the image (k of 6 blind orders)",
                 fontsize=10.5)
    note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig4_A_split.png"), dpi=200)


def main():
    os.makedirs(OUT, exist_ok=True)
    dfs = {m: load(m) for m in MODELS}
    curves = {m: pd.read_csv(f"results/tier3/{ADIR[m]}/curves.csv") for m in MODELS}
    fig1(curves)
    fig2(dfs, curves)
    d = fig3()
    fig4()
    print(d.round(2).to_string(index=False))
    print("saved", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
