"""Figures for the balanced runs and the baselines (reads results/tier3/baselines, run scripts.baselines first).
fig5_balanced_curves.png : change in G image benefit over training, balanced mixes, text10 vs text90, per model
fig6_baselines.png       : baseline error (MAE, pts) predicting the final change, two splits
Usage: python -m scripts.plot_baselines
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MIX_COLOR = {10: "#2a78d6", 90: "#eb6834"}  # same validated pair as plot_final
MIX_LABEL = {10: "text10 (10% text-answerable)", 90: "text90 (90% text-answerable)"}
MODEL_TITLE = {"qwen2vl": "Qwen2-VL-2B", "smolvlm": "SmolVLM-2.2B", "internvl": "InternVL3-2B"}
NOTE = "Feasibility sketch, not evidence (3 seeds per mix, no multiple-testing correction)."
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def fig5(t):
    t = t[t.mix_type == "balanced"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, model in zip(axes, ["qwen2vl", "smolvlm", "internvl"]):
        for ts in (10, 90):
            g = t[(t.model == model) & (t.text_share == ts)]
            cols = [c for c in ["dG_5", "dG_10", "dG_20", "dG_40", "dG_100"] if g[c].notna().all()]
            x = [0] + [int(c.split("_")[1]) for c in cols]
            ys = pd.concat([pd.Series(0.0, index=g.index, name="dG_0"), g[cols]], axis=1).to_numpy()
            ax.fill_between(x, ys.min(0), ys.max(0), color=MIX_COLOR[ts], alpha=0.15, linewidth=0)
            ax.plot(x, ys.mean(0), color=MIX_COLOR[ts], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=MIX_LABEL[ts])
            ax.text(102, ys.mean(0)[-1], f"text{ts}", color=INK2, fontsize=8, va="center")
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(MODEL_TITLE[model], fontsize=10)
        ax.set_xlabel("% of training")
        ax.set_xlim(-3, 115)
    axes[0].set_ylabel("change in G image benefit (pts)\n(real - blank, vs step 0)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Balanced mixes: image benefit on the G set during training (line = mean of 3 seeds, band = seed range)",
                 fontsize=10)
    fig.text(0.995, 0.005, NOTE + " SmolVLM: only 0/40/100% checkpoints.", ha="right", va="bottom", fontsize=8,
             color=INK2, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig5_balanced_curves.png"), dpi=200)


def fig6(s):
    s = s[s.subset == "all_runs"]
    names = {"zero": "nothing changes", "static": "static (model + mix)", "loss": "training loss", "early_G40": "G at 40%"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (split, title) in zip(axes, [("loo", "Leave one run out (config seen)"),
                                         ("leave_one_model_out", "Leave one model out (new model)")]):
        g = s[s.split == split].set_index("baseline").loc[list(names)]
        y = range(len(g))
        ax.barh(y, g.MAE, height=0.55, color="#2a78d6", edgecolor=SURFACE, linewidth=2)
        ax.errorbar(g.MAE, y, xerr=[g.MAE - g.MAE_lo, g.MAE_hi - g.MAE], fmt="none", ecolor=INK2, capsize=3, linewidth=1)
        for yi, (m, pa) in zip(y, zip(g.MAE, g.pair_acc)):
            ax.text(g.MAE_hi.iloc[yi] + 0.15, yi, f"{m:.1f} pts  |  pairs right {100 * pa:.0f}%", va="center",
                    fontsize=8, color=INK)
        ax.set_yticks(list(y), [names[k] for k in g.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 7.5)
        ax.set_title(title, fontsize=10)
    fig.supxlabel("mean absolute error predicting the final change (pts, lower = better)", fontsize=10, y=0.06)
    fig.suptitle("Simple baselines for the final drop in G image benefit (36 runs, 18 text10/text90 pairs)", fontsize=10)
    fig.text(0.995, 0.005, NOTE + " Bars: 95% bootstrap CI over runs.", ha="right", va="bottom", fontsize=8,
             color=INK2, style="italic")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig6_baselines.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig5(pd.read_csv("results/tier3/baselines/run_table.csv"))
    fig6(pd.read_csv("results/tier3/baselines/baseline_scores.csv"))
