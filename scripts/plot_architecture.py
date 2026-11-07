"""Block diagram of the pilot: what was built and run (solid, green) vs the proposed next phase (dashed, gray).
Writes results/figures/architecture.png. Usage: python -m scripts.plot_architecture
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "results/figures/architecture.png"
BUILT = dict(fc="#e9f4ec", ec="#2f7d4a", ls="-")
PROP = dict(fc="#f4f4f2", ec="#8a8a8a", ls=(0, (5, 3)))
SUB_BUILT = dict(fc="#ffffff", ec="#7fb48f", ls="-")
SUB_PROP = dict(fc="#ffffff", ec="#b0b0b0", ls=(0, (5, 3)))
INK, MUTED = "#1f1f1f", "#6b6b6b"


def box(ax, x, y, w, h, style, lw=1.8, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=style["fc"], ec=style["ec"], ls=style["ls"], lw=lw))


def title(ax, x, y, s, color=INK, size=13):
    ax.text(x, y, s, fontsize=size, fontweight="bold", color=color, va="top")


def bullets(ax, x, y, items, size=9.6, line=0.2, gap=0.22, color=INK):
    for s in items:
        c = MUTED if s.startswith("(proposed)") else color
        ax.text(x, y, "\u2022", fontsize=size, color=c, va="top")
        ax.text(x + 0.18, y, s, fontsize=size, color=c, va="top", linespacing=1.25)
        y -= line * (1 + s.count("\n")) + gap


def arrow(ax, p, q, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color="#8a8a8a" if dashed else "#333333",
                                 ls=(0, (5, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


def sub(ax, x, y, w, h, head, body, style=SUB_BUILT):
    box(ax, x, y, w, h, style, lw=1.2, r=0.08)
    ax.text(x + w / 2, y + h - 0.12, head, fontsize=9.8, fontweight="bold", ha="center", va="top", color=INK)
    ax.text(x + w / 2, y + h - 0.45, body, fontsize=8.6, ha="center", va="top", color=INK, linespacing=1.25)


def main():
    fig, ax = plt.subplots(figsize=(17, 9.6))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    ax.text(8.5, 9.45, "Project architecture: feasibility pilot (done) → regression forecasting (main goal)", fontsize=19, fontweight="bold",
            ha="center", va="top", color=INK)
    # legend
    box(ax, 4.3, 8.55, 0.5, 0.3, BUILT, r=0.05)
    ax.text(4.95, 8.7, "Done: feasibility pilot (Tiers 1-3)", fontsize=10.5, va="center", color=INK)
    box(ax, 8.6, 8.55, 0.5, 0.3, PROP, r=0.05)
    ax.text(9.25, 8.7, "Main goal, next phase: forecasting (not built yet)", fontsize=10.5, va="center", color=INK)

    # 1. data
    box(ax, 0.2, 5.0, 3.6, 3.25, BUILT)
    title(ax, 0.4, 8.1, "1. Data & setup")
    bullets(ax, 0.4, 7.6, [
        "The Cauldron pool: VSR, A-OKVQA,\nCOCO-QA, TallyQA (3161 items)",
        "Labels: text-answerable vs\nimage-necessary (loss gap)",
        "Two mixes: 10% vs 90%\ntext-answerable, same sources\n(+ yes/no-balanced follow-up)",
        "Leakage guard vs all eval sets\n(fails loudly on any overlap)"], size=9.3, line=0.215, gap=0.2)

    # 2. LoRA
    box(ax, 4.5, 6.0, 7.0, 2.25, BUILT)
    title(ax, 4.7, 8.1, "2. LoRA fine-tuning")
    sub(ax, 4.75, 6.2, 2.45, 1.45, "Frozen VLM",
        "Qwen2-VL-2B (4-bit)\nSmolVLM-2.2B (bf16)\nInternVL3-2B (bf16)")
    ax.text(7.37, 6.92, "+", fontsize=18, ha="center", va="center", color=INK)
    sub(ax, 7.55, 6.2, 1.75, 1.45, "LoRA adapters", "r = 16\nlanguage-model\nlayers only")
    sub(ax, 9.5, 6.2, 1.8, 1.45, "Fine-tuning", "120-125 steps\n2 mixes x 3 seeds\nper model")
    arrow(ax, (9.3, 6.92), (9.5, 6.92))

    # 3. dataset analysis
    box(ax, 0.2, 0.75, 3.6, 3.65, BUILT)
    title(ax, 0.4, 4.25, "3. Dataset analysis")
    bullets(ax, 0.4, 3.75, [
        "Visual-necessity score = loss gap\n(answer NLL blank \u2212 NLL real image)",
        "Source-dataset balance (matched)",
        "Answer balance (yes/no):\nkey pilot lesson, it flipped\nSmolVLM's paired-image result",
        "(proposed) question-type and\ndiversity statistics"], size=9.3, line=0.215, gap=0.2)

    # 4. signals + evaluation
    box(ax, 4.5, 0.75, 7.0, 4.65, BUILT)
    title(ax, 4.7, 5.25, "4. Training signals & checkpoint evaluation")
    sub(ax, 4.75, 3.15, 2.1, 1.6, "Training loss", "per class\n(text-answerable /\nimage-necessary)")
    sub(ax, 7.0, 3.15, 2.1, 1.6, "Gradient norm", "every step")
    sub(ax, 9.25, 3.15, 2.05, 1.6, "Attention on\nimage tokens", "\n100 held-out items\n(supporting only)")
    sub(ax, 4.75, 0.95, 3.2, 2.0, "MMStar: A and G sets",
        "A = 500 general items\nG set = 581 image-necessary items\nimages: real / blank / swapped\n"
        "$G = A_{\\mathrm{real}} - A_{\\mathrm{blank}}$ (on the G set)\nswap benefit $A_{\\mathrm{real}} - A_{\\mathrm{swap}}$: separate")
    sub(ax, 8.1, 0.95, 3.2, 2.0, "Paired-image tests",
        "MMVP (150 pairs)\nNaturalBench (298 samples)\nsame answer to both images\n+ answer distribution\n(\"Yes\" rate, option A rate)")

    # 5. forecasting (proposed)
    box(ax, 12.2, 2.3, 2.45, 4.2, PROP)
    title(ax, 12.35, 6.38, "5. Forecaster\n(main goal)", color=MUTED, size=12)
    sub(ax, 12.32, 4.2, 2.2, 1.5, "Inputs",
        "static dataset features (3)\n+ early-checkpoint\nsignals (4)\nmodel: e.g. XGBoost / MLP", SUB_PROP)
    sub(ax, 12.32, 2.45, 2.2, 1.55, "Predicts",
        "final change in image\nbenefit on G, vs step 0:\n$\\Delta\\hat{G} = \\hat{G}_T - G_0$\n"
        "($G = A_{\\mathrm{real}} - A_{\\mathrm{blank}}$)", SUB_PROP)
    arrow(ax, (13.42, 4.2), (13.42, 4.0), dashed=True)

    # 6. outputs (proposed)
    box(ax, 14.9, 2.3, 1.95, 4.2, PROP)
    title(ax, 15.05, 6.38, "6. Outputs", color=MUTED, size=12)
    sub(ax, 15.01, 4.95, 1.73, 0.95, "Regression risk", "high / medium / low", SUB_PROP)
    sub(ax, 15.01, 3.75, 1.73, 1.05, "Predicted\nchange in G", "\n$\\Delta\\hat{G}$ (image benefit)", SUB_PROP)
    sub(ax, 15.01, 2.42, 1.73, 1.2, "Guidance", "LR, LoRA rank,\ndata mix\n(future extension)", SUB_PROP)

    # pilot status note
    box(ax, 12.2, 6.62, 4.65, 1.68, dict(fc="#fff8e6", ec="#c98500", ls="-"), lw=1.4)
    ax.text(12.35, 8.17, "Pilot: is the forecast target measurable?", fontsize=10.5, fontweight="bold", va="top", color=INK)
    ax.text(12.35, 7.88, "Yes, at ~2-pt resolution (90% vs 10% text mix, 3 seeds):\n"
            "• InternVL3: overall accuracy A flat; on the image-\n"
            "   necessary G set: accuracy −2.6, image benefit −3.7\n"
            "   (real−blank) and −4.2 (real−swap) pts\n"
            "• Qwen2-VL: A −2.6; G real−blank image benefit −2.5 pts\n"
            "• SmolVLM: paired-image tests track its answer bias",
            fontsize=8.4, va="top", color=INK, linespacing=1.22)

    # arrows
    arrow(ax, (3.8, 7.1), (4.5, 7.1))
    arrow(ax, (2.0, 5.0), (2.0, 4.4))
    arrow(ax, (8.0, 6.0), (8.0, 5.4))
    arrow(ax, (11.5, 3.9), (12.2, 3.9), dashed=True)
    ax.text(11.85, 4.0, "early\nsignals", fontsize=8.3, ha="center", va="bottom", color=MUTED, linespacing=1.1)
    arrow(ax, (14.65, 4.6), (14.9, 4.6), dashed=True)
    # dataset features -> forecaster (runs under box 4)
    ax.plot([3.8, 3.95, 3.95, 11.85, 11.85], [1.5, 1.5, 0.45, 0.45, 2.6], color="#8a8a8a", ls=(0, (5, 3)), lw=1.6)
    arrow(ax, (11.85, 2.55), (12.2, 2.75), dashed=True)
    ax.text(10.4, 0.45, " static dataset features ", fontsize=8.6, ha="center", va="center", color=MUTED,
            bbox=dict(fc="white", ec="none", pad=1))
    # feedback loop (proposed)
    ax.plot([15.85, 15.85, 2.0], [2.3, 0.2, 0.2], color="#8a8a8a", ls=(0, (5, 3)), lw=1.6)
    arrow(ax, (2.0, 0.2), (2.0, 0.75), dashed=True)
    ax.text(8.9, 0.2, "  Early stopping / training adaptation (proposed)  ", fontsize=10, ha="center", va="center",
            color=MUTED, bbox=dict(fc="white", ec="#8a8a8a", ls=(0, (5, 3)), boxstyle="round,pad=0.3"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
