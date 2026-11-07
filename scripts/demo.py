"""Review demo: three short parts that show the pilot's real pipeline and saved outputs (no training, no forecasting).
  --part eval     one MMStar question scored live with the real, a blank and a swapped image (GPU, ~1-2 min),
                  then the saved Tier 1 result (Qwen2-VL real vs blank on 500 items, McNemar)
  --part data     training pool, loss-gap labels, the two mixtures, one example of each label, leakage check (CPU)
  --part results  LoRA config, training loss, 90%-vs-10% contrasts for all three models, figure paths (CPU)
Add --cached to --part eval to print the saved scores for the same item instead of loading the model.
Images are written to results/demo/. Usage: python -m scripts.demo --part eval [--model smolvlm|internvl|qwen2vl]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import blank_like, load_mmstar_by_ids
from src.eval import swap_map
from src.overlap import assert_no_overlap

OUT = "results/demo"
E = "results/tier3/eval_v2"
PREFIX = {"qwen2vl": "", "smolvlm": "smolvlm_", "internvl": "internvl_"}
ADIR = {"qwen2vl": "analysis_v2", "smolvlm": "analysis_v2_smolvlm", "internvl": "analysis_v2_internvl"}


def head(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def saved_base(model):
    """Base-model (step 0) per-item results on the A set for real / blank / swap, from the eval v2 cache."""
    f = f"{E}/{PREFIX[model]}matched_text10_step0000_A_{{}}.csv"
    return {c: pd.read_csv(f.format(c)).set_index("id") for c in ("real", "blank", "swap")}


def part_eval(cfg, model, cached):
    base = saved_base(model)
    ok = base["real"].correct.eq(1) & base["blank"].correct.eq(0) & base["swap"].correct.eq(0)
    # demo item: right only with its own image, a perception question, distinct options, most confident with the real image
    r = base["real"]
    p_right = np.exp(r[["lp_A", "lp_B", "lp_C", "lp_D"]].to_numpy(float))
    p_right = pd.Series(p_right[np.arange(len(r)), r.answer.to_numpy(int)] / p_right.sum(1), index=r.index)
    ok &= r.category.str.contains("perception")
    item_id = p_right[ok].sort_values(ascending=False).index[0]
    sets = json.load(open("results/tier3/eval_sets_v2/eval_sets.json"))
    items = load_mmstar_by_ids(cfg["tier1"], sets["A"])
    it = next(x for x in items if x["id"] == item_id)
    imgs = {"real": it["image"], "blank": blank_like(it["image"], cfg["tier1"]["blank_gray"]),
            "swap": swap_map(items, seed=0)[item_id]}  # same swapped image as in the saved evaluation
    os.makedirs(OUT, exist_ok=True)
    for c, im in imgs.items():
        im.save(f"{OUT}/eval_{c}.png")
    head(f"PART 1: does the answer depend on the image?  model = {model}, item = {item_id}")
    print("Q:", it["question"])
    for L, o in zip("ABCD", it["options"]):
        print(f"   {L}. {o}")
    print(f"correct answer: {'ABCD'[it['answer']]}    images saved to {OUT}/eval_{{real,blank,swap}}.png")
    if cached:
        lp = {c: base[c].loc[item_id, ["lp_A", "lp_B", "lp_C", "lp_D"]].to_numpy(float) for c in imgs}
        print("(saved scores from the evaluation run)")
    else:
        from src.vlm import make_model, select_model
        select_model(cfg, model)
        vlm = make_model(cfg["model"]).load()
        lp = {c: np.asarray(vlm.score_options(im, it["question"], it["options"])) for c, im in imgs.items()}
    print(f"\n{'image':8s}" + "".join(f"  P({L})" for L in "ABCD") + "   answer")
    for c, v in lp.items():
        p = np.exp(v - np.logaddexp.reduce(v))
        print(f"{c:8s}" + "".join(f"  {x:4.0%}" for x in p) + f"   {'ABCD'[int(np.argmax(v))]}"
              + ("  (right)" if int(np.argmax(v)) == it["answer"] else "  (wrong)"))

    head("Saved Tier 1 result: Qwen2-VL on 500 MMStar items")
    t1 = json.load(open("results/tier1/full/analysis.json"))
    print(f"real image   {t1['real_acc']:.1%}  (95% CI {t1['real_wilson95'][0]:.1%}-{t1['real_wilson95'][1]:.1%})")
    print(f"blank image  {t1['blank_acc']:.1%}  (95% CI {t1['blank_wilson95'][0]:.1%}-{t1['blank_wilson95'][1]:.1%})"
          "   chance = 25%")
    t = t1["table_2x2"]
    print(f"McNemar: {t['real_right_blind_wrong']} items right->wrong without the image, "
          f"{t['real_wrong_blind_right']} wrong->right, exact p = {t1['mcnemar_exact_p']:.1e}")
    head("Base models, 500-item A set (saved): accuracy with real / blank / swapped image")
    for m in PREFIX:
        b = saved_base(m)
        print(f"{m:9s}" + "".join(f"  {c} {b[c].correct.mean():5.1%}" for c in b))


def part_data(cfg):
    T2 = "results/tier2/full"
    pool = pd.read_csv(f"{T2}/pool_labeled_matched.csv")
    filt = json.load(open(f"{T2}/pool_filter.json"))
    head("PART 2a: training pool (The Cauldron, one question per image)")
    print(pool.groupby("subset").size().rename("items").to_string(), f"\ntotal {len(pool)}")
    print(f"streamed {filt['n_streamed']}, dropped {filt['n_in_pool_duplicate_images']} duplicate images and "
          f"{filt['n_dropped_overlap_with_eval']} items overlapping MMStar -> {filt['n_final']} items")
    head("PART 2b: label = loss gap = answer NLL with blank image - NLL with real image")
    print("small gap: the answer is guessable without the image -> text-answerable")
    print("large gap: the image is needed                          -> image-necessary")
    print("thresholds are the 40th / 60th percentile within each subset:")
    th = json.load(open(f"{T2}/summary_matched.json"))["within_subset_thresholds"]
    print(pd.DataFrame(th).T.round(3).to_string())
    print("\nlabel counts:\n" + pool.groupby(["subset", "label"]).size().unstack(fill_value=0).to_string())
    os.makedirs(OUT, exist_ok=True)
    for lab in ("text_answerable", "image_necessary"):
        d = pool[(pool.subset == "aokvqa") & (pool.label == lab)]
        d = d[d.answer.astype(str).str.len().lt(25) & ~d.answer.astype(str).str.contains("Answer|\n")]  # skip rationale-style answers
        x = d.loc[(d.gap - d.gap.median()).abs().idxmin()]  # a typical item, not an extreme one
        Image.open(x.image_path).convert("RGB").save(f"{OUT}/example_{lab}.png")
        print(f"\nexample {lab} (gap {x.gap:+.2f}), image -> {OUT}/example_{lab}.png")
        print("  Q:", x.user.strip().splitlines()[0][:150], "\n  A:", str(x.answer)[:80])
    head("PART 2c: the two training mixtures (same sources, 250 items per subset)")
    for m in ("matched_text10", "matched_text90"):
        d = pd.read_csv(f"{T2}/mixture_{m}.csv")
        tab = d.groupby(["subset", "label"]).size().unstack(fill_value=0)
        print(f"\n{m}: {len(d)} items, {d.label.eq('text_answerable').mean():.0%} text-answerable, "
              f"mean loss gap {d.gap.mean():.2f}\n{tab.to_string()}")
    head("PART 2d: leakage guard (hard rule: any overlap with evaluation data raises an error)")
    fp = json.load(open("results/tier2/mmstar_fingerprints.json"))
    for m in ("matched_text10", "matched_text90"):
        d = pd.read_csv(f"{T2}/mixture_{m}.csv")
        assert_no_overlap(d.id.tolist(), d.dhash.map(lambda h: int(h, 16)).tolist(), d.qnorm.tolist(), fp,
                          cfg["tier2"]["overlap_hamming"], m)
    ex = json.load(open("results/tier3/paired_bench/excluded_overlap.json"))
    print(f"MMVP / NaturalBench: checked against the training pool in run_paired_bench, "
          f"{len({e['pair'] for e in ex})} NaturalBench samples excluded (results/tier3/paired_bench/excluded_overlap.json)")
    print("(tests/test_overlap.py plants a duplicate and checks that the guard raises)")


def part_results(cfg):
    head("PART 3a: LoRA training setup (configs/config.yaml)")
    t3 = cfg["tier3"]
    print(f"LoRA r={t3['lora_r']}, alpha={t3['lora_alpha']}, dropout={t3['lora_dropout']}, lr={t3['lr']}, "
          f"batch {t3['batch_size']}, language-model layers only")
    print("Qwen2-VL-2B 4-bit NF4 (120 steps) | SmolVLM-2.2B bf16 (125) | InternVL3-2B bf16 (125); "
          "2 mixtures x 3 seeds each = 18 runs")
    head("PART 3b: training loss (mean of first / last 10 steps, seed 0)")
    for m in PREFIX:
        for mix in ("matched_text10", "matched_text90"):
            tl = pd.read_csv(f"results/tier3/{PREFIX[m]}{mix}/train_log.csv")
            print(f"{m:9s} {mix:15s} loss {tl.loss.head(10).mean():.2f} -> {tl.loss.tail(10).mean():.2f}   "
                  f"({len(tl)} steps)")
    head("PART 3c: 90% minus 10% text-answerable run at the final step (pts; 95% bootstrap CI over items)")
    names = {("A", "real"): "A accuracy (overall)", ("G", "real"): "G accuracy (image-necessary)",
             ("G", "ben_blank"): "G image benefit, real-blank", ("G", "ben_swap"): "G image benefit, real-swap"}
    for m in PREFIX:
        c = pd.read_csv(f"results/tier3/{ADIR[m]}/change_vs_step0.csv")
        c = c[c.comparison == "contrast_90_minus_10"]
        print(f"\n{m}")
        for (s, k), lab in names.items():
            r = c[(c.set == s) & (c.metric == k)]
            if len(r):
                r = r.iloc[0]
                print(f"  {lab:30s} {r.change_pts:+5.1f}  [{r.ci_lo:+5.1f}, {r.ci_hi:+5.1f}]"
                      f"{'  *' if r.ci_excludes_0 else '   '}  seeds {r.per_seed_pts}")
    print("\n* = CI excludes 0. Feasibility sketch, not evidence (3 seeds, ~20 contrasts per model, no correction).")
    print("Figures: results/figures/fig1_blindness.png, fig2_training_curves.png, fig3_contrast_forest.png, "
          "fig4_A_split.png, architecture.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["eval", "data", "results"])
    ap.add_argument("--model", default="smolvlm", choices=list(PREFIX))
    ap.add_argument("--cached", action="store_true", help="--part eval: use saved scores, do not load the model")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    pd.set_option("display.width", 160)
    {"eval": lambda: part_eval(cfg, a.model, a.cached), "data": lambda: part_data(cfg),
     "results": lambda: part_results(cfg)}[a.part]()


if __name__ == "__main__":
    main()
