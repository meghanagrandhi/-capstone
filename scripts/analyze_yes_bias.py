"""Answer-prior check for the paired benchmarks (CPU only). Explains SmolVLM's reversed paired-image result.
  1. yes/no balance of the training mixtures (only VSR has yes/no answers)
  2. per model and mixture (mean over seeds): % "Yes" on NaturalBench yes/no questions (truth: 50%)
  3. same-answer rate (answer identical for both images) split by NaturalBench question type
Writes results/tier3/paired_bench/yes_bias.csv.
Usage: python -m scripts.analyze_yes_bias [--dir results/tier3/paired_bench_balanced --mix balanced]
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ioutil import safe_to_csv

D = "results/tier3/paired_bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    ap.add_argument("--mix", default="matched", choices=["matched", "balanced"])
    a = ap.parse_args()
    for m in ("text10", "text90"):
        d = pd.read_csv(f"results/tier2/full/mixture_{a.mix}_{m}.csv")
        ans = d[d.subset == "vsr"].answer.astype(str).str.strip().str.lower()
        print(f"mixture {m}: VSR answers yes {100 * ans.str.startswith('yes').mean():.1f}%, "
              f"no {100 * ans.str.startswith('no').mean():.1f}% (n={len(ans)})")
    rows = []
    for f in glob.glob(os.path.join(a.dir, "*_step*.csv")):
        m = re.match(r"(?:(smolvlm|internvl)_)?((?:matched|balanced)_text\d+)(?:_s(\d+))?_step(\d+)\.csv", os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f)
        d = d[d.category.str.startswith("nb_")]
        d["q"] = d.id.str.rsplit("-i", n=1).str[0]
        same = d.groupby(["category", "q"]).pred.nunique().eq(1).groupby("category").mean()
        yn = d[d.category == "nb_yes_no"]  # options are always ["Yes", "No"], so pred 0 = "Yes"
        rows.append(dict(model=m.group(1) or "qwen2vl", mixture="base" if int(m.group(4)) == 0 else m.group(2),
                         seed=int(m.group(3) or 0), yes_pct=100 * (yn.pred == 0).mean(),
                         yes_truth_pct=100 * (yn.answer == 0).mean(),
                         same_answer_yes_no=100 * same["nb_yes_no"], same_answer_mc=100 * same["nb_multiple_choice"]))
    r = pd.DataFrame(rows).sort_values(["model", "mixture", "seed"])
    safe_to_csv(r, os.path.join(a.dir, "yes_bias.csv"))
    pd.set_option("display.width", 200)
    print(r.round(1).to_string(index=False))
    print(r.groupby(["model", "mixture"])[["yes_pct", "same_answer_yes_no", "same_answer_mc"]].mean().round(1))


if __name__ == "__main__":
    main()
