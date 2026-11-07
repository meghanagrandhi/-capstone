"""Stats for the paired-image benchmarks (MMVP, NaturalBench), both models, CPU only.
Unit = one question asked about two images (a "pair"). Per pair: both_right (pair accuracy) and same_answer
(model gives the same answer for both images = image ignored).
Reports, per model and benchmark, mean over 3 seeds of: change from base (step 0) for each mixture, and the contrast
text90 - text10 at the final step, with 95% bootstrap CIs over pairs and per-seed signs.
Usage: python -m scripts.analyze_paired_bench [--dir results/tier3/paired_bench_balanced --mix balanced]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ioutil import safe_to_csv

D = "results/tier3/paired_bench"
B = 2000


def qkey(i):
    """question-level pair key from item id: mmvp-<n> -> pair (n-1)//2; nb-<s>-q<q>-i<i> -> nb-<s>-q<q>."""
    if i.startswith("mmvp-"):
        return f"mmvp-p{(int(i[5:]) - 1) // 2}"
    return i.rsplit("-i", 1)[0]


def per_pair(f):
    d = pd.read_csv(f, usecols=["id", "category", "pred", "correct"])
    d["bench"] = np.where(d.category == "mmvp", "mmvp", "naturalbench")
    d["qkey"] = d.id.map(qkey)
    g = d.groupby(["bench", "qkey"])
    return pd.DataFrame({"both_right": g.correct.min(), "same_answer": (g.pred.nunique() == 1).astype(int)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D)
    ap.add_argument("--mix", default="matched", choices=["matched", "balanced"])
    a = ap.parse_args()
    t10, t90 = f"{a.mix}_text10", f"{a.mix}_text90"
    runs = {}
    for f in glob.glob(os.path.join(a.dir, "*_step*.csv")):
        m = re.match(r"(?:(smolvlm|internvl)_)?((?:matched|balanced)_text\d+)(?:_s(\d+))?_step(\d+)\.csv", os.path.basename(f))
        if m:
            runs[(m.group(1) or "qwen2vl", m.group(2), int(m.group(3) or 0), int(m.group(4)))] = per_pair(f)
    rng = np.random.RandomState(0)
    rows = []
    for model in ("qwen2vl", "smolvlm", "internvl"):
        keys = [k for k in runs if k[0] == model]
        if not keys:
            continue
        base = runs[[k for k in keys if k[3] == 0][0]]
        final = {(mix, s): runs[k] for k in keys if k[3] > 0 for mix, s in [(k[1], k[2])]}
        seeds = sorted({s for _, s in final})
        for bench in ("mmvp", "naturalbench"):
            b = base.loc[bench]
            for metric in ("both_right", "same_answer"):
                x = {mix: np.stack([final[(mix, s)].loc[bench][metric].reindex(b.index).to_numpy() for s in seeds])
                     for mix in (t10, t90)}
                comps = {"text10 vs base": x[t10] - b[metric].to_numpy(),
                         "text90 vs base": x[t90] - b[metric].to_numpy(),
                         "text90 - text10": x[t90] - x[t10]}
                for name, v in comps.items():
                    n = v.shape[1]
                    bs = v.mean(0)[rng.randint(0, n, size=(B, n))].mean(1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    rows.append(dict(model=model, bench=bench, metric=metric, comparison=name, n_pairs=n,
                                     base=100 * b[metric].mean(), change_pts=100 * v.mean(), ci_lo=100 * lo,
                                     ci_hi=100 * hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                                     per_seed_pts=" ".join(f"{100 * s:+.1f}" for s in v.mean(1))))
    out = pd.DataFrame(rows)
    safe_to_csv(out, os.path.join(a.dir, "paired_bench_stats.csv"))
    pd.set_option("display.width", 220)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
