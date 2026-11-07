"""Tier 3 v2 analysis across 3 seeds (CPU only, reads results/tier3/eval_v2).
For every (mixture, step) and set (A, G):
  acc_real, image benefit vs blank (real - blank) and vs swap (real - swap), averaged over seeds
Change from step 0 -> final step, per mixture and the contrast text90 - text10:
  bootstrap 95% CI over items (same resampled items for every seed/condition), sign consistency over seeds,
  per-seed exact McNemar on real-image correctness (final vs step 0).
Extra diagnostics: prior agreement (real-image prediction == base model's blank prediction), letter distribution,
A split by Tier 1 blind k (k<=1 vs k>=4), per-category change.
NOTE: G was selected for low blind accuracy on orders 0-5, and eval uses order 0, so G_blank at step 0 is biased low
(3.8% vs 7.3% on fresh orders). Any model change regresses it upward -> real-blank on G overstates the drop.
real-swap and the text90-vs-text10 contrast are not affected by this selection.
Usage: python -m scripts.analyze_tier3_v2 [--model smolvlm|internvl]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ioutil import safe_to_csv

D = "results/tier3/eval_v2"
OUT = "results/tier3/analysis_v2"
B = 2000
MIX = ("matched_text10", "matched_text90")


def load(model):
    rows = []
    for f in glob.glob(os.path.join(D, "*_step*_*_*.csv")):
        m = re.match(r"(?:(smolvlm|internvl)_)?(matched_text\d+)(?:_s(\d+))?_step(\d+)_([AG])_(real|swap|blank)\.csv",
                     os.path.basename(f))
        if not m or (m.group(1) or "qwen2vl") != model:
            continue
        d = pd.read_csv(f, usecols=["id", "category", "pred", "answer", "correct", "pred_letter_shown"])
        d["mixture"], d["seed"], d["step"], d["set"], d["cond"] = \
            m.group(2), int(m.group(3) or 0), int(m.group(4)), m.group(5), m.group(6)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def cube(df, mix, step, s, cond, ids):
    """seeds x items matrix of correctness, items in `ids` order."""
    x = df[(df.mixture == mix) & (df.step == step) & (df.set == s) & (df.cond == cond)]
    return x.pivot(index="seed", columns="id", values="correct")[ids].sort_index().to_numpy(float)


def metrics(df, mix, step, s, ids):
    r = cube(df, mix, step, s, "real", ids)
    return {"real": r, "ben_blank": r - cube(df, mix, step, s, "blank", ids), "ben_swap": r - cube(df, mix, step, s, "swap", ids)}


def boot_ci(x, rng):
    """x: seeds x items per-item values; CI of the mean over seeds and items, resampling items."""
    n = x.shape[1]
    idx = rng.randint(0, n, size=(B, n))
    bs = x.mean(0)[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mcnemar(a, b):
    """exact McNemar: a, b = 0/1 vectors (before, after)."""
    lose, gain = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(gain, gain + lose, 0.5).pvalue if gain + lose else 1.0
    return gain, lose, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2vl", choices=["qwen2vl", "smolvlm", "internvl"])
    model = ap.parse_args().model
    out = OUT if model == "qwen2vl" else f"{OUT}_{model}"
    os.makedirs(out, exist_ok=True)
    df = load(model)
    rng = np.random.RandomState(0)
    steps = sorted(df.step.unique())
    s0, sF = steps[0], steps[-1]
    ids = {s: sorted(df[df.set == s].id.unique()) for s in ("A", "G")}
    seeds = sorted(df.seed.unique())
    print(f"seeds={seeds} steps={steps} A={len(ids['A'])} G={len(ids['G'])}")

    # 1) curves: seed-averaged metrics per checkpoint
    curve = []
    for mix in MIX:
        for st in steps:
            for s in ("A", "G"):
                m = metrics(df, mix, st, s, ids[s])
                curve.append(dict(mixture=mix, step=st, set=s, **{k: float(v.mean()) for k, v in m.items()},
                                  **{f"{k}_seed_sd": float(v.mean(1).std(ddof=1)) for k, v in m.items()}))
    curve = pd.DataFrame(curve)
    safe_to_csv(curve, os.path.join(out, "curves.csv"))

    # 2) change from step 0 to final, per mixture, plus contrast text90 - text10
    ch = []
    for s in ("A", "G"):
        base = metrics(df, MIX[0], s0, s, ids[s])  # step 0 is the same base model for both mixtures
        delta = {mix: {k: v - base[k] for k, v in metrics(df, mix, sF, s, ids[s]).items()} for mix in MIX}
        for k in ("real", "ben_blank", "ben_swap"):
            for mix in MIX + ("contrast_90_minus_10",):
                x = delta[MIX[1]][k] - delta[MIX[0]][k] if mix.startswith("contrast") else delta[mix][k]
                mean, lo, hi = boot_ci(x, rng)
                per_seed = x.mean(1)
                ch.append(dict(set=s, metric=k, comparison=mix, change_pts=100 * mean, ci_lo=100 * lo, ci_hi=100 * hi,
                               ci_excludes_0=bool(lo > 0 or hi < 0),
                               seeds_negative=int((per_seed < 0).sum()), seeds_positive=int((per_seed > 0).sum()),
                               per_seed_pts=" ".join(f"{100 * v:+.1f}" for v in per_seed)))
    ch = pd.DataFrame(ch)
    safe_to_csv(ch, os.path.join(out, "change_vs_step0.csv"))

    # 3) per-seed McNemar on real-image correctness: final vs step 0, and text90 vs text10 at final step
    mc = []
    for s in ("A", "G"):
        b = cube(df, MIX[0], s0, s, "real", ids[s])[0]
        f = {mix: cube(df, mix, sF, s, "real", ids[s]) for mix in MIX}
        for i, sd in enumerate(seeds):
            for mix in MIX:
                g, l, p = mcnemar(b, f[mix][i])
                mc.append(dict(set=s, seed=sd, test=f"{mix} final vs step0", gained=g, lost=l, p=p))
            g, l, p = mcnemar(f[MIX[0]][i], f[MIX[1]][i])
            mc.append(dict(set=s, seed=sd, test="text90 vs text10 at final", gained=g, lost=l, p=p))
    mc = pd.DataFrame(mc)
    safe_to_csv(mc, os.path.join(out, "mcnemar.csv"))

    # 4) diagnostics: prior agreement, letter distribution, per-category, A split by Tier 1 blind k
    diag = {}
    prior = {s: df[(df.mixture == MIX[0]) & (df.step == s0) & (df.set == s) & (df.cond == "blank") & (df.seed == 0)]
             .set_index("id").pred for s in ("A", "G")}
    pa = []
    for mix in MIX:
        for st in (s0, sF):
            for s in ("A", "G"):
                x = df[(df.mixture == mix) & (df.step == st) & (df.set == s) & (df.cond == "real")]
                agree = (x.pred.to_numpy() == prior[s].reindex(x.id).to_numpy()).mean()
                letters = x.pred_letter_shown.value_counts(normalize=True).reindex(list("ABCD")).fillna(0)
                pa.append(dict(mixture=mix, step=st, set=s, prior_agreement=float(agree),
                               **{f"pct_{L}": float(v) for L, v in letters.items()}))
    diag["prior_agreement_and_letters"] = pd.DataFrame(pa)

    t1 = pd.read_csv("results/tier1/full/per_item.csv")
    kb = t1[t1.condition == "blank"].groupby("id").correct.agg(["sum", "count"])
    kb = kb["sum"].where(kb["count"] == 6)
    split = []
    for mix in MIX:
        for st in (s0, sF):
            x = df[(df.mixture == mix) & (df.step == st) & (df.set == "A") & (df.cond == "real")].copy()
            x["k"] = kb.reindex(x.id).to_numpy()
            for name, sel in (("blind_k<=1 (needs image)", x.k <= 1), ("blind_k>=4 (text-guessable)", x.k >= 4)):
                split.append(dict(mixture=mix, step=st, group=name, n_items=int(sel.sum() / len(seeds)),
                                  acc_real=float(x[sel].correct.mean())))
    diag["A_split_by_blind_k"] = pd.DataFrame(split)

    cat = df[(df.cond == "real") & (df.step.isin([s0, sF]))].groupby(["set", "mixture", "step", "category"]).correct.mean()
    cat = cat.unstack("step")
    cat["change_pts"] = 100 * (cat[sF] - cat[s0])
    diag["per_category"] = cat.reset_index()

    for k, v in diag.items():
        safe_to_csv(v, os.path.join(out, f"{k}.csv"))

    pd.set_option("display.width", 200)
    print("\n== change step0 -> final (pts, mean over 3 seeds, 95% bootstrap CI over items) ==")
    print(ch.round(2).to_string(index=False))
    print("\n== McNemar (real correctness) ==")
    print(mc.round(4).to_string(index=False))
    for k, v in diag.items():
        print(f"\n== {k} ==")
        print(v.round(3).to_string(index=False))
    json.dump({"seeds": [int(x) for x in seeds], "steps": [int(x) for x in steps], "n_boot": B,
               "note": "G blank at step 0 biased low by selection (orders 0-5 used to pick G, eval uses order 0)."},
              open(os.path.join(out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
