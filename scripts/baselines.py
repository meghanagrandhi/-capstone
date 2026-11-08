"""Run table + simple baselines for predicting a run's final drop in G image benefit (CPU only). NOT a forecaster.

Run table (results/tier3/baselines/run_table.csv): one row per training run (matched + balanced, 3 models x 2 mixes x 3 seeds),
  G image benefit (G real - G blank accuracy, pts) and A accuracy at every evaluated checkpoint (% of training),
  early training-loss features (first 40% of steps), and the target:
  dG_final = G benefit at the last step - G benefit at step 0 (pts; negative = the run lost image benefit).

Baselines (each predicts dG_final):
  zero      : predicts 0 ("nothing changes")
  static    : mean target of the other runs with the same model, mix type and text share (only what is known before training)
  early_G40 : linear fit of dG_final on the G-benefit change at 40% of training (every run has a 40% checkpoint)
  loss      : linear fit on early loss features (mean loss, image-necessary minus text-answerable loss, first 40% of steps)
  early_G5/10/20: same as early_G40 at 5/10/20%; only the 12 balanced Qwen/InternVL runs have these checkpoints
Two splits: leave-one-run-out (the run's own config was seen, via other seeds) and leave-one-model-out (predict a model
never seen in fitting; static then uses the other models' mean for that mix).
Scores: MAE (pts, bootstrap 95% CI over runs), Spearman rho (pred vs actual), pair accuracy = share of (model, mix type, seed)
pairs where the prediction gets right whether text90 loses more than text10 (ties count 0.5). Small n: feasibility sketch.
Usage: python -m scripts.baselines
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ioutil import safe_to_csv

OUT = "results/tier3/baselines"
B = 2000
PCTS = (0, 5, 10, 20, 40, 100)


def run_name(model, mixture, seed):
    pfx = "" if model == "qwen2vl" else f"{model}_"
    return f"{pfx}{mixture}" + (f"_s{seed}" if seed else "")


def build_table():
    ck = pd.concat([pd.read_csv("results/tier3/eval_v2/checkpoint_eval_v2.csv"),
                    pd.read_csv("results/tier3/eval_v2_balanced/checkpoint_eval_v2.csv")], ignore_index=True)
    ck = ck[ck.set.isin(["A", "G"])]
    p = ck.pivot_table(index=["run", "model", "mixture", "seed", "step"], columns=["set", "cond"], values="acc") * 100
    p = pd.DataFrame({"G_ben": p[("G", "real")] - p[("G", "blank")], "A": p[("A", "real")]}).reset_index()
    rows = []
    for (run, model, mixture, seed), g in p.groupby(["run", "model", "mixture", "seed"]):
        meta = json.load(open(f"results/tier3/{run}/run_meta.json"))
        n = int(meta["steps"])
        r = dict(run=run, model=model, mix_type=mixture.split("_")[0], text_share=int(mixture.split("text")[1]),
                 seed=int(seed), n_steps=n)
        g = g.set_index("step")
        for pct in PCTS:
            st = [s for s in g.index if round(100 * s / n) == pct]
            r[f"G_ben_{pct}"] = float(g.loc[st[0], "G_ben"]) if st else np.nan
            r[f"A_{pct}"] = float(g.loc[st[0], "A"]) if st else np.nan
        tl = pd.read_csv(f"results/tier3/{run}/train_log.csv")
        e = tl[tl.step <= 0.4 * n]
        r["loss_early_mean"] = float(e.loss.mean())
        r["loss_early_gap_img_minus_text"] = float(e.loss_image_necessary.mean() - e.loss_text_answerable.mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    for pct in PCTS[1:]:
        t[f"dG_{pct}"] = t[f"G_ben_{pct}"] - t["G_ben_0"]
    t["dA_100"] = t["A_100"] - t["A_0"]
    t["dG_final"] = t["dG_100"]
    return t.sort_values(["model", "mix_type", "seed", "text_share"]).reset_index(drop=True)


def linfit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), (Xte - mu) / sd] @ w


def predict(t, name, split):
    y = t.dG_final.to_numpy()
    pred = np.full(len(t), np.nan)
    feats = {"early_G40": ["dG_40"], "early_G20": ["dG_20"], "early_G10": ["dG_10"], "early_G5": ["dG_5"],
             "loss": ["loss_early_mean", "loss_early_gap_img_minus_text"]}
    for i in range(len(t)):
        tr = (t.index != i) if split == "loo" else (t.model != t.model[i]).to_numpy()
        if name == "zero":
            pred[i] = 0.0
        elif name == "static":
            same = tr & (t.mix_type == t.mix_type[i]).to_numpy() & (t.text_share == t.text_share[i]).to_numpy()
            if split == "loo":
                same &= (t.model == t.model[i]).to_numpy()
            pred[i] = y[same].mean()
        else:
            f = feats[name]
            ok = tr & t[f].notna().all(1).to_numpy()
            if t.loc[i, f].isna().any() or ok.sum() < 4:
                continue
            pred[i] = linfit_predict(t.loc[ok, f].to_numpy(), y[ok], t.loc[[i], f].to_numpy())[0]
    return pred


def score(t, pred, rng):
    ok = ~np.isnan(pred)
    y, p = t.dG_final.to_numpy()[ok], pred[ok]
    err = np.abs(y - p)
    bs = err[rng.randint(0, len(err), size=(B, len(err)))].mean(1)
    rho = spearmanr(p, y).statistic if np.std(p) > 0 else np.nan
    tt = t[ok].assign(pred=p)
    hits = []
    for _, g in tt.groupby(["model", "mix_type", "seed"]):
        if len(g) != 2:
            continue
        a, b = g[g.text_share == 10].iloc[0], g[g.text_share == 90].iloc[0]
        dp, dy = b.pred - a.pred, b.dG_final - a.dG_final
        hits.append(0.5 if dp == 0 or dy == 0 else float(np.sign(dp) == np.sign(dy)))
    return dict(n_runs=int(ok.sum()), MAE=float(err.mean()), MAE_lo=float(np.percentile(bs, 2.5)),
                MAE_hi=float(np.percentile(bs, 97.5)), spearman=float(rho) if rho == rho else np.nan,
                n_pairs=len(hits), pair_acc=float(np.mean(hits)) if hits else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    t = build_table()
    safe_to_csv(t, os.path.join(OUT, "run_table.csv"))
    rng = np.random.RandomState(0)
    res = []
    early = t[t.dG_5.notna()].reset_index(drop=True)  # the 12 runs with 5/10/20% checkpoints
    for subset, d, names in (("all_runs", t, ["zero", "static", "loss", "early_G40"]),
                             ("early_ckpt_runs", early, ["zero", "static", "loss", "early_G5", "early_G10", "early_G20",
                                                          "early_G40"])):
        for split in ("loo", "leave_one_model_out"):
            for nm in names:
                res.append(dict(subset=subset, split=split, baseline=nm, **score(d, predict(d, nm, split), rng)))
    res = pd.DataFrame(res)
    safe_to_csv(res, os.path.join(OUT, "baseline_scores.csv"))
    pd.set_option("display.width", 220)
    show = ["run", "dG_5", "dG_10", "dG_20", "dG_40", "dG_final", "dA_100", "loss_early_mean"]
    print(t[show].round(1).to_string(index=False))
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
