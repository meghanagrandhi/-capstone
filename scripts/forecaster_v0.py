"""Forecaster v0 (first sketch, CPU only): predict a run's final change in G image benefit (dG_final, pts) from signals
available EARLY in training, and compare with the simple baselines (scripts.baselines). NOT the final forecaster.

Features at a cutoff c% of training (c = 5, 10, 20, 40), all as change from step 0, from results/tier3/<run>/probe_log.csv
(100-item in-training probe, 50 image-necessary + 50 text-answerable items):
  d_gap_img   : change in (blank-image NLL - real-image NLL) on image-necessary probe items (how much the image helps)
  d_gap_text  : same on text-answerable probe items
  d_attn_img  : change in attention mass on image tokens, image-necessary items
  d_attn_text : same, text-answerable items
  plus optionally text_share (static, known before training) and mean training loss up to c%.
Model: ridge regression on standardized features, ridge strength picked by inner leave-one-out on the training fold only
(sklearn RidgeCV), so the held-out run never influences the fit. 36 runs is too few for LightGBM (TuneAhead) - later.
Splits and scores as in scripts.baselines (leave-one-run-out, leave-one-model-out; MAE with bootstrap CI, Spearman,
text90-vs-text10 pair accuracy). Feasibility sketch, not evidence.
Usage: python -m scripts.forecaster_v0   (run scripts.baselines first)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.baselines import score
from src.ioutil import safe_to_csv

OUT = "results/tier3/forecaster_v0"
CUTS = (5, 10, 20, 40)
PROBE = {"d_gap_img": "probe_gap_image_necessary", "d_gap_text": "probe_gap_text_answerable",
         "d_attn_img": "probe_attn_img_mass_image_necessary", "d_attn_text": "probe_attn_img_mass_text_answerable"}
SETS = {"probe": list(PROBE), "probe+static": list(PROBE) + ["text_share"],
        "probe+static+loss": list(PROBE) + ["text_share", "loss_mean"]}


def features(t, cut):
    rows = []
    for _, r in t.iterrows():
        p = pd.read_csv(f"results/tier3/{r.run}/probe_log.csv").set_index("step")
        st = [s for s in p.index if round(100 * s / r.n_steps) == cut]
        assert st, (r.run, cut)
        f = {k: float(p.loc[st[0], c] - p.loc[0, c]) for k, c in PROBE.items()}
        tl = pd.read_csv(f"results/tier3/{r.run}/train_log.csv")
        f["loss_mean"] = float(tl[tl.step <= st[0]].loss.mean())
        rows.append(f)
    return pd.concat([t[["run", "model", "mix_type", "text_share", "seed", "dG_final"]], pd.DataFrame(rows)], axis=1)


def predict(d, cols, split):
    y = d.dG_final.to_numpy()
    X = d[cols].to_numpy(float)
    pred = np.full(len(d), np.nan)
    for i in range(len(d)):
        tr = (d.index != i) if split == "loo" else (d.model != d.model[i]).to_numpy()
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 21)))
        m.fit(X[tr], y[tr])
        pred[i] = m.predict(X[[i]])[0]
    return pred


def main():
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv("results/tier3/baselines/run_table.csv")
    rng = np.random.RandomState(0)
    res, coefs = [], []
    for cut in CUTS:
        d = features(t, cut)
        safe_to_csv(d, os.path.join(OUT, f"features_{cut}pct.csv"))
        for name, cols in SETS.items():
            for split in ("loo", "leave_one_model_out"):
                res.append(dict(cutoff_pct=cut, features=name, split=split, **score(d, predict(d, cols, split), rng)))
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 21))).fit(d[cols], d.dG_final)
            coefs.append(dict(cutoff_pct=cut, features=name, alpha=float(m[-1].alpha_),
                              **{c: float(w) for c, w in zip(cols, m[-1].coef_)}))
    res, coefs = pd.DataFrame(res), pd.DataFrame(coefs)
    safe_to_csv(res, os.path.join(OUT, "forecaster_v0_scores.csv"))
    safe_to_csv(coefs, os.path.join(OUT, "coefficients_full_fit.csv"))
    base = pd.read_csv("results/tier3/baselines/baseline_scores.csv")
    base = base[base.subset == "all_runs"]
    pd.set_option("display.width", 220)
    print("== baselines (36 runs) ==")
    print(base[["split", "baseline", "MAE", "MAE_lo", "MAE_hi", "spearman", "pair_acc"]].round(2).to_string(index=False))
    print("== forecaster v0 (36 runs) ==")
    print(res[["cutoff_pct", "features", "split", "MAE", "MAE_lo", "MAE_hi", "spearman", "pair_acc"]].round(2)
          .to_string(index=False))
    print("== standardized coefficients (fit on all 36 runs; pts per 1 SD) ==")
    print(coefs.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
