"""Post-hoc statistics on results/tier1/full/per_item.csv (no model needed).
Usage: python -m scripts.analyze_tier1
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "results/tier1/full"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    d = pd.read_csv(f"{D}/per_item.csv")
    n = d.id.nunique()
    out = {"n_items": n}

    # 1. Wilson CIs + McNemar (canonical order, seed 0)
    s0 = d[d.seed == 0]
    r = s0[s0.condition == "real"].set_index("id").correct
    b = s0[s0.condition == "blank"].set_index("id").correct.reindex(r.index)
    for name, x in (("real", r), ("blank", b)):
        lo, hi = wilson(int(x.sum()), n)
        out[f"{name}_acc"] = float(x.mean())
        out[f"{name}_wilson95"] = [float(lo), float(hi)]
    t = pd.crosstab(r.rename("real"), b.rename("blank")).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    n_rb, n_rB = int(t.loc[1, 1]), int(t.loc[1, 0])  # real right: blind right / blind wrong
    n_Rb, n_RB = int(t.loc[0, 1]), int(t.loc[0, 0])  # real wrong: blind right / blind wrong
    p_mc = float(stats.binomtest(n_rB, n_rB + n_Rb, 0.5).pvalue)  # exact McNemar = two-sided binomial on discordant pairs
    out["table_2x2"] = {"real_right_blind_right": n_rb, "real_right_blind_wrong": n_rB,
                        "real_wrong_blind_right": n_Rb, "real_wrong_blind_wrong": n_RB}
    out["mcnemar_exact_p"] = p_mc
    print("== 1. canonical order, n=%d" % n)
    print(f"real  {out['real_acc']:.3f}  Wilson95 [{out['real_wilson95'][0]:.3f}, {out['real_wilson95'][1]:.3f}]")
    print(f"blank {out['blank_acc']:.3f}  Wilson95 [{out['blank_wilson95'][0]:.3f}, {out['blank_wilson95'][1]:.3f}]  (chance 0.25)")
    print("2x2 (rows real, cols blank):\n             blank right  blank wrong\n"
          f"real right   {n_rb:11d}  {n_rB:11d}\nreal wrong   {n_Rb:11d}  {n_RB:11d}")
    print(f"discordant: real-only-right {n_rB}, blank-only-right {n_Rb}; exact McNemar p = {p_mc:.3g}")

    # 2. k-of-6 blind histogram vs Binomial(6, 0.25), and real acc by blind k
    bl = d[d.condition == "blank"].pivot(index="id", columns="seed", values="correct")
    assert bl.shape[1] == 6
    k = bl.sum(1).astype(int)
    obs = k.value_counts().reindex(range(7), fill_value=0)
    exp = pd.Series(stats.binom.pmf(range(7), 6, 0.25) * n, index=range(7))
    print("\n== 2. blind: items correct in k of 6 option orders")
    print(pd.DataFrame({"observed": obs, "Binom(6,.25) expected": exp.round(1)}).T.to_string())
    print("(the 6 orders are not independent draws of one item, so no p-value is claimed for this comparison)")
    out["blind_k_hist_observed"] = obs.tolist()
    out["blind_k_hist_binom_expected"] = exp.round(2).tolist()
    rl = d[d.condition == "real"].pivot(index="id", columns="seed", values="correct")
    real_mean = rl.mean(1)  # per-item mean over the 6 orders = per-order-averaged accuracy when averaged over items
    for name, m in (("k<=1", k <= 1), ("k>=4", k >= 4), ("k in 2..3", (k >= 2) & (k <= 3))):
        ids = k.index[m]
        per_order = rl.loc[ids].mean(0)  # accuracy per order over these items
        out[f"real_acc_blind_{name}"] = {"n": int(m.sum()), "mean_over_orders": float(per_order.mean()),
                                         "per_order": per_order.round(3).tolist()}
        print(f"real-image acc on blind {name}: n={int(m.sum()):3d}  mean over 6 orders {per_order.mean():.3f}  per order {per_order.round(3).tolist()}")

    # 3. predicted-letter distribution (REAL), shown positions, shuffled seeds vs canonical
    real = d[d.condition == "real"]
    sh = real[real.seed > 0]
    print("\n== 3. REAL condition, predicted letter by SHOWN position")
    dist_sh = sh.pred_letter_shown.value_counts(normalize=True).reindex(list("ABCD")).round(3)
    dist_0 = real[real.seed == 0].pred_letter_shown.value_counts(normalize=True).reindex(list("ABCD")).round(3)
    perm_arr = sh.perm.astype(str).str.zfill(4)
    gold_pos = [str(p).index(str(a)) for p, a in zip(perm_arr, sh.answer)]
    gold_dist = pd.Series(gold_pos).map(lambda i: "ABCD"[i]).value_counts(normalize=True).reindex(list("ABCD")).round(3)
    print("canonical (seed 0):      ", dist_0.to_dict())
    print("shuffled (seeds 1-5):    ", dist_sh.to_dict())
    print("gold answer shown-position, shuffled (should be ~uniform):", gold_dist.to_dict())
    out["real_pred_letter_canonical"] = dist_0.to_dict()
    out["real_pred_letter_shuffled"] = dist_sh.to_dict()
    out["gold_position_shuffled"] = gold_dist.to_dict()
    sm = json.load(open(f"{D}/summary.json"))
    print("model:", sm["model"], "revision:", sm.get("model_revision", "(not recorded in summary.json)"))

    json.dump(out, open(f"{D}/analysis.json", "w"), indent=2)
    print(f"\nsaved {D}/analysis.json")


if __name__ == "__main__":
    main()
