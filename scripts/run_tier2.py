"""Tier 2: label a training pool (text-answerable vs image-necessary) and build two fixed-size mixtures.
Stages (each cached/resumable):
  pool   stream subsets, drop in-pool duplicate images, drop anything overlapping MMStar (CPU/network)
  label  loss-gap labeling with the base model (GPU)
  mix    thresholds by quantile, histogram, two mixtures, HARD overlap assertions (CPU)
Usage: python -m scripts.run_tier2 --stage {pool,label,mix,all} [--smoke] [--extra-eval-fp fp.json ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.overlap import assert_no_overlap, find_overlaps, mmstar_fingerprints


def merge_fps(fps: list[dict]) -> dict:
    return {k: sum((fp[k] for fp in fps), []) for k in ("ids", "dhash", "q")}


def stage_pool(cfg, t2, out, eval_fp):
    from src.pool import build_pool
    t0 = time.time()
    raw = build_pool(t2, out)
    raw.to_csv(os.path.join(out, "pool_raw.csv"), index=False)
    n0 = len(raw)
    # in-pool duplicate images: keep the first
    h = raw.dhash.map(lambda x: int(x, 16)).tolist()
    keep, seen = [], []
    for i, v in enumerate(h):
        if any((v ^ s).bit_count() == 0 for s in seen):
            continue
        seen.append(v)
        keep.append(i)
    pool = raw.iloc[keep].reset_index(drop=True)
    n_dup = n0 - len(pool)
    # overlap with eval sets -> drop
    bad = find_overlaps(pool.dhash.map(lambda x: int(x, 16)).tolist(), pool.qnorm.tolist(), eval_fp, t2["overlap_hamming"])
    bad_idx = {i for i, _ in bad}
    reasons = pd.Series([r for _, r in bad]).value_counts().to_dict() if bad else {}
    pool = pool.drop(index=list(bad_idx)).reset_index(drop=True)
    # belt and braces: the filtered pool must now pass the hard check
    assert_no_overlap(pool.id.tolist(), pool.dhash.map(lambda x: int(x, 16)).tolist(), pool.qnorm.tolist(),
                      eval_fp, t2["overlap_hamming"], "filtered pool")
    pool.to_csv(os.path.join(out, "pool.csv"), index=False)
    info = dict(n_streamed=n0, n_in_pool_duplicate_images=n_dup, n_dropped_overlap_with_eval=len(bad_idx),
                overlap_reasons=reasons, n_final=len(pool), per_subset=pool.subset.value_counts().to_dict(),
                runtime_min=(time.time() - t0) / 60)
    json.dump(info, open(os.path.join(out, "pool_filter.json"), "w"), indent=2)
    print("pool:", info)
    return pool


def stage_label(cfg, t2, out):
    from src.labeling import label_pool
    from src.vlm import make_model
    pool = pd.read_csv(os.path.join(out, "pool.csv"))
    mc = dict(cfg["model"])
    mc["max_answer_tokens"] = t2["max_answer_tokens"]
    model = make_model(mc).load()
    t0 = time.time()
    lab = label_pool(model, pool, os.path.join(out, "labels.csv"), cfg["tier1"]["blank_gray"])
    print(f"labeled {len(lab)}/{len(pool)} items in {(time.time()-t0)/60:.1f} min")


def stage_mix(cfg, t2, out, eval_fp, smoke):
    pool = pd.read_csv(os.path.join(out, "pool.csv"))
    lab = pd.read_csv(os.path.join(out, "labels.csv")).drop_duplicates("id")
    df = pool.merge(lab, on="id", how="inner")
    print(f"{len(df)} labeled items of {len(pool)} in pool")
    lo, hi = df.gap.quantile([t2["low_quantile"], t2["high_quantile"]]).tolist()
    df["label"] = np.where(df.gap <= lo, "text_answerable", np.where(df.gap >= hi, "image_necessary", "middle"))
    df.to_csv(os.path.join(out, "pool_labeled.csv"), index=False)  # all labeled items incl. 'middle'; used for Tier 3 probe
    n_t, n_i = int((df.label == "text_answerable").sum()), int((df.label == "image_necessary").sum())
    print(f"thresholds by quantile: text-answerable gap <= {lo:.3f} nats (q{t2['low_quantile']}), "
          f"image-necessary gap >= {hi:.3f} nats (q{t2['high_quantile']}); counts text={n_t}, image={n_i}, middle={len(df)-n_t-n_i}")

    # --- mixtures: fixed size, ~10% and ~90% text-answerable, fixed seed ---
    M = t2["smoke_mixture_size"] if smoke else t2["mixture_size"]
    rng = np.random.RandomState(t2["mixture_seed"])
    T, I = df[df.label == "text_answerable"], df[df.label == "image_necessary"]
    comp = {}
    for frac in t2["mixture_text_fracs"]:
        nt = int(round(frac * M))
        ni = M - nt
        if nt > len(T) or ni > len(I):
            raise ValueError(f"mixture {frac}: need {nt} text/{ni} image items but pool has {len(T)}/{len(I)}")
        pick = pd.concat([T.iloc[rng.permutation(len(T))[:nt]], I.iloc[rng.permutation(len(I))[:ni]]])
        pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
        name = f"mixture_text{int(round(frac*100))}"
        pick[["id", "subset", "label", "gap", "nll_real", "nll_blank", "user", "answer", "image_path", "dhash", "qnorm"]].to_csv(
            os.path.join(out, f"{name}.csv"), index=False)
        # HARD RULE: no overlap with any eval item (all 1500 MMStar items + any extra eval fingerprints)
        assert_no_overlap(pick.id.tolist(), pick.dhash.map(lambda x: int(x, 16)).tolist(), pick.qnorm.tolist(),
                          eval_fp, t2["overlap_hamming"], name)
        comp[name] = dict(n=len(pick), n_text_answerable=nt, n_image_necessary=ni,
                          text_share=nt / len(pick), by_subset=pick.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict("index"))
    ids = {n: set(pd.read_csv(os.path.join(out, f"{n}.csv")).id) for n in comp}
    comp["shared_items_between_mixtures"] = len(set.intersection(*ids.values()))

    # --- stats ---
    summ = dict(n_pool_labeled=len(df), thresholds=dict(low=lo, high=hi, low_q=t2["low_quantile"], high_q=t2["high_quantile"]),
                share_gap_le_0=float((df.gap <= 0).mean()), share_gap_le_0p1=float((df.gap <= 0.1).mean()),
                share_gap_le_0p5=float((df.gap <= 0.5).mean()), gap_quantiles=df.gap.quantile([.05, .25, .5, .75, .95]).round(3).tolist(),
                per_subset=df.groupby("subset").gap.agg(["count", "median", "mean"]).round(3).to_dict("index"),
                label_by_subset=df.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict("index"), mixtures=comp)
    mc = df[df.mc_valid == 1]
    if len(mc):
        summ["mc"] = dict(n=len(mc), real_acc=float(mc.mc_real_correct.mean()), blank_acc=float(mc.mc_blank_correct.mean()),
                          mean_gap_blank_correct=float(mc[mc.mc_blank_correct == 1].gap.mean()) if (mc.mc_blank_correct == 1).any() else None,
                          mean_gap_blank_wrong=float(mc[mc.mc_blank_correct == 0].gap.mean()) if (mc.mc_blank_correct == 0).any() else None)
    summ["model_revision"] = cfg["model"].get("revision")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("mixtures", "per_subset", "label_by_subset")}, indent=1, default=float))
    print("per-subset gap:", summ["per_subset"])
    print("mixtures:", {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != 'by_subset'}) for k, v in comp.items()})

    # --- histogram ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, GRAY = "#1f2937", "#6b7280", "#2a6fdb", "#9ca3af"
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=150, gridspec_kw=dict(width_ratios=[1.6, 1]))
    a.hist(df.gap, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    for v, lab_, ha in ((lo, f"q{int(t2['low_quantile']*100)} = {lo:.2f} ", "right"),
                        (hi, f" q{int(t2['high_quantile']*100)} = {hi:.2f}", "left")):
        a.axvline(v, color=INK, ls="--", lw=1)
        a.text(v, a.get_ylim()[1] * 0.97, lab_, color=INK, fontsize=8, va="top", ha=ha)
    a.set_xlabel("loss gap = NLL(blank) - NLL(real), nats/token", color=INK)
    a.set_ylabel("items", color=INK)
    a.set_title(f"Loss-gap score, pool n={len(df)} (left of q{int(t2['low_quantile']*100)}: text-answerable)", color=INK, fontsize=9, loc="left")
    sub = df.groupby("subset").gap.median().sort_values()
    b.barh(sub.index, sub.values, color=BLUE, height=0.5)
    b.set_xlabel("median gap", color=INK)
    b.set_title("median gap by subset", color=INK, fontsize=9, loc="left")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y" if ax is a else "x", color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gap_hist.png"))
    print("saved", os.path.join(out, "gap_hist.png"))


def stage_mix_matched(cfg, t2, out, eval_fp, smoke):
    """Subset-matched mixtures: label by WITHIN-subset gap quantiles, then give every mixture the same number of items
    per subset (M / n_subsets), so the two mixtures differ only in the share of (relatively) text-answerable items."""
    pool = pd.read_csv(os.path.join(out, "pool.csv"))
    lab = pd.read_csv(os.path.join(out, "labels.csv")).drop_duplicates("id")
    df = pool.merge(lab, on="id", how="inner")
    parts, thr = [], {}
    for s, g in df.groupby("subset"):
        lo, hi = g.gap.quantile([t2["low_quantile"], t2["high_quantile"]]).tolist()
        thr[s] = dict(low=lo, high=hi)
        g = g.copy()
        g["label"] = np.where(g.gap <= lo, "text_answerable", np.where(g.gap >= hi, "image_necessary", "middle"))
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(out, "pool_labeled_matched.csv"), index=False)
    M = t2["smoke_mixture_size"] if smoke else t2["mixture_size"]
    subs = sorted(df.subset.unique())
    per = M // len(subs)
    rng = np.random.RandomState(t2["mixture_seed"])
    comp = {}
    for frac in t2["mixture_text_fracs"]:
        picks = []
        for s in subs:
            nt = int(round(frac * per))
            ni = per - nt
            T = df[(df.subset == s) & (df.label == "text_answerable")]
            I = df[(df.subset == s) & (df.label == "image_necessary")]
            if nt > len(T) or ni > len(I):
                raise ValueError(f"matched {frac} {s}: need {nt}/{ni}, have {len(T)}/{len(I)}")
            picks += [T.iloc[rng.permutation(len(T))[:nt]], I.iloc[rng.permutation(len(I))[:ni]]]
        pick = pd.concat(picks)
        pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
        name = f"mixture_matched_text{int(round(frac*100))}"
        pick[["id", "subset", "label", "gap", "nll_real", "nll_blank", "user", "answer", "image_path", "dhash", "qnorm"]].to_csv(
            os.path.join(out, f"{name}.csv"), index=False)
        assert_no_overlap(pick.id.tolist(), pick.dhash.map(lambda x: int(x, 16)).tolist(), pick.qnorm.tolist(),
                          eval_fp, t2["overlap_hamming"], name)
        comp[name] = dict(n=len(pick), text_share=float((pick.label == "text_answerable").mean()),
                          mean_gap=float(pick.gap.mean()), median_gap=float(pick.gap.median()),
                          by_subset=pick.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict("index"))
    a, b = (set(pd.read_csv(os.path.join(out, f"{n}.csv")).id) for n in comp)
    comp["shared_items_between_mixtures"] = len(a & b)
    json.dump(dict(within_subset_thresholds=thr, mixtures=comp), open(os.path.join(out, "summary_matched.json"), "w"),
              indent=2, default=float)
    print("within-subset thresholds:", {s: {k: round(v, 3) for k, v in t.items()} for s, t in thr.items()})
    for n, c in comp.items():
        print(n, c if not isinstance(c, dict) else {k: v for k, v in c.items()})


def yes_no(answers: pd.Series) -> pd.Series:
    a = answers.astype(str).str.strip().str.lower()
    return pd.Series(np.where(a.str.startswith("yes"), "yes", np.where(a.str.startswith("no"), "no", None)), index=answers.index)


def stage_mix_balanced(cfg, t2, out, eval_fp, smoke):
    """Answer-balanced version of mix_matched (added after the runs: the matched mixtures had 75% vs 36% "yes" among VSR
    answers, because Qwen's image-necessary VSR items are mostly "yes"). Same within-subset labels; every yes/no subset
    (all answers yes/no, i.e. VSR) gets exactly 50% "yes" in BOTH mixtures. The pool has only 62 image-necessary "no"
    items, so yes/no subsets get `balanced_yesno_items` (150) items per mixture instead of 250; mixtures are 900 items."""
    df = pd.read_csv(os.path.join(out, "pool_labeled_matched.csv"))
    df["yn"] = yes_no(df.answer)
    subs = sorted(df.subset.unique())
    per = (t2["smoke_mixture_size"] if smoke else t2["mixture_size"]) // len(subs)
    n_bin = t2["balanced_yesno_items"]
    rng = np.random.RandomState(t2["mixture_seed"])
    comp = {}
    for frac in t2["mixture_text_fracs"]:
        picks = []
        for s in subs:
            g = df[df.subset == s]
            binary = g.yn.notna().all()
            n = n_bin if binary else per
            nt = int(round(frac * n))
            want = {"text_answerable": nt, "image_necessary": n - nt}
            if binary:  # majority label as close to 50/50 as the pool allows, minority label fills up to n/2 "yes"
                maj, mino = sorted(want, key=want.get, reverse=True)
                avail = {(lab, yn): int(((g.label == lab) & (g.yn == yn)).sum()) for lab in want for yn in ("yes", "no")}
                y_maj = min(max(round(want[maj] / 2), want[maj] - avail[(maj, "no")]), avail[(maj, "yes")])
                y_min = n // 2 - y_maj
                cells = {(maj, "yes"): y_maj, (maj, "no"): want[maj] - y_maj,
                         (mino, "yes"): y_min, (mino, "no"): want[mino] - y_min}
            else:
                cells = {(lab, None): k for lab, k in want.items()}
            for (lab, yn), k in cells.items():
                c = g[(g.label == lab) & ((g.yn == yn) if yn else True)]
                if not 0 <= k <= len(c):
                    raise ValueError(f"balanced {frac} {s} {lab} {yn}: need {k}, have {len(c)}")
                picks.append(c.iloc[rng.permutation(len(c))[:k]])
        pick = pd.concat(picks)
        pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
        name = f"mixture_balanced_text{int(round(frac*100))}"
        pick[["id", "subset", "label", "gap", "nll_real", "nll_blank", "user", "answer", "image_path", "dhash", "qnorm"]].to_csv(
            os.path.join(out, f"{name}.csv"), index=False)
        assert_no_overlap(pick.id.tolist(), pick.dhash.map(lambda x: int(x, 16)).tolist(), pick.qnorm.tolist(),
                          eval_fp, t2["overlap_hamming"], name)
        b = pick[pick.yn.notna()]
        comp[name] = dict(n=len(pick), text_share=float((pick.label == "text_answerable").mean()),
                          mean_gap=float(pick.gap.mean()), yes_share_of_yes_no=float((b.yn == "yes").mean()),
                          by_subset=pick.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict("index"),
                          yes_no_cells={f"{l}/{y}": int(k) for (l, y), k in b.groupby(["label", "yn"]).size().items()})
    json.dump(comp, open(os.path.join(out, "summary_balanced.json"), "w"), indent=2, default=str)
    for n, c in comp.items():
        print(n, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["pool", "label", "mix", "mix_matched", "mix_balanced", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--extra-eval-fp", nargs="*", default=[], help="extra eval fingerprint json files (same format as mmstar_fingerprints.json)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t2 = dict(cfg["tier2"])
    if a.smoke:
        t2["n_per_subset"] = t2["smoke_n_per_subset"]
    out = os.path.join(t2["out_dir"], "smoke" if a.smoke else "full")
    os.makedirs(out, exist_ok=True)
    os.makedirs(t2["out_dir"], exist_ok=True)
    eval_fp = merge_fps([mmstar_fingerprints(cfg["tier1"], os.path.join(t2["out_dir"], "mmstar_fingerprints.json"))]
                        + [json.load(open(p)) for p in a.extra_eval_fp])
    print(f"eval fingerprints: {len(eval_fp['ids'])} items (all of MMStar val + extras)")
    if a.stage in ("pool", "all"):
        stage_pool(cfg, t2, out, eval_fp)
    if a.stage in ("label", "all"):
        stage_label(cfg, t2, out)
    if a.stage in ("mix", "all"):
        stage_mix(cfg, t2, out, eval_fp, a.smoke)
    if a.stage in ("mix_matched", "all"):
        stage_mix_matched(cfg, t2, out, eval_fp, a.smoke)
    if a.stage == "mix_balanced":  # not part of "all": added after the main runs
        stage_mix_balanced(cfg, t2, out, eval_fp, a.smoke)


if __name__ == "__main__":
    main()
