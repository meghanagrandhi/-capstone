"""Multiple-choice evaluator by option-letter log-likelihood. Per-item results cached to CSV (resumable)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .data import blank_like
from .ioutil import safe_append_csv

COLS = ["id", "seed", "condition", "category", "n_options", "pred", "answer", "correct",
        "pred_letter_shown", "perm", "lp_A", "lp_B", "lp_C", "lp_D"]


def load_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLS)


def swap_map(items, seed: int = 0) -> dict:
    """Derangement: each item gets the image of a DIFFERENT item (seeded). A model that really reads image content
    should not answer the same as with its own image."""
    rng = np.random.RandomState(20_000 + seed)
    order = rng.permutation(len(items))
    donor = np.roll(order, 1)  # rotating a shuffled order guarantees no item keeps its own image
    return {items[i]["id"]: items[j]["image"] for i, j in zip(order, donor)}


def evaluate(model, items, seed: int, condition: str, cache_path: str, gray: int = 127, log_every: int = 50,
             swap: dict | None = None):
    """seed 0 = canonical option order; seed>0 = item order and option order shuffled with that seed.
    condition: real | blank (gray image of same size) | swap (another item's image; needs `swap` from swap_map).
    Option log-probs are stored in ORIGINAL option order (lp_A.. = original option A..).
    Resumable: rows already in cache_path (id, seed, condition) are skipped."""
    assert condition in ("real", "blank", "swap")
    assert condition != "swap" or swap is not None, "swap condition needs a swap map"
    cache = load_cache(cache_path)
    done = set(zip(cache["id"], cache["seed"], cache["condition"]))
    rng = np.random.RandomState(10_000 + seed)
    order = np.arange(len(items)) if seed == 0 else rng.permutation(len(items))
    # perms drawn in fixed item order -> independent of resume state; same perms for real and blank
    perms = {it["id"]: (np.arange(len(it["options"])) if seed == 0 else rng.permutation(len(it["options"])))
             for it in items}
    new = []
    for k, ii in enumerate(order):
        it = items[ii]
        if (it["id"], seed, condition) in done:
            continue
        perm = perms[it["id"]]  # perm[j] = original index of option shown at position j
        img = (it["image"] if condition == "real" else swap[it["id"]] if condition == "swap"
               else blank_like(it["image"], gray))
        lp_shown = model.score_options(img, it["question"], [it["options"][p] for p in perm])
        lp = np.full(4, np.nan)
        lp[perm] = lp_shown
        j = int(np.argmax(lp_shown))
        pred = int(perm[j])
        new.append(dict(id=it["id"], seed=seed, condition=condition, category=it["category"],
                        n_options=len(perm), pred=pred, answer=it["answer"], correct=int(pred == it["answer"]),
                        pred_letter_shown="ABCD"[j], perm="".join(map(str, perm)),
                        **{f"lp_{'ABCD'[i]}": lp[i] for i in range(4)}))
        if len(new) % log_every == 0:
            _flush(cache_path, new)
            new = []
            print(f"  seed={seed} {condition}: {k+1}/{len(items)}", flush=True)
    _flush(cache_path, new)
    df = load_cache(cache_path)
    ids = {it["id"] for it in items}
    return df[(df.seed == seed) & (df.condition == condition) & df.id.isin(ids)]


def _flush(path, rows):
    if not rows:
        return
    safe_append_csv(pd.DataFrame(rows, columns=COLS), path)
