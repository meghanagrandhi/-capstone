"""Loss-gap labeling: gap = mean answer-token NLL (blank image) - mean answer-token NLL (real image).
Small gap = text-answerable, large gap = image-necessary. Multiple-choice items additionally get
blank/real option-letter correctness using the same scorer as Tier 1."""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from .data import blank_like

COLS = ["id", "nll_real", "nll_blank", "gap", "n_ans_tokens", "mc_valid", "mc_real_correct", "mc_blank_correct"]


def parse_mc(user: str, answer: str):
    """A-OKVQA style: 'STEM\\n...\\nOptions: a, b, c, d.' with a text answer. None if not a clean 4-option MC."""
    m = re.search(r"\nOptions: (.+)$", user, re.S)
    if not m:
        return None
    opts = [o.strip() for o in m.group(1).strip().rstrip(".").split(", ")]
    if len(opts) != 4 or not all(opts):
        return None
    gold = [o.lower() for o in opts]
    a = answer.strip().rstrip(".").lower()
    if a not in gold:
        return None
    return user.split("\n")[0].strip(), opts, gold.index(a)


def label_pool(model, pool: pd.DataFrame, cache_path: str, gray: int = 127, log_every: int = 50) -> pd.DataFrame:
    done = set(pd.read_csv(cache_path).id) if os.path.exists(cache_path) else set()
    new, n_err = [], 0
    for k, r in enumerate(pool.itertuples()):
        if r.id in done:
            continue
        try:
            img = Image.open(r.image_path).convert("RGB")
            blank = blank_like(img, gray)
            nll_r, ntok = model.answer_nll(img, r.user, r.answer)
            nll_b, _ = model.answer_nll(blank, r.user, r.answer)
            mc = parse_mc(r.user, r.answer)
            if mc:
                stem, opts, gold = mc
                rc = int(np.argmax(model.score_options(img, stem, opts)) == gold)
                bc = int(np.argmax(model.score_options(blank, stem, opts)) == gold)
            row = dict(id=r.id, nll_real=nll_r, nll_blank=nll_b, gap=nll_b - nll_r, n_ans_tokens=ntok,
                       mc_valid=int(bool(mc)), mc_real_correct=rc if mc else np.nan, mc_blank_correct=bc if mc else np.nan)
            new.append(row)
        except Exception as e:
            n_err += 1
            print(f"  label failed for {r.id}: {type(e).__name__}: {str(e)[:100]}")
        if len(new) and len(new) % log_every == 0:
            _flush(cache_path, new)
            new = []
            print(f"  labeled {k+1}/{len(pool)} (errors {n_err})", flush=True)
    _flush(cache_path, new)
    return pd.read_csv(cache_path)


def _flush(path, rows):
    if rows:
        pd.DataFrame(rows, columns=COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
