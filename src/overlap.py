"""Train/eval leakage guard: near-duplicate image (dHash) or identical long question text. Fails loudly."""
from __future__ import annotations
import json
import os
import re
import numpy as np
from PIL import Image


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


MIN_Q_CHARS = 30  # shorter question text ("how many?") is too generic to count as a match on its own


def mmstar_fingerprints(cfg: dict, cache_path: str) -> dict:
    """Fingerprints of ALL 1500 MMStar items (not just the 500 sampled), so any A/G subset is covered too."""
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    from datasets import load_dataset
    from .data import parse_mmstar_question
    ds = load_dataset(cfg["dataset_id"], split=cfg["split"])
    fp = {"ids": [], "dhash": [], "q": []}
    for i in range(len(ds)):
        r = ds[i]
        fp["ids"].append(f"mmstar-{r['index']}")
        fp["dhash"].append(dhash(r["image"].convert("RGB")))
        fp["q"].append(norm_q(parse_mmstar_question(r["question"])[0]))
    json.dump(fp, open(cache_path, "w"))
    return fp


def find_overlaps(pool_dhash: list[int], pool_q: list[str], fp: dict, ham: int) -> list[tuple[int, str]]:
    """Returns [(pool_index, reason)] for every pool item that overlaps an eval item."""
    ev_q = {q for q in fp["q"] if len(q) >= MIN_Q_CHARS}
    ev_h = [int(h) for h in fp["dhash"]]
    bad = []
    for i, (h, q) in enumerate(zip(pool_dhash, pool_q)):
        if any((h ^ e).bit_count() <= ham for e in ev_h):
            bad.append((i, "image near-duplicate of an eval image"))
        elif len(q) >= MIN_Q_CHARS and q in ev_q:
            bad.append((i, "identical question text to an eval item"))
    return bad


def assert_no_overlap(train_ids, train_dhash, train_q, eval_fp: dict, ham: int, train_name="train"):
    """HARD RULE: raises AssertionError listing offenders if any training item overlaps any eval item."""
    id_hit = set(train_ids) & set(eval_fp["ids"])
    bad = find_overlaps(list(train_dhash), list(train_q), eval_fp, ham)
    if id_hit or bad:
        raise AssertionError(
            f"TRAIN/EVAL OVERLAP in {train_name}: {len(id_hit)} shared ids {sorted(id_hit)[:5]}, "
            f"{len(bad)} content overlaps e.g. {[(train_ids[i], r) for i, r in bad[:5]]}")
    print(f"[overlap check OK] {train_name}: {len(train_ids)} items vs {len(eval_fp['ids'])} eval items "
          f"(dHash<= {ham}, question text >= {MIN_Q_CHARS} chars) -> 0 overlaps")
