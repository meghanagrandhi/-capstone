"""MMStar loading, option parsing, sampling, blank image."""
from __future__ import annotations
import re
import numpy as np
from PIL import Image


def parse_mmstar_question(q: str) -> tuple[str, list[str]]:
    """MMStar stores options inside the question string, in two formats. Fails loudly if neither fits."""
    if "\n(A) " in q:  # "Hint: ...\nQuestion: STEM\n[Choices:\n](A) x\n(B) y ..."
        head, tail = q.split("\n(A) ", 1)
        tail = "(A) " + tail
        stem = head.split("Question:", 1)[-1].strip()
        stem = re.sub(r"\s*Choices:\s*$", "", stem)
        parts = re.split(r"(?:^|\n)\(([A-H])\) ?", tail)
        letters, texts = parts[1::2], [t.strip() for t in parts[2::2]]
    elif "\nOptions: A: " in q:  # "STEM\nOptions: A: x, B: y, C: z, D: w"
        stem, tail = q.split("\nOptions: A: ", 1)
        texts, rest, letters = [], tail, ["A"]
        for nxt in "BCDEFGH":
            j = rest.find(f", {nxt}: ")
            if j < 0:
                break
            texts.append(rest[:j].strip())
            rest = rest[j + len(f", {nxt}: "):]
            letters.append(nxt)
        texts.append(rest.strip())
        stem = stem.strip()
    else:
        raise ValueError(f"unrecognised MMStar format: {q[:200]!r}")
    if letters != list("ABCDEFGH"[: len(letters)]) or len(texts) != len(letters) or len(texts) < 2 or not all(texts):
        raise ValueError(f"bad option parse {letters} {texts} from {q[:200]!r}")
    return stem, texts


def load_mmstar(cfg: dict, n: int, seed: int) -> list[dict]:
    """Sample n items with a fixed seed. Items that do not have exactly 4 options (1 in MMStar val) are
    skipped so that chance is a uniform 25%; the skip count is printed."""
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_id"], split=cfg["split"])
    items, skipped = [], 0
    for i in np.random.RandomState(seed).permutation(len(ds)).tolist():
        if len(items) == n:
            break
        r = ds[int(i)]
        stem, opts = parse_mmstar_question(r["question"])
        if len(opts) != 4:
            skipped += 1
            continue
        items.append(dict(id=f"mmstar-{r['index']}", question=stem, options=opts,
                          answer="ABCD".index(r["answer"]), category=r["category"],
                          image=r["image"].convert("RGB")))
    print(f"load_mmstar: {len(items)} items, skipped {skipped} non-4-option items")
    return sorted(items, key=lambda x: int(x["id"].split("-")[1]))


def load_mmstar_rest(cfg: dict, n_sample: int, seed: int) -> list[dict]:
    """All 4-option MMStar items that are NOT in the calibration sample drawn by load_mmstar(cfg, n_sample, seed).
    Uses the same permutation, so the sample is the first n_sample valid items and this is the remainder."""
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_id"], split=cfg["split"])
    valid = []
    for i in np.random.RandomState(seed).permutation(len(ds)).tolist():
        r = ds[int(i)]
        stem, opts = parse_mmstar_question(r["question"])
        if len(opts) == 4:
            valid.append(dict(id=f"mmstar-{r['index']}", question=stem, options=opts,
                              answer="ABCD".index(r["answer"]), category=r["category"],
                              image=r["image"].convert("RGB")))
    sample_ids = {it["id"] for it in valid[:n_sample]}
    rest = [it for it in valid[n_sample:]]
    assert not (sample_ids & {it["id"] for it in rest}), "calibration sample and G-candidate pool overlap"
    print(f"load_mmstar_rest: {len(rest)} items outside the {n_sample}-item calibration sample")
    return sorted(rest, key=lambda x: int(x["id"].split("-")[1]))


def load_mmstar_by_ids(cfg: dict, ids: list[str]) -> list[dict]:
    """Items for the given ids ('mmstar-<index>'), in the order given. Fails if an id is missing or not 4-option."""
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_id"], split=cfg["split"])
    by_index = {int(r): i for i, r in enumerate(ds["index"])}
    items = []
    for id_ in ids:
        r = ds[by_index[int(id_.split("-")[1])]]
        stem, opts = parse_mmstar_question(r["question"])
        assert len(opts) == 4, f"{id_} does not have 4 options"
        items.append(dict(id=id_, question=stem, options=opts, answer="ABCD".index(r["answer"]),
                          category=r["category"], image=r["image"].convert("RGB")))
    return items


def blank_like(img: Image.Image, gray: int = 127) -> Image.Image:
    return Image.new("RGB", img.size, (gray, gray, gray))
