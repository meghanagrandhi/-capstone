
import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.overlap import find_overlaps

ROOT = r"G:\Backup D\Capstone"
BASELINE = os.path.join(
    ROOT, "capstone_run", "results", "tier2", "full", "pool.csv"
)
DATASETS = os.path.join(ROOT, "datasets")
OUT = os.path.join(
    ROOT, "capstone_run", "results", "screen_pools"
)
FP_PATH = os.path.join(
    ROOT, "capstone_run", "results", "tier2",
    "mmstar_fingerprints.json"
)

os.makedirs(OUT, exist_ok=True)

baseline = pd.read_csv(BASELINE)
with open(FP_PATH, encoding="utf-8") as f:
    eval_fp = json.load(f)

def clean_hashes(df):
    df = df.copy()
    df["dhash"] = df["dhash"].map(
        lambda x: format(int(str(x), 16), "x")
    )
    return df

baseline = clean_hashes(baseline)

def prepare(name, extra_path=None):
    frames = [baseline]

    if extra_path:
        extra = pd.read_csv(extra_path)

        # TextVQA: one question per image.
        if "image_id" in extra.columns:
            extra = extra.drop_duplicates(
                subset="image_id", keep="first"
            )

        extra = clean_hashes(extra)
        frames.append(extra)

    pool = pd.concat(frames, ignore_index=True)

    # Preserve the baseline first; remove duplicate IDs and exact dHashes.
    pool = pool.drop_duplicates("id", keep="first")
    pool = pool.drop_duplicates("dhash", keep="first")
    pool = pool.reset_index(drop=True)

    bad = find_overlaps(
        pool.dhash.map(lambda x: int(x, 16)).tolist(),
        pool.qnorm.fillna("").tolist(),
        eval_fp,
        4,
    )
    bad_indices = {i for i, _ in bad}
    pool = pool.drop(index=list(bad_indices)).reset_index(drop=True)

    path = os.path.join(OUT, f"{name}.csv")
    pool.to_csv(path, index=False)

    print(
        f"{name}: {len(pool)} rows; "
        f"subsets={pool.subset.value_counts().to_dict()}; "
        f"eval overlaps removed={len(bad_indices)}"
    )

prepare("baseline")
prepare(
    "baseline_okvqa",
    os.path.join(DATASETS, "OK-VQA", "pool.csv")
)
prepare(
    "baseline_textvqa",
    os.path.join(DATASETS, "TextVQA", "pool.csv")
)
prepare(
    "baseline_textcaps",
    os.path.join(DATASETS, "TextCaps", "pool.csv")
)

print("Candidate pool construction complete.")
