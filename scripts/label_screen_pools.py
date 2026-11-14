import sys
import re
sys.modules.setdefault("regex", re)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pandas as pd
from src.vlm import make_model
from src.labeling import label_pool

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
model = make_model(dict(cfg["model"])).load()

jobs = [
    ("baseline_okvqa.csv", "okvqa"),
    ("baseline_textvqa.csv", "textvqa"),
    ("baseline_textcaps.csv", "textcaps"),
]

N_ROWS = int(os.environ.get("LABEL_LIMIT", 850))

for filename, subset in jobs:
    path = Path("results/screen_pools") / filename
    pool = pd.read_csv(path)
    pool = pool[pool["subset"] == subset].head(N_ROWS).copy()

    output = Path("results/screen_pools") / f"labels_{subset}.csv"
    print(f"\nLabelling {subset}: {len(pool)} rows", flush=True)

    label_pool(model, pool, output, gray=127, log_every=50)
    print(f"Saved: {output}", flush=True)

print("\nALL THREE SOURCES LABELLED", flush=True)
