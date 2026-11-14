"""Build the 8 mixtures for the visual-grounding regression experiment:
4 conditions x 2 text-answerable shares (10% and 90%).
Each mixture contains exactly 1,050 examples.

Conditions:
1. Baseline:
   - A-OKVQA: 300
   - COCO-QA: 300
   - TallyQA: 300
   - VSR: 150
2. Baseline + OK-VQA:
   - A-OKVQA: 200
   - COCO-QA: 200
   - TallyQA: 200
   - VSR: 150
   - OK-VQA: 300
3. Baseline + TextVQA:
   - A-OKVQA: 200
   - COCO-QA: 200
   - TallyQA: 200
   - VSR: 150
   - TextVQA: 300
4. Baseline + TextCaps:
   - A-OKVQA: 200
   - COCO-QA: 200
   - TallyQA: 200
   - VSR: 150
   - TextCaps: 300

Methodology:
- Within-source quantile labelling (low=0.4, high=0.6).
- VSR 50/50 yes/no balance preserved exactly (75 yes, 75 no).
- Exact 10% (105 text) and 90% (945 text) proportions.
- Seed 0 deterministic sampling.
- Hard MMStar eval overlap assertions.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.overlap import assert_no_overlap
from scripts.run_tier2 import yes_no

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2_DIR = os.path.join(ROOT, "results", "tier2", "full")
SCREEN_DIR = os.path.join(ROOT, "results", "screen_pools")
FP_PATH = os.path.join(ROOT, "results", "tier2", "mmstar_fingerprints.json")

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml")))
t2 = cfg["tier2"]
seed = t2["mixture_seed"]
eval_fp = json.load(open(FP_PATH))

# 1. Load baseline labeled matched
base_df = pd.read_csv(os.path.join(T2_DIR, "pool_labeled_matched.csv"))
base_df["yn"] = yes_no(base_df.answer)

# 2. Load and quantile-label new sources
new_subsets = ["okvqa", "textvqa", "textcaps"]
new_dfs = []
quantiles_info = {}

for s in new_subsets:
    pool_path = os.path.join(SCREEN_DIR, f"baseline_{s}.csv")
    label_path = os.path.join(SCREEN_DIR, f"labels_{s}.csv")
    raw_pool = pd.read_csv(pool_path)
    raw_sub = raw_pool[raw_pool["subset"] == s].copy()
    raw_labels = pd.read_csv(label_path)
    sub_df = raw_sub.merge(raw_labels, on="id", how="inner")
    
    lo, hi = sub_df.gap.quantile([t2["low_quantile"], t2["high_quantile"]]).tolist()
    quantiles_info[s] = {"low": lo, "high": hi, "count": len(sub_df)}
    sub_df["label"] = np.where(sub_df.gap <= lo, "text_answerable",
                               np.where(sub_df.gap >= hi, "image_necessary", "middle"))
    sub_df["yn"] = yes_no(sub_df.answer)
    new_dfs.append(sub_df)

full_pool = pd.concat([base_df] + new_dfs, ignore_index=True)
full_pool.to_csv(os.path.join(T2_DIR, "pool_labeled_screen.csv"), index=False)
full_pool.to_csv(os.path.join(SCREEN_DIR, "pool_labeled_screen.csv"), index=False)
print("Saved pool_labeled_screen.csv with", len(full_pool), "rows.")

# 3. Define composition conditions
conditions = {
    "baseline": {
        "aokvqa": 300,
        "cocoqa": 300,
        "tallyqa": 300,
        "vsr": 150
    },
    "okvqa": {
        "aokvqa": 200,
        "cocoqa": 200,
        "tallyqa": 200,
        "vsr": 150,
        "okvqa": 300
    },
    "textvqa": {
        "aokvqa": 200,
        "cocoqa": 200,
        "tallyqa": 200,
        "vsr": 150,
        "textvqa": 300
    },
    "textcaps": {
        "aokvqa": 200,
        "cocoqa": 200,
        "tallyqa": 200,
        "vsr": 150,
        "textcaps": 300
    }
}

COLS = ["id", "subset", "label", "gap", "nll_real", "nll_blank", "user", "answer", "image_path", "dhash", "qnorm"]

summary = {}

for cond_name, spec in conditions.items():
    for frac in [0.1, 0.9]:
        share_str = f"text{int(round(frac * 100))}"
        mix_name = f"screen_{cond_name}_{share_str}"
        
        rng = np.random.RandomState(seed * 1000 + int(round(frac * 100)))
        picks = []
        
        for s, n in spec.items():
            g = full_pool[full_pool.subset == s]
            binary = g.yn.notna().all()
            nt = int(round(frac * n))
            ni = n - nt
            want = {"text_answerable": nt, "image_necessary": ni}
            
            if binary:  # VSR yes/no 50/50 balancing
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
                    raise ValueError(f"{mix_name} {s} {lab} {yn}: need {k}, have {len(c)}")
                picks.append(c.iloc[rng.permutation(len(c))[:k]])
                
        pick = pd.concat(picks)
        pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
        
        # Verify exact requirements
        assert len(pick) == 1050, f"Expected 1050 items, got {len(pick)}"
        expected_text = int(round(frac * 1050))
        actual_text = int((pick.label == "text_answerable").sum())
        assert actual_text == expected_text, f"Expected {expected_text} text items, got {actual_text}"
        
        vsr_part = pick[pick.subset == "vsr"]
        assert len(vsr_part) == 150, f"Expected 150 VSR, got {len(vsr_part)}"
        yes_cnt = int((vsr_part.yn == "yes").sum())
        no_cnt = int((vsr_part.yn == "no").sum())
        assert yes_cnt == 75 and no_cnt == 75, f"VSR balance failed: yes={yes_cnt}, no={no_cnt}"
        
        # Check MMStar overlap
        assert_no_overlap(pick.id.tolist(), pick.dhash.map(lambda x: int(str(x), 16)).tolist(),
                          pick.qnorm.fillna("").tolist(), eval_fp, t2["overlap_hamming"], mix_name)
        
        out_csv1 = os.path.join(T2_DIR, f"mixture_{mix_name}.csv")
        out_csv2 = os.path.join(SCREEN_DIR, f"mixture_{mix_name}.csv")
        pick[COLS].to_csv(out_csv1, index=False)
        pick[COLS].to_csv(out_csv2, index=False)
        
        summary[mix_name] = {
            "n": len(pick),
            "text_share": float((pick.label == "text_answerable").mean()),
            "by_subset": pick.groupby("subset").size().to_dict(),
            "by_label": pick.groupby(["subset", "label"]).size().unstack(fill_value=0).to_dict("index"),
            "vsr_yes_no": {"yes": yes_cnt, "no": no_cnt}
        }
        print(f"Built {mix_name}: n={len(pick)}, text_share={actual_text/len(pick):.2f}, VSR 50/50 OK")

summary_path = os.path.join(SCREEN_DIR, "mixtures_summary.json")
with open(summary_path, "w") as f:
    json.dump({"quantiles": quantiles_info, "mixtures": summary}, f, indent=2)
print("Saved summary to", summary_path)
