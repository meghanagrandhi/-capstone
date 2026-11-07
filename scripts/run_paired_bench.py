"""Paired-image tests (MMVP + NaturalBench) at the start and end checkpoints of the Tier 3 runs.
Both benchmarks ask the SAME question about two DIFFERENT images whose correct answers differ, so a model that
ignores the image tends to give the same answer to both.
  MMVP/MMVP:           150 pairs (300 questions), 2 options "(a) X (b) Y"
  BaiqiL/NaturalBench: n_samples x (2 images x 2 questions); yes/no -> options [Yes, No], MC -> parsed A/B options
Metrics per checkpoint: question accuracy, pair accuracy (both images right; NaturalBench also group = all 4 right),
same-answer rate (same answer to one question on both images = image ignored).
HARD RULE: fails loudly if any benchmark image/question overlaps the training mixtures (dHash + question text).
Usage: python -m scripts.run_paired_bench --runs matched_text10 matched_text90 ... [--limit 20 for a smoke test]
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import evaluate
from src.ioutil import safe_to_csv
from src.overlap import dhash, find_overlaps, norm_q
from src.vlm import make_model, select_model

OUT = "results/tier3/paired_bench"
HUB = os.path.expanduser("~/.cache/huggingface/hub")
MMVP_REV = "37eafecab8a3940c50c2ade5b36de69dbc99a8cf"
NB_REV = "ba41a7d564877a9b64c094b08015ca493cc3e54b"


def load_mmvp():
    root = os.path.join(HUB, "datasets--MMVP--MMVP", "snapshots", MMVP_REV)
    q = pd.read_csv(os.path.join(root, "Questions.csv")).rename(columns={"Correct Answer": "correct"})
    items = []
    for r in q.itertuples(index=False):
        opts = re.findall(r"\(([ab])\)\s*(.*?)\s*(?=\([ab]\)|$)", r.Options)
        assert [o[0] for o in opts] == ["a", "b"], f"MMVP options not parsed: {r.Options!r}"
        img = Image.open(os.path.join(root, "MMVP Images", f"{r.Index}.jpg")).convert("RGB")
        items.append(dict(id=f"mmvp-{r.Index}", pair=f"mmvp-p{(r.Index - 1) // 2}", qkey=f"mmvp-p{(r.Index - 1) // 2}",
                          image=img, question=r.Question, options=[o[1] for o in opts],
                          answer="ab".index(r.correct.strip()[1]), category="mmvp"))
    return items


def load_naturalbench(n_samples, seed=0):
    fs = sorted(glob.glob(os.path.join(HUB, "datasets--BaiqiL--NaturalBench", "snapshots", NB_REV, "data", "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.iloc[np.random.RandomState(seed).permutation(len(df))]
    items, kept, skipped = [], 0, []
    for r in df.itertuples(index=False):
        if kept == n_samples:
            break
        qs = []
        for qi in (0, 1):
            qtext = getattr(r, f"Question_{qi}")
            if r.Question_Type == "yes_no":
                qs.append((qtext, ["Yes", "No"], ["Yes", "No"]))
                continue
            m = re.match(r"(.*?)\s*Option:\s*A:(.*?);\s*B:(.*?);?\s*$", qtext, re.S)
            if m:
                qs.append((m.group(1), [m.group(2).strip(), m.group(3).strip()], ["A", "B"]))
        answers = [[getattr(r, f"Image_{ii}_Question_{qi}").strip() for ii in (0, 1)] for qi in (0, 1)]
        if len(qs) < 2 or any(a not in qs[qi][2] for qi in (0, 1) for a in answers[qi]):
            skipped.append(int(r.Index))  # malformed in the source data (e.g. mangled "B:" option label)
            continue
        kept += 1
        imgs = [Image.open(io.BytesIO(r.Image_0["bytes"])).convert("RGB"),
                Image.open(io.BytesIO(r.Image_1["bytes"])).convert("RGB")]
        for qi, (question, opts, letters) in enumerate(qs):
            for ii in (0, 1):
                items.append(dict(id=f"nb-{r.Index}-q{qi}-i{ii}", pair=f"nb-{r.Index}", qkey=f"nb-{r.Index}-q{qi}",
                                  image=imgs[ii], question=question, options=opts,
                                  answer=letters.index(answers[qi][ii]), category=f"nb_{r.Question_Type}"))
    print(f"NaturalBench: kept {kept} samples, skipped {len(skipped)} malformed: {skipped}", flush=True)
    return items


def find_bench_overlaps(items, ham):
    """[(item index, reason)] for benchmark items that overlap ANY training mixture (dHash or question text)."""
    mixes = [pd.read_csv(f) for f in glob.glob("results/tier2/full/mixture_*.csv")]
    mix = pd.concat(mixes, ignore_index=True).drop_duplicates("id")
    train_fp = {"dhash": mix.dhash.map(lambda x: int(x, 16)).tolist(), "q": mix.qnorm.tolist()}
    return find_overlaps([dhash(it["image"]) for it in items], [norm_q(it["question"]) for it in items], train_fp, ham)


def drop_overlaps(items, ham, out):
    """Training is already done, so overlapping benchmark items are removed from the EVAL side: the whole pair /
    NaturalBench sample goes, so paired metrics stay intact. Logged to excluded_overlap.json; re-checked to be zero."""
    hits = find_bench_overlaps(items, ham)
    bad_pairs = {items[i]["pair"] for i, _ in hits}
    log = [dict(id=items[i]["id"], pair=items[i]["pair"], question=items[i]["question"], reason=why) for i, why in hits]
    kept = [it for it in items if it["pair"] not in bad_pairs]
    if hits:
        print(f"!!! OVERLAP: {len(hits)} benchmark questions match training data -> dropping {len(bad_pairs)} "
              f"pairs/samples ({len(items) - len(kept)} questions): {sorted(bad_pairs)}", flush=True)
        json.dump(log, open(os.path.join(out, "excluded_overlap.json"), "w"), indent=1)
    left = find_bench_overlaps(kept, ham)
    assert not left, f"overlap still present after dropping: {left}"
    return kept


def summarize(d, items):
    meta = pd.DataFrame([{k: it[k] for k in ("id", "pair", "qkey", "category")} for it in items])
    d = d.merge(meta, on=["id", "category"])
    out = {}
    for bench, x in (("mmvp", d[d.category == "mmvp"]), ("naturalbench", d[d.category.str.startswith("nb")])):
        if not len(x):
            continue
        byq = x.groupby("qkey")
        out[bench] = dict(n_questions=len(x), acc=float(x.correct.mean()),
                          pair_acc=float(byq.correct.min().mean()),  # both images right for that question
                          same_answer_rate=float((byq.pred.nunique() == 1).mean()))
        if bench == "naturalbench":
            out[bench]["group_acc"] = float(x.groupby("pair").correct.min().mean())  # all 4 (2 img x 2 q) right
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nb-samples", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per benchmark")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default="qwen2vl", choices=["qwen2vl", "smolvlm", "internvl"])
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    select_model(cfg, a.model)
    os.makedirs(a.out, exist_ok=True)
    mm, nb = load_mmvp(), load_naturalbench(a.nb_samples)
    if a.limit:
        mm, nb = mm[: a.limit], nb[: a.limit]
    items = drop_overlaps(mm + nb, cfg["tier2"]["overlap_hamming"], a.out)
    n_mm = sum(it["category"] == "mmvp" for it in items)
    print(f"MMVP={n_mm} NaturalBench={len(items) - n_mm} questions; no overlap with training mixtures", flush=True)

    from peft import PeftModel, load_peft_weights, set_peft_model_state_dict
    vlm = make_model(cfg["model"]).load()
    base, pm, t0, rows = vlm.model, None, time.time(), []
    for run in a.runs:
        cks = {int(re.search(r"step_(\d+)", c).group(1)): c
               for c in glob.glob(os.path.join("results/tier3", run, "adapters", "step_*"))}
        assert cks, f"no adapters for {run}"
        # step 0 = base model for every run (LoRA B starts at zero), so evaluate it once, on the first run
        for step in ((min(cks), max(cks)) if run == a.runs[0] else (max(cks),)):
            if pm is None:
                pm = PeftModel.from_pretrained(base, cks[step], is_trainable=False)
            else:
                set_peft_model_state_dict(pm, load_peft_weights(cks[step], device="cuda"))
            pm.eval()
            vlm.model = pm
            d = evaluate(vlm, items, 0, "real", os.path.join(a.out, f"{run}_step{step:04d}.csv"), log_every=1000)
            s = summarize(d, items)
            sm = re.match(r"(?:(smolvlm|internvl)_)?((?:matched|balanced)_text\d+|text\d+)(?:_s(\d+))?$", run)
            for bench, m in s.items():
                rows.append(dict(run=run, model=sm.group(1) or "qwen2vl", mixture=sm.group(2), seed=int(sm.group(3) or 0), step=step, bench=bench, **m))
            print(f"  {run} step {step}: " + json.dumps({b: {k: round(v, 3) for k, v in m.items()} for b, m in s.items()})
                  + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if not a.limit:
        safe_to_csv(pd.DataFrame(rows), os.path.join(a.out, f"paired_bench_summary_{a.model}.csv"))
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
