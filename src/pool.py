"""Training pool: stream the first rows of small Cauldron subsets (no bulk download), save images locally."""
from __future__ import annotations
import os
import time
import pandas as pd
from .overlap import dhash, norm_q


def stream_subset(dataset_id: str, sub: str, n: int, img_dir: str, retries: int = 3) -> list[dict]:
    """First n rows of the stream; first QA pair per image only (so no two items share an image)."""
    from datasets import load_dataset
    for attempt in range(retries):
        try:
            ds = load_dataset(dataset_id, sub, split="train", streaming=True)
            rows = []
            for ridx, r in enumerate(ds):
                if len(rows) == n:
                    break
                if len(r["images"]) != 1 or not r["texts"]:
                    continue
                t = r["texts"][0]
                user, ans = t["user"].strip(), t["assistant"].strip()
                if not user or not ans:
                    continue
                img = r["images"][0].convert("RGB")
                iid = f"cauldron-{sub}-{ridx}"
                path = os.path.join(img_dir, f"{iid}.jpg")
                img.save(path, quality=90)
                rows.append(dict(id=iid, subset=sub, row=ridx, user=user, answer=ans, image_path=path,
                                 w=img.size[0], h=img.size[1], dhash=format(dhash(img), "x"),
                                 qnorm=norm_q(user.split("\n")[0])))
            return rows
        except Exception as e:  # network hiccups on streaming
            print(f"  stream {sub} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(5)
    raise RuntimeError(f"could not stream subset {sub}")


def build_pool(cfg: dict, out_dir: str) -> pd.DataFrame:
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    frames = []
    for sub in cfg["subsets"]:
        t0 = time.time()
        rows = stream_subset(cfg["pool_dataset"], sub, cfg["n_per_subset"], img_dir)
        mb = sum(os.path.getsize(r["image_path"]) for r in rows) / 1e6
        print(f"  {sub}: {len(rows)} items, {mb:.1f} MB of images saved, {time.time()-t0:.0f}s")
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)
