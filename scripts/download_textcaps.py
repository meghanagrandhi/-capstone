
import csv
import io
import json
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
import numpy as np

ROOT = r"G:\Backup D\Capstone\datasets\TextCaps"
IMAGE_DIR = os.path.join(ROOT, "images")
CSV_PATH = os.path.join(ROOT, "pool.csv")

API = "https://datasets-server.huggingface.co/rows"
DATASET = "lmms-lab-encoder/TextCaps"
BATCH = 100
WORKERS = 8

os.makedirs(IMAGE_DIR, exist_ok=True)

FIELDS = [
    "id", "subset", "row", "user", "answer",
    "image_path", "w", "h", "dhash", "qnorm",
    "answers_json", "question_id"
]


def get_json(offset, length):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }

    for attempt in range(8):
        try:
            response = requests.get(API, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == 7:
                raise
            wait = min(15 * (2 ** attempt), 300)
            print(f"API retry at offset {offset}; waiting {wait}s")
            time.sleep(wait)


def dhash(img, size=8):
    g = np.asarray(
        img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR),
        dtype=np.int16
    )
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def norm_q(q):
    import re
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


def download_image(item):
    row_index, row = item
    image_id = str(row["image_id"])
    image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")

    if os.path.exists(image_path):
        return row_index, image_path

    url = row["image"]["src"]

    for attempt in range(6):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()

            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(image_path, format="JPEG", quality=95)
            return row_index, image_path

        except Exception:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    existing = {}

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["question_id"]] = row

    total = get_json(0, 1)["num_rows_total"]
    print("Total TextCaps training rows:", total)
    print("Already completed:", len(existing))

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            writer.writeheader()

        for offset in range(0, total, BATCH):
            if str(offset) in existing:
                continue

            data = get_json(offset, min(BATCH, total - offset))
            records = data["rows"]

            downloads = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {
                    pool.submit(download_image, (offset + i, item["row"])):
                    i for i, item in enumerate(records)
                }

                for future in as_completed(futures):
                    i, path = future.result()
                    downloads[i - offset] = path

            output = []
            for i, item in enumerate(records):
                r = item["row"]
                image_path = downloads[i]

                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    image_hash = dhash(img)

                question = r["question"]
                caption = r["caption_str"]
                references = r.get("reference_strs") or []

                output.append({
                    "id": f"textcaps-{r['question_id']}",
                    "subset": "textcaps",
                    "row": offset + i,
                    "user": question,
                    "answer": caption,
                    "image_path": image_path,
                    "w": width,
                    "h": height,
                    "dhash": image_hash,
                    "qnorm": norm_q(question),
                    "answers_json": json.dumps(
                        references, ensure_ascii=False
                    ),
                    "question_id": r["question_id"],
                })

            writer.writerows(output)
            f.flush()

            print(
                f"Progress: {min(offset + len(records), total)}/{total} "
                f"questions; {len(os.listdir(IMAGE_DIR))} images"
            )

    print("TextCaps download complete.")


if __name__ == "__main__":
    main()
