
import csv
import io
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

DATASET = "Multimodal-Fatima/OK-VQA_train"
API = "https://datasets-server.huggingface.co/rows"
OUT = r"G:\Backup D\Capstone\datasets\OK-VQA"
IMG_DIR = os.path.join(OUT, "images")
CSV_PATH = os.path.join(OUT, "pool.csv")

PAGE_SIZE = 100
WORKERS = 8

FIELDS = [
    "id", "subset", "row", "user", "answer", "image_path",
    "w", "h", "dhash", "qnorm", "answers_json", "question_id"
]


def get_json(url, params=None):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def make_dhash(image):
    image = image.convert("L").resize(
        (9, 8), Image.Resampling.BILINEAR
    )
    pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(
                pixels[y * 9 + x + 1] > pixels[y * 9 + x]
            )
    return format(value, "x")


def majority_answer(answers):
    normalized = [str(a).strip().lower() for a in answers if str(a).strip()]
    if not normalized:
        raise ValueError("No usable answers")

    counts = Counter(normalized)
    winner = counts.most_common(1)[0][0]
    return next(
        str(a).strip() for a in answers
        if str(a).strip().lower() == winner
    )


def download_one(item):
    row = item["row"]
    qid = str(row["question_id"])
    item_id = f"okvqa-{qid}"
    image_path = os.path.join(IMG_DIR, item_id + ".jpg")

    for attempt in range(5):
        try:
            response = requests.get(
                row["image"]["src"], timeout=90
            )
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            image.load()
            image = image.convert("RGB")

            image.save(image_path, format="JPEG", quality=95)

            question = str(row["question"]).strip()
            answers = row["answers"]

            return {
                "id": item_id,
                "subset": "okvqa",
                "row": item["row_idx"],
                "user": question,
                "answer": majority_answer(answers),
                "image_path": image_path,
                "w": image.width,
                "h": image.height,
                "dhash": make_dhash(image),
                "qnorm": re.sub(
                    r"[^a-z0-9 ]", "", question.lower()
                ).strip(),
                "answers_json": json.dumps(answers),
                "question_id": qid,
            }

        except Exception as exc:
            if attempt == 4:
                print(f"FAILED {item_id}: {exc}")
                return None
            time.sleep(3 * (attempt + 1))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    completed = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            completed = {
                row["id"] for row in csv.DictReader(f)
            }

    first = get_json(API, {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 1,
    })

    total = first["num_rows_total"]
    print(f"Dataset rows: {total}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {total - len(completed)}")

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for offset in range(0, total, PAGE_SIZE):
                page = get_json(API, {
                    "dataset": DATASET,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": min(PAGE_SIZE, total - offset),
                })

                pending = [
                    item for item in page["rows"]
                    if f"okvqa-{item['row']['question_id']}"
                    not in completed
                ]

                results = executor.map(download_one, pending)

                for result in results:
                    if result is not None:
                        writer.writerow(result)
                        f.flush()
                        completed.add(result["id"])

                print(
                    f"Progress: {len(completed)}/{total}",
                    flush=True,
                )

    print("\nDownload finished.")
    print(f"Successful records: {len(completed)}")
    print(f"Manifest: {CSV_PATH}")
    print(f"Images: {IMG_DIR}")


if __name__ == "__main__":
    main()
