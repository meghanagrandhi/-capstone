
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

DATASET = "Multimodal-Fatima/TextVQA_train"
API = "https://datasets-server.huggingface.co/rows"
OUT = r"G:\Backup D\Capstone\datasets\TextVQA"
IMG_DIR = os.path.join(OUT, "images")
CSV_PATH = os.path.join(OUT, "pool.csv")

PAGE_SIZE = 100
WORKERS = 8

FIELDS = [
    "id", "subset", "row", "user", "answer", "image_path",
    "w", "h", "dhash", "qnorm", "answers_json",
    "question_id", "image_id"
]



def get_json(params):
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, timeout=120)

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(15 * (2 ** attempt), 300)
                print(
                    f"API HTTP {r.status_code}; retrying in {wait}s "
                    f"(attempt {attempt + 1}/8)",
                    flush=True,
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as exc:
            if attempt == 7:
                raise
            wait = min(15 * (2 ** attempt), 300)
            print(f"API error: {exc}; retrying in {wait}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("Hugging Face API unavailable after 8 attempts")



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
    normalized = [
        str(a).strip().lower()
        for a in answers if str(a).strip()
    ]
    if not normalized:
        raise ValueError("No usable answers")

    winner = Counter(normalized).most_common(1)[0][0]
    return next(
        str(a).strip() for a in answers
        if str(a).strip().lower() == winner
    )


def download_image(image_id, image_url):
    path = os.path.join(IMG_DIR, f"{image_id}.jpg")

    if os.path.exists(path):
        try:
            with Image.open(path) as im:
                im.verify()
            return path
        except Exception:
            os.remove(path)

    for attempt in range(5):
        try:
            r = requests.get(image_url, timeout=90)
            r.raise_for_status()

            im = Image.open(io.BytesIO(r.content))
            im.load()
            im = im.convert("RGB")
            im.save(path, format="JPEG", quality=95)
            return path

        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    completed = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            completed = {
                row["id"] for row in csv.DictReader(f)
            }

    first = get_json({
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 1,
    })

    total = first["num_rows_total"]
    print(f"Training questions: {total}")
    print(f"Already completed: {len(completed)}")

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            start_offset = max(
                (int(x.split("-", 1)[1]) for x in completed),
                default=-1
            ) + 1

            print(f"Resuming from API offset: {start_offset}")

            for offset in range(start_offset, total, PAGE_SIZE):
            
                page = get_json({
                    "dataset": DATASET,
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": min(PAGE_SIZE, total - offset),
                })

                rows = []
                for item in page["rows"]:
                    row = item["row"]
                    qid = str(row["question_id"])
                    item_id = f"textvqa-{qid}"

                    if item_id not in completed:
                        rows.append((item, item_id))

                # Download each distinct image only once per page.
                image_urls = {}
                for item, _ in rows:
                    row = item["row"]
                    image_urls[str(row["image_id"])] = row["image"]["src"]

                futures = {
                    image_id: executor.submit(
                        download_image, image_id, url
                    )
                    for image_id, url in image_urls.items()
                }

                image_paths = {}
                for image_id, future in futures.items():
                    image_paths[image_id] = future.result()

                records = []
                for item, item_id in rows:
                    row = item["row"]
                    image_id = str(row["image_id"])
                    path = image_paths[image_id]

                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        width, height = im.size
                        dhash = make_dhash(im)

                    question = str(row["question"]).strip()
                    answers = row["answers"]

                    records.append({
                        "id": item_id,
                        "subset": "textvqa",
                        "row": item["row_idx"],
                        "user": question,
                        "answer": majority_answer(answers),
                        "image_path": path,
                        "w": width,
                        "h": height,
                        "dhash": dhash,
                        "qnorm": re.sub(
                            r"[^a-z0-9 ]", "",
                            question.lower()
                        ).strip(),
                        "answers_json": json.dumps(answers),
                        "question_id": row["question_id"],
                        "image_id": image_id,
                    })

                for record in records:
                    writer.writerow(record)
                    completed.add(record["id"])

                f.flush()
                print(
                    f"Progress: {len(completed)}/{total} questions",
                    flush=True,
                )

    print("\nTextVQA download finished.")
    print(f"Questions: {len(completed)}")
    print(f"Manifest: {CSV_PATH}")
    print(f"Images: {IMG_DIR}")


if __name__ == "__main__":
    main()
