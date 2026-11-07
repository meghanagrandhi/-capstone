"""Smoke test for a non-default model (GPU, ~3 min): loads it, checks image-token budget, that the answer letter is predicted
the way we score it, option scoring on real vs blank images, and answer NLL.
Usage: python tests/smoke_model.py smolvlm|internvl"""
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import blank_like, load_mmstar
from src.vlm import LETTERS, make_model, select_model

cfg = yaml.safe_load(open("configs/config.yaml"))
select_model(cfg, sys.argv[1])
t0 = time.time()
vlm = make_model(cfg["model"]).load()
print(f"loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
items = load_mmstar(cfg["tier1"], 12, 0)
it = items[0]
print("image tokens:", vlm.n_image_tokens(it["image"]), "image size:", it["image"].size)
print("letter ids:", {L: vlm.tok.decode([i]) for L, i in zip(LETTERS[:4], vlm.letter_ids)})

# what does the model actually generate first? (should be one of the scored letter tokens)
opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
inp = vlm._inputs(it["image"], f"{it['question']}\n{opts}\nAnswer with the option's letter from the given choices directly.",
                  cfg["model"].get("score_reply_prefix", ""))
print("prompt tail:", repr(vlm.processor.tokenizer.decode(inp["input_ids"][0, -12:])))
with torch.no_grad():
    logits = vlm.model(**inp).logits[0, -1].float()
top = torch.topk(torch.log_softmax(logits, -1), 5)
print("top-5 next tokens:", [(repr(vlm.tok.decode([i])), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
mass = float(torch.softmax(logits, -1)[vlm.letter_ids[:4]].sum())
print(f"prob mass on scored letters: {mass:.3f}")
assert mass > 0.5, "scored letter tokens do not carry most of the probability -> fix letter_prefix"

acc = {"real": [], "blank": []}
t0 = time.time()
for it in items:
    for cond in acc:
        img = it["image"] if cond == "real" else blank_like(it["image"], 127)
        acc[cond].append(int(np.argmax(vlm.score_options(img, it["question"], it["options"])) == it["answer"]))
print(f"12 items: real acc {np.mean(acc['real']):.2f}, blank acc {np.mean(acc['blank']):.2f}, "
      f"{(time.time() - t0) / 24:.2f} s/item")
print("answer NLL:", vlm.answer_nll(items[0]["image"], "Describe the image.", "A picture."))
print("end token id:", vlm.tok.convert_tokens_to_ids(vlm.end_token))
print("SMOKE OK")
