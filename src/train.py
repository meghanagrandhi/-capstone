"""QLoRA fine-tuning of the Qwen2-VL *language model* on a mixture. Vision tower and merger are frozen.
Logs: per-step loss split by text-answerable / image-necessary items, grad norm, lr; and at every checkpoint a
held-out probe (loss real/blank by class, mean attention mass on image tokens by class)."""
from __future__ import annotations
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .data import blank_like
from .ioutil import retry, safe_to_csv
from .vlm import build_train_example

LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_PARTS = ("visual", "vision_model", "vision_tower", "connector", "multi_modal_projector")  # Qwen2-VL, Idefics3, InternVL image modules


def attach_lora(vlm, r=16, alpha=32, dropout=0.05):
    """LoRA on the language-model linear layers only (names ending in LORA_SUFFIXES and not in the vision tower)."""
    from peft import LoraConfig, get_peft_model
    base = vlm.model
    names = [n for n, m in base.named_modules()
             if n.split(".")[-1] in LORA_SUFFIXES and not any(v in n for v in VISION_PARTS) and hasattr(m, "weight")]
    assert names, "no LoRA target modules found; check module names"
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=names, bias="none")
    pm = get_peft_model(base, cfg)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"LoRA: {len(names)} target modules (language model only), {n_train/1e6:.2f}M trainable params")
    vlm.model = pm
    return pm


def _example_loss(vlm, image, user, answer, max_ans):
    inp, n_p, a = build_train_example(vlm, image, user, answer, max_ans)
    logits = vlm.model(**inp).logits[0, n_p - 1: -1].float()
    return F.cross_entropy(logits, a)


@torch.no_grad()
def probe_metrics(vlm, probe: pd.DataFrame, gray=127, max_ans=96) -> dict:
    """Held-out probe: answer NLL with real and blank image, and mean attention mass on image tokens
    (answer-position queries, averaged over heads and layers), split by label. Attention needs the eager
    implementation (sdpa does not return weights); switched on only for this call."""
    pm = vlm.model
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    was_training = pm.training
    pm.eval()
    rows = []
    attn_ok = True
    try:
        base.set_attn_implementation("eager")
    except Exception as e:
        attn_ok = False
        print(f"  [probe] could not switch to eager attention ({type(e).__name__}); attention mass = NaN")
    img_id = base.config.image_token_id
    for r in probe.itertuples():
        img = Image.open(r.image_path).convert("RGB")
        out = dict(label=r.label)
        for cond, im in (("real", img), ("blank", blank_like(img, gray))):
            inp, n_p, a = build_train_example(vlm, im, r.user, r.answer, max_ans)
            if cond == "real" and attn_ok:
                try:
                    o = pm(**inp, output_attentions=True)
                    is_img = inp["input_ids"][0] == img_id
                    q = torch.arange(inp["input_ids"].shape[1], device=is_img.device) >= n_p - 1  # answer-predicting positions
                    m = [at[0][:, q][:, :, is_img].sum(-1).mean().item() for at in o.attentions if at is not None]
                    out["attn_img_mass"] = float(np.mean(m)) if m else float("nan")
                    logits = o.logits[0, n_p - 1: -1].float()
                except Exception as e:
                    attn_ok = False
                    print(f"  [probe] attention extraction failed ({type(e).__name__}: {str(e)[:80]}); attention mass = NaN")
                    logits = pm(**inp).logits[0, n_p - 1: -1].float()
            else:
                logits = pm(**inp).logits[0, n_p - 1: -1].float()
            out[f"nll_{cond}"] = F.cross_entropy(logits, a).item()
        rows.append(out)
    if attn_ok:
        base.set_attn_implementation("sdpa")
    if was_training:
        pm.train()
    df = pd.DataFrame(rows)
    res = {}
    for lab, g in df.groupby("label"):
        res[f"probe_nll_real_{lab}"] = g.nll_real.mean()
        res[f"probe_nll_blank_{lab}"] = g.nll_blank.mean()
        res[f"probe_gap_{lab}"] = (g.nll_blank - g.nll_real).mean()
        res[f"probe_attn_img_mass_{lab}"] = g.attn_img_mass.mean() if "attn_img_mass" in g else float("nan")
    return res


def ckpt_steps_for(total: int, pcts=(0, 5, 10, 20, 40, 60, 80, 100)) -> list[int]:
    return sorted({int(round(p / 100 * total)) for p in pcts})


def train(vlm, mix: pd.DataFrame, probe: pd.DataFrame, steps: int, bs: int, lr: float, seed: int, out_dir: str,
          max_ans: int = 96, warmup_frac: float = 0.05, clip: float = 1.0, gray: int = 127):
    """Identical schedule for every run: same steps, bs, lr, warmup, seed. Resumable at checkpoint granularity is NOT
    implemented (runs are short); a finished run leaves adapters/step_XXXX + train_log.csv + probe_log.csv."""
    os.makedirs(os.path.join(out_dir, "adapters"), exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n_ep = math.ceil(steps * bs / len(mix))
    order = np.concatenate([rng.permutation(len(mix)) for _ in range(n_ep)])
    pm = vlm.model
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(round(warmup_frac * steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    cks = set(ckpt_steps_for(steps))
    log, plog = [], []

    def checkpoint(step):
        retry(lambda: pm.save_pretrained(os.path.join(out_dir, "adapters", f"step_{step:04d}")), f"save adapter step {step}")
        m = probe_metrics(vlm, probe, gray, max_ans)
        m["step"] = step
        plog.append(m)
        safe_to_csv(pd.DataFrame(plog), os.path.join(out_dir, "probe_log.csv"))
        print(f"  [ckpt step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "step" and 'attn' in k or 'gap' in k), flush=True)

    pm.train()
    t0 = time.time()
    if 0 in cks:
        checkpoint(0)
    for step in range(1, steps + 1):
        losses = {"text_answerable": [], "image_necessary": []}
        for j in range(bs):
            r = mix.iloc[int(order[(step - 1) * bs + j])]
            img = Image.open(r.image_path).convert("RGB")
            loss = _example_loss(vlm, img, r.user, r.answer, max_ans)
            (loss / bs).backward()
            losses[r.label].append(loss.item())
        gn = float(torch.nn.utils.clip_grad_norm_(params, clip))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        allv = losses["text_answerable"] + losses["image_necessary"]
        log.append(dict(step=step, loss=float(np.mean(allv)),
                        loss_text_answerable=float(np.mean(losses["text_answerable"])) if losses["text_answerable"] else float("nan"),
                        loss_image_necessary=float(np.mean(losses["image_necessary"])) if losses["image_necessary"] else float("nan"),
                        n_text=len(losses["text_answerable"]), n_image=len(losses["image_necessary"]),
                        grad_norm=gn, lr=sched.get_last_lr()[0], elapsed_s=time.time() - t0))
        if step % 10 == 0 or step == steps:
            print(f"  step {step}/{steps} loss {log[-1]['loss']:.3f} gn {gn:.2f} {(time.time()-t0)/step:.1f}s/step", flush=True)
            safe_to_csv(pd.DataFrame(log), os.path.join(out_dir, "train_log.csv"))
        if step in cks:
            checkpoint(step)
    safe_to_csv(pd.DataFrame(log), os.path.join(out_dir, "train_log.csv"))
    return pd.DataFrame(log), pd.DataFrame(plog)
