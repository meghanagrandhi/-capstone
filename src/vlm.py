"""Swappable VLM interface + Qwen2-VL wrapper (4-bit, sdpa)."""
from __future__ import annotations
import abc
import os

# transformers 5.x loads weights with a 4-thread pool; on this Windows machine that intermittently segfaults
# (access violation reading the memory-mapped safetensors). Single-threaded loading is stable and only ~24 s.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
import numpy as np
import torch

LETTERS = "ABCDEFGH"


class VLM(abc.ABC):
    @abc.abstractmethod
    def load(self) -> "VLM": ...

    @abc.abstractmethod
    def score_options(self, image, question: str, options: list[str]) -> np.ndarray:
        """Log-prob (full-vocab log-softmax) of each option letter as the next token. shape (n_options,)."""

    @abc.abstractmethod
    def answer_nll(self, image, prompt: str, answer: str) -> tuple[float, int]:
        """(mean NLL per answer token, n_answer_tokens) of `answer` given image+prompt."""


class Qwen2VL(VLM):
    end_token = "<|im_end|>"  # appended after training answers

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None

    def load(self):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
        c = self.cfg
        kw = dict(attn_implementation=c.get("attn_implementation", "sdpa"), device_map={"": 0}, revision=c.get("revision"))
        if c.get("load_4bit", True):
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(c["hf_id"], **kw).eval()
        self.processor = AutoProcessor.from_pretrained(
            c["hf_id"], revision=c.get("revision"), min_pixels=c["min_pixels"], max_pixels=c["max_pixels"])
        self._set_letters()
        return self

    def _set_letters(self, prefix=""):
        self.tok = self.processor.tokenizer
        ids = [self.tok.encode(prefix + l, add_special_tokens=False) for l in LETTERS]
        assert all(len(i) == 1 for i in ids), f"letters must be single tokens: {ids}"
        self.letter_ids = [i[0] for i in ids]

    # ---- helpers -------------------------------------------------------
    def _inputs(self, image, text: str, reply_prefix: str = ""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + reply_prefix
        inp = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inp.items()}

    def n_image_tokens(self, image) -> int:
        inp = self._inputs(image, "x")
        return int((inp["input_ids"] == self.model.config.image_token_id).sum())

    @torch.no_grad()
    def score_options(self, image, question, options):
        mo = self.cfg.get("max_option_chars", 400)
        opts = "\n".join(f"{LETTERS[i]}. {o[:mo]}" for i, o in enumerate(options))
        text = f"{question}\n{opts}\nAnswer with the option's letter from the given choices directly."
        inp = self._inputs(image, text, self.cfg.get("score_reply_prefix", ""))
        logits = self.model(**inp).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return lp[self.letter_ids[: len(options)]].cpu().numpy()

    @torch.no_grad()
    def answer_nll(self, image, prompt, answer):
        inp = self._inputs(image, prompt)
        a = self.tok.encode(answer, add_special_tokens=False)[: self.cfg.get("max_answer_tokens", 64)]
        a_t = torch.tensor([a], device=self.model.device)
        n_p = inp["input_ids"].shape[1]
        for k, v in list(inp.items()):  # other per-token tensors (e.g. mm_token_type_ids in transformers 5.x)
            if k not in ("input_ids", "attention_mask") and v.dim() == 2 and v.shape[1] == n_p:
                inp[k] = torch.cat([v, torch.zeros_like(a_t, dtype=v.dtype)], 1)
        inp["input_ids"] = torch.cat([inp["input_ids"], a_t], 1)
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        logits = self.model(**inp).logits[0, n_p - 1: -1].float()
        nll = -torch.log_softmax(logits, -1).gather(1, a_t[0][:, None]).squeeze(1)
        return float(nll.mean()), len(a)


def build_train_example(vlm: "Qwen2VL", image, prompt: str, answer: str, max_answer_tokens: int = 96):
    """Chat prompt (image + prompt) followed by answer tokens + the model's end-of-turn token. Returns (inputs, n_prompt_tokens, answer_ids).
    Loss is taken only on the answer positions logits[n_p-1:-1] vs answer_ids."""
    inp = vlm._inputs(image, prompt)
    end_id = vlm.tok.convert_tokens_to_ids(vlm.end_token)
    ids = vlm.tok.encode(answer, add_special_tokens=False)[: max_answer_tokens - 1] + [end_id]
    a_t = torch.tensor([ids], device=vlm.model.device)
    n_p = inp["input_ids"].shape[1]
    for k, v in list(inp.items()):  # other per-token tensors (e.g. mm_token_type_ids in transformers 5.x)
        if k not in ("input_ids", "attention_mask") and v.dim() == 2 and v.shape[1] == n_p:
            inp[k] = torch.cat([v, torch.zeros_like(a_t, dtype=v.dtype)], 1)
    inp["input_ids"] = torch.cat([inp["input_ids"], a_t], 1)
    inp["attention_mask"] = torch.ones_like(inp["input_ids"])
    return inp, n_p, a_t[0]


class SmolVLM(Qwen2VL):
    """HuggingFaceTB/SmolVLM-Instruct (Idefics3: SigLIP + SmolLM2-1.7B). Same scoring/NLL code as Qwen2VL; differs in
    model class, image budget (longest_edge -> tiles of 384 px, 81 tokens each + 1 global tile) and chat format,
    whose generation prompt ends in "Assistant:" so the answer letter comes as " A" (leading space)."""
    end_token = "<end_of_utterance>"

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        c = self.cfg
        kw = dict(attn_implementation=c.get("attn_implementation", "sdpa"), device_map={"": 0}, revision=c.get("revision"))
        if c.get("load_4bit", True):
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(c["hf_id"], **kw).eval()
        self.processor = AutoProcessor.from_pretrained(c["hf_id"], revision=c.get("revision"), **self._processor_kw())
        self._set_letters(c.get("letter_prefix", " "))
        return self

    def _processor_kw(self):
        return {"size": {"longest_edge": self.cfg["longest_edge"]}}


class InternVL(SmolVLM):
    """OpenGVLab/InternVL3-2B-hf (InternViT-300M + Qwen2.5-1.5B). Qwen-style chat format: the answer letter follows
    "<|im_start|>assistant\\n" directly (no leading space) and turns end in <|im_end|>. One 448 px tile
    (crop_to_patches=False) = 256 image tokens, the same budget as Qwen2-VL here."""
    end_token = "<|im_end|>"

    def _processor_kw(self):
        return {"crop_to_patches": self.cfg.get("crop_to_patches", False)}


def select_model(cfg: dict, model: str) -> str:
    """--model switch: puts cfg["model_<model>"] in cfg["model"] (in place). Returns the run-name prefix ("" for qwen2vl)."""
    if model == "qwen2vl":
        return ""
    cfg["model"] = cfg[f"model_{model}"]
    return f"{model}_"


def make_model(cfg: dict) -> VLM:
    if cfg["name"] == "qwen2vl":
        return Qwen2VL(cfg)
    if cfg["name"] == "smolvlm":
        return SmolVLM(cfg)
    if cfg["name"] == "internvl":
        return InternVL(cfg)
    raise ValueError(f"unknown model {cfg['name']}; add a wrapper implementing VLM")
