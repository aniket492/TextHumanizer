"""Prompting and generation for the rewrite model.

``build_messages`` is shared with training so fine-tuned adapters see the same format
at train and inference time.
"""
from __future__ import annotations

import logging
import re
import time

from ml.inference.model_loader import LoadedModel, load_model

logger = logging.getLogger("humanizer.predictor")

SYSTEM_PROMPT = (
    "You are a skilled human editor. Rewrite the user's passage so it reads like natural, "
    "fluent writing by a thoughtful person: vary sentence length, prefer plain words over "
    "inflated ones, and drop filler phrases and stacked transitions. "
    "Keep the meaning exactly. Keep every fact, name, number, date, citation, URL, quotation "
    "and technical term unchanged. Do not add new information, opinions or headings. "
    "Reply with the rewritten passage only."
)

STYLE_GUIDES = {
    "natural": "Natural: clear, relaxed everyday prose; contractions are fine.",
    "professional": "Professional: polished business register, confident and direct; no slang, no contractions.",
    "academic": "Academic: formal scholarly register, precise and measured; no contractions.",
    "casual": "Casual: friendly and conversational, like explaining to a friend; contractions welcome.",
    "simple": "Simple: short sentences and common words a 12-year-old could follow.",
    "concise": "Concise: remove redundancy and filler; make it noticeably shorter without losing content.",
    "detailed": "Detailed: slightly fuller explanations that clarify what is already stated, without inventing facts.",
}

INTENSITY_GUIDES = {
    "light": "Make light edits: keep most sentences, smooth only awkward or mechanical phrasing.",
    "medium": "Make moderate edits: rephrase sentences freely while keeping the paragraph structure.",
    "strong": "Rewrite thoroughly: restructure sentences and reorder clauses where it reads better.",
}


def build_messages(text: str, style: str = "natural", intensity: str = "medium") -> list[dict]:
    user = (
        f"Style - {STYLE_GUIDES.get(style, STYLE_GUIDES['natural'])}\n"
        f"Degree of change - {INTENSITY_GUIDES.get(intensity, INTENSITY_GUIDES['medium'])}\n\n"
        f"Passage:\n{text}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_seq2seq_prompt(text: str, style: str = "natural", intensity: str = "medium") -> str:
    return (
        f"Rewrite the passage in a {style} style ({intensity} changes). Keep all facts, names and "
        f"numbers. Passage: {text}"
    )


_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is) (?:the |a |your )?(?:rewritten|revised|edited)[^:\n]*:|"
    r"(?:rewritten|revised|edited) (?:passage|text|version)\s*:)\s*",
    re.IGNORECASE,
)
_TRAILING_NOTE = re.compile(r"\n\s*(?:\(?note|explanation|changes made)\b.*$", re.IGNORECASE | re.DOTALL)


def clean_generation(text: str) -> str:
    text = text.strip()
    text = _PREAMBLE.sub("", text)
    text = _TRAILING_NOTE.sub("", text)
    text = text.strip()
    if len(text) > 2 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text


class ModelTimeoutError(RuntimeError):
    pass


class Predictor:
    def __init__(self, name_or_path: str, device: str = "auto", max_new_tokens: int = 384):
        self.loaded: LoadedModel = load_model(name_or_path, device)
        self.max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return self.loaded.name

    def _encode(self, text: str, style: str, intensity: str):
        tok = self.loaded.tokenizer
        if self.loaded.kind == "causal" and getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(
                build_messages(text, style, intensity), tokenize=False, add_generation_prompt=True
            )
        elif self.loaded.kind == "causal":
            msgs = build_messages(text, style, intensity)
            prompt = f"{msgs[0]['content']}\n\n{msgs[1]['content']}\n\nRewritten passage:\n"
        else:
            prompt = build_seq2seq_prompt(text, style, intensity)
        return tok(prompt, return_tensors="pt").to(self.loaded.device)

    def generate(
        self,
        text: str,
        style: str = "natural",
        intensity: str = "medium",
        num_candidates: int = 2,
        max_time: float | None = None,
        seed: int = 13,
    ) -> list[str]:
        """Generate up to ``num_candidates`` rewrites (1 greedy + sampled alternatives)."""
        import torch

        tok = self.loaded.tokenizer
        model = self.loaded.model
        inputs = self._encode(text, style, intensity)
        input_len = inputs["input_ids"].shape[1]
        source_tokens = len(tok(text)["input_ids"])
        budget = min(self.max_new_tokens, int(source_tokens * (2.0 if style == "detailed" else 1.5)) + 24)
        eos_ids = {tok.eos_token_id}
        gen_eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        eos_ids.discard(None)
        eos_tensor = torch.tensor(sorted(eos_ids), device=self.loaded.device)
        common = dict(
            max_new_tokens=budget,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
            eos_token_id=sorted(eos_ids),
        )
        temperature = {"light": 0.5, "medium": 0.7, "strong": 0.9}.get(intensity, 0.7)
        start = time.perf_counter()
        outputs: list[str] = []
        with torch.inference_mode():
            # Clear sampling params inherited from the model's generation_config for the greedy pass.
            plans = [dict(do_sample=False, temperature=None, top_p=None, top_k=None)]
            if num_candidates > 1:
                plans.append(dict(do_sample=True, temperature=temperature, top_p=0.9, num_return_sequences=num_candidates - 1))
            for plan in plans:
                remaining = None if max_time is None else max_time - (time.perf_counter() - start)
                if remaining is not None and remaining <= 1.0:
                    break
                torch.manual_seed(seed)
                gen = model.generate(**inputs, **common, **plan, max_time=remaining)
                for seq in gen:
                    new_tokens = seq[input_len:] if self.loaded.kind == "causal" else seq
                    # Only accept generations that finished naturally (not cut by budget/max_time).
                    hit_eos = bool(torch.isin(new_tokens, eos_tensor).any())
                    decoded = clean_generation(tok.decode(new_tokens, skip_special_tokens=True))
                    if decoded and hit_eos:
                        outputs.append(decoded)
                    elif decoded:
                        logger.info("Discarding truncated generation (budget or time limit reached).")
        if not outputs and max_time is not None and time.perf_counter() - start >= max_time - 1.0:
            raise ModelTimeoutError(f"Generation exceeded {max_time:.0f}s")
        # De-duplicate while keeping order.
        seen: set[str] = set()
        return [o for o in outputs if not (o in seen or seen.add(o))]
