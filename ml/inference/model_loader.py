"""Load a Hugging Face rewrite model (causal or seq2seq), optionally with a LoRA adapter."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("humanizer.model_loader")


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    kind: str  # "causal" or "seq2seq"
    device: str
    name: str
    load_seconds: float


def resolve_device(preference: str = "auto") -> str:
    import torch

    pref = (preference or "auto").lower()
    if pref != "auto":
        if pref.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; using CPU.")
            return "cpu"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _adapter_base(path: Path) -> str | None:
    cfg = path / "adapter_config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path")
    return None


def load_model(name_or_path: str, device: str = "auto") -> LoadedModel:
    """Load a model by Hub id or local path. A local LoRA adapter dir is merged into its base."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    start = time.perf_counter()
    device = resolve_device(device)
    path = Path(name_or_path)
    adapter_base = _adapter_base(path) if path.is_dir() else None
    base_id = adapter_base or name_or_path

    config = AutoConfig.from_pretrained(base_id)
    kind = "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model_cls = AutoModelForSeq2SeqLM if kind == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(base_id, dtype=dtype, low_cpu_mem_usage=True)
    tokenizer_src = name_or_path if (path / "tokenizer_config.json").is_file() else base_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    if adapter_base:
        from peft import PeftModel

        logger.info("Applying LoRA adapter from %s onto %s", path, adapter_base)
        model = PeftModel.from_pretrained(model, str(path))
        model = model.merge_and_unload()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if kind == "causal":
        tokenizer.padding_side = "left"

    model.to(device)
    model.eval()
    elapsed = time.perf_counter() - start
    logger.info("Loaded %s (%s) on %s in %.1fs", name_or_path, kind, device, elapsed)
    return LoadedModel(model=model, tokenizer=tokenizer, kind=kind, device=device, name=name_or_path, load_seconds=elapsed)
