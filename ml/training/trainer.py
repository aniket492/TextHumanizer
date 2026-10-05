"""LoRA supervised fine-tuning on chat-formatted humanization pairs.

Loss is computed on the assistant (target) tokens only; the system/user prompt is masked.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch

from ml.preprocessing.prepare_dataset import read_jsonl
from ml.training.config import TrainConfig

logger = logging.getLogger("humanizer.trainer")
IGNORE_INDEX = -100


def tokenize_example(messages: list[dict], tokenizer, max_len: int) -> dict | None:
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(prompt_ids, "input_ids"):  # some tokenizers return BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) > max_len:
        return None  # template mismatch or too long: skip rather than truncate the target
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int, limit: int | None = None):
        rows = read_jsonl(path)
        if limit:
            rows = rows[:limit]
        self.items = [ex for ex in (tokenize_example(r["messages"], tokenizer, max_len) for r in rows) if ex]
        self.skipped = len(rows) - len(self.items)
        logger.info("%s: %d examples (%d skipped as too long)", path.name, len(self.items), self.skipped)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train(cfg: TrainConfig) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none",
        ),
    )
    model.print_trainable_parameters()

    train_ds = SFTDataset(cfg.train_file, tokenizer, cfg.max_seq_length, cfg.max_train_samples)
    eval_ds = SFTDataset(cfg.validation_file, tokenizer, cfg.max_seq_length, cfg.max_eval_samples)
    if len(train_ds) == 0:
        raise SystemExit("No training examples fit max_seq_length; increase --max-seq-length.")

    args = TrainingArguments(
        output_dir=str(cfg.checkpoint_dir),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if len(eval_ds) else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        report_to=[],
        seed=cfg.seed,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds if len(eval_ds) else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    metrics = {"train": result.metrics}
    if len(eval_ds):
        metrics["eval"] = trainer.evaluate()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))  # LoRA adapter (adapter_config.json + weights)
    tokenizer.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.tokenizer_dir))
    cfg_dump = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    (cfg.output_dir / "training_summary.json").write_text(
        json.dumps({"config": cfg_dump, "metrics": metrics, "train_examples": len(train_ds),
                    "skipped_too_long": train_ds.skipped}, indent=2),
        encoding="utf-8",
    )
    logger.info("Adapter saved to %s", cfg.output_dir)
    return cfg.output_dir
