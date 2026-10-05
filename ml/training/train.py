"""CLI for LoRA fine-tuning.

Examples:
  # Smoke test on CPU (a few steps, proves the pipeline end to end)
  python -m ml.training.train --max-steps 2 --max-train-samples 8 --output-dir models/checkpoints/smoke

  # Full run (GPU strongly recommended)
  python -m ml.training.train --epochs 2

Then serve it with MODEL_PATH=models/final (the adapter is merged into its base at load time).
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.config import TrainConfig  # noqa: E402
from ml.training.trainer import train  # noqa: E402


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=cfg.base_model)
    p.add_argument("--train-file", type=Path, default=cfg.train_file)
    p.add_argument("--validation-file", type=Path, default=cfg.validation_file)
    p.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    p.add_argument("--epochs", type=float, default=cfg.num_epochs)
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--batch-size", type=int, default=cfg.per_device_batch_size)
    p.add_argument("--grad-accum", type=int, default=cfg.gradient_accumulation_steps)
    p.add_argument("--max-seq-length", type=int, default=cfg.max_seq_length)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=cfg.max_eval_samples)
    p.add_argument("--lora-r", type=int, default=cfg.lora_r)
    p.add_argument("--eval-steps", type=int, default=cfg.eval_steps)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    a = p.parse_args()
    cfg.base_model = a.base_model
    cfg.train_file, cfg.validation_file = a.train_file, a.validation_file
    cfg.output_dir = a.output_dir if a.output_dir.is_absolute() else PROJECT_ROOT / a.output_dir
    cfg.num_epochs, cfg.max_steps, cfg.learning_rate = a.epochs, a.max_steps, a.lr
    cfg.per_device_batch_size, cfg.gradient_accumulation_steps = a.batch_size, a.grad_accum
    cfg.max_seq_length, cfg.max_train_samples, cfg.max_eval_samples = a.max_seq_length, a.max_train_samples, a.max_eval_samples
    cfg.lora_r, cfg.lora_alpha = a.lora_r, a.lora_r * 2
    cfg.eval_steps, cfg.seed = a.eval_steps, a.seed
    cfg.gradient_checkpointing = not a.no_gradient_checkpointing
    if cfg.max_steps > 0:
        cfg.eval_steps = min(cfg.eval_steps, cfg.max_steps)
        cfg.logging_steps = 1
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse_args()
    logging.info("Config: %s", {f.name: getattr(cfg, f.name) for f in fields(cfg)})
    out = train(cfg)
    print(f"\nDone. Serve with: MODEL_PATH={out}")


if __name__ == "__main__":
    main()
