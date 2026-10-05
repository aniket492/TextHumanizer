"""Fine-tuning configuration (LoRA/PEFT). Every field can be overridden from the CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    base_model: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    train_file: Path = PROJECT_ROOT / "data" / "train" / "sft.jsonl"
    validation_file: Path = PROJECT_ROOT / "data" / "validation" / "sft.jsonl"
    output_dir: Path = PROJECT_ROOT / "models" / "final"
    checkpoint_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    tokenizer_dir: Path = PROJECT_ROOT / "models" / "tokenizer"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Optimisation
    learning_rate: float = 2e-4
    num_epochs: float = 2.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_seq_length: int = 1024
    max_steps: int = -1
    max_train_samples: int | None = None
    max_eval_samples: int | None = 200
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 42
    gradient_checkpointing: bool = True
