"""Convert the splits into chat-format SFT files (same prompt as inference).

Usage:  python scripts/prepare_dataset.py
Outputs data/{train,validation,test}/sft.jsonl
"""
from __future__ import annotations

import _common  # noqa: F401
from ml.preprocessing.prepare_dataset import DATA_DIR, read_jsonl, to_sft_record, write_jsonl


def main() -> None:
    for split in ("train", "validation", "test"):
        src = DATA_DIR / split / f"{split}.jsonl"
        if not src.exists():
            raise SystemExit(f"{src} missing. Run scripts/create_splits.py first.")
        rows = read_jsonl(src)
        n = write_jsonl(DATA_DIR / split / "sft.jsonl", (to_sft_record(r) for r in rows))
        print(f"{split}: {n} chat examples -> {DATA_DIR / split / 'sft.jsonl'}")


if __name__ == "__main__":
    main()
