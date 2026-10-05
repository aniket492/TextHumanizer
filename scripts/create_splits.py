"""Create deterministic train/validation/test splits from the cleaned data.

- test       = the dataset's official test split (keeps results comparable with the dataset card)
- validation = a stratified (by category) slice of the cleaned train split
- train      = the rest

Usage:  python scripts/create_splits.py [--val-fraction 0.1] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import _common  # noqa: F401
from ml.preprocessing.prepare_dataset import DATA_DIR, PROCESSED_DIR, SEED, read_jsonl, stratified_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_path, test_path = PROCESSED_DIR / "train_clean.jsonl", PROCESSED_DIR / "test_clean.jsonl"
    if not train_path.exists():
        raise SystemExit("Cleaned data missing. Run scripts/clean_dataset.py first.")
    train_all = read_jsonl(train_path)
    test = read_jsonl(test_path)
    train, validation = stratified_split(train_all, args.val_fraction, seed=args.seed)

    summary = {}
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_jsonl(DATA_DIR / name / f"{name}.jsonl", rows)
        summary[name] = {"rows": len(rows), "category": dict(Counter(r.get("category") for r in rows))}
        print(f"{name:>10}: {len(rows):5d} rows  {summary[name]['category']}")
    assert not ({r["fingerprint"] for r in train} & {r["fingerprint"] for r in validation}), "train/validation overlap"
    summary["seed"] = args.seed
    (PROCESSED_DIR / "splits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
