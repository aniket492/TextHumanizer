"""Clean the raw dataset into data/processed/ (raw files are only read, never modified).

Steps: whitespace normalisation (punctuation and wording preserved), removal of empty,
corrupted, invalid and duplicate pairs, and removal of train records that leak into test.

Usage:  python scripts/clean_dataset.py
"""
from __future__ import annotations

import json

import _common  # noqa: F401
from ml.preprocessing.prepare_dataset import PROCESSED_DIR, RAW_DIR, clean_records, read_jsonl, remove_leakage, write_jsonl


def main() -> None:
    if not (RAW_DIR / "train.jsonl").exists():
        raise SystemExit("Raw data missing. Run scripts/download_dataset.py first.")
    report = {}
    cleaned = {}
    for split in ("train", "test"):
        raw = read_jsonl(RAW_DIR / f"{split}.jsonl")
        kept, dropped = clean_records(raw, split)
        cleaned[split] = kept
        report[split] = {"raw": len(raw), "kept": len(kept), "dropped": dict(dropped)}

    cleaned["train"], leaked = remove_leakage(cleaned["train"], cleaned["test"])
    report["train"]["dropped"]["leaked_into_test"] = leaked
    report["train"]["kept"] = len(cleaned["train"])

    for split, rows in cleaned.items():
        n = write_jsonl(PROCESSED_DIR / f"{split}_clean.jsonl", rows)
        print(f"{split}: kept {n} / {report[split]['raw']}  dropped={report[split]['dropped']}")
    (PROCESSED_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {PROCESSED_DIR / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
