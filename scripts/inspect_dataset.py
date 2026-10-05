"""Inspect the raw (or a processed) dataset: structure, samples, missing values, duplicates, lengths.

Usage:  python scripts/inspect_dataset.py [--dir data/raw/gohumanize] [--samples 2]
Writes a JSON summary to reports/dataset_inspection.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from ml.preprocessing.normalize_text import word_count
from ml.preprocessing.prepare_dataset import PROJECT_ROOT, RAW_DIR, read_jsonl


def length_stats(values: list[int]) -> dict:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": int(arr.min()), "p5": float(np.percentile(arr, 5)), "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 1), "p95": float(np.percentile(arr, 95)), "max": int(arr.max()),
    }


def inspect_split(name: str, rows: list[dict], n_samples: int) -> dict:
    columns = sorted({k for r in rows for k in r})
    missing = {c: sum(1 for r in rows if r.get(c) in (None, "", [])) for c in columns}
    inputs = [" ".join(str(r.get("input") or "").split()) for r in rows]
    outputs = [" ".join(str(r.get("output") or "").split()) for r in rows]
    pairs = Counter(zip(inputs, outputs))
    report = {
        "rows": len(rows),
        "columns": columns,
        "missing_values": {c: n for c, n in missing.items() if n},
        "duplicate_pairs": sum(c - 1 for c in pairs.values() if c > 1),
        "duplicate_inputs": sum(c - 1 for c in Counter(inputs).values() if c > 1),
        "duplicate_outputs": sum(c - 1 for c in Counter(outputs).values() if c > 1),
        "identical_input_output": sum(1 for a, b in zip(inputs, outputs) if a and a == b),
        "input_words": length_stats([word_count(t) for t in inputs]),
        "output_words": length_stats([word_count(t) for t in outputs]),
        "input_chars": length_stats([len(t) for t in inputs]),
        "category": dict(Counter(str(r.get("category")) for r in rows)),
        "generator": dict(Counter(str(r.get("generator")) for r in rows)),
        "source": dict(Counter(str(r.get("source")) for r in rows)),
    }
    print(f"\n=== {name}: {len(rows)} records ===")
    print("Columns:", ", ".join(columns))
    for key in ("missing_values", "duplicate_pairs", "duplicate_inputs", "duplicate_outputs", "identical_input_output",
                "input_words", "output_words", "category", "generator", "source"):
        print(f"{key:>24}: {report[key]}")
    for i, row in enumerate(rows[:n_samples]):
        print(f"\n--- sample {i + 1} (id={row.get('id')}, category={row.get('category')}, style={row.get('style')}) ---")
        print("INPUT :", (row.get("input") or "")[:500])
        print("OUTPUT:", (row.get("output") or "")[:500])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=RAW_DIR)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in {args.dir}. Run scripts/download_dataset.py first.")
    summary = {}
    all_outputs: dict[str, set] = {}
    for f in files:
        rows = read_jsonl(f)
        summary[f.stem] = inspect_split(f.stem, rows, args.samples)
        all_outputs[f.stem] = {" ".join(str(r.get("output") or "").split()) for r in rows}
    names = list(all_outputs)
    overlap = {f"{a}&{b}": len(all_outputs[a] & all_outputs[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    summary["cross_split_output_overlap"] = overlap
    print("\nCross-split output overlap:", overlap)

    out = PROJECT_ROOT / "reports" / "dataset_inspection.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
