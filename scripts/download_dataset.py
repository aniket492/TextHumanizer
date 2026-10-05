"""Download the GoHumanize Open Humanizer Dataset (CC-BY 4.0) into data/raw/.

Usage:  python scripts/download_dataset.py [--revision SHA] [--force]

The raw files are written once and never modified by later pipeline steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import _common  # noqa: F401
from ml.preprocessing.prepare_dataset import DATASET_ID, DATASET_REVISION, RAW_DIR


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DATASET_REVISION, help="Dataset git revision (pinned for reproducibility).")
    parser.add_argument("--force", action="store_true", help="Re-download even if raw files exist.")
    args = parser.parse_args()

    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Raw dataset already present at {RAW_DIR} (use --force to re-download).")
        print(manifest_path.read_text(encoding="utf-8"))
        return

    from datasets import load_dataset

    print(f"Downloading {DATASET_ID}@{args.revision[:10]} ...")
    ds = load_dataset(DATASET_ID, revision=args.revision)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, data in ds.items():
        out = RAW_DIR / f"{split}.jsonl"
        data.to_json(str(out), lines=True, force_ascii=False)
        files[split] = {"file": out.name, "rows": data.num_rows, "sha256": sha256(out), "columns": data.column_names}
        print(f"  {split}: {data.num_rows} rows -> {out}")

    manifest = {
        "dataset": DATASET_ID,
        "url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": args.revision,
        "license": "CC-BY-4.0",
        "citation": "GoHumanize team (2026). GoHumanize Open Humanizer Dataset. https://doi.org/10.5281/zenodo.22843083",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "splits": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
