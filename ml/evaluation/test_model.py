"""Rewrite one text with an engine and print the quality metrics.

Usage:
  python -m ml.evaluation.test_model --engine baseline --text "Furthermore, it is important to note that ..."
  python -m ml.evaluation.test_model --engine model --style casual --intensity strong   # uses a test-split sample
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.evaluate import build_engine  # noqa: E402
from ml.evaluation.metrics import evaluate_pair, get_similarity  # noqa: E402
from ml.preprocessing.prepare_dataset import DATA_DIR, read_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default="baseline", choices=["baseline", "model"])
    p.add_argument("--text", default=None)
    p.add_argument("--style", default="natural")
    p.add_argument("--intensity", default="medium")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    text, reference = args.text, None
    if not text:
        test_file = DATA_DIR / "test" / "test.jsonl"
        if not test_file.exists():
            raise SystemExit("Pass --text or prepare the dataset first.")
        rec = read_jsonl(test_file)[0]
        text, reference = rec["input"], rec["output"]

    out = build_engine(args.engine, args.style, args.intensity)(text)
    sim = get_similarity(os.getenv("SIMILARITY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    print("INPUT:\n" + text + "\n\nOUTPUT:\n" + out + "\n")
    print(json.dumps(evaluate_pair(text, out, sim, reference=reference), indent=2))


if __name__ == "__main__":
    main()
