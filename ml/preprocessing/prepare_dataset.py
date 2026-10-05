"""Dataset cleaning, splitting and SFT formatting (library code used by scripts/)."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from ml.preprocessing.clean_text import clean_text, pair_problem
from ml.preprocessing.normalize_text import word_count

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_ID = "gohumanize/gohumanize-open-humanizer-dataset"
DATASET_REVISION = "badba6c07f9bc32fd6d0f827e32f7787f6275be4"
RAW_DIR = DATA_DIR / "raw" / "gohumanize"
PROCESSED_DIR = DATA_DIR / "processed"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _fingerprint(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def clean_records(records: list[dict], split: str) -> tuple[list[dict], Counter]:
    """Clean one split. Never mutates the input records."""
    kept: list[dict] = []
    dropped: Counter = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    seen_inputs: set[str] = set()
    seen_outputs: set[str] = set()
    for rec in records:
        src = clean_text(rec.get("input"))
        tgt = clean_text(rec.get("output"))
        if not src or not tgt:
            dropped["empty"] += 1
            continue
        problem = pair_problem(src, tgt)
        if problem:
            dropped[problem] += 1
            continue
        fp_in, fp_out = _fingerprint(src), _fingerprint(tgt)
        if (fp_in, fp_out) in seen_pairs:
            dropped["duplicate_pair"] += 1
            continue
        if fp_in in seen_inputs:
            dropped["duplicate_input"] += 1
            continue
        if fp_out in seen_outputs:
            dropped["duplicate_output"] += 1
            continue
        seen_pairs.add((fp_in, fp_out))
        seen_inputs.add(fp_in)
        seen_outputs.add(fp_out)
        new = {k: v for k, v in rec.items() if k not in {"input", "output"}}
        new.update(
            input=src,
            output=tgt,
            split_origin=split,
            input_words=word_count(src),
            output_words=word_count(tgt),
            fingerprint=fp_out,
        )
        kept.append(new)
    return kept, dropped


def remove_leakage(train: list[dict], test: list[dict]) -> tuple[list[dict], int]:
    """Drop train records whose input or output text also appears in the test split."""
    test_fps = {r["fingerprint"] for r in test} | {_fingerprint(r["input"]) for r in test}
    filtered = [r for r in train if r["fingerprint"] not in test_fps and _fingerprint(r["input"]) not in test_fps]
    return filtered, len(train) - len(filtered)


def stratified_split(records: list[dict], fraction: float, key: str = "category", seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Deterministically split records into (keep, holdout) preserving the ``key`` distribution."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in sorted(records, key=lambda r: r["fingerprint"]):
        groups[str(rec.get(key, "unknown"))].append(rec)
    rng = random.Random(seed)
    keep, holdout = [], []
    for name in sorted(groups):
        items = groups[name][:]
        rng.shuffle(items)
        n_hold = max(1, round(len(items) * fraction)) if len(items) > 1 else 0
        holdout.extend(items[:n_hold])
        keep.extend(items[n_hold:])
    rng.shuffle(keep)
    rng.shuffle(holdout)
    return keep, holdout


def to_sft_record(rec: dict, style: str = "natural", intensity: str = "medium") -> dict:
    """Chat-format training example using the exact inference prompt."""
    from ml.inference.predictor import build_messages

    messages = build_messages(rec["input"], style, intensity)
    messages.append({"role": "assistant", "content": rec["output"]})
    return {"id": rec.get("id"), "category": rec.get("category"), "messages": messages}
