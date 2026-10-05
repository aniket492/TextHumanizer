"""Evaluate rewrite engines on the held-out test split and write a report.

Engines:
  identity  - returns the input unchanged (control row)
  baseline  - rule-based rewriter (CPU, instant)
  model     - the configured LLM (MODEL_PATH or MODEL_NAME); slow on CPU, use --limit

Metrics are transparent writing-quality measures (meaning preservation, key-item preservation,
factual consistency, grammar, readability, repetition, stock-phrase rate, length) plus similarity to
the human-written reference. No AI detector is used; detector scores are probabilistic signals,
not ground truth, and are deliberately not part of this report.

Usage:
  python -m ml.evaluation.evaluate --engines identity baseline --limit 294
  python -m ml.evaluation.evaluate --engines model --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.metrics import evaluate_pair, get_similarity, safe_mean  # noqa: E402
from ml.inference.baseline import BaselineRewriter  # noqa: E402
from ml.preprocessing.prepare_dataset import DATA_DIR, read_jsonl  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports"
AGG_KEYS = [
    "semantic_similarity", "entity_preservation", "factual_consistency", "grammar_score",
    "readability_before", "readability_after", "repetition_before", "repetition_after",
    "stock_phrases_before", "stock_phrases_after", "length_ratio", "reference_similarity",
    "sentence_length_cv_before", "sentence_length_cv_after",
]


def passes(m: dict, min_sim: float) -> bool:
    return m["semantic_similarity"] >= min_sim and m["factual_consistency"] >= 0.85 and m["grammar_score"] >= 0.5


def build_engine(name: str, style: str, intensity: str):
    if name == "identity":
        return lambda text: text
    if name == "baseline":
        rw = BaselineRewriter()
        return lambda text: rw.rewrite(text, style, intensity)[0]
    if name == "model":
        from ml.inference.predictor import Predictor

        model_id = os.getenv("MODEL_PATH") or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        predictor = Predictor(model_id, device=os.getenv("MODEL_DEVICE", "auto"))

        def run(text: str) -> str:
            from ml.inference.predictor import ModelTimeoutError

            try:
                outs = predictor.generate(text, style, intensity, num_candidates=1, max_time=240)
            except ModelTimeoutError:
                run.timeouts += 1  # type: ignore[attr-defined]
                return text  # counted as a failed record below
            return outs[0] if outs else text

        run.timeouts = 0  # type: ignore[attr-defined]

        run.model_id = model_id  # type: ignore[attr-defined]
        return run
    raise ValueError(f"Unknown engine {name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines", nargs="+", default=["identity", "baseline"], choices=["identity", "baseline", "model"])
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--style", default="natural")
    p.add_argument("--intensity", default="medium")
    p.add_argument("--similarity-model", default=os.getenv("SIMILARITY_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    p.add_argument("--min-similarity", type=float, default=0.82)
    p.add_argument("--tag", default="", help="Suffix for the report file names.")
    args = p.parse_args()

    path = DATA_DIR / args.split / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing. Run the dataset scripts first (see README).")
    rows = read_jsonl(path)[: args.limit]
    similarity = get_similarity(args.similarity_model)
    print(f"Evaluating {len(rows)} {args.split} records; similarity={similarity.method}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "records": len(rows), "style": args.style, "intensity": args.intensity,
        "similarity_method": similarity.method, "min_similarity": args.min_similarity, "engines": {},
        "note": "Quality metrics only. No AI-detector scores are reported; such scores are probabilistic, not ground truth.",
    }
    for name in args.engines:
        engine = build_engine(name, args.style, args.intensity)
        per_record, start = [], time.perf_counter()
        for i, rec in enumerate(rows, 1):
            timeouts_before = getattr(engine, "timeouts", 0)
            out = engine(rec["input"])
            m = evaluate_pair(rec["input"], out, similarity, reference=rec["output"])
            m.pop("grammar_issues", None)
            timed_out = getattr(engine, "timeouts", 0) > timeouts_before
            per_record.append({"id": rec.get("id"), "category": rec.get("category"), **m, "timed_out": timed_out,
                               "passed": passes(m, args.min_similarity) and not timed_out})
            if name == "model" and (i % 5 == 0 or i == len(rows)):
                print(f"  [{name}] {i}/{len(rows)}", flush=True)
        elapsed = time.perf_counter() - start
        summary = {k: round(safe_mean([r[k] for r in per_record if k in r]), 4) for k in AGG_KEYS}
        summary["pass_rate"] = round(sum(r["passed"] for r in per_record) / max(len(per_record), 1), 4)
        summary["fully_preserved_key_items"] = round(
            sum(r["entity_preservation"] == 1.0 for r in per_record) / max(len(per_record), 1), 4)
        summary["seconds_per_record"] = round(elapsed / max(len(per_record), 1), 3)
        summary["timeouts"] = getattr(engine, "timeouts", 0)
        by_cat = {}
        for cat in sorted({r["category"] for r in per_record}):
            subset = [r for r in per_record if r["category"] == cat]
            by_cat[cat] = {k: round(safe_mean([r[k] for r in subset]), 4)
                           for k in ("semantic_similarity", "reference_similarity", "stock_phrases_after")}
        report["engines"][name] = {
            "model": getattr(engine, "model_id", None), "summary": summary, "by_category": by_cat,
            "records": per_record,
        }
        print(f"\n== {name} ==")
        for k, v in summary.items():
            print(f"  {k:>28}: {v}")

    REPORT_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = REPORT_DIR / f"eval_report{suffix}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = REPORT_DIR / f"eval_report{suffix}.md"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"\nReports written to {json_path} and {md_path}")


def to_markdown(report: dict) -> str:
    engines = report["engines"]
    keys = ["semantic_similarity", "reference_similarity", "entity_preservation", "factual_consistency",
            "grammar_score", "readability_after", "repetition_after", "stock_phrases_after", "length_ratio",
            "pass_rate", "timeouts", "seconds_per_record"]
    lines = [
        "# Evaluation report", "",
        f"- Created: {report['created_at']}",
        f"- Split: `{report['split']}` ({report['records']} records), style `{report['style']}`, intensity `{report['intensity']}`",
        f"- Similarity: `{report['similarity_method']}`; pass = similarity >= {report['min_similarity']}, "
        "factual consistency >= 0.85, grammar >= 0.5",
        f"- {report['note']}", "",
        "| metric | " + " | ".join(engines) + " |",
        "|---|" + "---|" * len(engines),
    ]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(e['summary'].get(k, '')) for e in engines.values()) + " |")
    first = next(iter(engines.values()))["summary"]
    lines += ["", f"Input reference values: readability {first['readability_before']}, repetition "
              f"{first['repetition_before']}, stock phrases/100 words {first['stock_phrases_before']}.", "",
              "`reference_similarity` compares the output with the human-written original (the dataset target)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
