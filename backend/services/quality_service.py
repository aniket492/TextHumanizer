"""Quality gating and ranking of rewrite candidates."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from core.constants import DEFAULT_LENGTH_BOUNDS, LENGTH_BOUNDS, MIN_SIMILARITY, Intensity, Style
from ml.evaluation.metrics import (
    SemanticSimilarity,
    entity_preservation,
    extract_key_items,
    factual_consistency,
    grammar_check,
    length_ratio,
    readability,
    repetition_score,
    stock_phrase_rate,
)
from ml.preprocessing.normalize_text import words

# Items that must survive verbatim; named entities are scored softly (paraphrase may
# legitimately say "the agency" instead of repeating its full name).
STRICT_CATEGORIES = ("urls", "emails", "citations", "numbers", "dates")
MIN_FACTUAL = 0.85
MIN_GRAMMAR = 0.5
# TF-IDF similarity runs lower than embedding similarity for genuine paraphrases.
TFIDF_SIMILARITY = {Intensity.light: 0.62, Intensity.medium: 0.5, Intensity.strong: 0.4}
CHANGE_TARGETS = {Intensity.light: (0.03, 0.35), Intensity.medium: (0.12, 0.6), Intensity.strong: (0.25, 0.9)}


@dataclass
class Candidate:
    text: str
    source: str  # "model" | "baseline" | "original"
    changes: list[str] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    similarity: float
    strict_missing: list[str]
    entity_score: float
    grammar: float
    factual: float
    length_ratio: float
    change_ratio: float
    score: float
    passed: bool
    reasons: list[str]


def change_ratio(source: str, output: str) -> float:
    a, b = [w.lower() for w in words(source)], [w.lower() for w in words(output)]
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


class QualityService:
    def __init__(self, similarity: SemanticSimilarity):
        self.similarity = similarity

    def min_similarity(self, intensity: Intensity) -> float:
        if self.similarity.method == "tfidf":
            return TFIDF_SIMILARITY[intensity]
        return MIN_SIMILARITY[intensity]

    def score(self, source: str, candidates: list[Candidate], style: Style, intensity: Intensity) -> list[ScoredCandidate]:
        if not candidates:
            return []
        items = extract_key_items(source)
        strict_items = {k: v for k, v in items.items() if k in STRICT_CATEGORIES}
        soft_items = {k: v for k, v in items.items() if k not in STRICT_CATEGORIES}
        sims = self.similarity.batch_similarity(source, [c.text for c in candidates])
        min_sim = self.min_similarity(intensity)
        lo, hi = LENGTH_BOUNDS.get(style, DEFAULT_LENGTH_BOUNDS)
        c_lo, c_hi = CHANGE_TARGETS[intensity]
        rep_before = repetition_score(source)
        fre_before = readability(source)["flesch_reading_ease"]

        scored: list[ScoredCandidate] = []
        for cand, sim in zip(candidates, sims):
            reasons: list[str] = []
            text = cand.text.strip()
            _, strict_missing = entity_preservation(source, text, strict_items)
            ent_score, _ = entity_preservation(source, text, soft_items)
            grammar, _ = grammar_check(text) if text else (0.0, [])
            factual = factual_consistency(source, text) if text else 0.0
            ratio = length_ratio(source, text)
            change = change_ratio(source, text)

            if not text:
                reasons.append("empty output")
            if sim < min_sim:
                reasons.append(f"meaning drift (similarity {sim:.2f} < {min_sim:.2f})")
            if strict_missing:
                reasons.append(f"lost key items: {', '.join(strict_missing[:3])}")
            if factual < MIN_FACTUAL:
                reasons.append("numbers or negations changed")
            if cand.source != "original" and not (lo <= ratio <= hi):
                reasons.append(f"length ratio {ratio:.2f} outside {lo}-{hi}")
            if grammar < MIN_GRAMMAR:
                reasons.append("grammar check failed")

            if c_lo <= change <= c_hi:
                change_fit = 1.0
            elif change < c_lo:
                change_fit = change / c_lo if c_lo else 1.0
            else:
                change_fit = max(0.0, 1.0 - (change - c_hi) * 3)
            rep_gain = max(0.0, min(1.0, 0.5 + (rep_before - repetition_score(text)) * 5))
            fre_after = readability(text)["flesch_reading_ease"] if text else 0.0
            if style in (Style.simple, Style.casual, Style.natural, Style.concise):
                read_fit = max(0.0, min(1.0, 0.5 + (fre_after - fre_before) / 40))
            else:
                read_fit = 1.0 - min(1.0, abs(fre_after - fre_before) / 60)
            # Absolute scale: 0 stock phrases per 100 words -> 1.0, >= 4 per 100 words -> 0.0.
            # (A relative gain saturates on very dense inputs and stops discriminating.)
            stock_gain = max(0.0, 1.0 - stock_phrase_rate(text) / 4) if text else 0.0
            sim_norm = max(0.0, min(1.0, (sim - min_sim) / max(1.0 - min_sim, 1e-6)))
            score = (
                0.3 * sim_norm + 0.18 * grammar + 0.12 * ent_score + 0.15 * change_fit
                + 0.12 * stock_gain + 0.07 * rep_gain + 0.06 * read_fit
            )
            scored.append(
                ScoredCandidate(cand, sim, strict_missing, ent_score, grammar, factual, ratio, change, round(score, 4), not reasons, reasons)
            )
        return scored

    def select(self, source: str, candidates: list[Candidate], style: Style, intensity: Intensity) -> tuple[ScoredCandidate | None, list[ScoredCandidate]]:
        scored = self.score(source, candidates, style, intensity)
        passing = [s for s in scored if s.passed]
        best = max(passing, key=lambda s: s.score) if passing else None
        return best, scored

    def report(self, source: str, output: str, intensity: Intensity) -> dict:
        sim = self.similarity.similarity(source, output)
        ent_score, missing = entity_preservation(source, output)
        strict = {k: v for k, v in extract_key_items(source).items() if k in STRICT_CATEGORIES}
        _, strict_missing = entity_preservation(source, output, strict)
        grammar, _ = grammar_check(output)
        factual = factual_consistency(source, output)
        return {
            "semantic_similarity": round(sim, 4),
            "similarity_method": self.similarity.method,
            "readability_before": readability(source)["flesch_reading_ease"],
            "readability_after": readability(output)["flesch_reading_ease"],
            "grammar_score": grammar,
            "repetition_before": round(repetition_score(source), 4),
            "repetition_after": round(repetition_score(output), 4),
            "length_ratio": round(length_ratio(source, output), 4),
            "entity_preservation": round(ent_score, 4),
            "missing_items": missing[:20],
            "factual_consistency": factual,
            "passed": sim >= self.min_similarity(intensity) and not strict_missing and factual >= MIN_FACTUAL,
        }
