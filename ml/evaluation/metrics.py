"""Measurable writing-quality metrics.

Used both by the API (per-request quality checks) and by offline evaluation
(``ml/evaluation/evaluate.py``). Nothing here is an "AI detector": these are
transparent, explainable measures of meaning preservation and writing quality.
"""
from __future__ import annotations

import logging
import math
import os
import re
import string
import threading
from collections import Counter
from functools import lru_cache

import numpy as np
import textstat
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.preprocessing.normalize_text import split_paragraphs, split_sentences, words

logger = logging.getLogger("humanizer.metrics")

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | {"s", "t", "ll", "ve", "re", "d", "m"}


# ---------------------------------------------------------------------------
# spaCy (optional, used for NER and POS-based grammar checks)
# ---------------------------------------------------------------------------
_nlp_lock = threading.Lock()
_nlp = None
_nlp_failed = False


def get_nlp():
    """Return a cached spaCy pipeline, or None if spaCy/the model is unavailable."""
    global _nlp, _nlp_failed
    if _nlp is not None or _nlp_failed:
        return _nlp
    with _nlp_lock:
        if _nlp is None and not _nlp_failed:
            try:
                import spacy

                _nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"), disable=["lemmatizer"])
            except Exception as exc:  # pragma: no cover - depends on environment
                logger.warning("spaCy model unavailable (%s); using regex fallbacks.", exc)
                _nlp_failed = True
    return _nlp


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
def sentence_lengths(text: str) -> list[int]:
    return [len(words(s)) for s in split_sentences(text) if words(s)]


def sentence_stats(text: str) -> dict:
    lengths = sentence_lengths(text)
    if not lengths:
        return {"count": 0, "mean": 0.0, "std": 0.0, "cv": 0.0}
    arr = np.array(lengths, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    return {"count": len(lengths), "mean": mean, "std": std, "cv": std / mean if mean else 0.0}


def vocabulary_diversity(text: str, window: int = 50) -> float:
    """Moving-average type/token ratio (MATTR); robust to text length."""
    tokens = [w.lower() for w in words(text)]
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, len(tokens)):
        out_tok, in_tok = tokens[i - window], tokens[i]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        total += len(counts)
    return total / (len(tokens) - window + 1) / window


def lexical_density(text: str) -> float:
    tokens = [w.lower() for w in words(text)]
    if not tokens:
        return 0.0
    return sum(t not in STOPWORDS for t in tokens) / len(tokens)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repetition_score(text: str, n: int = 3) -> float:
    """Share of tokens that fall inside an n-gram occurring more than once (0 = no repetition)."""
    tokens = [w.lower() for w in words(text)]
    if len(tokens) < n:
        return 0.0
    grams = _ngrams(tokens, n)
    counts = Counter(grams)
    covered = [False] * len(tokens)
    for i, g in enumerate(grams):
        if counts[g] > 1 and not all(t in STOPWORDS for t in g):
            for j in range(i, i + n):
                covered[j] = True
    return sum(covered) / len(tokens)


def repeated_words(text: str, min_count: int = 3, top: int = 10) -> list[dict]:
    tokens = [w.lower() for w in words(text)]
    counts = Counter(t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit())
    return [{"text": w, "count": c} for w, c in counts.most_common() if c >= min_count][:top]


def repeated_phrases(text: str, min_count: int = 2, top: int = 10) -> list[dict]:
    tokens = [w.lower() for w in words(text)]
    found: dict[str, int] = {}
    for n in (5, 4, 3):
        for gram, c in Counter(_ngrams(tokens, n)).items():
            if c < min_count or all(t in STOPWORDS for t in gram):
                continue
            phrase = " ".join(gram)
            # Skip sub-phrases of an already reported longer phrase with the same count.
            if any(phrase in longer and c <= lc for longer, lc in found.items()):
                continue
            found[phrase] = c
    ranked = sorted(found.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [{"text": p, "count": c} for p, c in ranked[:top]]


def punctuation_distribution(text: str) -> dict[str, int]:
    marks = {
        "period": ".", "comma": ",", "semicolon": ";", "colon": ":", "question": "?",
        "exclamation": "!", "dash": "-", "em_dash": "—", "parenthesis": "(",
        "quote": '"',
    }
    dist = {name: text.count(ch) for name, ch in marks.items()}
    dist["quote"] += text.count("“")
    dist["apostrophe"] = text.count("'") + text.count("’")
    dist["ellipsis"] = text.count("...") + text.count("…")
    return dist


def readability(text: str) -> dict:
    if not words(text):
        return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0, "gunning_fog": 0.0, "label": "n/a"}
    fre = float(textstat.flesch_reading_ease(text))
    if fre >= 80:
        label = "very easy"
    elif fre >= 60:
        label = "easy"
    elif fre >= 50:
        label = "fairly difficult"
    elif fre >= 30:
        label = "difficult"
    else:
        label = "very difficult"
    return {
        "flesch_reading_ease": round(fre, 2),
        "flesch_kincaid_grade": round(float(textstat.flesch_kincaid_grade(text)), 2),
        "gunning_fog": round(float(textstat.gunning_fog(text)), 2),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------
class SemanticSimilarity:
    """Sentence-Transformers cosine similarity with a TF-IDF fallback.

    The fallback is used when no model name is configured or the model cannot be
    loaded (e.g. offline). The method actually used is always reported.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or ""
        self._model = None
        self._failed = not bool(self.model_name)
        self._lock = threading.Lock()

    @property
    def method(self) -> str:
        self._ensure_model()
        return f"embedding:{self.model_name}" if self._model is not None else "tfidf"

    def _ensure_model(self):
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device="cpu")
                    logger.info("Loaded similarity model %s", self.model_name)
                except Exception as exc:
                    logger.warning("Similarity model unavailable (%s); using TF-IDF.", exc)
                    self._failed = True

    def similarity(self, a: str, b: str) -> float:
        return self.batch_similarity(a, [b])[0]

    def batch_similarity(self, source: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        if not source.strip():
            return [0.0 for _ in candidates]
        self._ensure_model()
        if self._model is not None:
            emb = self._model.encode([source, *candidates], normalize_embeddings=True, show_progress_bar=False)
            sims = emb[1:] @ emb[0]
            return [float(max(0.0, min(1.0, s))) for s in sims]
        return [tfidf_similarity(source, c) for c in candidates]


def tfidf_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([a, b])
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit([a, b])
    w = cosine_similarity(word_vec.transform([a]), word_vec.transform([b]))[0, 0]
    c = cosine_similarity(char_vec.transform([a]), char_vec.transform([b]))[0, 0]
    return float(0.4 * w + 0.6 * c)


# ---------------------------------------------------------------------------
# Key-item (entity) extraction and preservation
# ---------------------------------------------------------------------------
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+[^\s<>\"').,;:\]]")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CITATION = re.compile(
    r"\[\d+(?:[,–-]\s*\d+)*\]"  # [1], [2, 3], [4-6]
    r"|\((?:[A-Z][A-Za-z'\-]+(?: et al\.)?(?:,? (?:and|&) [A-Z][A-Za-z'\-]+)?),? \d{4}[a-z]?\)"  # (Smith, 2020)
)
_NUMBER = re.compile(r"(?<![\w.])[$£€]?\d[\d,]*(?:\.\d+)?%?")
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
    rf"|(?:{_MONTHS})\.? \d{{1,2}}(?:st|nd|rd|th)?(?:, \d{{4}})?|\d{{1,2}} (?:{_MONTHS}) \d{{4}})\b"
)
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
_CAP_SEQ = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+(?:of\s+|the\s+)?[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z0-9]*)\b")


def normalize_number(token: str) -> str:
    return token.replace(",", "").lstrip("$£€").rstrip("%").rstrip(".")


def extract_key_items(text: str, use_ner: bool = True) -> dict[str, list[str]]:
    """Extract items that must survive a rewrite unchanged."""
    urls = _URL.findall(text)
    emails = _EMAIL.findall(text)
    citations = _CITATION.findall(text)
    dates = _DATE.findall(text)
    masked = text
    for item in urls + emails:
        masked = masked.replace(item, " ")
    numbers = [n for n in _NUMBER.findall(masked) if any(ch.isdigit() for ch in n)]
    entities: list[str] = []
    nlp = get_nlp() if use_ner else None
    if nlp is not None:
        doc = nlp(masked)
        for ent in doc.ents:
            if ent.label_ in _ENTITY_LABELS:
                ent_text = ent.text.strip(string.punctuation + " ")
                if ent_text.lower().startswith("the "):
                    ent_text = ent_text[4:]
                if len(ent_text) > 1 and any(c.isupper() for c in ent_text):
                    entities.append(ent_text)
    else:
        entities = [m for m in _CAP_SEQ.findall(masked)]
    return {
        "urls": _dedupe(urls),
        "emails": _dedupe(emails),
        "citations": _dedupe(citations),
        "dates": _dedupe(dates),
        "numbers": _dedupe(numbers),
        "entities": _dedupe(entities),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _number_set(text: str) -> Counter:
    masked = text
    for item in _URL.findall(text) + _EMAIL.findall(text):
        masked = masked.replace(item, " ")
    return Counter(normalize_number(n) for n in _NUMBER.findall(masked) if any(c.isdigit() for c in n))


def entity_preservation(source: str, output: str, items: dict[str, list[str]] | None = None) -> tuple[float, list[str]]:
    """Fraction of key items from ``source`` that are still present in ``output``."""
    items = items if items is not None else extract_key_items(source)
    out_lower = output.lower()
    out_numbers = set(_number_set(output))
    required: list[str] = []
    missing: list[str] = []
    for category, values in items.items():
        for value in values:
            required.append(value)
            if category == "numbers":
                ok = normalize_number(value) in out_numbers
            elif category == "entities":
                ok = value.lower() in out_lower
            else:
                ok = value in output
            if not ok:
                missing.append(value)
    if not required:
        return 1.0, []
    return 1.0 - len(missing) / len(required), missing


_NEGATIONS = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without)\b|n't\b", re.I)


def factual_consistency(source: str, output: str) -> float:
    """Heuristic factual consistency: number agreement (both directions) and negation polarity."""
    src_nums, out_nums = _number_set(source), _number_set(output)
    total = sum((src_nums | out_nums).values())
    if total:
        mismatched = sum((src_nums - out_nums).values()) + sum((out_nums - src_nums).values())
        number_score = max(0.0, 1.0 - mismatched / total)
    else:
        number_score = 1.0
    src_neg, out_neg = len(_NEGATIONS.findall(source)), len(_NEGATIONS.findall(output))
    if src_neg or out_neg:
        negation_score = 1.0 - min(1.0, abs(src_neg - out_neg) / max(src_neg, out_neg, 1))
        # Allow small drift: rewrites can legitimately merge or rephrase one negation.
        negation_score = min(1.0, negation_score + 0.25)
    else:
        negation_score = 1.0
    return round(0.7 * number_score + 0.3 * negation_score, 4)


# ---------------------------------------------------------------------------
# Grammar / fluency heuristics
# ---------------------------------------------------------------------------
_REPEATED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.I)
_A_AN = re.compile(r"\b(a)\s+([aeiou]\w+)|\b(an)\s+([bcdfgjklmnpqrstvwxz]\w+)", re.I)
_A_AN_EXCEPTIONS = re.compile(r"^(?:uni|use|usu|uti|eu|one|once|ur|hour|honest|honou?r|heir)", re.I)


def grammar_check(text: str) -> tuple[float, list[str]]:
    """Lightweight grammar/fluency check. Returns (score 0-1, issues).

    Checks sentence capitalisation and terminal punctuation, doubled words,
    a/an agreement, unbalanced brackets/quotes, run-on length and (with spaCy)
    sentences without a verb. It is a heuristic, not a full grammar checker.
    """
    sentences = split_sentences(text)
    if not sentences:
        return 0.0, ["empty text"]
    issues: list[str] = []
    for s in sentences:
        stripped = s.lstrip("\"'“‘([")
        if stripped and stripped[0].isalpha() and stripped[0].islower():
            issues.append(f"lowercase sentence start: {s[:40]!r}")
        if s[-1] not in ".!?\"')”’]:" and len(words(s)) > 3:
            issues.append(f"missing terminal punctuation: {s[-40:]!r}")
        if len(words(s)) > 60:
            issues.append(f"very long sentence ({len(words(s))} words)")
    for m in _REPEATED_WORD.finditer(text):
        if m.group(1).lower() not in {"that", "had", "is", "very", "bye", "ha", "no"}:
            issues.append(f"doubled word: {m.group(0)!r}")
    for m in _A_AN.finditer(text):
        nxt = m.group(2) or m.group(4)
        article = (m.group(1) or m.group(3)).lower()
        if nxt.isupper() or _A_AN_EXCEPTIONS.match(nxt):
            continue  # acronyms ("an FBI", "a USB") and sound-based exceptions
        if article == "a" or not nxt[0].isupper():
            issues.append(f"article agreement: {m.group(0)!r}")
    for open_c, close_c in (("(", ")"), ("[", "]")):
        if text.count(open_c) != text.count(close_c):
            issues.append(f"unbalanced {open_c}{close_c}")
    if text.count('"') % 2:
        issues.append("unbalanced quotes")
    nlp = get_nlp()
    if nlp is not None and "tagger" in nlp.pipe_names:
        long_sents = [s for s in sentences if len(words(s)) >= 6]
        for sent, doc in zip(long_sents, nlp.pipe(long_sents, disable=["ner", "parser"])):
            if not any(t.pos_ in {"VERB", "AUX"} for t in doc):
                issues.append(f"no verb: {sent[:40]!r}")
    score = max(0.0, 1.0 - len(issues) / (len(sentences) * 2))
    return round(score, 4), issues


_STOCK_PHRASES = re.compile(
    r"\b(?:it is (?:important|worth|crucial|essential) to (?:note|mention|highlight|emphasi[sz]e)|"
    r"it should be noted|furthermore|moreover|additionally|in conclusion|in summary|"
    r"in today's (?:fast-paced|modern|digital) (?:world|landscape)|plays? an? (?:crucial|pivotal|vital|key) role|"
    r"delv(?:e|es|ing) into|a (?:wide|vast|diverse) (?:range|array|variety) of|plethora|myriad|"
    r"utili[sz](?:e|es|ed|ing)|leverag(?:e|es|ed|ing)|seamless(?:ly)?|navigat(?:e|ing) the complexities|"
    r"in order to|due to the fact that|a testament to|ever-evolving|notably|ultimately|"
    r"it is evident that|tapestry|embark(?:s|ed)? on|(?:rapidly |ever-)?(?:evolving|changing) (?:\w+ )?landscape|"
    r"intricacies|(?:diverse|wide|vast) array of|underscor(?:e|es|ed|ing)|pivotal|in the realm of|meticulous(?:ly)?|"
    r"stands? out as|vital component|it's (?:essential|crucial|important) to (?:recogni[sz]e|note|understand))\b",
    re.IGNORECASE,
)


def stock_phrase_rate(text: str) -> float:
    """Mechanical stock phrases per 100 words (lower reads more natural)."""
    n = len(words(text))
    return 100.0 * len(_STOCK_PHRASES.findall(text)) / n if n else 0.0


def length_ratio(source: str, output: str) -> float:
    return len(words(output)) / max(len(words(source)), 1)


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------
def evaluate_pair(source: str, output: str, similarity: SemanticSimilarity, reference: str | None = None) -> dict:
    """Compute every quality metric for one (source, output) pair."""
    sim = similarity.similarity(source, output)
    ent_score, missing = entity_preservation(source, output)
    grammar, issues = grammar_check(output)
    result = {
        "semantic_similarity": round(sim, 4),
        "readability_before": readability(source)["flesch_reading_ease"],
        "readability_after": readability(output)["flesch_reading_ease"],
        "grammar_score": grammar,
        "grammar_issues": issues[:10],
        "repetition_before": round(repetition_score(source), 4),
        "repetition_after": round(repetition_score(output), 4),
        "length_ratio": round(length_ratio(source, output), 4),
        "entity_preservation": round(ent_score, 4),
        "missing_items": missing,
        "factual_consistency": factual_consistency(source, output),
        "sentence_length_cv_before": round(sentence_stats(source)["cv"], 4),
        "sentence_length_cv_after": round(sentence_stats(output)["cv"], 4),
        "vocabulary_diversity_after": round(vocabulary_diversity(output), 4),
        "stock_phrases_before": round(stock_phrase_rate(source), 3),
        "stock_phrases_after": round(stock_phrase_rate(output), 3),
    }
    if reference is not None:
        result["reference_similarity"] = round(similarity.similarity(reference, output), 4)
        result["reference_length_ratio"] = round(length_ratio(reference, output), 4)
    return result


@lru_cache(maxsize=4)
def get_similarity(model_name: str) -> SemanticSimilarity:
    return SemanticSimilarity(model_name)


def safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0


__all__ = [
    "SemanticSimilarity", "get_similarity", "evaluate_pair", "extract_key_items", "entity_preservation",
    "factual_consistency", "grammar_check", "readability", "repetition_score", "repeated_words",
    "repeated_phrases", "punctuation_distribution", "sentence_stats", "vocabulary_diversity",
    "lexical_density", "length_ratio", "split_paragraphs", "split_sentences", "safe_mean",
]
