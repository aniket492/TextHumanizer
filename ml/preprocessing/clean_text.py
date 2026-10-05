"""Record-level cleaning rules for the humanizer dataset."""
from __future__ import annotations

import re
from collections import Counter

from ml.preprocessing.normalize_text import normalize_whitespace, word_count, words

_REPLACEMENT_CHAR = "�"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_MOJIBAKE = re.compile(r"(?:Ã.|â€.|Â\s)")
_MARKUP = re.compile(r"(<\s*/?\s*(?:div|span|p|br|html|body|script)\b[^>]*>|\[\s*Illustration)", re.I)


def clean_text(text: str | None) -> str:
    """Normalise whitespace and strip control characters; keep punctuation and wording."""
    if not text:
        return ""
    return normalize_whitespace(_CONTROL.sub("", str(text)))


def corruption_reason(text: str) -> str | None:
    """Return why a text looks corrupted, or None if it looks fine."""
    if not text:
        return "empty"
    if _REPLACEMENT_CHAR in text:
        return "replacement_character"
    if len(_MOJIBAKE.findall(text)) >= 2:
        return "mojibake"
    if _MARKUP.search(text):
        return "markup"
    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < 0.6:
        return "low_letter_ratio"
    if re.search(r"(.)\1{9,}", text):
        return "repeated_characters"
    return None


_NUM = re.compile(r"\d+(?:\.\d+)?")


def internal_repetition(text: str, n: int = 6) -> float:
    """Share of tokens covered by 6-grams that occur more than once (duplicated passages)."""
    tokens = [w.lower() for w in words(text)]
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(tokens)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return sum(covered) / len(tokens)


def pair_problem(
    source: str,
    target: str,
    min_words: int = 20,
    max_words: int = 600,
    min_ratio: float = 0.4,
    max_ratio: float = 2.5,
) -> str | None:
    """Validate an (AI-styled input, human output) pair. Returns a reason string if invalid."""
    for name, text in (("input", source), ("output", target)):
        reason = corruption_reason(text)
        if reason:
            return f"{name}_{reason}"
        n = word_count(text)
        if n < min_words:
            return f"{name}_too_short"
        if n > max_words:
            return f"{name}_too_long"
    if source.strip() == target.strip():
        return "identical_pair"
    if internal_repetition(target) > 0.25:
        return "output_internal_repetition"
    # Numbers in the target that never appear in the input teach the model to invent facts.
    src_nums, tgt_nums = set(_NUM.findall(source.replace(",", ""))), set(_NUM.findall(target.replace(",", "")))
    unsupported = tgt_nums - src_nums
    if len(unsupported) >= 2 and len(unsupported) / max(len(tgt_nums), 1) > 0.5:
        return "output_unsupported_numbers"
    ratio = word_count(target) / max(word_count(source), 1)
    if not (min_ratio <= ratio <= max_ratio):
        return "length_ratio_out_of_range"
    return None
