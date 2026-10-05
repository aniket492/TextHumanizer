"""Meaning-preserving text normalisation and segmentation helpers.

Only invisible/whitespace characters are normalised. Punctuation, quotes and casing are kept.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_UNICODE_SPACES = re.compile(r"[   -   　\t\f\v]")
_MULTI_SPACE = re.compile(r" {2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?%)\]}])")
_MULTI_BLANK = re.compile(r"\n\s*\n+")


def normalize_whitespace(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks (blank lines)."""
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _UNICODE_SPACES.sub(" ", text)
    paragraphs = _MULTI_BLANK.split(text)
    cleaned = []
    for para in paragraphs:
        # Single newlines inside a paragraph become spaces.
        para = " ".join(line.strip() for line in para.split("\n") if line.strip())
        para = _MULTI_SPACE.sub(" ", para).strip()
        if para:
            cleaned.append(para)
    return "\n\n".join(cleaned)


def tidy_spacing(text: str) -> str:
    """Fix spacing artefacts introduced by edits (double spaces, space before punctuation)."""
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(r"\.\s*\.(?!\.)", ".", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _MULTI_BLANK.split(text.strip()) if p.strip()]


# Abbreviations that should not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig",
    "no", "vol", "inc", "ltd", "co", "corp", "u.s", "u.k", "al", "approx", "dept", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (fast, dependency-free, abbreviation aware)."""
    sentences: list[str] = []
    for para in split_paragraphs(text):
        start = 0
        for match in _SENT_BOUNDARY.finditer(para):
            candidate = para[start:match.start()].strip()
            last_word = candidate.rstrip(".!?\"')]”’").split()[-1:] or [""]
            token = last_word[0].lower().rstrip(".")
            # Skip abbreviations and single initials ("J. Smith").
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
            sentences.append(para[start:match.end()].strip())
            start = match.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if s]


_WORD = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_count(text: str) -> int:
    return len(words(text))
