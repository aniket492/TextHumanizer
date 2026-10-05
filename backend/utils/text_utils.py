"""Segmentation helpers used by the humanizer pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from ml.preprocessing.normalize_text import split_paragraphs, split_sentences, word_count


@dataclass
class Chunk:
    paragraph: int
    text: str


def chunk_text(text: str, max_words: int = 110) -> list[Chunk]:
    """Split text into paragraph-aligned chunks of whole sentences (<= max_words when possible)."""
    chunks: list[Chunk] = []
    for p_idx, para in enumerate(split_paragraphs(text)):
        current: list[str] = []
        current_words = 0
        for sent in split_sentences(para):
            n = word_count(sent)
            if current and current_words + n > max_words:
                chunks.append(Chunk(p_idx, " ".join(current)))
                current, current_words = [], 0
            current.append(sent)
            current_words += n
        if current:
            chunks.append(Chunk(p_idx, " ".join(current)))
    return chunks


def assemble(chunks: list[Chunk], outputs: list[str]) -> str:
    paragraphs: dict[int, list[str]] = {}
    for chunk, out in zip(chunks, outputs):
        paragraphs.setdefault(chunk.paragraph, []).append(out.strip())
    return "\n\n".join(" ".join(parts) for _, parts in sorted(paragraphs.items()))


def preview(text: str, length: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1].rsplit(" ", 1)[0] + "…"
