"""Structured text analysis (statistics, readability, repetition, similarity)."""
from __future__ import annotations

from ml.evaluation import metrics as m
from ml.preprocessing.normalize_text import split_paragraphs, words


def analyze_text(text: str, compare_text: str | None = None, similarity: m.SemanticSimilarity | None = None) -> dict:
    stats = m.sentence_stats(text)
    read = m.readability(text)
    result = {
        "word_count": len(words(text)),
        "character_count": len(text),
        "character_count_no_spaces": sum(not c.isspace() for c in text),
        "sentence_count": stats["count"],
        "paragraph_count": len(split_paragraphs(text)),
        "average_sentence_length": round(stats["mean"], 2),
        "sentence_length_std": round(stats["std"], 2),
        "sentence_length_variation": round(stats["cv"], 4),
        "vocabulary_diversity": round(m.vocabulary_diversity(text), 4),
        "lexical_density": round(m.lexical_density(text), 4),
        "readability_score": read["flesch_reading_ease"],
        "readability": read,
        "repetition_score": round(m.repetition_score(text), 4),
        "repeated_words": m.repeated_words(text),
        "repeated_phrases": m.repeated_phrases(text),
        "punctuation": m.punctuation_distribution(text),
        "semantic_similarity": None,
        "similarity_method": None,
    }
    if compare_text and compare_text.strip() and similarity is not None:
        result["semantic_similarity"] = round(similarity.similarity(text, compare_text), 4)
        result["similarity_method"] = similarity.method
    return result
