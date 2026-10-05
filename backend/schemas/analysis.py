from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(description="Text to analyse.")
    compare_text: str | None = Field(
        default=None, description="Optional second text (e.g. a rewrite) to compare against."
    )


class RepeatedItem(BaseModel):
    text: str
    count: int


class ReadabilityScores(BaseModel):
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    label: str


class AnalysisResponse(BaseModel):
    word_count: int
    character_count: int
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int
    average_sentence_length: float
    sentence_length_std: float
    sentence_length_variation: float = Field(description="Coefficient of variation of sentence lengths.")
    vocabulary_diversity: float = Field(description="Moving-average type/token ratio (0-1).")
    lexical_density: float
    readability_score: float = Field(description="Flesch reading ease (higher = easier).")
    readability: ReadabilityScores
    repetition_score: float = Field(description="Share of tokens inside repeated 3-grams (0-1).")
    repeated_words: list[RepeatedItem]
    repeated_phrases: list[RepeatedItem]
    punctuation: dict[str, int]
    semantic_similarity: float | None = None
    similarity_method: str | None = None
