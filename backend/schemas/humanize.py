from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from core.constants import Intensity, Style


class HumanizeRequest(BaseModel):
    text: str = Field(description="Text to rewrite.")
    tone: Style = Style.natural
    intensity: Intensity = Intensity.medium
    save_history: bool = True


class QualityReport(BaseModel):
    semantic_similarity: float
    similarity_method: str
    readability_before: float
    readability_after: float
    grammar_score: float
    repetition_before: float
    repetition_after: float
    length_ratio: float
    entity_preservation: float
    missing_items: list[str]
    factual_consistency: float
    passed: bool


class HumanizeResponse(BaseModel):
    id: int | None = None
    output_text: str
    tone: Style
    intensity: Intensity
    engine: str
    fallback_used: bool
    notes: list[str]
    input_word_count: int
    output_word_count: int
    chunks: int
    processing_ms: int
    quality: QualityReport


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tone: str
    intensity: str
    word_count: int
    created_at: datetime
    preview: str


class HistoryDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_text: str
    output_text: str
    tone: str
    intensity: str
    word_count: int
    quality: dict | None
    created_at: datetime


class HistoryList(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class UsageResponse(BaseModel):
    request_count: int
    words_processed: int
    history_count: int
    updated_at: datetime | None
