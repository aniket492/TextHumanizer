from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_models
from config import settings
from schemas.analysis import AnalysisResponse, AnalyzeRequest
from services.model_service import ModelService
from services.text_analysis_service import analyze_text
from utils.validators import validate_text

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: AnalyzeRequest, models: ModelService = Depends(get_models)) -> AnalysisResponse:
    limits = dict(min_words=1, max_words=settings.max_input_words * 2, max_chars=settings.max_input_chars * 2)
    text = validate_text(payload.text, **limits)
    compare = validate_text(payload.compare_text, **limits) if payload.compare_text else None
    result = analyze_text(text, compare, models.similarity if compare else None)
    return AnalysisResponse(**result)
