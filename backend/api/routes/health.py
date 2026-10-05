from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_models
from config import settings
from database import check_database
from services.model_service import ModelService

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "message": "AI Humanizer API: natural-language rewriting with meaning-preservation checks.",
    }


@router.get("/health")
def health(models: ModelService = Depends(get_models)) -> JSONResponse:
    db_ok = check_database()
    body = {
        "status": "ok" if db_ok else "degraded",
        "version": VERSION,
        "database": "ok" if db_ok else "unavailable",
        "model": models.info(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)
