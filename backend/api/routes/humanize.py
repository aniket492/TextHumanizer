from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_current_user, get_humanizer
from database import get_db
from models import HumanizationHistory, Usage, User
from schemas.humanize import HumanizeRequest, HumanizeResponse
from services.humanizer_service import HumanizerService

router = APIRouter(prefix="/api", tags=["humanize"])


@router.post("/humanize", response_model=HumanizeResponse)
def humanize(
    payload: HumanizeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: HumanizerService = Depends(get_humanizer),
) -> HumanizeResponse:
    # Sync endpoint: FastAPI runs it in a worker thread, so CPU inference does not block the event loop.
    result = service.humanize(payload.text, payload.tone, payload.intensity)

    history_id = None
    if payload.save_history:
        entry = HumanizationHistory(
            user_id=user.id,
            original_text=result.original_text,
            output_text=result.output_text,
            tone=result.tone.value,
            intensity=result.intensity.value,
            word_count=result.input_word_count,
            quality={**result.quality, "engine": result.engine, "notes": result.notes},
        )
        db.add(entry)
    usage = db.query(Usage).filter(Usage.user_id == user.id).with_for_update().one_or_none()
    if usage is None:
        usage = Usage(user_id=user.id, request_count=0, words_processed=0)
        db.add(usage)
    usage.request_count += 1
    usage.words_processed += result.input_word_count
    db.commit()
    if payload.save_history:
        history_id = entry.id

    return HumanizeResponse(
        id=history_id,
        output_text=result.output_text,
        tone=result.tone,
        intensity=result.intensity,
        engine=result.engine,
        fallback_used=result.fallback_used,
        notes=result.notes,
        input_word_count=result.input_word_count,
        output_word_count=result.output_word_count,
        chunks=result.chunks,
        processing_ms=result.processing_ms,
        quality=result.quality,
    )
