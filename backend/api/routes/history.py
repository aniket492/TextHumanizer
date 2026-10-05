from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import get_current_user
from core.constants import HISTORY_PAGE_SIZE
from database import get_db
from models import HumanizationHistory, Usage, User
from schemas.humanize import HistoryDetail, HistoryItem, HistoryList, UsageResponse
from utils.text_utils import preview

router = APIRouter(prefix="/api", tags=["history"])


def _get_owned(db: Session, user: User, history_id: int) -> HumanizationHistory:
    entry = db.get(HumanizationHistory, history_id)
    # Same 404 for "missing" and "not yours" so ids of other users are not revealed.
    if entry is None or entry.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "History item not found.")
    return entry


@router.get("/history", response_model=HistoryList)
def list_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(HISTORY_PAGE_SIZE, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HistoryList:
    total = db.scalar(select(func.count()).select_from(HumanizationHistory).where(HumanizationHistory.user_id == user.id))
    rows = db.scalars(
        select(HumanizationHistory)
        .where(HumanizationHistory.user_id == user.id)
        .order_by(HumanizationHistory.created_at.desc(), HumanizationHistory.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [
        HistoryItem(
            id=r.id, tone=r.tone, intensity=r.intensity, word_count=r.word_count,
            created_at=r.created_at, preview=preview(r.output_text),
        )
        for r in rows
    ]
    return HistoryList(items=items, total=total or 0, page=page, page_size=page_size)


@router.get("/history/{history_id}", response_model=HistoryDetail)
def get_history(history_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> HistoryDetail:
    return HistoryDetail.model_validate(_get_owned(db, user, history_id))


@router.delete("/history/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(history_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    db.delete(_get_owned(db, user, history_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/usage", response_model=UsageResponse)
def get_usage(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UsageResponse:
    usage = db.scalar(select(Usage).where(Usage.user_id == user.id))
    history_count = db.scalar(
        select(func.count()).select_from(HumanizationHistory).where(HumanizationHistory.user_id == user.id)
    )
    return UsageResponse(
        request_count=usage.request_count if usage else 0,
        words_processed=usage.words_processed if usage else 0,
        history_count=history_count or 0,
        updated_at=usage.updated_at if usage else None,
    )
