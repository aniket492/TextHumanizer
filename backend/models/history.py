from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class HumanizationHistory(Base):
    __tablename__ = "humanization_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    output_text: Mapped[str] = mapped_column(Text, nullable=False)
    tone: Mapped[str] = mapped_column(String(32), nullable=False)
    intensity: Mapped[str] = mapped_column(String(16), nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Quality report and engine metadata for the request (similarity, readability, ...).
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True, nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="history")  # noqa: F821
