"""Saved analysis record (cloud history/progress, per user).

Stores the full frontend history entry as a JSON blob plus a few indexed
columns for querying. ``client_id`` is the entry id generated on the device
(``h<timestamp>``) so imports/deletes are idempotent across local <-> cloud.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        UniqueConstraint("user_id", "client_id", name="uq_analysis_user_client"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    client_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    sport: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
