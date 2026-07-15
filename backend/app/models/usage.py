"""Analysis usage events -- one row per analysis, used for per-user quotas.

DB-backed (unlike the in-memory IP limiter) so counts survive restarts and are
shared across workers. Counting a window is a filtered COUNT over
``created_at_ms``.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # Epoch milliseconds (UTC) when the analysis was accepted. Indexed so window
    # counts stay cheap as the table grows.
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    # "video" | "photo" -- for reporting; quotas count both together.
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
