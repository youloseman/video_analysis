"""The link between an athlete's own account and a coach's roster row.

Model A of the coach area (docs/COACH_AREA_PLAN_RU.md): the athlete has a
Flapp account and a plan of their own, enters the coach's code, ticks the
consent box, and from then on the coach reads the athlete's history -- what
the athlete's plan shows the athlete, no more -- and writes notes on it.

A row of its own rather than a flag, because the consent is the thing: when
it was given, by which account, and when it was taken back. ``revoked_at_ms``
NULL is the live link; revoking keeps the row, so "who could see my clips in
October" has an answer. ``athletes.linked_user_id`` mirrors the live link for
cheap joins and is cleared when the link is revoked.

The privacy policy (section 3) promises that no human sees footage outside a
bought review unless the athlete did exactly this, so nothing reads across
accounts without a live row here.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# How the link was made. "code" = the athlete typed the coach's code.
SOURCE_CODE = "code"


class CoachLink(Base):
    __tablename__ = "coach_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(
        ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # The athlete's own account.
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    consented_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    revoked_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
