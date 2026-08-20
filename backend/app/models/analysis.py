"""Saved analysis record (cloud history/progress, per user).

Stores the full frontend history entry as a JSON blob plus a few indexed
columns for querying. ``client_id`` is the entry id generated on the device
(``h<timestamp>``) so imports/deletes are idempotent across local <-> cloud.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
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
    # The upload that produced this analysis. The footage lives under this id in
    # the uploads directory, and this column is the only thing joining the two:
    # history entries are keyed by a client-minted ``client_id`` while
    # directories are named by ``job_id``, so without it a stored analysis and
    # the clip it came from were unrelated objects. That is why an Expert Review
    # bought a week later had nothing but the annotated keyframe to work from --
    # not because the file had necessarily expired, but because nothing could
    # find it. Indexed: the storage sweeper looks jobs up by it.
    job_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    # Which setup this was filmed on (models/profile.py). A real column rather
    # than a field inside ``data`` because history and trends filter on it, and
    # because the client must not be the one deciding what it belongs to.
    # NULL is legitimate: everything filmed before profiles existed, and
    # anything the athlete has not filed.
    profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("profiles.id", ondelete="SET NULL"), index=True, nullable=True,
    )
    # The analyzer's own output, complete, written by the server -- NOT by the
    # client, which only ever receives what its plan allows and would otherwise
    # be the authority on its own entitlements. ``data`` above is the client's
    # rendering of an analysis; this is the analysis.
    #
    # It exists so that a report can be sold later. Trimming on the way in threw
    # the paid half away before anything stored it, which quietly decided that a
    # free analysis could never become a paid one: unlocking had nothing to
    # reveal, and subscribing could not open the history somebody had built
    # while free.
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # This analysis was the account's one free preview. Server-set, so that
    # re-opening it months later still shows what it showed on the day.
    preview: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False,
    )
    # Paid for individually (the one-off unlock). The Order is the record of the
    # payment; this is the entitlement, denormalised onto the row so that
    # reading a report is not a join per analysis.
    unlocked_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sport: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
