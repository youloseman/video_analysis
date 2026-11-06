"""A clip waiting in the coach's lane, and a report sent out from it.

``CoachUpload``: one file of a coach's batch, or one clip an athlete dropped
on their intake link. The coach walks away while they run, so the state has to
outlive the process: Railway restarts the service on every deploy, and an
in-memory queue would quietly lose whatever had not run yet. A row survives,
and the lane picks it up again at startup (services/coach_lane.py).

``ReportShare``: a read-only link to one report, for an athlete who has no
account (model B). A row rather than a signed token, because a coach must be
able to switch a link off.

New tables, built by ``create_all``: nothing here alters an existing one.
"""

from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# queued -> scanning -> running -> done | failed, or scanning -> no_go.
STATUS_QUEUED = "queued"
STATUS_SCANNING = "scanning"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_NO_GO = "no_go"
STATUS_FAILED = "failed"
OPEN_STATUSES = (STATUS_QUEUED, STATUS_SCANNING, STATUS_RUNNING)

SOURCE_BATCH = "batch"
SOURCE_INTAKE = "intake"

MAX_FILENAME_CHARS = 200
MAX_ERROR_CHARS = 300


class CoachUpload(Base):
    __tablename__ = "coach_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    athlete_id: Mapped[int | None] = mapped_column(
        ForeignKey("athletes.id", ondelete="SET NULL"), index=True, nullable=True,
    )
    profile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Client-minted for a batch, so a file that fails to upload can be retried
    # into the same one. NULL for a clip from the intake link.
    batch_id: Mapped[str | None] = mapped_column(String(24), index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(8), nullable=False, default=SOURCE_BATCH)
    # The history row and the footage behind it.
    client_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(MAX_FILENAME_CHARS), nullable=True)
    sport: Mapped[str] = mapped_column(String(8), nullable=False)
    # Not "position": it is an SQL keyword (see models/profile.py).
    bike_position: Mapped[str | None] = mapped_column(String(16), nullable=True)
    camera_side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(String(10), index=True, nullable=False, default=STATUS_QUEUED)
    # What GO / NO-GO said about the clip: [{code, level, text}, ...].
    reasons: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String(MAX_ERROR_CHARS), nullable=True)
    # The coach looked at a NO-GO and said run it anyway.
    run_anyway: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    finished_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class ReportShare(Base):
    __tablename__ = "report_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(48), unique=True, index=True, nullable=False)
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    athlete_id: Mapped[int | None] = mapped_column(
        ForeignKey("athletes.id", ondelete="SET NULL"), nullable=True,
    )
    # The coach's own history row: a linked athlete reads theirs in the app.
    client_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    revoked_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
