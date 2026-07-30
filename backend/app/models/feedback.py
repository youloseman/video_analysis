"""Micro-feedback on an analysis result -- "does this match what you see?".

One row per rating. The point is not a support inbox: it is a *labelled dataset*
over the analysis pipeline. Each row carries the rating plus the machine context
that produced the result (sport, position, score, quality-gate flags, confidence
tier), so the 👎 rate can be sliced by those flags and the scoring thresholds /
coaching prompt tuned against evidence instead of intuition.

Feedback is private: only the admin tier reads it (see ``app/api/feedback.py``).
The annotated keyframe is stored in its own column so listings never drag a
~150 KB base64 blob per row -- same split as ``Analysis`` in ``app/api/me.py``.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# Triage lifecycle. "new" until Artur looks at it.
STATUS_NEW = "new"
VALID_STATUSES = (STATUS_NEW, "triaged", "fixed", "wontfix")

# What the rating is about. Only "analysis" is submitted today; the column
# exists so a general idea/bug channel can share the table later without a
# migration.
KIND_ANALYSIS = "analysis"


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)

    # Signed-in author, if any. SET NULL (not CASCADE): the pipeline signal is
    # worth keeping after an account goes away. NOTE: when an account-deletion
    # flow lands, it must also null ``email`` on these rows -- the FK going null
    # does not scrub the contact copy below.
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True,
    )
    # Copied at submission so a reply is possible without a join. Null for
    # anonymous (free, not-signed-in) visitors.
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    # Salted hash of the client IP -- never the IP itself. Used to dedupe and
    # rate-limit anonymous submissions.
    ip_hash: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)

    kind: Mapped[str] = mapped_column(
        String(16), default=KIND_ANALYSIS, nullable=False,
    )
    # +1 = "looks right", -1 = "something's off", NULL = comment with no verdict.
    rating: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    # Quick-pick reasons, e.g. ["didnt_detect_me", "angles_wrong"].
    tags: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The history entry id the rating belongs to (``h<timestamp>``, same id as
    # ``Analysis.client_id``) -- joins the rating to the stored analysis for
    # signed-in users. Not a FK: anonymous visitors have no stored analysis.
    analysis_client_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True,
    )
    # Machine context of the rated result (sport, score, gate flags, ...). This
    # is the column that makes the table analysable -- see ``/admin/feedback/stats``.
    context: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # Annotated keyframe (data URI), only when the user ticked the consent box.
    keyframe: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), default=STATUS_NEW, server_default=STATUS_NEW,
        index=True, nullable=False,
    )
    admin_note: Mapped[str | None] = mapped_column(Text, nullable=True)
