"""A rider's word on a part: "it fitted" or "it did not", on their frame.

A queue for review, never a write into the data. The data is files read by
people (app/data/parts/README.md); a report becomes a record only when
somebody has looked at it and found a source or a second owner. So the
statuses in the files keep meaning what they say, and one confused rider --
or one bored one -- cannot mark a part as fitting a frame.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

MAX_NOTE_CHARS = 400


class PartReport(Base):
    __tablename__ = "part_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    frame_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    # A part on record, or NULL for "a part you do not list" (named in the note).
    part_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    fits: Mapped[bool] = mapped_column(Boolean, nullable=False)
    note: Mapped[str | None] = mapped_column(String(MAX_NOTE_CHARS), nullable=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    # Set when somebody has looked at it (and, if it held, edited the file).
    reviewed_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
