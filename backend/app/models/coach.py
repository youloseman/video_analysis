"""The coach behind a Full account: a name, a mark, a library of cues.

A Full subscriber who coaches hands these reports to athletes. What the
report lacked was the coach: it said Flapp at the top and nothing about who
read it, and every note the coach wanted to add lived in a message beside
the link. This is the account-level half of that -- who the coach is and the
phrases they reach for -- while the note on each report lives on the
analysis entry itself (``Analysis.data.coachNote``), where the report is.

One row per user, JSON inside: the fields are a handful of strings and one
small image, and their shape will move faster than a schema should. A new
table rather than a column on ``users``: ``create_all`` builds it on any
database, and the users table's hand-written migration list is exactly the
kind of thing that took production down for an afternoon once.
"""

from __future__ import annotations

from sqlalchemy import JSON, BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

MAX_NAME_CHARS = 60
MAX_TAGLINE_CHARS = 120
MAX_CUES = 40
MAX_CUE_CHARS = 200
# A mark, not a poster: the printed header shows it ~48 px tall. Resized
# client-side before upload; this is the ceiling the server holds.
MAX_LOGO_BYTES = 200 * 1024
LOGO_PREFIXES = ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")


class CoachProfile(Base):
    __tablename__ = "coach_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True,
    )
    updated_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # {name, tagline, logo (data URI), cues: [str]}
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
