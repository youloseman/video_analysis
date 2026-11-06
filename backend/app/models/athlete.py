"""An athlete on a coach's roster.

The coach runs other people's clips. Until now the only place to say whose
clip it was came from the equipment profile -- "one profile per athlete" -- which
fights what a profile is: a setup with a crank length and a position, not a
person. A coach with six athletes on two bikes each ran out of profiles, and
"Two setups" started comparing people instead of bikes.

So the person gets a row of their own. An athlete is a label the coach owns:
analyses and profiles point at it, history and trends filter by it, the squad
table has one row per athlete, and the printed report names them instead of
the account's email. It is NOT an account -- the athlete does not sign in;
``linked_user_id`` is where that goes if they ever do.

``upload_token`` is the intake link: a public page at ``/a/{token}`` where the
athlete drops a clip without signing up, and it lands in this row under the
coach's quota. Minted on creation so the link exists the moment the athlete
does; rotated by the coach if it leaks.

A new table rather than columns anywhere else: ``create_all`` builds it on any
database, and the hand-written ALTER list for ``users`` is exactly the kind
of thing that took production down for an afternoon once.
"""

from __future__ import annotations

import secrets

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

MAX_NAME_CHARS = 60
MAX_NOTE_CHARS = 240
# What the athlete does with this coach. "both" is the triathlete.
ATHLETE_SPORTS = ("run", "bike", "both")
DEFAULT_SPORT = "both"
TOKEN_CHARS = 22


def new_upload_token() -> str:
    """URL-safe, 22 characters: 128 bits, unguessable, short enough to read
    out loud at a pool deck."""
    return secrets.token_urlsafe(16)[:TOKEN_CHARS]


class Athlete(Base):
    __tablename__ = "athletes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    name: Mapped[str] = mapped_column(String(MAX_NAME_CHARS), nullable=False)
    # Where a report goes when the coach sends it (phase 3). Optional: a coach
    # who hands reports over in person never needs it.
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    sport: Mapped[str] = mapped_column(
        String(8), default=DEFAULT_SPORT, server_default=DEFAULT_SPORT, nullable=False,
    )
    # The coach's one line about them: goal race, what they are working on.
    note: Mapped[str | None] = mapped_column(String(MAX_NOTE_CHARS), nullable=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Left the squad, but still owns the history filmed while on it.
    archived: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False,
    )
    upload_token: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False, default=new_upload_token,
    )
    # The athlete's own Flapp account, once they have one and the coach has
    # linked it (phase 5). NULL today for everyone.
    linked_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
