"""What a coach keeps about a squad beyond the roster row itself.

Three things a coach asked the Squad screen for and the roster could not say:

* **Races** -- when each athlete races. It is how a coach decides whose clip
  to open first; the free-text ``athletes.note`` could hold "70.3 in December"
  but nothing could sort by it.
* **Clip reviews** -- "I have looked at this one". Until now the only trace of
  a coach's attention was a written note, and not every clip earns a note.
  The mark is the coach's own: one row per coach per analysis, so a linked
  athlete's clip can be marked without writing anything onto the athlete's
  entry, and two coaches never share a mark.
* **Groups** -- a club has squads inside it (the long-course group, the
  juniors). An athlete can sit in several.

New tables, all of them, rather than columns on ``athletes`` or ``analyses``:
``create_all`` builds a new table on any database, and the hand-written ALTER
list is the thing that took production down once (core/db.py). No booleans and
no server defaults here either, for the same reason.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

MAX_RACE_NAME_CHARS = 80
MAX_RACES_PER_ATHLETE = 12
# A = the race the season is built around, C = a training race.
RACE_PRIORITIES = ("A", "B", "C")
DEFAULT_PRIORITY = "A"

MAX_GROUP_NAME_CHARS = 40
MAX_GROUPS = 30


class AthleteRace(Base):
    __tablename__ = "athlete_races"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    athlete_id: Mapped[int] = mapped_column(
        ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # The roster's owner, kept on the row so every read scopes by it.
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    name: Mapped[str] = mapped_column(String(MAX_RACE_NAME_CHARS), nullable=False)
    # ISO date (YYYY-MM-DD), no time of day -- the convention ``cda_measurements``
    # set. A string sorts and compares correctly in that form on any database.
    race_on: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    priority: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ClipReview(Base):
    __tablename__ = "clip_reviews"
    __table_args__ = (
        UniqueConstraint("coach_user_id", "analysis_id", name="uq_clip_review_coach_analysis"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # The analysis row itself, the coach's own or a linked athlete's: deleting
    # the analysis takes the mark with it.
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    reviewed_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AthleteGroup(Base):
    __tablename__ = "athlete_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    coach_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    name: Mapped[str] = mapped_column(String(MAX_GROUP_NAME_CHARS), nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AthleteGroupMember(Base):
    __tablename__ = "athlete_group_members"

    group_id: Mapped[int] = mapped_column(
        ForeignKey("athlete_groups.id", ondelete="CASCADE"), primary_key=True,
    )
    athlete_id: Mapped[int] = mapped_column(
        ForeignKey("athletes.id", ondelete="CASCADE"), primary_key=True, index=True,
    )
