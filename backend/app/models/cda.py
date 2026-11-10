"""Measured CdA: a drag number the athlete (or their coach) brought from a
field test, a wind tunnel or a CFD service, kept on the setup it was measured
on.

Flapp does not calculate these. The aero panel's zone is an estimate from the
trunk angle and says so; a row here is somebody's measurement, with where it
came from and how it was taken, because one ordinary ride can move by more
than the difference between two setups (John Bye, 2026-11-08). The source and
the protocol are what decide whether two numbers may be compared at all.

A measurement belongs to a profile and goes with it: deleting the bike deletes
what was measured on it. Archiving keeps both.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# Where the number came from. "field_test" is the rider's own protocol
# (Chung / virtual elevation on loops) without a named tool.
CDA_SOURCES = ("windsock", "notio", "airo", "wind_tunnel", "field_test", "other")
# How it was taken. "not_a_ride" is a tunnel or a CFD model: no wind, no road.
CDA_PROTOCOLS = ("repeated_loops", "single_ride", "not_a_ride")
# Sources that are never a ride, whatever the form says.
NOT_A_RIDE_SOURCES = frozenset({"airo", "wind_tunnel"})

# m^2. A pro TT position is ~0.18-0.20, an upright road one ~0.35-0.40. Outside
# this it is a typo or another unit (W/CdA, ft^2), not a position.
CDA_RANGE = (0.12, 0.60)
MAX_NOTE_CHARS = 280


class CdaMeasurement(Base):
    __tablename__ = "cda_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # The profile's owner, kept on the row so every read scopes by it.
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    value_m2: Mapped[float] = mapped_column(Float, nullable=False)
    # ISO date (YYYY-MM-DD): the day it was measured, no time of day.
    measured_on: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(String(MAX_NOTE_CHARS), nullable=True)
    # Who typed it: the owner, or a linked coach writing on the athlete's
    # setup. The athlete sees "added by your coach" and can delete it.
    entered_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
