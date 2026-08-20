"""Equipment profiles: which bike (or which shoes) an analysis was filmed on.

Technique is measured against a setup, and the setup changes. A rider on a road
bike and the same rider on a TT bike are two different sets of angles, both
correct; averaging them into one history produces a trend line that describes
nobody. Without somewhere to say which was which, "before / after" could only
ever mean "before and after in time", which is the wrong axis the moment a
second bike exists.

That is the Full tier's actual differentiator, and the honest reason the
comparison it enables cannot be sold as a one-off report: two profiles are, by
definition, more than one analysis.

Archived rather than deleted by default -- a retired bike still owns the
history that was filmed on it, and removing the profile would orphan the only
explanation for why those numbers look different.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# What a profile is. Kept loose on purpose: it labels a setup, and the useful
# labels are the ones the athlete would use out loud.
PROFILE_KINDS = ("road", "tt", "trainer", "gravel", "mtb", "shoes", "other")
DEFAULT_KIND = "other"

MAX_NAME_CHARS = 60


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    name: Mapped[str] = mapped_column(String(MAX_NAME_CHARS), nullable=False)
    # "run" | "bike" -- which analyses may be filed under it. A profile is not
    # cross-sport: comparing a running cadence to a pedalling one is nonsense,
    # and letting the two share a profile would invite exactly that chart.
    sport: Mapped[str] = mapped_column(String(16), nullable=False)
    kind: Mapped[str] = mapped_column(
        String(16), default=DEFAULT_KIND, server_default=DEFAULT_KIND,
        nullable=False,
    )
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Retired, but still owns its history. Hidden from pickers, kept in filters.
    archived: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False,
    )
