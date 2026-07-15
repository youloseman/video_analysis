"""User account model."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# Subscription tiers. Order matters for capability checks (higher = more).
TIER_STARTER = "starter"
TIER_ENTHUSIAST = "enthusiast"
TIER_FULL = "full"
TIER_ADMIN = "admin"
VALID_TIERS = (TIER_STARTER, TIER_ENTHUSIAST, TIER_FULL, TIER_ADMIN)

# Tiers that unlock the full (non-teaser) output + no watermark.
PAID_TIERS = (TIER_ENTHUSIAST, TIER_FULL, TIER_ADMIN)

# Analysis quota per tier: (max analyses, window). "month" = current calendar
# month (UTC); "day" = rolling 24h. Admin is deliberately a small DAILY cap
# (per Artur's request) rather than unlimited.
TIER_LIMITS: dict[str, tuple[int, str]] = {
    TIER_STARTER: (3, "month"),
    TIER_ENTHUSIAST: (30, "month"),
    TIER_FULL: (120, "month"),
    TIER_ADMIN: (5, "day"),
}


def tier_limit(tier: str) -> tuple[int, str]:
    """(max, window) for a tier; unknown tiers fall back to the free plan."""
    return TIER_LIMITS.get(tier, TIER_LIMITS[TIER_STARTER])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    # Subscription tier -- drives limits + teaser gating. Replaces the old
    # ``is_pro`` boolean (kept as a derived property for back-compat with the
    # frontend/token payload). New accounts start on the free ``starter`` tier.
    tier: Mapped[str] = mapped_column(
        String(20), default=TIER_STARTER, server_default=TIER_STARTER, nullable=False,
    )
    is_pro: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    @property
    def is_paid(self) -> bool:
        """True for any tier that unlocks full output (Enthusiast/Full/Admin)."""
        return self.tier in PAID_TIERS
