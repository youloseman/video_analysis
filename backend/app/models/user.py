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
# month (UTC); "day" = rolling 24h. Admin is deliberately a DAILY cap (per
# Artur's request) rather than unlimited -- it bounds how much CPU our own
# testing can burn. Raised 5 -> 10 because a live demo of the product runs
# through five clips before the conversation is over.
TIER_LIMITS: dict[str, tuple[int, str]] = {
    TIER_STARTER: (3, "month"),
    TIER_ENTHUSIAST: (30, "month"),
    TIER_FULL: (120, "month"),
    TIER_ADMIN: (10, "day"),
}


# Expert Reviews the Full tier includes per paid term. The pricing card says
# "1 Expert Review included ($39 value)", and this is the number that makes
# that sentence true rather than aspirational.
FULL_EXPERT_CREDITS = 1


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
    # --- Billing (Stripe, Stage 4). Null until the user starts a checkout. ---
    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True,
    )
    # Latest subscription status from Stripe: active | trialing | past_due |
    # canceled | unpaid | incomplete ... (None = no subscription).
    subscription_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Unspent Expert Reviews the account is entitled to (the Full tier includes
    # one). A credit is a *paid deliverable we owe*, so it is a balance on the
    # account rather than a flag derived from the tier: it must survive a
    # downgrade (they paid for the year), and spending it must be a single
    # decrement that cannot happen twice.
    expert_credits: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False,
    )
    # The Stripe object (Checkout Session today, invoice once renewals grant
    # too) that last topped the balance up. Stripe retries a webhook until it
    # gets a 2xx, so "granted on subscription purchase" without this is
    # "granted once per delivery attempt".
    expert_credit_grant_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    @property
    def is_paid(self) -> bool:
        """True for any tier that unlocks full output (Enthusiast/Full/Admin)."""
        return self.tier in PAID_TIERS
