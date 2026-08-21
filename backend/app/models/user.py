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
    # Raised 3 -> 10 when reading became the paid thing rather than uploading.
    # A teaser costs a CPU slot and nothing else -- no Gemini call, no overlay
    # render -- so a low cap bought us nothing and cost the visitor the second
    # clip, which is the one they film after changing something. This is an
    # anti-abuse ceiling now, not a product tier.
    TIER_STARTER: (10, "month"),
    TIER_ENTHUSIAST: (30, "month"),
    TIER_FULL: (120, "month"),
    TIER_ADMIN: (10, "day"),
}


# Equipment profiles per tier (see models/profile.py). One is enough to say
# "this is my bike"; keeping several -- road, TT, trainer, two pairs of shoes --
# is what makes a comparison BETWEEN setups possible, and that is the thing a
# single bought report structurally cannot give you. So it is where Full earns
# its price, rather than in a quota nobody reaches.
PROFILE_LIMITS: dict[str, int] = {
    TIER_STARTER: 1,
    TIER_ENTHUSIAST: 1,
    TIER_FULL: 10,
    TIER_ADMIN: 10,
}
DEFAULT_PROFILE_LIMIT = PROFILE_LIMITS[TIER_STARTER]


def profile_limit(tier: str | None) -> int:
    """How many live profiles this tier may keep; unknown tiers get the free one."""
    return PROFILE_LIMITS.get(tier or "", DEFAULT_PROFILE_LIMIT)


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
    # The analysis that spent this account's one free preview -- the single
    # report a free user gets in a form worth judging us by (see
    # services/result_gating.py). Stored as the JOB id rather than a timestamp
    # so a failed analysis can hand the preview back: burning somebody's one
    # good look at the product on a clip that errored is the worst possible
    # first impression, and it is exactly the run most likely to fail.
    free_preview_job_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    # The Stripe object (Checkout Session today, invoice once renewals grant
    # too) that last topped the balance up. Stripe retries a webhook until it
    # gets a 2xx, so "granted on subscription purchase" without this is
    # "granted once per delivery attempt".
    expert_credit_grant_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    # Standing height in centimetres. The only body measurement the analysis
    # asks for, and it buys one specific thing: a real scale. A side view
    # measures everything in fractions of the picture, so turning any of it
    # into centimetres needs one known length. Without this the pipeline
    # assumes a population-average torso, which corresponds to a 156 cm
    # person -- see running_analyzer._lock_body_scale.
    #
    # Nullable, and must stay so. It is optional to give, the analysis works
    # without it, and a required body measurement on a sign-up form is a
    # reason not to sign up.
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    @property
    def is_paid(self) -> bool:
        """True for any tier that unlocks full output (Enthusiast/Full/Admin)."""
        return self.tier in PAID_TIERS
