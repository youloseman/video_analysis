"""The $4 report, credited to the plan that follows it.

Paywall read of 2026-10-28: nine people bought a single $4 report and none of
them went on to a plan. The $4 and the $9 read as the same thing bought
twice, so the second purchase feels like paying for the first again. This
makes the first one count: a paid unlock within the window takes US$4 off
the first invoice of a subscription started afterwards, with a Stripe coupon
(``STRIPE_COUPON_UNLOCK_CREDIT``, fixed amount, duration once).

Decisions (Artur, 2026-10-29): a seven-day window; monthly and yearly alike;
several unlocks in the window credit once, off the latest. The credit is
shown only AFTER the unlock is paid -- never on the teaser card, where it
would read as a reason to buy the $4 first.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.order import ORDER_PAID, Order
from app.models.user import User

# Stripe plan key of a paid single report. A granted free unlock is recorded
# as "unlock_credit" with amount 0 and credits nothing: there was no $4.
PAID_UNLOCK_PLAN = "unlock"
DAY_MS = 86_400_000


def enabled() -> bool:
    """Only promise the credit when the coupon that pays it is configured."""
    return bool(settings.stripe_coupon_unlock_credit)


def window_ms() -> int:
    return settings.unlock_credit_window_days * DAY_MS


async def credited_unlock(db: AsyncSession, user: User, now_ms: int) -> Order | None:
    """The latest paid $4 report inside the window, or None.

    Paid means Stripe took money (``amount_total > 0``) and the order is still
    ``paid``: a refunded unlock credits nothing, and a report opened on a
    granted credit never cost anything to credit.
    """
    if not enabled():
        return None
    since = now_ms - window_ms()
    return (
        await db.execute(
            select(Order)
            .where(
                Order.user_id == user.id,
                Order.plan == PAID_UNLOCK_PLAN,
                Order.status == ORDER_PAID,
                Order.amount_total > 0,
                Order.created_at_ms >= since,
            )
            .order_by(Order.created_at_ms.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def credit_state(order: Order | None) -> dict[str, Any] | None:
    """What /auth/me tells the page: how much, until when, for which report."""
    if order is None:
        return None
    return {
        "amount_minor": int(order.amount_total or 0),
        "until_ms": int(order.created_at_ms) + window_ms(),
        "entry_id": order.analysis_client_id,
        "order_id": order.id,
    }
