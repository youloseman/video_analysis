"""One-time purchase record (currently: the $29 Expert Review).

Subscriptions live on ``users.tier`` / ``users.subscription_status`` -- they are
self-describing, because the tier IS the thing the customer bought. A one-time
purchase has no such trace: it grants no tier and changes nothing about the
account, so without a row here a paid Expert Review exists only as a line in
the application log and a charge in Stripe. The customer sees no
acknowledgement, and nothing tracks whether the review was actually delivered.

``stripe_session_id`` is unique: Stripe retries webhooks, so the insert has to
be idempotent.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

# Fulfilment states for a one-time purchase.
ORDER_PAID = "paid"           # money in, not yet picked up
ORDER_IN_REVIEW = "in_review"  # Artur is working on it
ORDER_DELIVERED = "delivered"  # sent to the customer
ORDER_REFUNDED = "refunded"
ORDER_STATUSES = (ORDER_PAID, ORDER_IN_REVIEW, ORDER_DELIVERED, ORDER_REFUNDED)

# What the customer is told each state means. Kept server-side so the wording
# is identical in the account view and the admin queue.
ORDER_STATUS_LABEL = {
    ORDER_PAID: "Payment received — we'll start your review shortly.",
    ORDER_IN_REVIEW: "A coach is reviewing your clip right now.",
    ORDER_DELIVERED: "Your review has been delivered.",
    ORDER_REFUNDED: "This order was refunded.",
}


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # Checkout Session id -- the idempotency key for webhook retries.
    stripe_session_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False,
    )
    # Frontend plan key ("expert"), carried through Checkout metadata.
    plan: Mapped[str] = mapped_column(String(32), nullable=False)
    # Minor units exactly as Stripe reports them (2900 = $29.00).
    amount_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default=ORDER_PAID, server_default=ORDER_PAID, nullable=False,
    )
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    updated_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # Internal triage note; never shown to the customer.
    admin_note: Mapped[str | None] = mapped_column(Text, nullable=True)
