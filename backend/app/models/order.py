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

from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.core.db import Base

# Postgres in production, SQLite locally -- same as models/analysis.py.
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")

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

    # Which analysis was bought for review. Chosen by the athlete at checkout and
    # carried through Stripe metadata; without it the reviewer is guessing which
    # clip the customer meant, which is exactly how this used to work.
    # It is the CLIENT id (``analyses.client_id``), the same id history uses.
    analysis_client_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # The deliverable itself (see services/expert_review.py for the shape).
    # Written as a draft while status is ``in_review`` and only shown to the
    # customer once the order is ``delivered`` -- a half-finished report reaching
    # a paying customer is worse than a late one.
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE, nullable=True)
    # When the report was published. Distinct from ``updated_at_ms``, which moves
    # on every internal edit, including ones made after delivery.
    delivered_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # When the athlete first opened it. Drives the "you have an unread review"
    # card on the dashboard, and is the only signal we have that the thing they
    # paid for was actually read -- worth knowing before selling more of them.
    read_at_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
