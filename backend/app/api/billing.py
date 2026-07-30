"""Stripe billing: hosted Checkout + webhook + customer portal (Stage 4).

Flow
    POST /billing/checkout  -> creates a Checkout Session, returns its URL.
    (Stripe hosts the payment page; card data never touches this server.)
    POST /billing/webhook   -> Stripe calls this on payment/subscription events;
    we verify the signature and set the user's ``tier`` accordingly.
    POST /billing/portal    -> link to Stripe's hosted subscription-management page.

Subscriptions map to tiers (enthusiast/full); the one-time Expert Review does
not change tier (it's fulfilled manually). Admin accounts are never downgraded
by billing events.

All endpoints 503 when Stripe isn't configured (no STRIPE_SECRET_KEY), so the
frontend degrades to a "coming soon" message instead of erroring.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import stripe
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.core.security import get_current_user, require_admin
from app.models.order import (
    ORDER_PAID,
    ORDER_STATUS_LABEL,
    ORDER_STATUSES,
    Order,
)
from app.models.user import TIER_ADMIN, TIER_STARTER, User


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

logger = structlog.get_logger()
router = APIRouter(prefix="/billing", tags=["billing"])
# Admin routes live off /admin (not /billing), so they need their own router.
admin_router = APIRouter(tags=["billing"])


class CheckoutIn(BaseModel):
    # Matches the frontend startCheckout() keys.
    plan: str


def log_billing_configuration() -> None:
    """Report, once at startup, whether we can actually take money.

    Every misconfiguration here is silent by construction: the pricing page
    renders all three tiers whatever the environment says, so a missing price ID
    shows up only as one dead button on the highest-intent screen in the
    product, and a test-mode key shows up only as real cards being declined at
    the very end of checkout. Neither raises anything. So say it out loud while
    someone is still reading the deploy log.
    """
    if not settings.stripe_enabled:
        logger.info("BILLING_DISABLED", reason="STRIPE_SECRET_KEY unset")
        return
    missing = sorted(p for p, price in settings.plan_price_map.items() if not price)
    if missing:
        logger.error("BILLING_PRICES_MISSING", plans=missing)
    else:
        logger.info("BILLING_READY", plans=sorted(settings.plan_price_map))
    if str(settings.stripe_secret_key or "").startswith("sk_test_"):
        logger.warning(
            "BILLING_TEST_MODE",
            detail="Stripe key is a TEST key -- live cards will be declined",
        )


def _require_stripe() -> None:
    if not settings.stripe_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Billing is not configured yet.",
        )
    stripe.api_key = settings.stripe_secret_key


def _base_url(request: Request) -> str:
    return settings.public_base_url or str(request.base_url).rstrip("/")


# A subscription in one of these states is still billing the customer, so a
# second Checkout would stack a second subscription on the same card rather than
# switch plans. ``incomplete``/``canceled``/``unpaid`` are deliberately absent:
# those never charge again, so a fresh Checkout is the right move.
LIVE_SUB_STATUSES = ("active", "trialing", "past_due")


def has_live_subscription(user: User) -> bool:
    """True when the user already pays for a plan (so send them to the portal)."""
    return bool(
        user.stripe_customer_id
        and user.subscription_status in LIVE_SUB_STATUSES
    )


def _portal_url(customer_id: str, return_to: str) -> str:
    return stripe.billing_portal.Session.create(
        customer=customer_id, return_url=return_to,
    ).url


def tier_for_plan(plan: str) -> str | None:
    """Subscription tier a plan grants (None for the one-time Expert Review)."""
    return {
        "enthusiast_monthly": "enthusiast",
        "enthusiast_yearly": "enthusiast",
        "full_yearly": "full",
    }.get(plan)


def tier_for_price(price_id: str | None) -> str | None:
    """Tier a Stripe price maps to (from config); None if unknown/one-time."""
    if not price_id:
        return None
    return settings.price_tier_map.get(price_id)


@router.post("/checkout")
async def create_checkout(
    body: CheckoutIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    _require_stripe()
    # Two different failures were sharing one 400 and one internal-sounding
    # string, which the client puts straight in front of the customer.
    if body.plan not in settings.plan_price_map:
        # A plan key we do not sell. The caller is wrong; 400 is right.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unknown plan.")
    price_id = settings.plan_price_map[body.plan]
    if not price_id:
        # A plan we DO sell, whose Stripe price was never configured. That is
        # our mistake, not the caller's -- and it is invisible until somebody
        # clicks Buy and gets a dead button. Shout in the logs, and give the
        # customer a sentence written for them rather than for us.
        logger.error("PLAN_PRICE_UNCONFIGURED", plan=body.plan, user_id=user.id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "This plan can't be purchased right now. Please try again shortly "
            "— or email support@getflapp.com and we'll sort it out.",
        )

    is_subscription = body.plan != "expert"
    base = _base_url(request)

    # Already subscribed? A second Checkout does not upgrade anyone -- it creates
    # a SECOND subscription and charges for both. Stripe's portal is the only
    # place a plan can actually be switched, so send them there and let the
    # client explain why.
    if is_subscription and has_live_subscription(user):
        try:
            url = _portal_url(str(user.stripe_customer_id), f"{base}/")
        except stripe.StripeError as e:  # noqa: BLE001
            logger.warning("PORTAL_FAILED", err=str(e), user_id=user.id)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "Could not open billing portal.",
            )
        logger.info("CHECKOUT_REDIRECTED_TO_PORTAL", user_id=user.id, plan=body.plan)
        return {"url": url, "mode": "portal"}

    # Reuse the user's Stripe customer across purchases; create on first use.
    customer_id = user.stripe_customer_id
    if not customer_id:
        customer = stripe.Customer.create(
            email=user.email, metadata={"user_id": str(user.id)},
        )
        customer_id = customer.id
        user.stripe_customer_id = customer_id
        await db.commit()

    try:
        session_obj = stripe.checkout.Session.create(
            mode="subscription" if is_subscription else "payment",
            customer=customer_id,
            client_reference_id=str(user.id),
            line_items=[{"price": price_id, "quantity": 1}],
            # Carry the plan back so the client can acknowledge a one-time Expert
            # Review (which grants no tier, so tier-polling would never confirm it).
            success_url=f"{base}/?checkout=success&plan={body.plan}",
            cancel_url=f"{base}/?checkout=cancel",
            allow_promotion_codes=True,
            metadata={
                "user_id": str(user.id),
                "plan": body.plan,
                "tier": tier_for_plan(body.plan) or "",
            },
        )
    except stripe.StripeError as e:  # noqa: BLE001
        logger.warning("CHECKOUT_FAILED", err=str(e), user_id=user.id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not start checkout.")
    return {"url": session_obj.url}


# --------------------------------------------------------------------------
# Orders (one-time purchases). The customer needs somewhere to see that their
# Expert Review was received and where it stands; the admin needs a queue.
# --------------------------------------------------------------------------
def _order_out(o: Order) -> dict[str, Any]:
    return {
        "id": o.id,
        "plan": o.plan,
        "status": o.status,
        "status_label": ORDER_STATUS_LABEL.get(o.status, o.status),
        "amount_total": o.amount_total,
        "currency": o.currency,
        "created_at_ms": o.created_at_ms,
        "updated_at_ms": o.updated_at_ms,
    }


@router.get("/orders")
async def my_orders(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """The caller's one-time purchases, newest first. No Stripe call needed."""
    rows = (
        await db.execute(
            select(Order)
            .where(Order.user_id == user.id)
            .order_by(Order.created_at_ms.desc())
            .limit(50)
        )
    ).scalars().all()
    return [_order_out(o) for o in rows]


@admin_router.get("/admin/orders")
async def admin_orders(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """Fulfilment queue: every one-time purchase with the buyer's email."""
    rows = (
        await db.execute(
            select(Order, User.email)
            .join(User, User.id == Order.user_id)
            .order_by(Order.created_at_ms.desc())
            .limit(200)
        )
    ).all()
    return [
        {**_order_out(o), "email": email, "admin_note": o.admin_note}
        for o, email in rows
    ]


class OrderPatch(BaseModel):
    status: str | None = None
    admin_note: str | None = None


@admin_router.patch("/admin/orders/{order_id}")
async def admin_update_order(
    order_id: int,
    body: OrderPatch,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Move an order through the fulfilment states / leave an internal note."""
    order = await db.get(Order, order_id)
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if body.status is not None:
        if body.status not in ORDER_STATUSES:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"status must be one of {list(ORDER_STATUSES)}",
            )
        order.status = body.status
    if body.admin_note is not None:
        order.admin_note = body.admin_note.strip()[:2000] or None
    order.updated_at_ms = _now_ms()
    await db.commit()
    return {**_order_out(order), "admin_note": order.admin_note}


@router.post("/portal")
async def customer_portal(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    _require_stripe()
    if not user.stripe_customer_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "No billing account yet — subscribe first.",
        )
    try:
        portal = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id, return_url=f"{_base_url(request)}/",
        )
    except stripe.StripeError as e:  # noqa: BLE001
        logger.warning("PORTAL_FAILED", err=str(e), user_id=user.id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not open billing portal.")
    return {"url": portal.url}


@router.post("/webhook")
async def webhook(
    request: Request, db: AsyncSession = Depends(get_session),
) -> dict[str, bool]:
    if not settings.stripe_enabled or not settings.stripe_webhook_secret:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Billing is not configured.",
        )
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        # Verify the signature (raises on tampering); we ignore the returned
        # StripeObject and process the raw JSON as a plain dict instead -- the
        # StripeObject's attribute/.get() semantics differ from a dict and broke
        # the handlers, and this keeps prod on the exact shape the tests cover.
        stripe.Webhook.construct_event(
            payload, sig, settings.stripe_webhook_secret,
        )
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid signature")
    event = json.loads(payload)
    await _apply_event(db, event)
    return {"received": True}


# --------------------------------------------------------------------------
# Event handling (kept separate from the HTTP layer so it is unit-testable
# without a live Stripe / a real signature).
# --------------------------------------------------------------------------
async def _user_by_id(db: AsyncSession, uid: Any) -> User | None:
    try:
        return await db.get(User, int(uid))
    except (TypeError, ValueError):
        return None


async def _user_by_customer(db: AsyncSession, customer_id: Any) -> User | None:
    if not customer_id:
        return None
    return (
        await db.execute(
            select(User).where(User.stripe_customer_id == str(customer_id))
        )
    ).scalar_one_or_none()


def _set_tier(user: User, tier: str) -> None:
    """Set a user's tier unless they're admin (billing never touches admins)."""
    if user.tier != TIER_ADMIN:
        user.tier = tier


async def _apply_event(db: AsyncSession, event: Any) -> None:
    etype = event["type"]
    obj = event["data"]["object"]
    if etype == "checkout.session.completed":
        await _on_checkout_completed(db, obj)
    elif etype in ("customer.subscription.created", "customer.subscription.updated"):
        await _on_subscription_change(db, obj)
    elif etype == "customer.subscription.deleted":
        await _on_subscription_deleted(db, obj)
    else:
        return
    await db.commit()


async def _record_order(db: AsyncSession, user: User, obj: Any) -> None:
    """Insert (once) the one-time purchase this Checkout Session paid for.

    Idempotent on the session id: Stripe retries a webhook until it gets a 2xx,
    so the same ``checkout.session.completed`` can arrive several times.
    """
    session_id = str(obj.get("id") or "")
    if not session_id:
        logger.warning("ORDER_NO_SESSION_ID", user_id=user.id)
        return
    existing = (
        await db.execute(
            select(Order).where(Order.stripe_session_id == session_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return
    now = _now_ms()
    db.add(Order(
        user_id=user.id,
        stripe_session_id=session_id,
        plan=str((obj.get("metadata") or {}).get("plan") or "expert")[:32],
        amount_total=obj.get("amount_total"),
        currency=(obj.get("currency") or None),
        status=ORDER_PAID,
        created_at_ms=now,
        updated_at_ms=now,
    ))
    logger.info(
        "ORDER_RECORDED", user_id=user.id, session=session_id,
        amount=obj.get("amount_total"), currency=obj.get("currency"),
    )


async def _on_checkout_completed(db: AsyncSession, obj: Any) -> None:
    user = await _user_by_id(db, obj.get("client_reference_id"))
    if user is None:
        user = await _user_by_customer(db, obj.get("customer"))
    if user is None:
        # NB: not ``event=`` -- structlog reserves that kwarg for the message
        # itself, so passing it raises TypeError and turns an unmatched webhook
        # into a 500 that Stripe then retries forever.
        logger.warning("WEBHOOK_NO_USER", stripe_event="checkout.session.completed")
        return
    # Keep the customer id linked (in case it was created Stripe-side).
    if obj.get("customer") and not user.stripe_customer_id:
        user.stripe_customer_id = str(obj["customer"])

    if obj.get("mode") == "subscription":
        meta = obj.get("metadata") or {}
        tier = meta.get("tier") or None
        if tier:
            _set_tier(user, tier)
            user.subscription_status = "active"
            logger.info("SUB_ACTIVATED", user_id=user.id, tier=tier)
    else:
        # One-time Expert Review: grants no tier, so nothing about the account
        # would otherwise change. Record the order -- it is the only trace the
        # customer (and the fulfilment queue) has that they paid.
        await _record_order(db, user, obj)


def _price_from_subscription(obj: Any) -> str | None:
    try:
        return obj["items"]["data"][0]["price"]["id"]
    except (KeyError, IndexError, TypeError):
        return None


async def _on_subscription_change(db: AsyncSession, obj: Any) -> None:
    user = await _user_by_customer(db, obj.get("customer"))
    if user is None:
        return
    status_str = obj.get("status")
    user.subscription_status = status_str
    tier = tier_for_price(_price_from_subscription(obj))
    if status_str in ("active", "trialing") and tier:
        _set_tier(user, tier)
        logger.info("SUB_SYNCED", user_id=user.id, tier=tier, status=status_str)


async def _on_subscription_deleted(db: AsyncSession, obj: Any) -> None:
    user = await _user_by_customer(db, obj.get("customer"))
    if user is None:
        return
    user.subscription_status = "canceled"
    _set_tier(user, TIER_STARTER)
    logger.info("SUB_CANCELED", user_id=user.id)
