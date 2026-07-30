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
from typing import Any

import stripe
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.core.security import get_current_user
from app.models.user import TIER_ADMIN, TIER_STARTER, User

logger = structlog.get_logger()
router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutIn(BaseModel):
    # Matches the frontend startCheckout() keys.
    plan: str


def _require_stripe() -> None:
    if not settings.stripe_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Billing is not configured yet.",
        )
    stripe.api_key = settings.stripe_secret_key


def _base_url(request: Request) -> str:
    return settings.public_base_url or str(request.base_url).rstrip("/")


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
    price_id = settings.plan_price_map.get(body.plan)
    if not price_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Unknown or unconfigured plan.",
        )

    # Reuse the user's Stripe customer across purchases; create on first use.
    customer_id = user.stripe_customer_id
    if not customer_id:
        customer = stripe.Customer.create(
            email=user.email, metadata={"user_id": str(user.id)},
        )
        customer_id = customer.id
        user.stripe_customer_id = customer_id
        await db.commit()

    is_subscription = body.plan != "expert"
    base = _base_url(request)
    try:
        session_obj = stripe.checkout.Session.create(
            mode="subscription" if is_subscription else "payment",
            customer=customer_id,
            client_reference_id=str(user.id),
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{base}/?checkout=success",
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


async def _on_checkout_completed(db: AsyncSession, obj: Any) -> None:
    user = await _user_by_id(db, obj.get("client_reference_id"))
    if user is None:
        user = await _user_by_customer(db, obj.get("customer"))
    if user is None:
        logger.warning("WEBHOOK_NO_USER", event="checkout.completed")
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
        # One-time Expert Review: no tier change; fulfilled manually.
        logger.info("EXPERT_REVIEW_PURCHASED", user_id=user.id)


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
