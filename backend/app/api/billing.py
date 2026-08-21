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
from pathlib import Path
from typing import Any

import stripe
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import jobs
from app.core.config import settings
from app.core.db import get_session
from app.core.security import get_current_user, require_admin
from app.models.analysis import Analysis
from app.models.order import (
    ORDER_DELIVERED,
    ORDER_PAID,
    ORDER_REFUNDED,
    ORDER_STATUS_LABEL,
    ORDER_STATUSES,
    Order,
)
from app.models.user import (
    FULL_EXPERT_CREDITS,
    TIER_ADMIN,
    TIER_FULL,
    TIER_STARTER,
    User,
)
from app.services import analytics, expert_review, notify, pricing


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _amount_major(cents: Any) -> float | None:
    """Stripe's minor units as a plain amount, for the revenue properties.

    Analytics wants dollars; Stripe speaks cents. Anything unparseable comes
    back as ``None`` rather than as a zero-dollar sale.
    """
    try:
        return round(int(cents) / 100, 2)
    except (TypeError, ValueError):
        return None

logger = structlog.get_logger()
router = APIRouter(prefix="/billing", tags=["billing"])
# Admin routes live off /admin (not /billing), so they need their own router.
admin_router = APIRouter(tags=["billing"])


class CheckoutIn(BaseModel):
    # Matches the frontend startCheckout() keys.
    plan: str
    # Which analysis an Expert Review is being bought for. The athlete picks it
    # before paying; without it the reviewer has to email and ask which clip was
    # meant. Ignored for subscription plans. It is the client-side id, the same
    # one history and /me/analyses use.
    analysis_client_id: str | None = None


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

    # Two plans pointing at one Stripe price. Always a copy-paste slip -- the
    # ids are long, similar, and gathered by hand from five different screens --
    # and it is invisible from inside: checkout succeeds, Stripe charges, and
    # the customer is simply billed for the plan they did not choose. Monthly
    # and yearly are the pair this happens to, because they live in the same
    # product two rows apart.
    seen: dict[str, str] = {}
    for plan in sorted(settings.plan_price_map):
        price = settings.plan_price_map[plan]
        if not price:
            continue
        if price in seen:
            logger.error(
                "BILLING_PRICE_REUSED", price=price, plans=[seen[price], plan],
                detail=(
                    "two plans share one Stripe price -- whoever picks either "
                    "one is charged the same amount. Check the price ids."
                ),
            )
        else:
            seen[price] = plan
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


@router.get("/expert-review/slots")
async def expert_review_slots(
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Whether a review can be taken on right now. Public: the card asks before
    it offers a button, and "fully booked" is better said before payment."""
    left = await review_slots_left(db)
    return {"left": left, "open": left is None or left > 0}


@router.get("/plans")
def plans() -> dict[str, Any]:
    """The price list, for anyone -- signed in or not.

    Public and unauthenticated on purpose: it is the same information the
    landing page prints in plain HTML, and the pricing screen has to render
    for a visitor deciding whether to create an account at all.

    Every price carries an ``available`` flag derived from whether its Stripe
    id is configured, so a half-configured deploy renders an explanation
    instead of a button that 503s when pressed.
    """
    return pricing.catalog(settings.plan_price_map)


# --------------------------------------------------------------------------
# Expert Review credits (the "1 Expert Review included" on the Full tier).
#
# A credit is a deliverable we owe, not a capability the tier confers, so it
# lives as a balance on the account rather than as a check against the tier:
#   * it must outlive a downgrade -- they paid for the term that included it;
#   * spending it must be one decrement that cannot happen twice;
#   * granting it must be idempotent, because Stripe redelivers webhooks until
#     it gets a 2xx and "grant on purchase" would otherwise mean "grant per
#     delivery attempt".
# --------------------------------------------------------------------------
def grant_expert_credits(user: User, ref: str, count: int = FULL_EXPERT_CREDITS) -> bool:
    """Top the balance up once per Stripe object. True if anything changed.

    ``ref`` is the identity of the *payment* (a Checkout Session id today; an
    invoice id when renewals start granting too), not of the user or the
    subscription -- that is what makes a redelivered event a no-op while a
    genuine second purchase still counts.
    """
    if not ref or user.expert_credit_grant_ref == ref:
        return False
    user.expert_credits = (user.expert_credits or 0) + count
    user.expert_credit_grant_ref = ref
    return True


WEEK_MS = 7 * 24 * 60 * 60 * 1000


async def review_slots_left(db: AsyncSession) -> int | None:
    """How many Expert Reviews may still be taken on this week.

    ``None`` means no ceiling is configured. Counts orders *placed* in the last
    seven days rather than ones still open: the work is owed from the moment
    somebody pays, and letting the queue drain does not give back the hours
    already committed to it.

    This is the only product here whose supply is finite -- everything else
    scales with CPU. Selling more of it than one person can write turns a
    premium deliverable into a queue of people waiting on an apology.
    """
    cap = settings.expert_review_slots_per_week
    if cap <= 0:
        return None
    since = _now_ms() - WEEK_MS
    taken = int(
        await db.scalar(
            select(func.count(Order.id)).where(
                Order.plan.in_(("expert", "expert_credit")),
                Order.status != ORDER_REFUNDED,
                Order.created_at_ms >= since,
            )
        )
        or 0
    )
    return max(0, cap - taken)


async def _require_review_slot(db: AsyncSession) -> None:
    left = await review_slots_left(db)
    if left is None or left > 0:
        return
    # 503, not 402: they are not being asked for money, and the thing they
    # wanted exists -- there is simply none of this week left.
    logger.info("REVIEW_SLOTS_FULL", cap=settings.expert_review_slots_per_week)
    raise HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "This week's Expert Reviews are fully booked. They open again on "
        "Monday — email support@getflapp.com if yours is time-sensitive.",
    )


def _credit_session_key(user_id: int, analysis_client_id: str) -> str:
    """Idempotency key standing in for a Stripe session on a credit order.

    ``orders.stripe_session_id`` is NOT NULL and unique, which is exactly the
    guard a redemption needs: double-clicking "Use my included review" races
    two identical requests, and the second one loses to the constraint instead
    of spending a second credit.
    """
    return f"credit:{user_id}:{analysis_client_id}"


async def _redeem_expert_credit(
    db: AsyncSession, user: User, analysis_client_id: str,
) -> dict[str, Any]:
    """Spend one credit: create the order, decrement, never touch Stripe.

    Deliberately does not go through Checkout even for $0 -- a zero-amount
    Checkout Session still asks for a payment method, which is a strange thing
    to show someone who already paid for this a year ago.
    """
    key = _credit_session_key(user.id, analysis_client_id)
    now = _now_ms()
    order = Order(
        user_id=user.id,
        stripe_session_id=key,
        plan="expert_credit",
        amount_total=0,
        currency=pricing.CURRENCY,
        status=ORDER_PAID,
        created_at_ms=now,
        updated_at_ms=now,
        analysis_client_id=analysis_client_id,
    )
    db.add(order)
    user.expert_credits = max(0, (user.expert_credits or 0) - 1)
    try:
        await db.commit()
    except IntegrityError:
        # Same user, same clip, already redeemed -- the unique constraint did
        # its job. Hand back the order that exists rather than an error: from
        # where the customer is standing, the thing they asked for is done.
        await db.rollback()
        existing = (
            await db.execute(
                select(Order).where(Order.stripe_session_id == key)
            )
        ).scalar_one_or_none()
        if existing is None:
            raise
        # The rollback took the decrement with it -- which is the point, the
        # duplicate spends nothing -- and expired every object in the session
        # along the way. Re-read the balance rather than reporting the value
        # this request briefly imagined it had.
        await db.refresh(user)
        logger.info("CREDIT_REDEEM_DUPLICATE", user_id=user.id, order_id=existing.id)
        return {
            "mode": "credit", "order_id": existing.id,
            "credits_left": user.expert_credits,
        }
    logger.info(
        "CREDIT_REDEEMED", user_id=user.id, order_id=order.id,
        analysis=analysis_client_id, credits_left=user.expert_credits,
    )
    return {
        "mode": "credit", "order_id": order.id,
        "credits_left": user.expert_credits,
    }


@router.post("/checkout")
async def create_checkout(
    body: CheckoutIn,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    # An Expert Review the account already owns is spent here, before Stripe is
    # consulted at all: the Full tier includes one, and sending that customer
    # to a payment page to buy a thing they have already bought is how an
    # included benefit turns into a support email.
    if body.plan == "expert" and (user.expert_credits or 0) > 0:
        analysis_id = (body.analysis_client_id or "").strip()[:64]
        if not analysis_id:
            # The paid path can survive this (the reviewer picks a clip in the
            # queue), but a credit is spent irreversibly the moment it is
            # redeemed -- so refuse rather than burn it on an unknown clip.
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Choose which analysis you'd like reviewed first.",
            )
        await _require_review_slot(db)
        return await _redeem_expert_credit(db, user, analysis_id)

    # An unlock buys one specific report, so an unnamed one is meaningless --
    # and unlike the Expert Review there is no human downstream to work out
    # which was meant.
    if body.plan == "unlock" and not (body.analysis_client_id or "").strip():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Choose which analysis you'd like to unlock.",
        )

    if body.plan == "expert":
        await _require_review_slot(db)

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

    is_subscription = body.plan not in ("expert", "unlock")
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
            success_url=f"{base}/app?checkout=success&plan={body.plan}",
            cancel_url=f"{base}/app?checkout=cancel",
            allow_promotion_codes=True,
            metadata={
                "user_id": str(user.id),
                "plan": body.plan,
                "tier": tier_for_plan(body.plan) or "",
                # Round-trips through Stripe so the webhook can attach the order
                # to the analysis the athlete actually chose.
                "analysis_client_id": (body.analysis_client_id or "")[:64],
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
        "analysis_client_id": o.analysis_client_id,
        # Only a DELIVERED report exists as far as the customer is concerned; a
        # draft is working material. This flag is what puts the "Read your
        # review" link in their account, so it must not go true early.
        "has_report": bool(o.report) and o.status == ORDER_DELIVERED,
        "delivered_at_ms": o.delivered_at_ms,
        # Delivered but never opened. The dashboard turns this into a card,
        # because an email can be missed and Pricing is a strange place to go
        # looking for something you already bought.
        "unread": (
            bool(o.report) and o.status == ORDER_DELIVERED and not o.read_at_ms
        ),
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


@router.get("/orders/{order_id}/report")
async def my_report(
    order_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The delivered Expert Review for one of the caller's own orders.

    404 (not 403) for somebody else's order, and 404 for a report that has not
    shipped yet -- a draft is the reviewer's working material, not something the
    customer is entitled to watch being written.
    """
    order = (
        await db.execute(
            select(Order).where(Order.id == order_id, Order.user_id == user.id)
        )
    ).scalar_one_or_none()
    if order is None or not order.report or order.status != ORDER_DELIVERED:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No review to show yet.")
    # Which analysis this is about, so the athlete's copy carries the same
    # header as the sample they were shown before buying. Missing (deleted
    # history) degrades to a report with no context line, not to an error.
    entry = await _analysis_entry(db, user.id, order.analysis_client_id) or {}
    sport = entry.get("sport") or _sport_of(order.report)
    return {
        "order": _order_out(order),
        "report": order.report,
        "sections": expert_review.sections_for(sport),
        "context": {
            "sport": sport, "kind": entry.get("kind"),
            "score": entry.get("score"), "grade": entry.get("grade"),
            "delivered_at_ms": order.delivered_at_ms,
        },
    }


def _sport_of(report: Any) -> str | None:
    """Which sport's section list renders this report.

    Inferred from the report itself rather than stored twice: a report carrying
    fit rows is a cycling one. Keeps an older stored report renderable after the
    template grows.
    """
    return "bike" if (report or {}).get("fit") else "run"


# --------------------------------------------------------------------------
# Expert Review: the fulfilment queue and the report editor (admin tier).
# --------------------------------------------------------------------------
async def _analysis_entry(
    db: AsyncSession, user_id: int, client_id: str | None,
) -> dict[str, Any] | None:
    """The stored analysis an order is attached to (the frontend's own entry)."""
    if not client_id:
        return None
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == user_id, Analysis.client_id == client_id,
            )
        )
    ).scalar_one_or_none()
    return (row.data or None) if row is not None else None


def _analysis_brief(a: Analysis) -> dict[str, Any]:
    """One line per analysis for the reviewer's picker -- never the keyframe."""
    return {
        "id": a.client_id, "at": a.created_at_ms,
        "sport": a.sport, "kind": a.kind, "score": a.score,
    }


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
        {
            **_order_out(o), "email": email, "admin_note": o.admin_note,
            # The queue needs to distinguish "nothing written" from "drafted but
            # not sent"; `has_report` deliberately only speaks about delivery.
            "has_draft": not expert_review.is_empty(o.report),
        }
        for o, email in rows
    ]


@admin_router.get("/admin/orders/{order_id}")
async def admin_order_detail(
    order_id: int,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Everything needed to write the review on one screen.

    The order, the athlete's analysis (including its annotated keyframe -- the
    original clip is long gone, uploads expire in hours), the rest of their
    history so the reviewer can see a trend or re-point the order, the draft,
    and the section list that defines the template.
    """
    row = (
        await db.execute(
            select(Order, User.email)
            .join(User, User.id == Order.user_id)
            .where(Order.id == order_id)
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    order, email = row

    entry = await _analysis_entry(db, order.user_id, order.analysis_client_id)
    history = (
        await db.execute(
            select(Analysis)
            .where(Analysis.user_id == order.user_id)
            .order_by(Analysis.created_at_ms.desc())
            .limit(50)
        )
    ).scalars().all()
    sport = (entry or {}).get("sport") or "run"
    # An untouched order opens on a seeded draft rather than a blank page.
    report = order.report if not expert_review.is_empty(order.report) else \
        expert_review.prefill(entry)
    return {
        **_order_out(order),
        "email": email,
        "admin_note": order.admin_note,
        "analysis": entry,
        "history": [_analysis_brief(a) for a in history],
        "report": report,
        "sport": sport,
        "sections": expert_review.sections_for(sport),
        "missing_required": expert_review.missing_required(report, sport),
        # Whether the clip is still on disk. The player is only drawn when it
        # is -- a dead <video> element reads as a broken page, not as "this one
        # predates stored footage".
        "clip_available": await _clip_path(db, order) is not None,
    }


@admin_router.get("/admin/orders/{order_id}/clip")
async def admin_order_clip(
    order_id: int,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> FileResponse:
    """The footage this review was bought for.

    The whole product being sold here is a human watching the clip, and until
    uploads outlived their six-hour job the reviewer had only an annotated
    still to go on -- the template said so out loud. This is that repair.

    Admin-only and not signed with the athlete's job token: the reviewer is not
    the owner of the job, so ``authorized_job`` would (correctly) answer 404.
    """
    order = await db.get(Order, order_id)
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    path = await _clip_path(db, order)
    if path is None:
        # Expired, deleted by its owner, or from before analyses recorded which
        # upload produced them. All three are "there is no clip", and the queue
        # already says which by showing no player at all.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No clip stored for this order.")
    return FileResponse(path, filename=path.name)


async def _clip_path(db: AsyncSession, order: Order) -> Path | None:
    """The stored upload behind an order's analysis, if it is still there."""
    if not order.analysis_client_id:
        return None
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == order.user_id,
                Analysis.client_id == order.analysis_client_id,
            )
        )
    ).scalar_one_or_none()
    if row is None or not row.job_id:
        return None
    return jobs.job_file(row.job_id, "input")


class OrderPatch(BaseModel):
    status: str | None = None
    admin_note: str | None = None
    # The report draft, in the shape services/expert_review.py defines. Saved
    # as often as the editor likes; publishing is a separate, deliberate act.
    report: dict[str, Any] | None = None
    # Re-point the order at a different analysis (wrong clip picked at checkout,
    # or none picked at all).
    analysis_client_id: str | None = None


@admin_router.patch("/admin/orders/{order_id}")
async def admin_update_order(
    order_id: int,
    body: OrderPatch,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Move an order through the fulfilment states / save a draft / leave a note."""
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
    if body.analysis_client_id is not None:
        order.analysis_client_id = body.analysis_client_id.strip()[:64] or None
    if body.report is not None:
        entry = await _analysis_entry(db, order.user_id, order.analysis_client_id)
        sport = (entry or {}).get("sport") or "run"
        # Normalised server-side: the editor is not the last word on what may be
        # stored on an order row.
        order.report = expert_review.normalize_report(body.report, sport)
    order.updated_at_ms = _now_ms()
    await db.commit()
    return {
        **_order_out(order), "admin_note": order.admin_note,
        "report": order.report,
        "missing_required": expert_review.missing_required(
            order.report, _sport_of(order.report),
        ),
    }


@admin_router.post("/admin/orders/{order_id}/publish")
async def admin_publish_report(
    order_id: int,
    request: Request,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Deliver the review: the customer can read it from this moment on.

    Refuses on an incomplete report. The customer is shown "Your review has been
    delivered" the instant this succeeds, so the completeness gate belongs here
    rather than in the editor's markup, where it would be advisory.
    """
    order = await db.get(Order, order_id)
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    entry = await _analysis_entry(db, order.user_id, order.analysis_client_id)
    sport = (entry or {}).get("sport") or "run"
    missing = expert_review.missing_required(order.report, sport)
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Not ready to send — still empty: " + ", ".join(missing),
        )
    now = _now_ms()
    first_delivery = order.delivered_at_ms is None
    order.status = ORDER_DELIVERED
    order.delivered_at_ms = order.delivered_at_ms or now
    order.updated_at_ms = now
    # Re-publishing an edited report makes it unread again: the athlete has not
    # seen THIS version, and silently changing what they already read is worse
    # than telling them twice.
    order.read_at_ms = None
    await db.commit()
    logger.info("REVIEW_DELIVERED", order_id=order.id, user_id=order.user_id)

    # Tell them. Only on first delivery -- a typo fix should not re-email
    # someone. Blocking network call, so it goes to the threadpool; a mail
    # failure is logged and never fails a delivery that is otherwise complete.
    emailed = False
    if first_delivery:
        buyer = await db.get(User, order.user_id)
        if buyer is not None:
            subject, text, html = expert_review.ready_email(
                order.report, f"{_base_url(request)}/app",
            )
            emailed = await run_in_threadpool(
                notify.send_email, buyer.email, subject, text, html,
            )
    return {**_order_out(order), "report": order.report, "emailed": emailed}


@router.post("/orders/{order_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_report_read(
    order_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """The athlete opened their review. Clears the unread card; idempotent."""
    order = (
        await db.execute(
            select(Order).where(Order.id == order_id, Order.user_id == user.id)
        )
    ).scalar_one_or_none()
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if order.status == ORDER_DELIVERED and not order.read_at_ms:
        order.read_at_ms = _now_ms()
        await db.commit()


@admin_router.post("/admin/email-test")
async def admin_email_test(
    _admin: User = Depends(require_admin),
) -> dict[str, Any]:
    """Send a test message to the admin's own address.

    Mail config fails in ways nothing else catches: a token that lacks sending
    permission, a From: on a domain that was never verified, DNS that has not
    propagated. Every one of them shows up as a review that quietly announces
    itself to nobody. This makes the whole chain provable in one request instead
    of by delivering a real review and hoping.
    """
    if not notify.email_enabled():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Email is not configured — set EMAIL_PROVIDER, EMAIL_API_KEY and "
            "EMAIL_FROM, then restart.",
        )
    sent = await run_in_threadpool(
        notify.send_email, _admin.email,
        "Flapp email test",
        "If you are reading this, delivered Expert Reviews will reach your "
        "athletes.\n",
        "<p>If you are reading this, delivered Expert Reviews will reach your "
        "athletes.</p>",
    )
    if not sent:
        # The reason is in the log with the provider's own wording; repeating a
        # guess here would just be a second, worse explanation.
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The provider rejected it. See the server log for their reason "
            "(EMAIL_FAILED).",
        )
    return {"sent": True, "to": _admin.email, "provider": settings.email_provider}


@router.get("/expert-review/sample")
def expert_review_sample() -> dict[str, Any]:
    """A complete, real-shaped example report.

    Rendered by the same code as a delivered one and shown on the pricing page,
    because the Expert Review button otherwise sells something invisible: nobody
    knows what $29 buys until after they have spent it.
    """
    return {
        "context": expert_review.SAMPLE_CONTEXT,
        "report": expert_review.SAMPLE_REPORT,
        "sections": expert_review.sections_for(expert_review.SAMPLE_SPORT),
    }


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
            customer=user.stripe_customer_id, return_url=f"{_base_url(request)}/app",
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
    elif etype == "invoice.payment_succeeded":
        await _on_invoice_paid(db, obj)
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
    meta = obj.get("metadata") or {}
    analysis_id = str(meta.get("analysis_client_id") or "").strip()[:64] or None
    db.add(Order(
        user_id=user.id,
        stripe_session_id=session_id,
        plan=str(meta.get("plan") or "expert")[:32],
        amount_total=obj.get("amount_total"),
        currency=(obj.get("currency") or None),
        status=ORDER_PAID,
        created_at_ms=now,
        updated_at_ms=now,
        analysis_client_id=analysis_id,
    ))
    if not analysis_id:
        # Not fatal -- the reviewer can pick one in the queue -- but it means
        # somebody paid without telling us what to look at, which is worth
        # knowing about if it starts happening often.
        logger.warning("ORDER_WITHOUT_ANALYSIS", user_id=user.id, session=session_id)
    logger.info(
        "ORDER_RECORDED", user_id=user.id, session=session_id,
        amount=obj.get("amount_total"), currency=obj.get("currency"),
        analysis=analysis_id,
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

    # Money, recorded from the one place that actually knows a payment
    # happened. The browser's own `checkout_returned` only fires if it survived
    # the redirect -- an abandoned tab, a blocked script or a dead phone all
    # lose the sale from the funnel while the charge goes through anyway.
    meta_all = obj.get("metadata") or {}
    analytics.capture_bg(
        analytics.person_id(user), "purchase_completed",
        {
            "plan": str(meta_all.get("plan") or "") or None,
            "tier": meta_all.get("tier") or None,
            "kind": obj.get("mode"),          # subscription | payment
            "revenue": _amount_major(obj.get("amount_total")),
            "$revenue": _amount_major(obj.get("amount_total")),
            "currency": (obj.get("currency") or "").upper() or None,
        },
    )

    if obj.get("mode") == "subscription":
        meta = obj.get("metadata") or {}
        tier = meta.get("tier") or None
        if tier:
            _set_tier(user, tier)
            user.subscription_status = "active"
            logger.info("SUB_ACTIVATED", user_id=user.id, tier=tier)
        # The Full tier includes an Expert Review. Granted from the *purchase*
        # rather than from the resulting tier, so that an admin account (whose
        # tier billing deliberately never touches) still gets what it paid for.
        #
        # NB: this fires on purchase and on a re-subscribe, but not on an
        # automatic renewal -- Stripe reports those as `invoice.payment_
        # succeeded`, which this webhook does not listen for yet. First renewal
        # is a year out; granting on it is tracked with the rest of the Full
        # build-out.
        if tier == TIER_FULL and grant_expert_credits(user, str(obj.get("id") or "")):
            logger.info(
                "CREDIT_GRANTED", user_id=user.id,
                credits=user.expert_credits, reason="full_subscription",
            )
    else:
        # A one-time purchase grants no tier, so nothing about the account would
        # otherwise change. Record the order -- it is the only trace the customer
        # (and the fulfilment queue) has that they paid.
        await _record_order(db, user, obj)
        meta = obj.get("metadata") or {}
        if str(meta.get("plan") or "") == "unlock":
            await _apply_unlock(db, user, meta.get("analysis_client_id"))


async def _apply_unlock(
    db: AsyncSession, user: User, analysis_client_id: Any,
) -> None:
    """Open one stored report for good.

    Written onto the analysis rather than derived from the order on every read:
    a report is opened once and read many times, and joining orders per row on
    every history load would make the cheapest screen in the app the most
    expensive query.

    Idempotent by nature -- Stripe redelivers, and setting an already-set
    timestamp again would only move it. It is left alone.
    """
    client_id = str(analysis_client_id or "").strip()[:64]
    if not client_id:
        logger.warning("UNLOCK_WITHOUT_ANALYSIS", user_id=user.id)
        return
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == user.id, Analysis.client_id == client_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        # Deleted between paying and the webhook landing. The order stands as
        # the record that they paid; nothing here can be opened.
        logger.warning("UNLOCK_ANALYSIS_MISSING", user_id=user.id, analysis=client_id)
        return
    if row.unlocked_at_ms:
        return
    row.unlocked_at_ms = _now_ms()
    logger.info("UNLOCKED", user_id=user.id, analysis=client_id)


def _price_from_invoice(obj: Any) -> str | None:
    try:
        return obj["lines"]["data"][0]["price"]["id"]
    except (KeyError, IndexError, TypeError):
        return None


async def _on_invoice_paid(db: AsyncSession, obj: Any) -> None:
    """A subscription renewed. The Full tier includes a review per paid term,
    so a renewal owes another one.

    Only ``subscription_cycle``. Stripe also sends this for the invoice that
    opens a subscription, which ``checkout.session.completed`` has already
    granted against -- acting on both would hand out two credits for one
    purchase. The invoice id is the grant reference, so a redelivery of this
    event is a no-op for the same reason a redelivered checkout is.

    Until this existed, a Full subscriber's second year arrived with the
    included review silently missing: the card still promised it, and nothing
    would have agreed with them.
    """
    if obj.get("billing_reason") != "subscription_cycle":
        return
    user = await _user_by_customer(db, obj.get("customer"))
    if user is None:
        return
    price_tier = tier_for_price(_price_from_invoice(obj))
    # Recorded for every renewal, not only the Full one below: a renewal is
    # revenue, and retention is the number a subscription product lives on.
    analytics.capture_bg(
        analytics.person_id(user), "subscription_renewed",
        {
            "tier": price_tier,
            "revenue": _amount_major(obj.get("amount_paid")),
            "$revenue": _amount_major(obj.get("amount_paid")),
            "currency": (obj.get("currency") or "").upper() or None,
        },
    )
    if price_tier != TIER_FULL:
        return
    if grant_expert_credits(user, str(obj.get("id") or "")):
        logger.info(
            "CREDIT_GRANTED", user_id=user.id,
            credits=user.expert_credits, reason="renewal",
        )


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
    analytics.capture_bg(analytics.person_id(user), "subscription_canceled", {})
