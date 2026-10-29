"""The $4 report credited to the plan bought after it (services/unlock_credit).

Paywall read of 2026-10-28: nine $4 buyers, none went on to a plan. The $4
now comes off the first invoice of a subscription started within seven days,
by a Stripe coupon passed on the Checkout Session. Artur, 2026-10-29: seven
days; monthly and yearly alike; several unlocks credit once, off the latest;
the credit is shown only after the unlock is paid, never on the teaser card.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.api import auth, billing
from app.api.billing import CheckoutIn, create_checkout
from app.core.config import settings
from app.models.order import ORDER_PAID, ORDER_REFUNDED, Order
from app.models.user import TIER_STARTER
from app.services import unlock_credit

NOW = 1_800_000_000_000
DAY = unlock_credit.DAY_MS
STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


def _on(monkeypatch, coupon="UNLOCK_CREDIT_4", **more):
    """Stripe on, the coupon configured (or not), for every module that reads settings."""
    s = dataclasses.replace(
        settings, stripe_secret_key="sk_test_x", stripe_coupon_unlock_credit=coupon, **more,
    )
    monkeypatch.setattr(billing, "settings", s)
    monkeypatch.setattr(unlock_credit, "settings", s)


async def _order(db, user, *, at, plan="unlock", amount=400, status=ORDER_PAID, sid=None,
                 client_id="h1"):
    o = Order(
        user_id=user.id, stripe_session_id=sid or f"cs_{plan}_{at}", plan=plan,
        amount_total=amount, currency="usd", status=status,
        created_at_ms=at, updated_at_ms=at, analysis_client_id=client_id,
    )
    db.add(o)
    await db.commit()
    await db.refresh(o)
    return o


# --------------------------------------------------------------------------
# Which unlock counts
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_paid_unlock_inside_the_window_is_credited(db, make_user, monkeypatch):
    _on(monkeypatch)
    user = await make_user(TIER_STARTER)
    o = await _order(db, user, at=NOW - 3 * DAY)
    found = await unlock_credit.credited_unlock(db, user, NOW)
    assert found is not None and found.id == o.id
    state = unlock_credit.credit_state(found)
    assert state == {
        "amount_minor": 400, "until_ms": NOW - 3 * DAY + 7 * DAY, "entry_id": "h1",
        "order_id": o.id,
    }


@pytest.mark.asyncio
async def test_the_window_is_seven_days(db, make_user, monkeypatch):
    _on(monkeypatch)
    user = await make_user(TIER_STARTER)
    await _order(db, user, at=NOW - 8 * DAY)
    assert await unlock_credit.credited_unlock(db, user, NOW) is None
    assert settings.unlock_credit_window_days == 7


@pytest.mark.asyncio
async def test_several_unlocks_credit_once_off_the_latest(db, make_user, monkeypatch):
    _on(monkeypatch)
    user = await make_user(TIER_STARTER)
    await _order(db, user, at=NOW - 5 * DAY, client_id="old")
    latest = await _order(db, user, at=NOW - 1 * DAY, client_id="new")
    found = await unlock_credit.credited_unlock(db, user, NOW)
    assert found.id == latest.id and found.analysis_client_id == "new"


@pytest.mark.asyncio
async def test_a_refund_a_free_credit_and_another_product_credit_nothing(
    db, make_user, monkeypatch,
):
    _on(monkeypatch)
    user = await make_user(TIER_STARTER)
    await _order(db, user, at=NOW - DAY, status=ORDER_REFUNDED, sid="cs_r")
    await _order(db, user, at=NOW - DAY, plan="unlock_credit", amount=0, sid="cs_free")
    await _order(db, user, at=NOW - DAY, plan="notes", amount=1200, sid="cs_notes")
    assert await unlock_credit.credited_unlock(db, user, NOW) is None


@pytest.mark.asyncio
async def test_another_accounts_unlock_is_not_mine(db, make_user, monkeypatch):
    _on(monkeypatch)
    buyer = await make_user(TIER_STARTER)
    other = await make_user(TIER_STARTER)
    await _order(db, buyer, at=NOW - DAY)
    assert await unlock_credit.credited_unlock(db, other, NOW) is None


@pytest.mark.asyncio
async def test_without_the_coupon_nothing_is_promised(db, make_user, monkeypatch):
    """The line "your $4 counts" must never show for a checkout that cannot honour it."""
    _on(monkeypatch, coupon=None)
    user = await make_user(TIER_STARTER)
    await _order(db, user, at=NOW - DAY)
    assert not unlock_credit.enabled()
    assert await unlock_credit.credited_unlock(db, user, NOW) is None


# --------------------------------------------------------------------------
# Checkout: the coupon on the session, the promo box off
# --------------------------------------------------------------------------
class _FakeRequest:
    base_url = "https://getflapp.com/"


def _stripe_fakes(monkeypatch):
    calls = {}

    def fake_create(**kw):
        calls.update(kw)

        class _S:
            url = "https://checkout.stripe.test/s"

        return _S()

    class _Cust:
        id = "cus_new"

    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: _Cust())
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)
    monkeypatch.setattr(billing, "_now_ms", lambda: NOW)
    return calls


@pytest.mark.asyncio
async def test_a_credited_plan_checkout_carries_the_coupon_and_no_promo_box(
    db, make_user, monkeypatch,
):
    _on(monkeypatch)
    calls = _stripe_fakes(monkeypatch)
    user = await make_user(TIER_STARTER)
    o = await _order(db, user, at=NOW - 2 * DAY)
    for plan in ("enthusiast_monthly", "full_yearly"):   # monthly and yearly alike
        calls.clear()
        await create_checkout(CheckoutIn(plan=plan), _FakeRequest(), user, db)
        assert calls["mode"] == "subscription"
        assert calls["discounts"] == [{"coupon": "UNLOCK_CREDIT_4"}]
        assert "allow_promotion_codes" not in calls   # Stripe refuses both at once
        assert calls["metadata"]["unlock_credit_order"] == str(o.id)


@pytest.mark.asyncio
async def test_an_uncredited_checkout_is_as_before(db, make_user, monkeypatch):
    _on(monkeypatch)
    calls = _stripe_fakes(monkeypatch)
    user = await make_user(TIER_STARTER)
    await create_checkout(CheckoutIn(plan="enthusiast_monthly"), _FakeRequest(), user, db)
    assert calls["allow_promotion_codes"] is True
    assert "discounts" not in calls
    assert calls["metadata"]["unlock_credit_order"] == ""


@pytest.mark.asyncio
async def test_the_coupon_never_lands_on_a_one_time_purchase(db, make_user, monkeypatch):
    """A second $4 report, or a review, is not what the credit is for."""
    from tests.test_unlock import store

    _on(monkeypatch)
    calls = _stripe_fakes(monkeypatch)
    user = await make_user(TIER_STARTER)
    await _order(db, user, at=NOW - DAY, client_id="h0")
    await store(db, user, client_id="h2")
    await create_checkout(
        CheckoutIn(plan="unlock", analysis_client_id="h2"), _FakeRequest(), user, db,
    )
    assert calls["mode"] == "payment"
    assert "discounts" not in calls and calls["allow_promotion_codes"] is True


# --------------------------------------------------------------------------
# /auth/me tells the page; the page says it after the purchase, never before
# --------------------------------------------------------------------------
def test_me_carries_the_credit_and_the_sign_in_answers_do_not():
    fields = auth.UserOut.model_fields
    assert fields["unlock_credit"].default is None
    assert set(auth.UnlockCreditOut.model_fields) == {
        "amount_minor", "until_ms", "entry_id", "order_id",
    }
    src = Path(auth.__file__).read_text(encoding="utf-8")
    assert "out.unlock_credit = UnlockCreditOut(**credit) if credit else None" in src


def test_the_page_shows_the_credit_after_the_purchase_only():
    assert "function unlockCreditHtml(where){" in SPA
    assert "counts toward your first month" in SPA
    # Rendered where an opened report sits and on the Account page; the
    # teaser and preview cards never mention it (Artur: after purchase).
    assert SPA.count("unlockCreditHtml('report')") == 1
    assert SPA.count("unlockCreditHtml('account')") == 1
    teaser = SPA.split("function teaserCardHtml(", 1)[1].split("\n}\n", 1)[0]
    preview = SPA.split("function previewCardHtml(", 1)[1].split("\n}\n", 1)[0]
    assert "unlockCredit" not in teaser and "unlockCredit" not in preview
    # The unlock redirect refreshes /auth/me so the credit exists before the report opens.
    assert "auth.unlock_credit=u.unlock_credit||null;" in SPA
    assert "credit: !!(auth&&auth.unlock_credit)" in SPA   # on checkout_started
    assert ".unlock-credit{" in CSS
