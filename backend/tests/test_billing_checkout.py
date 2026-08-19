"""Checkout entry and billing configuration.

Both things covered here fail *quietly*, which is why they are worth tests.

A Stripe price that was never configured used to produce exactly one symptom:
a dead button on the highest-intent screen in the product. Nothing raised,
nothing was logged, and the customer was shown whatever string the endpoint
happened to put in ``detail``. The pricing page now reads availability off the
catalogue and renders "Coming soon" instead of a button that can only 503 --
but the endpoint is still the last line of that defence, and it is what a
stale client or a direct POST meets. A test-mode key is worse still: everything
works right up to the point where a real card is declined.
"""

from __future__ import annotations

import dataclasses

import pytest
import structlog
from fastapi import HTTPException

from app.api import billing
from app.api.billing import CheckoutIn, create_checkout, log_billing_configuration
from app.core.config import settings


def _settings(**overrides):
    """A copy of the real settings with a few fields swapped (it is frozen)."""
    return dataclasses.replace(settings, **overrides)


def _use(monkeypatch, **overrides) -> None:
    """Point the billing module at a doctored settings object."""
    monkeypatch.setattr(billing, "settings", _settings(**overrides))


# --------------------------------------------------------------------------
# create_checkout: telling a bad request apart from a bad deployment
# --------------------------------------------------------------------------

async def test_a_plan_we_do_not_sell_is_a_client_error(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_test_x")
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="platinum_lifetime"), None, user, db)
    assert exc.value.status_code == 400


async def test_an_unconfigured_price_is_a_server_error_not_a_client_one(
    db, make_user, monkeypatch,
):
    """The request is fine; the deployment is not. A 400 blames the customer,
    and the client's error handling treats 4xx and 5xx differently."""
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_enthusiast_y=None)
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="enthusiast_yearly"), None, user, db)
    assert exc.value.status_code == 503


async def test_the_customer_is_not_shown_our_internal_wording(db, make_user, monkeypatch):
    """This detail string goes straight into a toast in front of a paying
    customer, so it has to read like a sentence written for them."""
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_enthusiast_y=None)
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="enthusiast_yearly"), None, user, db)
    detail = exc.value.detail.lower()
    assert "unconfigured" not in detail
    assert "support@" in detail


async def test_an_unconfigured_price_is_logged_at_error(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_full_y=None)
    user = await make_user()
    with structlog.testing.capture_logs() as logs:
        with pytest.raises(HTTPException):
            await create_checkout(CheckoutIn(plan="full_yearly"), None, user, db)
    assert any(
        e["event"] == "PLAN_PRICE_UNCONFIGURED" and e["log_level"] == "error"
        for e in logs
    )


# --------------------------------------------------------------------------
# log_billing_configuration: the startup read-out
# --------------------------------------------------------------------------

def test_a_missing_price_is_named_at_startup(monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_live_x", stripe_price_full_y=None)
    with structlog.testing.capture_logs() as logs:
        log_billing_configuration()
    missing = [e for e in logs if e["event"] == "BILLING_PRICES_MISSING"]
    assert missing, "a plan with no Stripe price must be called out at startup"
    assert missing[0]["plans"] == ["full_yearly"]
    assert missing[0]["log_level"] == "error"


def test_a_test_mode_key_is_flagged_at_startup(monkeypatch):
    """The failure this catches is a launch where every real card is declined."""
    _use(monkeypatch, stripe_secret_key="sk_test_abc123")
    with structlog.testing.capture_logs() as logs:
        log_billing_configuration()
    assert any(e["event"] == "BILLING_TEST_MODE" for e in logs)


def test_a_live_key_with_every_price_configured_says_nothing_alarming(monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_live_abc123")
    with structlog.testing.capture_logs() as logs:
        log_billing_configuration()
    assert {e["log_level"] for e in logs} == {"info"}
    assert any(e["event"] == "BILLING_READY" for e in logs)


def test_billing_switched_off_is_reported_but_not_an_error(monkeypatch):
    """No Stripe key at all is a legitimate state (local dev, a fresh
    environment) -- it should not look like a broken deployment."""
    _use(monkeypatch, stripe_secret_key=None)
    with structlog.testing.capture_logs() as logs:
        log_billing_configuration()
    assert [e["event"] for e in logs] == ["BILLING_DISABLED"]
    assert logs[0]["log_level"] == "info"
