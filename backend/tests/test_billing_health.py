"""Can this deployment take money, and is every plan wired up?

The answer used to exist only as a line in the deploy log -- BILLING_READY,
BILLING_TEST_MODE, BILLING_PRICES_MISSING -- which scrolls away. So the single
most consequential setting in the app was the one nobody could look at, and
"did the live key actually take?" had no answer after the deploy that asked it.

What it must never do is leak: the mode and a count, never a key and never a
price id. That a deployment is in test mode is not a secret -- a real card
declining at the end of checkout announces it far more loudly -- but the
secret key and the price ids are.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import _billing_health, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def billing(monkeypatch):
    """Drive _billing_health with a chosen key and price map."""
    def _set(key, prices):
        monkeypatch.setattr(
            type(settings), "plan_price_map", property(lambda _self: prices),
        )
        object.__setattr__(settings, "stripe_secret_key", key)
        return _billing_health()
    return _set


FIVE = {
    "enthusiast_monthly": "price_a", "enthusiast_yearly": "price_b",
    "full_yearly": "price_c", "expert": "price_d", "unlock": "price_e",
}


# --------------------------------------------------------------------------
# the three modes
# --------------------------------------------------------------------------

def test_a_live_key_reads_live(billing):
    got = billing("sk_live_abc123", FIVE)
    assert got["mode"] == "live"
    assert got["plans_configured"] == 5
    assert got["plans_missing"] == []


def test_a_test_key_reads_test(billing):
    """The whole point after a go-live: `live` here is the difference between
    a working funnel and one that declines every real card at the last step."""
    assert billing("sk_test_abc123", FIVE)["mode"] == "test"


def test_no_key_reads_disabled(billing):
    assert billing("", FIVE)["mode"] == "disabled"
    assert billing(None, FIVE)["mode"] == "disabled"


# --------------------------------------------------------------------------
# the two ways a live key still cannot sell
# --------------------------------------------------------------------------

def test_a_missing_price_is_named(billing):
    """That plan's button renders "Coming soon" and nothing else says why."""
    prices = dict(FIVE, unlock=None, expert=None)
    got = billing("sk_live_abc123", prices)
    assert got["plans_configured"] == 3
    assert got["plans_missing"] == ["expert", "unlock"]


def test_one_price_in_two_plans_is_flagged(billing):
    """The likeliest slip in the whole go-live: five long ids copied by hand,
    and Enthusiast monthly and yearly sit two rows apart in one Stripe
    product. Checkout succeeds and the customer is billed for the plan they
    did not choose -- invisible from inside the app."""
    prices = dict(FIVE, enthusiast_yearly="price_a")   # same as monthly
    assert billing("sk_live_abc123", prices)["price_reused"] is True


def test_distinct_prices_are_not_flagged(billing):
    assert billing("sk_live_abc123", FIVE)["price_reused"] is False


def test_missing_prices_do_not_read_as_reuse(billing):
    """Two plans both unset share the value None, which is absence, not a
    shared price."""
    prices = dict(FIVE, expert=None, unlock=None)
    assert billing("sk_live_abc123", prices)["price_reused"] is False


# --------------------------------------------------------------------------
# it must not leak
# --------------------------------------------------------------------------

def test_no_key_or_price_id_is_exposed(billing):
    got = billing("sk_live_SUPERSECRET", FIVE)
    flat = repr(got)
    assert "SUPERSECRET" not in flat
    assert "sk_live" not in flat
    for price in FIVE.values():
        assert price not in flat


def test_health_carries_it(client):
    got = client.get("/health").json()
    assert "billing" in got
    assert set(got["billing"]) == {
        "mode", "plans_configured", "plans_total", "plans_missing",
        "price_reused",
    }


def test_health_still_answers_everything_it_did_before(client):
    """A health check is watched by a platform. Adding to it must not move
    what was already there."""
    got = client.get("/health").json()
    for key in ("status", "model_present", "overlay_encoder_present",
                "active_jobs", "running", "queued", "capacity", "stored_jobs"):
        assert key in got
    assert got["status"] == "ok"
