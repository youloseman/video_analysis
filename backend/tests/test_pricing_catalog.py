"""The price catalogue, and the drift it exists to prevent.

Prices used to be transcribed into four documents: the SPA's pricing markup,
the SPA's billing-period toggle, the landing page, and the Terms of Service.
They drifted, and the way they drifted mattered -- the pricing footer said the
amounts were in USD while the Terms said Canadian dollars, a ~37% difference
between two documents the same customer reads before paying, one of which is
the agreement.

So most of these tests are not about arithmetic. They are about there being
exactly one copy of each number, and about a card never advertising something
checkout cannot actually sell.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import settings
from app.services import pricing

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


# --------------------------------------------------------------------------
# The catalogue itself
# --------------------------------------------------------------------------
def test_every_purchasable_price_maps_to_a_plan_checkout_accepts():
    """A card offering a plan key ``/billing/checkout`` rejects is a Buy button
    that 400s. The two lists are separate structures; nothing but this keeps
    them agreeing."""
    for card in pricing.CARDS:
        for price in card.prices:
            if price.plan is not None:
                assert price.plan in settings.plan_price_map, (
                    f"{card.id} sells '{price.plan}', which checkout does not know"
                )


def test_the_free_tier_is_the_only_card_without_a_plan_key():
    keyless = {
        c.id for c in pricing.CARDS
        if any(p.plan is None for p in c.prices)
    }
    assert keyless == {"starter"}


def test_availability_follows_the_configured_stripe_prices():
    """An advertised plan with no Stripe price is not purchasable, and the
    payload has to say so -- otherwise the only symptom is a button whose one
    possible outcome is a 503."""
    configured = pricing.catalog({"enthusiast_monthly": "price_x"})
    by_plan = {
        p["plan"]: p["available"]
        for c in configured["cards"] for p in c["prices"]
    }
    assert by_plan["enthusiast_monthly"] is True
    assert by_plan["enthusiast_yearly"] is False
    assert by_plan["full_yearly"] is False
    # The free tier has nothing to configure, so it is always purchasable.
    free = [p for c in configured["cards"] for p in c["prices"] if p["plan"] is None]
    assert all(p["available"] for p in free)


def test_the_single_report_unlock_is_sellable():
    assert "unlock" in settings.plan_price_map
    assert pricing.CARD_BY_ID["unlock"].kind == "single"


def test_annual_saving_is_computed_from_the_amounts_not_written_down():
    card = pricing.CARD_BY_ID["enthusiast"]
    monthly = next(p for p in card.prices if p.interval == "month")
    yearly = next(p for p in card.prices if p.interval == "year")
    expected = int((1 - yearly.amount / (monthly.amount * 12)) * 100)
    assert pricing.annual_saving_pct() == expected
    # And it reaches the payload, which is what the toggle badge renders.
    assert pricing.catalog()["annual_saving_pct"] == expected


def test_the_yearly_note_carries_the_same_saving_as_the_badge():
    """Two claims about one discount. They are rendered from one number."""
    cat = pricing.catalog()
    ent = next(c for c in cat["cards"] if c["id"] == "enthusiast")
    yearly = next(p for p in ent["prices"] if p["interval"] == "year")
    assert f"save ~{cat['annual_saving_pct']}%" in yearly["note"]


@pytest.mark.parametrize(
    "amount,expected", [(0, "$0"), (900, "$9"), (6900, "$69"), (450, "$4.50")],
)
def test_money_drops_a_zero_cents_tail(amount, expected):
    assert pricing.money(amount) == expected


def test_one_currency_everywhere():
    assert pricing.CURRENCY == "usd"
    assert pricing.CURRENCY_LABEL in pricing.FOOTNOTE
    assert pricing.catalog()["currency"] == "usd"


# --------------------------------------------------------------------------
# The documents that used to hold their own copies
# --------------------------------------------------------------------------
# "$59 / year", "$39, one-time", "from <b>$9</b>/mo" -- an amount next to a
# billing word is what a transcribed price looks like. A bare "$4" in prose is
# not what broke. The optional tag is not paranoia: the upsell panel on every
# locked report carried its price inside a <b>, which is how it survived the
# first sweep.
_TRANSCRIBED_PRICE = re.compile(r"\$\d+\s*(?:</b>)?\s*(?:/|per\b|,\s*one-time)")


def _served(name: str) -> str:
    """A static document with its price tokens rendered, as a request sees it."""
    doc = (STATIC / name).read_text(encoding="utf-8")
    doc = doc.replace(pricing.LANDING_TOKEN, pricing.render_landing_pricing())
    doc = doc.replace(pricing.TERMS_TOKEN, pricing.render_terms_table())
    doc = doc.replace(pricing.EXPERT_PRICE_TOKEN, pricing.headline_price("expert"))
    return doc


@pytest.mark.parametrize("name", ["landing.html", "terms.html", "index.html"])
def test_no_document_holds_its_own_copy_of_a_price(name):
    raw = (STATIC / name).read_text(encoding="utf-8")
    found = _TRANSCRIBED_PRICE.findall(raw)
    assert not found, f"{name} transcribes a price: {found}"


def test_the_landing_page_prints_the_catalogue_into_its_html():
    """It is a marketing page: the numbers have to be in the document for
    crawlers and for the first paint, not fetched afterwards."""
    doc = _served("landing.html")
    for card in pricing.CARDS:
        if card.kind == "single":
            # Sold on the blurred report, at the moment somebody wants the rest
            # of their own score -- not on a page for comparing plans.
            assert card.name not in doc
            continue
        assert card.name in doc
        for price in card.prices:
            assert pricing.money(price.amount) in doc
    assert pricing.FOOTNOTE in doc
    assert pricing.LANDING_TOKEN not in doc


def test_the_terms_table_is_rendered_from_the_catalogue():
    doc = _served("terms.html")
    assert pricing.TERMS_TOKEN not in doc
    for card in pricing.CARDS:
        for price in card.prices:
            if price.interval != "forever":
                assert pricing.money(price.amount) in doc
    # The drift that started all of this.
    assert "Canadian dollars" not in doc
    assert "US dollars (USD)" in doc


def test_the_app_shell_renders_prices_from_the_api_not_from_markup():
    doc = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "/billing/plans" in doc
    assert 'id="tiersHost"' in doc and 'id="addonHost"' in doc


def test_the_plans_endpoint_is_public():
    """The pricing screen has to render for a visitor deciding whether to make
    an account at all, so this one cannot sit behind the bearer token."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from app.api.billing import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/billing/plans")

    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "usd"
    assert [c["id"] for c in body["cards"]] == [c.id for c in pricing.CARDS]


# --------------------------------------------------------------------------
# A bullet is a promise the code keeps
# --------------------------------------------------------------------------
def test_every_claim_on_the_full_card_has_something_behind_it():
    """This card once advertised multi-bike profiles, a branded PDF and an
    included Expert Review while none of the three existed in the codebase.

    The guard used to be a blacklist of those words, which stopped being useful
    the day they became true. So it checks the other direction now: for each
    claim, the thing that makes it true. A blacklist would have to be deleted to
    ship the feature; this has to be extended, which is the right amount of
    friction for putting a new promise on a price card.
    """
    from pathlib import Path as _P

    from app.models.user import (
        FULL_EXPERT_CREDITS,
        TIER_ENTHUSIAST,
        TIER_FULL,
        profile_limit,
    )

    text = " ".join(f.text.lower() for f in pricing.CARD_BY_ID["full"].features)
    spa = (STATIC / "index.html").read_text(encoding="utf-8")

    if "profile" in text:
        # Several setups, and strictly more than the tier below -- otherwise it
        # is not a reason to pay more.
        assert profile_limit(TIER_FULL) > profile_limit(TIER_ENTHUSIAST) > 0

    if "compare" in text:
        # The comparison needs two profiles, so the limit above IS its gate.
        assert "Two setups" in spa and "profileAggregate" in spa

    if "pdf" in text:
        assert "@media print" in spa and "printhead" in spa

    if "expert review" in text:
        assert FULL_EXPERT_CREDITS >= 1
        billing = _P(__file__).resolve().parents[1] / "app" / "api" / "billing.py"
        source = billing.read_text(encoding="utf-8")
        # Granted on purchase AND on renewal: a card promising one "every year"
        # against code that only grants on the first purchase is the same bug
        # this test exists for, one year deferred.
        assert "grant_expert_credits" in source
        assert "invoice.payment_succeeded" in source


def test_starter_admits_to_the_cloud_history_it_actually_gives():
    """/me never checked the tier, so free accounts always had synced history
    while this card said they did not. Promising less than you deliver still
    costs an upgrade argument."""
    features = pricing.CARD_BY_ID["starter"].features
    history = [f for f in features if "history" in f.text.lower()]
    assert history and all(f.on for f in history)
