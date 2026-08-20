"""The price list: one definition, every consumer.

Prices used to exist as literal strings in four places -- the SPA's pricing
markup, the SPA's billing-period toggle (which rewrites the same numbers a
second time), the landing page, and the Terms of Service. That is how the
pricing footer came to advertise **USD** while the Terms promised **CAD**: a
~37% difference between the two documents a customer reads before paying, in
one of which we are the party making the promise. Anything that names a price
now reads it from here.

What this module is NOT: the authority on what a card is actually charged.
Stripe holds that, in the Price object behind each ``plan`` key (see
``settings.plan_price_map``). These amounts are the *display* copy, and the two
have to be moved together by hand when a price changes. ``/billing/plans``
reports, per price, whether its Stripe id is configured -- so the failure mode
of forgetting one is a plan that renders without a Buy button, rather than a
customer meeting an error at the end of checkout.

The rule that governs the feature lists: **a bullet is a promise the code
keeps.** A card may only claim what a paying customer receives today. The Full
tier advertised multi-bike profiles, a branded PDF report and an included
Expert Review while none of the three existed anywhere in the codebase, which
is a refund conversation waiting to be had. It is deliberately short here
instead; each bullet comes back the day it becomes true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.user import tier_limit

# Everything is priced in one currency, and it is stated in exactly one place.
CURRENCY = "usd"
CURRENCY_LABEL = "USD"

# Shown under the tier grid. Names the biller because the Stripe statement
# descriptor says Clariva, not Flapp -- an unrecognised line on a card
# statement is a chargeback in waiting.
FOOTNOTE = (
    f"Prices in {CURRENCY_LABEL} · cancel anytime · secure checkout by Stripe "
    f"· billed by Clariva"
)


@dataclass(frozen=True)
class Price:
    """One purchasable price on a card.

    ``plan`` is the checkout key the frontend posts to ``/billing/checkout``;
    it must exist in ``settings.plan_price_map``. The free tier has no plan
    key (nothing to buy), which is why it is optional.
    """

    amount: int                 # minor units, like Stripe reports them
    per: str                    # the small text beside the amount
    note: str                   # the sub-line under the price
    interval: str               # "month" | "year" | "once" | "forever"
    plan: str | None = None


@dataclass(frozen=True)
class Feature:
    """One bullet. ``on=False`` renders as a struck-through 'not included'."""

    text: str                   # may contain <b>, and only <b>
    on: bool = True


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    tagline: str
    prices: tuple[Price, ...]
    features: tuple[Feature, ...]
    # "tier"   -> a subscription plan, matched against ``user.tier``
    # "addon"  -> a one-time purchase that grants no tier, sold on the pricing page
    # "single" -> a one-time purchase sold in context, never on the pricing page
    kind: str = "tier"
    tier: str | None = None
    badge: str | None = None
    # The landing page is static marketing and cannot know who is signed in, so
    # its buttons always point into the app. The SPA renders its own CTA from
    # the caller's tier (current-plan badge, Manage subscription, Checkout) and
    # ignores these two.
    cta_label: str = ""
    cta_href: str = "/app#pricing"


# --------------------------------------------------------------------------
# The catalogue.
# --------------------------------------------------------------------------
CARDS: tuple[Card, ...] = (
    Card(
        id="starter",
        name="Starter",
        tier="starter",
        tagline="See the tech work and try the process.",
        prices=(
            Price(amount=0, per="forever", note="No card required",
                  interval="forever"),
        ),
        features=(
            Feature("<b>10</b> analyses / month"),
            Feature("<b>Your first report in full</b> — findings + coaching"),
            Feature("After that: score + skeleton, or <b>$4</b> to unlock one"),
            # Signed-in free accounts have always had cloud history -- /me
            # never checked the tier -- while this card said they did not.
            # Promising less than you deliver is not the safe direction of a
            # mistake: it costs an upgrade argument and makes the rest of the
            # card look approximate.
            Feature("Cloud history, synced across devices"),
            Feature("Trends &amp; before / after", on=False),
            Feature("Overlay video", on=False),
        ),
        cta_label="Create free account",
        cta_href="/app#signup",
    ),
    Card(
        id="enthusiast",
        name="Enthusiast",
        tier="enthusiast",
        badge="Most popular",
        tagline="The full analysis for one discipline.",
        prices=(
            Price(plan="enthusiast_monthly", amount=900, per="/ month",
                  note="billed monthly", interval="month"),
            Price(plan="enthusiast_yearly", amount=6900, per="/ year",
                  note="billed annually", interval="year"),
        ),
        features=(
            Feature("<b>30</b> analyses / month"),
            Feature("Full AI coaching + action plan"),
            Feature("Exact joint angles, no watermark"),
            Feature("Overlay video + slow-motion"),
            Feature("Cloud history, trends &amp; before / after"),
            Feature("<b>1</b> equipment profile"),
        ),
        cta_label="Go Enthusiast",
    ),
    Card(
        id="full",
        name="Full",
        tier="full",
        tagline="Every setup you ride, and a human in the loop.",
        prices=(
            Price(plan="full_yearly", amount=9900, per="/ year",
                  note="billed annually", interval="year"),
        ),
        # These came off the card when it turned out none of them existed,
        # and go back one at a time as each becomes true. Every line here now
        # has code and a test behind it:
        #   profiles      -> models/profile.py, PROFILE_LIMITS
        #   comparison    -> the "Two setups" mode, which needs two profiles
        #                    and is therefore gated by the limit above rather
        #                    than by a check we could forget
        #   PDF           -> the print stylesheet
        #   Expert Review -> users.expert_credits, granted on purchase and on
        #                    renewal, spent without touching Stripe
        features=(
            Feature("Everything in Enthusiast, <b>120</b>/mo"),
            Feature("<b>10</b> equipment profiles — every bike and pair of shoes"),
            Feature("Compare one setup against another, not just week to week"),
            Feature("Branded PDF report to keep or hand to a coach"),
            Feature("<b>1 Expert Review included</b> ($39 value), every year"),
        ),
        cta_label="Go Full",
    ),
    # Not a card: the unlock is offered on the blurred report itself, at the
    # moment somebody is looking at their own score and wants the rest. A
    # pricing page is where you go to compare plans, which is a different
    # errand and a worse moment to interrupt with $4.
    Card(
        id="unlock",
        name="Unlock this report",
        kind="single",
        tagline=(
            "The measurements and the drills for one analysis. No plan, "
            "no card kept."
        ),
        prices=(
            Price(plan="unlock", amount=400, per="once",
                  note="per report", interval="once"),
        ),
        features=(
            Feature("Every joint angle we measured, against its range"),
            Feature("The corrective drills for what we found"),
            # Said out loud, because the fence IS the product boundary: one
            # report cannot be compared with anything, and comparison is what a
            # subscription is actually for.
            Feature("Trends, before / after and the overlay video", on=False),
        ),
        cta_label="Unlock this report",
    ),
    Card(
        id="expert",
        name="Expert Review",
        kind="addon",
        badge="Add-on · any plan",
        tagline=(
            "A real triathlete reads your analysis and tells you which single "
            "thing to change."
        ),
        prices=(
            Price(plan="expert", amount=3900, per="once",
                  note="per review · included with Full", interval="once"),
        ),
        # These are the report's actual sections (services/expert_review.py),
        # not a wishlist: the sample shown next to this card is rendered by the
        # same code that renders a delivered review, so the card, the sample
        # and the deliverable cannot drift apart.
        features=(
            Feature("A verdict, and which numbers to trust"),
            Feature("What's already working — so you don't &quot;fix&quot; it"),
            Feature("Ranked priorities with drills and a re-test date"),
            Feature("Position changes with their trade-offs (cycling)"),
        ),
        cta_label="Add Expert Review",
    ),
)

CARD_BY_ID = {c.id: c for c in CARDS}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def money(amount: int) -> str:
    """Minor units as display copy: ``$9``, ``$69``, ``$4.50``.

    Trailing ``.00`` is dropped -- every price we sell is a whole number of
    dollars, and "$9.00" on a pricing card reads like a form field.
    """
    whole, cents = divmod(int(amount), 100)
    return f"${whole}" if cents == 0 else f"${whole}.{cents:02d}"


def annual_saving_pct(card_id: str = "enthusiast") -> int:
    """How much the yearly price saves against paying monthly, rounded down.

    Computed rather than written down: the "save ~X%" badge on the billing
    toggle and the sub-line under the yearly price are the same claim as the
    two numbers above them, and a hand-maintained third copy of it is how a
    discount ends up advertised at a percentage it no longer is.
    """
    card = CARD_BY_ID.get(card_id)
    if card is None:
        return 0
    monthly = next((p for p in card.prices if p.interval == "month"), None)
    yearly = next((p for p in card.prices if p.interval == "year"), None)
    if not monthly or not yearly or monthly.amount <= 0:
        return 0
    full_year = monthly.amount * 12
    return max(0, int((1 - yearly.amount / full_year) * 100))


def _price_note(card: Card, price: Price) -> str:
    """The sub-line, with the computed saving appended to a yearly price."""
    if price.interval == "year":
        saved = annual_saving_pct(card.id)
        if saved > 0:
            return f"{price.note} · save ~{saved}%"
    return price.note


def _price_out(card: Card, price: Price, plan_price_map: dict[str, Any]) -> dict[str, Any]:
    # A plan we advertise but have no Stripe price for is not purchasable. Say
    # so in the payload so the client can render an explanation instead of a
    # button that 503s when pressed.
    available = price.plan is None or bool(plan_price_map.get(price.plan))
    return {
        "plan": price.plan,
        "amount": price.amount,
        "display": money(price.amount),
        "per": price.per,
        "interval": price.interval,
        "note": _price_note(card, price),
        "available": available,
    }


def catalog(plan_price_map: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole price list, JSON-safe. Served by ``GET /billing/plans``."""
    plan_price_map = plan_price_map or {}
    return {
        "currency": CURRENCY,
        "currency_label": CURRENCY_LABEL,
        "annual_saving_pct": annual_saving_pct(),
        "footnote": FOOTNOTE,
        "cards": [
            {
                "id": c.id,
                "kind": c.kind,
                "tier": c.tier,
                # The analysis allowance, from the tier table rather than from
                # the bullet text beside it. The client renders "4 of 10 left"
                # from this, and used to hold its own copy of the numbers --
                # which was wrong for two tiers at once by the time anyone
                # noticed.
                "quota": (
                    {"limit": tier_limit(c.tier)[0], "window": tier_limit(c.tier)[1]}
                    if c.tier else None
                ),
                "name": c.name,
                "tagline": c.tagline,
                "badge": c.badge,
                "cta_label": c.cta_label,
                "cta_href": c.cta_href,
                "prices": [_price_out(c, p, plan_price_map) for p in c.prices],
                "features": [{"text": f.text, "on": f.on} for f in c.features],
            }
            for c in CARDS
        ],
    }


# --------------------------------------------------------------------------
# Landing-page rendering.
#
# The landing page is served as one inlined static document for SEO and for a
# first paint that does not wait on an API call, so its prices cannot be
# fetched at runtime like the SPA's are. They are rendered into the shell on
# the way out instead (see main._serve_shell) -- same catalogue, no JavaScript,
# no second copy of the numbers.
# --------------------------------------------------------------------------
_CHECK = (
    '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true">'
    '<path d="M3 8.5l3 3 7-7" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
_CROSS = (
    '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true">'
    '<path d="M5 5l6 6M11 5l-6 6" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round"/></svg>'
)

# Where the rendered block is spliced into landing.html.
LANDING_TOKEN = "<!--PRICING-->"
# A price mentioned in running prose rather than on a card (the landing page
# points at the Expert Review mid-page, before the pricing section). Same
# catalogue, so an amount cannot be updated in one of the two and not the other.
EXPERT_PRICE_TOKEN = "<!--EXPERT-PRICE-->"


# The plan table inside the Terms of Service. That document is the one where a
# stale price is not a typo but a contradicted promise -- it is where "$59 /
# year" and "prices are in Canadian dollars" both outlived the truth.
TERMS_TOKEN = "<!--PLANS-TABLE-->"

_INTERVAL_WORD = {"month": "Monthly", "year": "Yearly"}


def headline_price(card_id: str) -> str:
    """Display amount of a card's first price -- for prose mentions."""
    card = CARD_BY_ID.get(card_id)
    return money(card.prices[0].amount) if card and card.prices else ""


def render_terms_table() -> str:
    """The plan/billing table for the Terms, rendered from this catalogue."""
    rows = ["<tr><th>Plan</th><th>Billing</th></tr>"]
    for card in CARDS:
        for price in card.prices:
            if price.interval == "forever":
                rows.append(
                    f"<tr><td>{card.name} (free)</td>"
                    f"<td>No charge. A small monthly analysis quota.</td></tr>"
                )
            elif price.interval == "once":
                rows.append(
                    f"<tr><td>{card.name}</td><td>{money(price.amount)}, "
                    f"one-time (per review).</td></tr>"
                )
            else:
                cycle = _INTERVAL_WORD.get(price.interval, price.interval)
                rows.append(
                    f"<tr><td>{card.name} — {cycle}</td>"
                    f"<td>{money(price.amount)} / {price.interval}, "
                    f"recurring.</td></tr>"
                )
    return '<table class="data">' + "".join(rows) + "</table>"


def _landing_features(card: Card) -> str:
    rows = []
    for f in card.features:
        cls = "" if f.on else ' class="off"'
        rows.append(
            f"<li{cls}>{_CHECK if f.on else _CROSS}<span>{f.text}</span></li>"
        )
    return "".join(rows)


def _landing_price(card: Card) -> str:
    """The headline price, plus the alternative cycle as a sub-line.

    A tier with both cycles shows the yearly amount and mentions the monthly
    one underneath, because the landing page has no billing toggle to switch
    between them and showing only "$9 / month" next to two annual prices is
    the unit mismatch that made the SPA's toggle necessary in the first place.
    """
    yearly = next((p for p in card.prices if p.interval == "year"), None)
    head = yearly or card.prices[0]
    monthly = next((p for p in card.prices if p.interval == "month"), None)
    if yearly and monthly:
        sub = (
            f"or {money(monthly.amount)} / month · "
            f"yearly saves ~{annual_saving_pct(card.id)}%"
        )
    else:
        sub = _price_note(card, head)
    return (
        f'<div class="tier-price"><span class="amt">{money(head.amount)}</span>'
        f'<span class="per">{head.per}</span></div>'
        f'<p class="tier-sub">{sub}</p>'
    )


def render_landing_pricing() -> str:
    """The landing page's tier grid + Expert Review card, as HTML.

    Class names mirror the ones already in landing.html's stylesheet, so this
    is a swap of the markup's *source*, not of its appearance.
    """
    tiers = [c for c in CARDS if c.kind == "tier"]
    addons = [c for c in CARDS if c.kind == "addon"]

    cards_html = []
    for i, card in enumerate(tiers, start=1):
        badge = (
            f'<span class="poptag">{card.badge}</span>' if card.badge else ""
        )
        btn = "btn-primary" if card.badge else "btn-outline"
        cards_html.append(
            f'<div class="tier{" pop" if card.badge else ""} rv" data-d="{i}">'
            f"{badge}"
            f'<p class="tier-name">{card.name}</p>'
            f'<p class="tier-tagline">{card.tagline}</p>'
            f"{_landing_price(card)}"
            f"<ul>{_landing_features(card)}</ul>"
            f'<a class="btn {btn}" href="{card.cta_href}">{card.cta_label}</a>'
            f"</div>"
        )

    out = f'<div class="tiers">{"".join(cards_html)}</div>'

    for card in addons:
        price = card.prices[0]
        out += (
            f'<div class="tier tier-solo rv">'
            f'<span class="poptag poptag-addon">{card.badge}</span>'
            f'<div class="solo-head">'
            f"<div>"
            f'<p class="tier-name">{card.name}</p>'
            f'<p class="tier-tagline">{card.tagline}</p>'
            f"</div>"
            f'<div class="solo-buy">'
            f'<div class="tier-price"><span class="amt">{money(price.amount)}</span>'
            f'<span class="per">{price.per}</span></div>'
            f'<p class="tier-sub">{_price_note(card, price)}</p>'
            f'<a class="btn btn-outline" href="{card.cta_href}">{card.cta_label}</a>'
            f"</div></div>"
            f'<ul class="solo-list">{_landing_features(card)}</ul>'
            f"</div>"
        )

    out += f'<p class="pricing-foot">{FOOTNOTE}</p>'
    return out
