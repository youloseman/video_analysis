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
    note: str = ""              # the grey line under it
    # "year" / "month": the bullet is true on that billing period only (the
    # Full plan's included review is a yearly-plan promise).
    period: str | None = None
    new: bool = False


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
    # Who the plan is for, as pills under its name ("Recommended for" Runner,
    # Cyclist), and the one-word version for the phone's three-column overview.
    fits: tuple[str, ...] = ()
    fits_lead: str = "Recommended for"
    fits_short: str = ""
    # The plan's bullets are what it ADDS to the one below it.
    plus_from: str = ""
    # The single line the phone's overview shows under the price.
    key: str = ""
    key_month: str = ""


# --------------------------------------------------------------------------
# The catalogue.
# --------------------------------------------------------------------------
# The one-time amounts, named because the plans' bullets quote them.
UNLOCK_AMOUNT = 400
EXPERT_AMOUNT = 3900
RECHECK_AMOUNT = 2700
# The order bump: an Expert Review added to an Enthusiast checkout, at a
# first-purchase price. The plans it rides on live in billing.BUMP_PLANS.
# It is also what an Enthusiast pays for a review later.
REVIEW_BUMP_AMOUNT = 3500
# The Expert Review for Full subscribers (yearly ones once their included review
# is spent; monthly ones always).
REVIEW_MEMBER_AMOUNT = 2900


def money(amount: int) -> str:
    """Minor units as display copy: ``$9``, ``$69``, ``$4.50``.

    Trailing ``.00`` is dropped -- every price we sell is a whole number of
    dollars, and "$9.00" on a pricing card reads like a form field.
    """
    whole, cents = divmod(int(amount), 100)
    return f"${whole}" if cents == 0 else f"${whole}.{cents:02d}"


def usd(amount: int) -> str:
    """``US$9``: the pricing page names the currency on every amount, because
    its readers are mostly in Canada, where a bare ``$`` means something else."""
    return "US" + money(amount)


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
        # What a free account gets, said plainly. The monthly allowance is not
        # on it (Artur, 2026-10-25): after the first report a free analysis is
        # a score and a skeleton, and "10 analyses" read as ten reports.
        features=(
            Feature("1 full report, free",
                    note=f"what's off and why · drills and angles for {usd(UNLOCK_AMOUNT)}"),
            Feature("A score on every clip"),
            Feature("AI coach notes", note="in your free report"),
            Feature("Move the hip and knee by hand", note="on a cycling photo"),
            # Signed-in free accounts have always had cloud history -- /me
            # never checked the tier -- while this card said they did not.
            # Promising less than you deliver is not the safe direction of a
            # mistake: it costs an upgrade argument and makes the rest of the
            # card look approximate.
            Feature("Your history, synced across devices"),
        ),
        cta_label="Start free",
        cta_href="/app#signup",
        fits=("Just curious",),
        fits_lead="Good for",
        fits_short="Just curious",
        key="1 full report",
    ),
    Card(
        id="enthusiast",
        name="Enthusiast",
        tier="enthusiast",
        # "Recommended", not "Most popular": the badge is a claim about other
        # buyers, and with the customer base this small it is not one the
        # numbers can back. Which tier we recommend is ours to say.
        badge="Recommended",
        tagline="The full analysis for one discipline.",
        prices=(
            Price(plan="enthusiast_monthly", amount=900, per="/ month",
                  note="billed monthly", interval="month"),
            Price(plan="enthusiast_yearly", amount=6900, per="/ year",
                  note="billed annually", interval="year"),
        ),
        features=(
            Feature(f"Every clip, up to <b>{tier_limit('enthusiast')[0]}</b> analyses a month"),
            Feature("Overlay video, frame by frame"),
            Feature("Every angle against the healthy range",
                    note="ranges from research and pro-athlete measures"),
            # Cycling only: the corrections endpoints refuse a run clip
            # (main._correction_access), so the bullet says so.
            Feature("Drag any joint by hand",
                    note="cycling photo and video · the angles recalculate"),
            Feature("Before / after, and trends of every angle"),
            Feature("Your earlier reports open in full"),
            Feature("<b>2</b> saved setups", note="the gear you ride and run with"),
            Feature("Send the report to ChatGPT or your coach"),
            Feature(f"Expert Review for {usd(REVIEW_BUMP_AMOUNT)}",
                    note=f"member price, instead of {usd(EXPERT_AMOUNT)}"),
        ),
        cta_label="Fix my technique",
        fits=("Runner", "Cyclist"),
        fits_short="Run or ride",
        plus_from="Starter",
        key=f"{tier_limit('enthusiast')[0]} clips a month",
    ),
    Card(
        id="full",
        name="Full",
        tier="full",
        tagline="Every setup you ride, and a human in the loop.",
        prices=(
            Price(plan="full_monthly", amount=1300, per="/ month",
                  note="billed monthly · Expert Review on the yearly plan",
                  interval="month"),
            Price(plan="full_yearly", amount=9900, per="/ year",
                  note="billed annually", interval="year"),
        ),
        # These came off the card when it turned out none of them existed,
        # and go back one at a time as each becomes true. Every line here now
        # has code and a test behind it:
        #   profiles      -> models/profile.py, PROFILE_LIMITS
        #   crank         -> biomechanics/hip_over_bb, gated by crank_access
        #   comparison    -> the "Two setups" mode, which needs two profiles
        #                    and is therefore gated by the limit above rather
        #                    than by a check we could forget
        #   PDF           -> the print stylesheet
        #   Expert Review -> users.expert_credits, granted on purchase and on
        #                    renewal, spent without touching Stripe
        features=(
            Feature(f"Up to <b>{tier_limit('full')[0]}</b> analyses a month"),
            Feature("Hip over the crank", new=True,
                    note="how far forward you sit over the crank axle, against the Kona pros"),
            Feature("<b>10</b> saved setups", note="every bike and every pair of shoes"),
            Feature("Compare two bikes or setups"),
            Feature("1 Expert Review a year, included", period="year",
                    note=f"a {usd(EXPERT_AMOUNT)} value · every other one {usd(REVIEW_MEMBER_AMOUNT)}"),
            Feature(f"Expert Review for {usd(REVIEW_MEMBER_AMOUNT)}", period="month",
                    note="member price · one a year is included on the yearly plan"),
            Feature("Branded PDF report", note="to keep, or to hand to a coach"),
        ),
        cta_label="Get a coach + every bike",
        fits=("Triathlete", "Road + TT bike"),
        fits_short="Triathlete",
        plus_from="Enthusiast",
        key=f"{tier_limit('full')[0]} clips + a pro review",
        key_month=f"{tier_limit('full')[0]} clips + every bike",
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
            Price(plan="unlock", amount=UNLOCK_AMOUNT, per="once",
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
    # Not a card either: offered after a delivered Coach Notes or Expert Review,
    # to somebody who changed what the coach said and filmed again. The review
    # is the same product; the lower price is for the second look at the same
    # athlete. Who may buy it, and once per delivery, is the checkout's rule.
    Card(
        id="recheck",
        name="Expert Review — re-check",
        kind="single",
        tagline=(
            "A full Expert Review of the clip you film after making the changes "
            "— open for a limited time after your last review or Coach Notes."
        ),
        prices=(
            Price(plan="expert_recheck", amount=RECHECK_AMOUNT, per="once",
                  note="per re-check", interval="once"),
        ),
        features=(
            Feature("The same full review, on the clip you filmed after the changes"),
            Feature("What moved, what did not, and what to try next"),
        ),
        cta_label="Book the re-check",
    ),
    # The rung under the Expert Review. Sold on the report itself, at the
    # moment somebody has read the AI's notes and wants a person's; also on
    # the pricing page so the ladder (AI -> Notes -> Review) is visible. What
    # makes it a different product rather than a cheaper one is the FORMAT --
    # see services/expert_review.py NOTES_SECTIONS.
    Card(
        id="notes",
        name="Coach Notes",
        kind="addon",
        badge="Add-on · any plan",
        tagline=(
            "A coach reads this analysis and writes back three things — "
            "on the report, within 72 hours."
        ),
        prices=(
            Price(plan="notes", amount=1200, per="once",
                  note="per report", interval="once"),
        ),
        features=(
            Feature("The first thing to change, in one line"),
            Feature("Three notes — what a coach sees that the numbers didn't"),
            Feature("Written on your report: prints and exports with it"),
            # The fence, said out loud: those are the Expert Review's, and a
            # buyer who wanted them should know before paying a third.
            Feature("Trust judgement, a four-week plan, position trade-offs", on=False),
        ),
        cta_label="Add Coach Notes",
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
            Price(plan="expert", amount=EXPERT_AMOUNT, per="once",
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


def review_bump(price_id: str | None) -> dict[str, Any]:
    """What the Enthusiast card offers as its checkbox, and whether it can."""
    full = CARD_BY_ID["expert"].prices[0].amount
    return {
        "amount": REVIEW_BUMP_AMOUNT,
        "display": money(REVIEW_BUMP_AMOUNT),
        "was": money(full),
        "available": bool(price_id),
    }


def review_member(price_id: str | None) -> dict[str, Any]:
    """What a Full subscriber pays for an Expert Review, and whether it is live."""
    return {
        "amount": REVIEW_MEMBER_AMOUNT,
        "display": money(REVIEW_MEMBER_AMOUNT),
        "was": money(CARD_BY_ID["expert"].prices[0].amount),
        "available": bool(price_id),
    }


def catalog(
    plan_price_map: dict[str, Any] | None = None, bump_price_id: str | None = None,
    member_price_id: str | None = None,
) -> dict[str, Any]:
    """The whole price list, JSON-safe. Served by ``GET /billing/plans``."""
    plan_price_map = plan_price_map or {}
    return {
        "currency": CURRENCY,
        "review_bump": review_bump(bump_price_id),
        "review_member": review_member(member_price_id),
        "currency_label": CURRENCY_LABEL,
        "annual_saving_pct": annual_saving_pct(),
        "footnote": FOOTNOTE,
        # The whole pricing block, rendered here like the landing page's, with
        # placeholders the app fills for whoever is signed in.
        "html": render_pricing("app"),
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
# Where the rendered block is spliced into landing.html.
LANDING_TOKEN = "<!--PRICING-->"
# The free allowance mentioned in landing-page prose (hero trust line, FAQ).
# Rendered from the tier table -- the number the server actually enforces --
# because the last hardcoded copy said "3" for a month after the limit became 10.
FREE_LIMIT_TOKEN = "<!--FREE-LIMIT-->"


def starter_monthly_limit() -> int:
    """Starter's analysis allowance, for prose mentions outside the cards."""
    return tier_limit("starter")[0]
# A price mentioned in running prose rather than on a card (the landing page
# points at the Expert Review mid-page, before the pricing section). Same
# catalogue, so an amount cannot be updated in one of the two and not the other.
EXPERT_PRICE_TOKEN = "<!--EXPERT-PRICE-->"
NOTES_PRICE_TOKEN = "<!--NOTES-PRICE-->"


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

# --------------------------------------------------------------------------
# The pricing block: one renderer for the landing page and the app.
#
# Desktop reads it as a table: the three plans as columns, every row a thing
# you get. A phone cannot hold three readable columns, so there it is three
# plans at a glance, one card per plan listing what it adds to the one below,
# and a comparison of any two plans the reader picks. Both layouts are in the
# markup; a container query shows one, so the app's narrower pane (beside its
# sidebar) switches on its own width rather than the window's.
#
# The app gets the same markup over /billing/plans and fills the parts that
# depend on who is looking -- the buttons, the bump, the review's price -- into
# the ``pr-slot`` placeholders. The landing page cannot know who is looking, so
# there those are plain links into the app.
#
# Interaction is inline attributes, not a script: markup inserted with
# innerHTML does not run <script>, but it does run onclick.
# --------------------------------------------------------------------------
YES, NO = "yes", "no"


@dataclass(frozen=True)
class Cell:
    text: str
    note: str = ""


@dataclass(frozen=True)
class Row:
    label: str
    sub: str
    # Starter, Enthusiast, Full. A value is YES, NO, plain text, a Cell, or a
    # {"year": ..., "month": ...} dict when the answer depends on the period.
    cells: tuple[Any, Any, Any]
    new: bool = False


_E_LIMIT = tier_limit("enthusiast")[0]
_F_LIMIT = tier_limit("full")[0]

COMPARE: tuple[tuple[str, tuple[Row, ...]], ...] = (
    ("What's off and what to change", (
        Row("Full reports", "", (
            Cell("<b>1 free report</b>",
                 f"what's off and why · the drills and angles open <b>only for {usd(UNLOCK_AMOUNT)}</b>"),
            Cell("<b>Every clip</b>", f"up to <b>{_E_LIMIT}</b> clips a month"),
            Cell("<b>Every clip</b>", f"up to <b>{_F_LIMIT}</b> clips a month"),
        )),
        Row("Score and angles", "", ("Score on every clip", YES, YES)),
        Row("Overlay video, frame by frame", "", (NO, YES, YES)),
        Row("AI coach notes, tips to improve, your angles vs the healthy range",
            "ranges from science research and pro-athlete measures", (
                Cell("AI notes in your free report", f"tips and angles: {usd(UNLOCK_AMOUNT)}"),
                YES, YES)),
        Row("Adjust landmarks manually",
            "cycling · drag a point and the angles recalculate", (
                Cell("Only hip and knee", "on a photo"),
                Cell("Every joint", "photo and video"),
                Cell("Every joint", "photo and video"))),
        Row("Your earlier reports open too", "subscribe and your history opens in full",
            (NO, YES, YES)),
    )),
    ("See that it worked", (
        Row("Before / after analysis", "", (NO, YES, YES)),
        Row("History / trends over time", "", ("Score only", "Every angle", "Every angle")),
    )),
    ("Your bikes and setups", (
        Row("Hip over the crank", "how far forward you sit over the crank axle",
            (NO, NO, YES), new=True),
        Row("Saved setups", "the gear you ride and run with", (NO, "2", "<b>10</b>")),
        Row("Compare two bikes or setups", "", (NO, NO, YES)),
    )),
    ("Advice from a person", (
        Row("Expert Review", "a pro triathlete's full plan for the next weeks", (
            usd(EXPERT_AMOUNT),
            Cell(usd(REVIEW_BUMP_AMOUNT), "member price"),
            {"year": Cell("<b>1 a year included</b>",
                          f"every other one {usd(REVIEW_MEMBER_AMOUNT)}"),
             "month": Cell(usd(REVIEW_MEMBER_AMOUNT),
                           "member price · 1 a year included on the yearly plan")},
        )),
    )),
    ("Keep and share", (
        Row("Send the report to ChatGPT or your coach", "",
            (f"With a {usd(UNLOCK_AMOUNT)} report", YES, YES)),
        Row("Branded PDF report", "", (NO, NO, YES)),
    )),
)

# The human add-ons, said as when to take each and what arrives.
COACH_TITLE = "A pro triathlete watches your clip. Then tells you what to change."
COACH_POINTS = (
    ("person", "Personal analysis"),
    ("eye", "A pro's eye"),
    ("target", "One main point, then the plan"),
)
COACH_ROWS = (
    ("Take it when", "You have one report and one question",
     "A race is coming, something hurts, or you got a new bike"),
    ("You get", "The first thing to change, and 3 notes pinned on your frames",
     "What he sees, what to fix in what order for 4 weeks, drills, and what to "
     "film to check it worked. Written for you, not copied from a list"),
    ("Arrives as", "Notes inside your report", "A PDF to keep, your numbers at the back"),
    ("Ready in", "72 hours", "5 days"),
)
# Who writes them. A name and verifiable results, nothing added.
COACH_CREDIT = (
    "Notes and reviews are written by <b>Bogdan Kovalenko</b> · Professional "
    "triathlete · 2nd at Challenge Gunsan 2026 · Kazakhstan's top-ranked pro (PTO)"
)
SAMPLE_REVIEW_PDF = "/media/expert-review-sample.pdf"
SUPPORT_EMAIL = "support@getflapp.com"

_TIERS = ("starter", "enthusiast", "full")
_COL = dict(zip(_TIERS, "SEF"))
_ICONS = {
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "target": ('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
               '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'),
}


def _icon(name: str) -> str:
    return (
        '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>'
    )


def _one(v: Any) -> str:
    if v == YES:
        return '<span class="pr-yes" role="img" aria-label="Included">✓</span>'
    if v == NO:
        return '<span class="pr-no" role="img" aria-label="Not included">—</span>'
    if isinstance(v, Cell):
        note = f"<small>{v.note}</small>" if v.note else ""
        return f"<span>{v.text}</span>{note}"
    return f"<span>{v}</span>"


def _value(v: Any) -> str:
    if isinstance(v, dict):
        return f'<span class="py">{_one(v["year"])}</span><span class="pm">{_one(v["month"])}</span>'
    return _one(v)


def _same(a: Any, b: Any) -> bool:
    """The same answer on both billing periods -- what "only the differences"
    may hide."""
    def at(v: Any, period: str) -> Any:
        return v[period] if isinstance(v, dict) else v
    return all(at(a, p) == at(b, p) for p in ("year", "month"))


def _pair_classes(cells: tuple[Any, Any, Any]) -> str:
    s, e, f = cells
    out = []
    if _same(s, e):
        out.append("same-SE")
    if _same(s, f):
        out.append("same-SF")
    if _same(e, f):
        out.append("same-EF")
    return " ".join(out)


def _prices(card: Card) -> tuple[Price | None, Price | None]:
    return (next((p for p in card.prices if p.interval == "year"), None),
            next((p for p in card.prices if p.interval == "month"), None))


def _price_html(card: Card) -> str:
    """The headline amount as a per-month figure on both periods, so the three
    plans are always in one unit, and what is actually billed under it."""
    yearly, monthly = _prices(card)
    if not yearly or not monthly:
        return (f'<span class="pr-amt">{usd(card.prices[0].amount)}</span>'
                f'<span class="pr-billed">free forever</span>')
    return (
        f'<span class="pr-amt"><span class="py">{usd(round(yearly.amount / 12))}</span>'
        f'<span class="pm">{usd(monthly.amount)}</span><span class="pr-per"> /mo</span></span>'
        f'<span class="pr-billed"><span class="py">billed yearly as {usd(yearly.amount)}</span>'
        f'<span class="pm">billed monthly</span></span>'
    )


def _cta(card: Card, mode: str) -> str:
    cls = f"pr-btn pr-btn-{card.id}"
    if mode == "app":
        yearly, monthly = _prices(card)
        if not yearly or not monthly:
            return f'<span class="pr-slot" data-slot="cta" data-card="{card.id}"></span>'
        return (f'<span class="pr-slot py" data-slot="cta" data-card="{card.id}" data-cycle="year"></span>'
                f'<span class="pr-slot pm" data-slot="cta" data-card="{card.id}" data-cycle="month"></span>')
    return f'<a class="{cls}" href="{card.cta_href}">{card.cta_label}</a>'


def _fits(card: Card) -> str:
    pills = "".join(f'<span class="pr-pill">{f}</span>' for f in card.fits)
    return (f'<span class="pr-lead">{card.fits_lead}</span>'
            f'<span class="pr-pills">{pills}</span>')


def _bump(mode: str) -> str:
    if mode == "app":
        return '<span class="pr-slot" data-slot="bump"></span>'
    # Not advertised before checkout can honour it: without its Stripe price
    # the app shows no box, and the landing page must not promise one.
    from app.core.config import settings

    if not settings.stripe_price_expert_bump:
        return ""
    return (
        '<p class="pr-bump pr-bump-info"><span class="pr-plus" aria-hidden="true">+</span>'
        "<span><b>Add a pro's review of your first clip</b> · "
        f'<b class="pr-good">{usd(REVIEW_BUMP_AMOUNT)}</b> '
        f"<span class=\"pr-soft\">instead of {usd(EXPERT_AMOUNT)} · tick it when you choose "
        "Enthusiast</span></span></p>"
    )


def _period_toggle() -> str:
    def btn(per: str, label: str, pressed: bool) -> str:
        js = ("var r=this.closest('.pr');r.dataset.period='" + per + "';"
              "r.querySelectorAll('[data-per]').forEach(function(b){"
              "b.setAttribute('aria-pressed',String(b.dataset.per===r.dataset.period))})")
        return (f'<button type="button" data-per="{per}" aria-pressed="{str(pressed).lower()}" '
                f'onclick="{js}">{label}</button>')
    saved = annual_saving_pct()
    badge = f' <span class="pr-save">Save {saved}%</span>' if saved else ""
    return (
        '<div class="pr-period" role="group" aria-label="Billing period">'
        + btn("month", "Monthly", False) + btn("year", f"Yearly{badge}", True)
        + "</div>"
    )


def _scroll_to(target: str) -> str:
    return (f"event.preventDefault();var t=document.getElementById('{target}');"
            "if(t)t.scrollIntoView({behavior:'smooth',block:'start'})")


def _desk_table(tiers: list[Card], mode: str) -> str:
    heads = "".join(
        f'<th scope="col" class="pr-col-{c.id}"><div class="pr-head">'
        f'<span class="pr-name">{c.name}</span>{_fits(c)}'
        f"{_price_html(c)}{_cta(c, mode)}</div></th>"
        for c in tiers
    )
    body = []
    for title, rows in COMPARE:
        body.append(f'<tr class="pr-grp"><th colspan="4" scope="colgroup">{title}</th></tr>')
        for r in rows:
            new = '<span class="pr-new">New</span>' if r.new else ""
            sub = f"<small>{r.sub}</small>" if r.sub else ""
            cells = "".join(f"<td>{_value(v)}</td>" for v in r.cells)
            body.append(f'<tr><th scope="row">{r.label}{new}{sub}</th>{cells}</tr>')
    return (
        '<div class="pr-desk"><div class="pr-tablewrap"><table class="pr-table">'
        f'<thead><tr><th class="pr-what"><span>What you get</span></th>{heads}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'<div class="pr-bumprow">{_bump(mode)}</div></div>'
    )


def _feature_li(f: Feature) -> str:
    cls = f' class="{"py" if f.period == "year" else "pm"}"' if f.period else ""
    new = '<span class="pr-new">New</span>' if f.new else ""
    note = f"<small>{f.note}</small>" if f.note else ""
    return f'<li{cls}><span class="pr-tick" aria-hidden="true">✓</span><span>{f.text}{new}{note}</span></li>'


def _mob(tiers: list[Card], mode: str) -> str:
    tiles = []
    for c in tiers:
        key = (f'<span class="py">{c.key}</span><span class="pm">{c.key_month}</span>'
               if c.key_month else c.key)
        tiles.append(
            f'<a class="pr-tile pr-col-{c.id}" href="#pr-{c.id}" onclick="{_scroll_to("pr-" + c.id)}">'
            f'<span class="pr-name">{c.name}</span><span class="pr-short">{c.fits_short}</span>'
            f"{_price_html(c)}<span class=\"pr-key\">{key}</span></a>"
        )
    cards = []
    for c in tiers:
        plus = f'<p class="pr-plusfrom">Everything in {c.plus_from}, plus</p>' if c.plus_from else ""
        bump = _bump(mode) if c.id == "enthusiast" else ""
        cards.append(
            f'<div class="pr-card pr-col-{c.id}" id="pr-{c.id}">'
            f'<div class="pr-cardhead"><span class="pr-name">{c.name}</span>{_fits(c)}</div>'
            f'<div class="pr-cardprice">{_price_html(c)}</div>{plus}'
            f'<ul class="pr-list">{"".join(_feature_li(f) for f in c.features)}</ul>'
            f"{bump}{_cta(c, mode)}</div>"
        )
    return (
        '<div class="pr-mob">'
        f'<div class="pr-tiles">{"".join(tiles)}</div>'
        '<p class="pr-tilenote">Each plan has everything in the one before it · tap one to see it all</p>'
        f'<a class="pr-tocompare" href="#pr-compare" onclick="{_scroll_to("pr-compare")}">'
        "Compare plans side by side</a>"
        f'{"".join(cards)}{_compare(tiers)}</div>'
    )


def _compare(tiers: list[Card]) -> str:
    """Any two plans, side by side, only where they differ unless asked."""
    def chip(c: Card) -> str:
        x = _COL[c.id]
        js = ("var r=this.closest('.pr'),p=r.dataset.pick.split(',');"
              f"if(p.indexOf('{x}')<0){{p=[p[1],'{x}'];r.dataset.pick=p.join(',');"
              "r.dataset.hide='SEF'.replace(p[0],'').replace(p[1],'');"
              "r.querySelectorAll('[data-chip]').forEach(function(b){"
              "b.setAttribute('aria-pressed',String(b.dataset.chip!==r.dataset.hide))})}")
        return (f'<button type="button" class="pr-chip pr-chip-{x}" data-chip="{x}" '
                f'aria-pressed="{str(x != "S").lower()}" onclick="{js}">{c.name}</button>')

    def row(label: str, cells: tuple[Any, Any, Any]) -> str:
        vals = "".join(f'<span class="pr-v" data-t="{x}">{_value(v)}</span>'
                       for x, v in zip("SEF", cells))
        return (f'<div class="pr-crow {_pair_classes(cells)}"><span class="pr-clabel">{label}</span>'
                f'<div class="pr-vs">{vals}</div></div>')

    price_cells = []
    for c in tiers:
        yearly, monthly = _prices(c)
        price_cells.append(
            {"year": usd(round(yearly.amount / 12)), "month": usd(monthly.amount)}
            if yearly and monthly else usd(c.prices[0].amount)
        )
    rows = [row("Price a month", tuple(price_cells))]
    rows += [row(r.label, r.cells) for _, group in COMPARE for r in group]
    heads = "".join(f'<span class="pr-v pr-vhead pr-col-{c.id}" data-t="{_COL[c.id]}">{c.name}</span>'
                    for c in tiers)
    return (
        '<div class="pr-compare" id="pr-compare">'
        '<h3 class="pr-h3">Compare plans</h3><p class="pr-soft">Pick any two.</p>'
        f'<div class="pr-chips">{"".join(chip(c) for c in tiers)}</div>'
        f'<div class="pr-cbox"><div class="pr-vs pr-chead">{heads}</div>{"".join(rows)}'
        '<label class="pr-all"><input type="checkbox" '
        "onchange=\"this.closest('.pr').classList.toggle('pr-showall',this.checked)\">"
        "<span>Show every row, not only the differences</span></label></div></div>"
    )


def _coach(mode: str) -> str:
    notes, expert = CARD_BY_ID["notes"], CARD_BY_ID["expert"]
    points = "".join(f'<span class="pr-point"><span class="pr-ic">{_icon(i)}</span>{t}</span>'
                     for i, t in COACH_POINTS)
    if mode == "app":
        e_price = '<span class="pr-slot" data-slot="expert-price"></span>'
        n_cta = '<span class="pr-slot" data-slot="notes-cta"></span>'
        e_cta = '<span class="pr-slot" data-slot="expert-cta"></span>'
        n_sub = '<span class="pr-slot" data-slot="notes-sub"></span>'
        e_sub = '<span class="pr-slot" data-slot="expert-sub"></span>'
        sample = '<span class="pr-slot" data-slot="sample"></span>'
    else:
        e_price = usd(expert.prices[0].amount)
        n_cta = '<a class="pr-btn-s" href="/app">Get notes on my report</a>'
        e_cta = '<a class="pr-btn-s" href="/app#pricing">Get my plan from a pro</a>'
        n_sub = f"Then: a full review of your next clip, {usd(RECHECK_AMOUNT)}"
        e_sub = "Included once a year with Full, on the yearly plan"
        sample = (f'<a href="{SAMPLE_REVIEW_PDF}" target="_blank" rel="noopener">'
                  "see a sample</a>")
    n_price = usd(notes.prices[0].amount)
    rows = []
    for label, a, b in COACH_ROWS:
        if label == "Arrives as":
            b = f"{b} · {sample}"
        rows.append(f'<tr><th scope="row">{label}</th><td>{a}</td><td>{b}</td></tr>')
    rows.append(f'<tr class="pr-coachcta"><th></th><td>{n_cta}<small>{n_sub}</small></td>'
                f"<td>{e_cta}<small class=\"pr-good-sub\">{e_sub}</small></td></tr>")
    table = (
        '<div class="pr-tablewrap pr-coachdesk"><table class="pr-coachtable"><thead><tr><th></th>'
        f'<th scope="col"><span class="pr-cname">Coach Notes <span>{n_price}</span></span></th>'
        f'<th scope="col"><span class="pr-cname">Expert Review <span>{e_price}</span></span></th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )
    cards = (
        '<div class="pr-coachmob">'
        f'<div class="pr-ccard"><div class="pr-cname">Coach Notes <span>{n_price}</span></div>'
        "<p><b>One report, one question.</b> The first thing to change and 3 notes on your frames.</p>"
        f'<p class="pr-mono">Ready in 72 hours · {n_sub}</p>{n_cta}</div>'
        f'<div class="pr-ccard"><div class="pr-cname">Expert Review <span>{e_price}</span></div>'
        "<p><b>Race coming, something hurts, new bike?</b> What to fix in what order for "
        "4 weeks, with drills.</p>"
        f'<p class="pr-mono">Ready in 5 days · a PDF to keep · {sample}</p>{e_cta}'
        f'<small class="pr-good-sub">{e_sub}</small></div></div>'
    )
    return (
        '<section class="pr-coach" aria-label="Advice from a person">'
        '<p class="pr-eyebrow">Optional · on top of any plan · a person, not AI</p>'
        f'<h3 class="pr-h2">{COACH_TITLE}</h3><div class="pr-points">{points}</div>'
        f'{table}{cards}<p class="pr-credit"><span class="pr-bk" aria-hidden="true">BK</span>'
        f"<span>{COACH_CREDIT}</span></p></section>"
    )


def _ask() -> str:
    return (
        '<div class="pr-ask"><div><h3 class="pr-h3">Not sure which plan fits?</h3>'
        "<p>Ask before you pay. A real person reads every question and writes back to you.</p></div>"
        f'<a class="pr-btn-o" href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a></div>'
    )


def _research() -> str:
    return (
        '<div class="pr-research"><h3 class="pr-h3">Built on our own research</h3><div>'
        '<a href="/academy/kona-2026-pro-bike-positions"><b>32 pro bike positions, measured</b>'
        "<span>The same tool you get, run on the Kona 2026 field. Read the study →</span></a>"
        '<a href="/academy/kona-2026-hip-over-crank"><b>Kona pros don\'t sit higher. They sit '
        'further forward</b><span>31 pros and 68 age-groupers: the pros\' hips sit 4–5 cm '
        "further over the crank. Read the study →</span></a></div></div>"
    )


def render_pricing(mode: str = "landing") -> str:
    """The whole pricing block for ``mode`` "landing" (links) or "app" (slots)."""
    tiers = [CARD_BY_ID[t] for t in _TIERS]
    research = _research() if mode == "app" else ""
    return (
        f"<style>{PRICING_CSS}</style>"
        f'<div class="pr" data-mode="{mode}" data-period="year" data-pick="E,F" data-hide="S">'
        f"{_period_toggle()}{_desk_table(tiers, mode)}{_mob(tiers, mode)}"
        f"{_coach(mode)}{_ask()}{research}"
        f'<p class="pr-foot">{FOOTNOTE}</p></div>'
    )


def render_landing_pricing() -> str:
    """The landing page's pricing block (see render_pricing)."""
    return render_pricing("landing")


# Colours come from tokens.css, which both pages load; the tints are mixed from
# them so the block follows the app's dark theme too.
PRICING_CSS = """
.pr{container-type:inline-size;display:flex;flex-direction:column;gap:22px;color:var(--c-ink);font-family:var(--f-body);margin-top:24px}
.pr[data-period=year] .pm,.pr[data-period=month] .py{display:none}
.pr small{display:block;font-size:12.5px;line-height:1.4;color:var(--c-ink-faint);font-weight:400;margin-top:2px}
.pr b{font-weight:700}
.pr-soft{color:var(--c-ink-faint)}
.pr-good{color:var(--st-good-fill)}
.pr-new{display:inline-block;margin-left:6px;vertical-align:2px;font:700 10px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:#fff;background:var(--c-coral-btn);padding:2px 7px;border-radius:999px}
.pr-period{align-self:center;display:inline-flex;gap:4px;padding:4px;background:var(--c-bg);border:1px solid var(--c-line);border-radius:999px}
.pr-period button{font:600 14px var(--f-body);padding:10px 18px;min-height:44px;border:0;border-radius:999px;background:transparent;color:var(--c-ink-soft);cursor:pointer}
.pr-period button[aria-pressed=true]{background:var(--c-brand-navy);color:#fff}
.pr-save{display:inline-block;margin-left:4px;font:700 11px var(--f-body);padding:3px 7px;border-radius:999px;background:var(--st-good-bg);color:var(--st-good)}
.pr-name{font-weight:700;font-size:18px;color:var(--c-head)}
.pr-lead{font:500 11px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--c-ink-faint)}
.pr-col-enthusiast .pr-lead{color:var(--c-coral-btn)} .pr-col-full .pr-lead{color:var(--c-blue)}
.pr-pills{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.pr-pill{font:600 12.5px/1 var(--f-body);padding:6px 11px;border-radius:999px;border:1px solid var(--c-line-strong);background:var(--c-panel);color:var(--c-ink-soft);white-space:nowrap}
.pr-col-enthusiast .pr-pill{background:color-mix(in srgb,var(--c-coral-btn) 12%,var(--c-bg));border-color:color-mix(in srgb,var(--c-coral-btn) 40%,var(--c-bg));color:var(--c-coral-dk)}
.pr-col-full .pr-pill{background:color-mix(in srgb,var(--c-blue) 11%,var(--c-bg));border-color:color-mix(in srgb,var(--c-blue) 35%,var(--c-bg));color:var(--c-head)}
.pr-amt{font-family:var(--f-brand);font-style:italic;font-weight:900;letter-spacing:-.02em;font-size:34px;line-height:1.05;color:var(--c-head)}
.pr-per{font:400 13px var(--f-mono);font-style:normal;letter-spacing:0;color:var(--c-ink-faint)}
.pr-billed{font-size:13px;color:var(--c-ink-soft);font-weight:500}
.pr-btn{display:flex;align-items:center;justify-content:center;width:100%;box-sizing:border-box;min-height:48px;padding:12px 14px;border-radius:8px;font:700 15px var(--f-body);text-decoration:none;cursor:pointer;border:1.5px solid transparent;text-align:center}
.pr-btn-starter{background:var(--c-bg);color:var(--c-head);border-color:var(--c-head)}
.pr-btn-enthusiast{background:var(--c-coral-btn);color:#fff}
.pr-btn-full{background:var(--c-brand-navy);color:#fff}
.pr-btn:hover{filter:brightness(.93)}
.pr-btn-dead{background:var(--c-panel);color:var(--c-ink-disabled);cursor:default}
.pr-slot{display:block;width:100%}
.pr-slot[data-slot=expert-price],.pr-slot[data-slot=sample]{display:inline;width:auto}
.pr-btn-s{display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;min-height:42px;padding:9px 18px;border-radius:999px;background:var(--c-brand-navy);color:#fff;font:700 14px var(--f-body);text-decoration:none;border:0;cursor:pointer}
.pr-btn-o{display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;min-height:44px;padding:10px 18px;border-radius:8px;border:1.5px solid var(--c-head);color:var(--c-head);font:700 15px var(--f-body);text-decoration:none;background:var(--c-bg)}
.pr-yes{color:var(--st-good-fill);font-weight:700;font-size:18px}
.pr-no{color:var(--c-line-strong);font-size:18px}
/* ---- desktop: the table ---- */
.pr-tablewrap{overflow-x:auto;background:var(--c-bg);border:1px solid var(--c-line);border-radius:10px}
.pr-table{width:100%;border-collapse:separate;border-spacing:0;font-size:15px;line-height:1.4;min-width:760px;table-layout:fixed}
.pr-table th,.pr-table td{padding:12px 16px;text-align:center;vertical-align:middle;border-bottom:1px solid var(--c-line)}
.pr-table thead th{vertical-align:top;padding:22px 14px 18px}
.pr-table tbody th{text-align:left;font-weight:600;width:31%}
.pr-table td{color:var(--c-ink-soft)}
.pr-table tr:last-child>*{border-bottom:0}
.pr-table .pr-what{text-align:left;vertical-align:bottom;font-size:13px;color:var(--c-ink-faint);font-weight:500}
.pr-table tr.pr-grp th{background:var(--c-ground);font:600 11px var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--c-ink-faint);padding:10px 16px}
.pr-head{display:flex;flex-direction:column;gap:7px;align-items:center}
.pr-head .pr-pills{min-height:26px}
.pr-head .pr-billed{min-height:18px;margin-bottom:6px}
.pr-bumprow{display:flex;justify-content:center;margin-top:14px}
.pr-bump{display:flex;gap:12px;align-items:center;margin:0;padding:12px 16px;border:1.5px dashed var(--c-line-strong);border-radius:10px;background:var(--c-bg);font-size:14.5px;line-height:1.45;max-width:660px;box-sizing:border-box}
.pr-bump input{width:20px;height:20px;margin:0;flex:none;accent-color:var(--c-brand-navy)}
.pr-plus{flex:none;width:22px;height:22px;border-radius:999px;background:var(--c-panel);color:var(--c-head);font-weight:700;display:flex;align-items:center;justify-content:center}
/* ---- phone: three at a glance, a card per plan, compare any two ---- */
.pr-mob{display:none;flex-direction:column;gap:16px}
.pr-tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.pr-tile{display:flex;flex-direction:column;align-items:center;text-align:center;gap:3px;padding:14px 6px 12px;background:var(--c-bg);border:1.5px solid var(--c-line);border-radius:14px;text-decoration:none;color:inherit}
.pr-tile.pr-col-enthusiast{border-color:color-mix(in srgb,var(--c-coral-btn) 40%,var(--c-bg))}
.pr-tile.pr-col-full{border:2px solid var(--c-brand-navy);background:color-mix(in srgb,var(--c-blue) 8%,var(--c-bg))}
.pr-tile .pr-name{font-size:15px}
.pr-short{font:500 9.5px var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--c-ink-faint)}
.pr-col-enthusiast .pr-short{color:var(--c-coral-dk)} .pr-col-full .pr-short{color:var(--c-blue)}
.pr-tile .pr-amt{font-size:22px;margin-top:6px}
.pr-tile .pr-per{display:none}
.pr-tile .pr-billed{font-size:11.5px;color:var(--c-ink-faint);font-weight:400}
.pr-key{align-self:stretch;margin-top:8px;padding-top:8px;border-top:1px solid var(--c-line);font-size:13px;font-weight:600;line-height:1.3}
.pr-tilenote{margin:-6px 0 0;text-align:center;font-size:14px;color:var(--c-ink-faint)}
.pr-tocompare{align-self:center;display:inline-flex;align-items:center;min-height:44px;padding:0 18px;border:1.5px solid var(--c-head);border-radius:999px;font-size:15px;font-weight:700;color:var(--c-head);text-decoration:none;background:var(--c-bg)}
.pr-card{background:var(--c-bg);border:1px solid var(--c-line);border-radius:16px;padding:22px 20px;display:flex;flex-direction:column;gap:14px;scroll-margin-top:72px}
.pr-card.pr-col-enthusiast{border:2px solid color-mix(in srgb,var(--c-coral-btn) 40%,var(--c-bg))}
.pr-card.pr-col-full{border:2px solid var(--c-brand-navy);box-shadow:var(--shadow)}
.pr-cardhead{display:flex;flex-direction:column;gap:6px;align-items:flex-start}
.pr-cardhead .pr-name{font-size:22px}
.pr-cardhead .pr-pills{justify-content:flex-start}
.pr-cardprice{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px}
.pr-plusfrom{margin:0;padding-top:14px;border-top:1px solid var(--c-line);font-size:15px;font-weight:700;color:var(--c-head)}
.pr-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:12px}
.pr-list li{display:grid;grid-template-columns:22px minmax(0,1fr);gap:10px;font-size:16px;line-height:1.4}
.pr-list small{font-size:14px}
.pr-tick{color:var(--st-good-fill);font-weight:700;font-size:18px;line-height:1.3}
.pr-card .pr-bump{font-size:15px}
.pr-compare{display:flex;flex-direction:column;gap:12px;scroll-margin-top:72px;margin-top:8px}
.pr-h3{margin:0;font-family:var(--f-brand);font-style:italic;font-weight:900;letter-spacing:-.02em;font-size:26px;line-height:1.1;color:var(--c-head)}
.pr-compare .pr-soft{margin:-6px 0 0;font-size:16px}
.pr-chips{display:flex;gap:8px}
.pr-chip{flex:1;min-height:46px;border-radius:999px;font:700 15px var(--f-body);cursor:pointer;background:var(--c-bg);color:var(--c-head);border:1.5px solid var(--c-line-strong)}
.pr-chip[aria-pressed=true]{background:var(--c-brand-navy);color:#fff;border-color:var(--c-brand-navy)}
.pr-chip[aria-pressed=true]::before{content:"✓ "}
.pr-chip[aria-pressed=false]::before{content:"+ "}
.pr-cbox{background:var(--c-bg);border:1px solid var(--c-line);border-radius:14px;overflow:hidden}
.pr-vs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}
.pr[data-hide=S] [data-t=S],.pr[data-hide=E] [data-t=E],.pr[data-hide=F] [data-t=F]{display:none}
.pr-chead{border-bottom:1px solid var(--c-line)}
.pr-vhead{padding:14px 16px;font-size:17px;font-weight:700;color:var(--c-head);background:var(--c-panel)}
.pr-vhead.pr-col-enthusiast{color:var(--c-coral-dk);background:color-mix(in srgb,var(--c-coral-btn) 9%,var(--c-bg))}
.pr-vhead.pr-col-full{background:color-mix(in srgb,var(--c-blue) 9%,var(--c-bg))}
.pr-crow{border-bottom:1px solid var(--c-line)}
.pr-clabel{display:block;padding:14px 16px 4px;font-size:14px;color:var(--c-ink-faint)}
.pr-v{padding:2px 16px 14px;font-size:17px;font-weight:600}
.pr-v small{font-size:13px}
.pr:not(.pr-showall)[data-hide=F] .same-SE,.pr:not(.pr-showall)[data-hide=E] .same-SF,.pr:not(.pr-showall)[data-hide=S] .same-EF{display:none}
.pr-all{display:flex;align-items:center;gap:12px;padding:12px 16px;min-height:44px;box-sizing:border-box;font-size:15px;font-weight:600;color:var(--c-head);cursor:pointer}
.pr-all input{width:22px;height:22px;margin:0;flex:none;accent-color:var(--c-brand-navy)}
/* ---- the human add-ons ---- */
.pr-coach{display:flex;flex-direction:column;gap:16px;background:var(--c-panel-blue);border:1.5px solid color-mix(in srgb,var(--c-blue) 35%,var(--c-bg));border-radius:14px;padding:26px 22px;margin-top:6px}
.pr-eyebrow{margin:0;font:500 11px var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--c-blue)}
.pr-h2{margin:0;font-family:var(--f-brand);font-style:italic;font-weight:900;letter-spacing:-.02em;font-size:30px;line-height:1.1;color:var(--c-head)}
.pr-points{display:flex;flex-wrap:wrap;gap:12px 36px}
.pr-point{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;color:var(--c-head)}
.pr-ic{width:36px;height:36px;flex:none;border-radius:999px;background:var(--c-bg);border:1px solid color-mix(in srgb,var(--c-blue) 35%,var(--c-bg));display:flex;align-items:center;justify-content:center;color:var(--c-blue)}
.pr-ic svg{width:18px;height:18px}
.pr-coachtable{width:100%;border-collapse:collapse;font-size:15px;line-height:1.45;min-width:600px;table-layout:fixed}
.pr-coachtable th,.pr-coachtable td{padding:14px 18px;text-align:left;vertical-align:top;border-bottom:1px solid var(--c-line)}
.pr-coachtable thead th{background:var(--c-ground)}
.pr-coachtable tbody th{font-weight:600;width:24%}
.pr-coachtable td{color:var(--c-ink-soft)}
.pr-coachtable tr:last-child>*{border-bottom:0}
.pr-coachtable small{margin-top:8px}
.pr-cname{display:flex;align-items:baseline;gap:10px;white-space:nowrap;font-family:var(--f-brand);font-style:italic;font-weight:900;letter-spacing:-.02em;font-size:24px;color:var(--c-head)}
.pr-cname span{font:inherit}
.pr-good-sub{color:var(--st-good-fill)!important;font-weight:600!important}
.pr-coachmob{display:none;flex-direction:column;gap:12px}
.pr-ccard{background:var(--c-bg);border:1px solid var(--c-line);border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:10px;align-items:flex-start}
.pr-ccard .pr-cname{justify-content:space-between;width:100%}
.pr-ccard p{margin:0;font-size:16px;line-height:1.45;color:var(--c-ink-soft)}
.pr-ccard p b{color:var(--c-ink)}
.pr-ccard .pr-mono{font-size:13.5px;color:var(--c-ink-faint)}
.pr-credit{display:flex;align-items:flex-start;gap:10px;margin:0;font-size:13.5px;line-height:1.5;color:var(--c-ink-soft)}
.pr-credit b{color:var(--c-head)}
.pr-bk{width:32px;height:32px;flex:none;border-radius:999px;background:var(--c-brand-navy);color:#fff;display:flex;align-items:center;justify-content:center;font:italic 900 12px var(--f-brand)}
.pr-ask{display:flex;align-items:center;justify-content:space-between;gap:16px 28px;background:var(--c-bg);border:1px solid var(--c-line);border-radius:10px;padding:22px}
.pr-ask p{margin:6px 0 0;font-size:15px;line-height:1.5;color:var(--c-ink-soft)}
.pr-research{display:flex;flex-direction:column;gap:12px}
.pr-research>div{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.pr-research a{display:flex;flex-direction:column;gap:4px;text-decoration:none;color:inherit;background:var(--c-bg);border:1px solid var(--c-line);border-radius:10px;padding:16px 18px}
.pr-research b{color:var(--c-head);font-size:15px}
.pr-research span{font-size:13.5px;color:var(--c-ink-soft)}
.pr-foot{margin:0;text-align:center;font:12px var(--f-mono);color:var(--c-ink-faint)}
/* Last, so it wins over the defaults above. */
@container (max-width:760px){
  .pr-desk,.pr-coachdesk{display:none}
  .pr-mob{display:flex}
  .pr-coachmob{display:flex}
  .pr-h2{font-size:26px}
  .pr-points{flex-direction:column;gap:10px}
  .pr-ask{flex-direction:column;align-items:stretch}
  .pr-research>div{grid-template-columns:1fr}
}
"""
