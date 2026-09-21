"""The native bundle shows no way to buy (Apple 3.1.1 / Play Billing).

``mobile/bridge/bridge.js`` hides the web app's purchase surfaces by
selector. That list is only as complete as the last person to add an offer
remembered to make it: the Coach Notes card ($12) shipped into the native
bundle for two days before its container was added. This test names every
purchase surface the SPA renders, so a new one fails here until the bridge
knows about it.

When a new offer lands in the SPA: add its container to ``PURCHASE_SURFACES``
AND to the bridge's hide list, in the same commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "mobile" / "bridge" / "bridge.js"
SPA = ROOT / "backend" / "app" / "static" / "index.html"

# selector in the bridge  ->  a string that proves the surface exists in the SPA
PURCHASE_SURFACES = {
    "#navPricing": 'id="navPricing"',
    # The pricing screen whole -- a price list with no in-app way to buy is
    # the same rejection as a buy button -- and every link into it.
    "#pricing": 'id="pricing"',
    'a[href="#pricing"]': 'href="#pricing"',
    ".plans-link": 'class="plans-link"',
    # The report's teaser card whole, and the per-report unlock wherever it is.
    ".upsell": 'class="upsell"',
    ".btn-unlock": "btn-unlock",
    ".solo-buy": 'class="solo-buy"',
    # The Coach Notes / Expert Review card in every state (offer, booked,
    # delivered-with-upsell). Delivered notes render in .cnote, not here.
    ".notes-offer": 'class="notes-offer',
    # Anything else wired inline to a purchase function.
    '[onclick*="startCheckout"]': "startCheckout(",
    '[onclick*="openPricing"]': "openPricing()",
    '[onclick*="openUpgrade"]': "openUpgrade",
}

# Things the bridge must NOT hide: they carry what a customer already paid
# for, and a stale selector once did exactly this (.addon-strip had become
# the orders box, so the store build hid delivered reviews).
MUST_STAY_VISIBLE = ("#ordersBox", ".addon-strip", "#history", ".cnote", "#coachNoteBlock")

# The functions every purchase path in the SPA ends in. Natively each is
# replaced by a no-op that says purchases are not available here -- so a
# call site the CSS list misses still sells nothing.
PURCHASE_FUNCTIONS = ("openPricing", "openUpgrade", "startCheckout", "openPortal")


@pytest.fixture(scope="module")
def bridge() -> str:
    return BRIDGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spa() -> str:
    return SPA.read_text(encoding="utf-8")


@pytest.mark.parametrize("selector,marker", sorted(PURCHASE_SURFACES.items()))
def test_every_purchase_surface_is_hidden_natively(bridge, spa, selector, marker):
    assert marker in spa, f"{selector}: the SPA no longer renders this -- drop it from both lists"
    assert f"'.flapp-native {selector}'" in bridge, (
        f"{selector} is sold in the SPA but not hidden by mobile/bridge/bridge.js"
    )


def test_every_checkout_call_site_has_a_known_container(spa):
    """Each ``startCheckout('<plan>'`` in the markup sits in one of the
    containers above. Coarse on purpose: a call site whose nearest class
    attribute is not on the list is a new surface, and the answer is to add
    it, not to loosen this."""
    known = ("tier-cta", "cta-row", "solo-buy", "no-cta", "btn-unlock",
             "data-notesbuy", "navPricing", "onclick=\"startCheckout")
    for i, line in enumerate(spa.splitlines(), 1):
        if "startCheckout('" not in line or "function startCheckout" in line:
            continue
        # A call site inside JS (an event handler wired by id) is covered by
        # the element it targets, which is listed above by id.
        if "onclick=" not in line and "data-" not in line:
            continue
        # Six lines before (the container the markup sits in) or twelve after
        # (a helper whose return value is dropped into one).
        lines = spa.splitlines()
        window = "\n".join(lines[max(0, i - 6):i + 12])
        assert any(k in window for k in known), (
            f"index.html:{i} sells something outside every container the "
            f"native bridge hides: {line.strip()[:120]}"
        )


@pytest.mark.parametrize("selector", MUST_STAY_VISIBLE)
def test_what_was_paid_for_is_not_hidden(bridge, selector):
    assert f"'.flapp-native {selector}'" not in bridge, (
        f"{selector} holds delivered purchases / history and must stay visible natively"
    )


def test_every_selector_in_the_bridge_is_accounted_for(bridge):
    """The other direction: a selector in the bridge that this file does not
    know is either a new surface (add it above) or a stale one (drop it)."""
    import re

    block = bridge.split("4. No purchase paths", 1)[1].split("5. Fit the phone", 1)[0]
    listed = set(re.findall(r"'\.flapp-native ([^']+)'", block))
    assert listed == set(PURCHASE_SURFACES), (
        f"bridge hides {sorted(listed - set(PURCHASE_SURFACES))} which this test "
        f"does not know, or is missing {sorted(set(PURCHASE_SURFACES) - listed)}"
    )


@pytest.mark.parametrize("name", PURCHASE_FUNCTIONS)
def test_every_purchase_function_is_neutralised_natively(bridge, spa, name):
    assert f"function {name}(" in spa, f"{name} is gone from the SPA -- drop it from PURCHASE_FUNCTIONS"
    assert f"'{name}'" in bridge, f"{name} is a purchase path in the SPA but the bridge does not override it"


def test_no_checkout_outside_the_named_functions(spa):
    """Every request to /billing/checkout or /billing/portal lives inside one
    of PURCHASE_FUNCTIONS, so overriding those four is overriding all of it."""
    lines = spa.splitlines()
    for i, line in enumerate(lines):
        if "'/billing/checkout'" not in line and "'/billing/portal'" not in line:
            continue
        if line.lstrip().startswith(("//", "*", "/*")) or "Checkout hits" in line:
            continue
        above = "\n".join(lines[max(0, i - 40):i])
        heads = [k for k in PURCHASE_FUNCTIONS if f"function {k}(" in above]
        assert heads, f"index.html:{i + 1} talks to billing outside every purchase function: {line.strip()[:100]}"
