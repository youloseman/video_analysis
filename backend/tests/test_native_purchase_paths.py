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
    ".upsell .cta-row": 'class="cta-row"',
    ".upsell .price-tag": "price-tag",
    "#pricing .tier-cta": 'class="tier-cta"',
    ".solo-buy": 'class="solo-buy"',
    ".no-price": 'class="no-price"',
    ".no-cta": 'class="no-cta"',
    ".btn-unlock": "btn-unlock",
}

# Things the bridge must NOT hide: they carry what a customer already paid
# for, and a stale selector once did exactly this (.addon-strip had become
# the orders box, so the store build hid delivered reviews).
MUST_STAY_VISIBLE = ("#ordersBox", ".addon-strip", "#history", ".no-card")


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
             "data-notesbuy", "navPricing")
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
