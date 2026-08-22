"""The mobility feature's client half, checked against the shipped SPA.

String checks rather than browser tests, for the same reason
``test_print_report.py`` is: the things that break here are a class name, a
key name, or a guard, and no Python test would otherwise ever look at them.

Two of these guard specific ways the feature would silently stop working. A
server field renamed without the client following it makes the card vanish
with no error anywhere. And a rider whose bike needs nothing changed would
never see the card at all if the fit block still hid itself on an empty
adjustment list -- which is exactly the reader the card exists for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.video_analysis.biomechanics import mobility as M

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return SPA.read_text(encoding="utf-8")


def test_the_panel_and_the_card_both_exist(html: str):
    assert 'id="mobBox"' in html
    assert 'id="mobScreens"' in html
    assert 'id="fitMobility"' in html


def test_the_client_reads_the_fields_the_server_sends(html: str):
    """A renamed key does not error -- the card just quietly disappears."""
    for field in ("mobility_fit", "tier_label", "unrestricted", "capture_warnings"):
        assert field in html, f"the SPA never reads {field}"


def test_the_card_renders_the_caveat(html: str):
    """The measurement is a measurement; the ceiling is a fitter's heuristic.
    Printing the numbers without the sentence that separates them would let
    the honest half carry the inferred one."""
    block = html[html.index("function renderMobilityFit("):]
    block = block[:block.index("\nasync function openPortal")]
    assert "fit.caveat" in block
    assert "fit.message" in block


def test_a_clean_fit_still_shows_the_card(html: str):
    """The rider whose bike needs nothing changed can still be riding below
    the range they measured. If an empty diagnostics list hides the block, that
    reader -- the one the card is for -- never sees it."""
    block = html[html.index("function renderFitPlan("):]
    block = block[:block.index("\n/* Analysis-confidence badge")]
    guard = re.search(r"if\(!diags\.length[^)]*\)\{", block)
    assert guard, "the empty-plan guard changed shape"
    assert "!mob" in guard.group(0), "an empty adjustment list still hides the mobility card"


def test_the_new_action_token_has_english(html: str):
    """The builder speaks in tokens. An unmapped one degrades to the raw token
    with underscores opened out, which reads as our schema leaking out."""
    assert "improve_mobility:" in html


def test_the_new_status_has_a_row_style(html: str):
    assert "mobility_limited:" in html


def test_every_tier_has_a_chip_class(html: str):
    block = html[html.index("const MOB_TIER_CLS={"):]
    block = block[:block.index("}")]
    for tier in M.TIER_ORDER:
        assert f"{tier}:" in block, f"tier {tier} has no chip class"


def test_the_capture_instructions_are_not_duplicated_in_the_client(html: str):
    """The setup steps and the checks that enforce them live in the same
    module server-side. A client copy would drift from the validation, and the
    athlete would be told off for following our own instructions."""
    for step in M.MOBILITY_SCREENS["hamstring"]["setup"]:
        assert step not in html, f"capture step hard-coded in the SPA: {step[:40]}"
    assert "s.setup.map" in html, "the SPA no longer renders the served steps"


def test_a_server_refusal_is_shown_verbatim(html: str):
    """"The raised knee is bent" is an instruction. Replacing it with a
    generic failure would leave the rider guessing at a problem we already
    diagnosed."""
    block = html[html.index("async function uploadMobScreen("):]
    block = block[:block.index("\nasync function saveMobGoal")]
    assert "j.detail" in block


def test_the_panel_is_bike_only_and_signed_in_only(html: str):
    block = html[html.index("function mobShouldShow("):]
    block = block[:block.index("}")]
    assert "auth" in block and "'bike'" in block and "photo" in block
