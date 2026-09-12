"""The free allowance is written in one place, and the pages quote it.

This copy has now drifted back twice. The limit became 10 a month; the landing
page went on promising 3 in five places, that was fixed with a
``<!--FREE-LIMIT-->`` token, and two spots survived the fix anyway -- the
pricing lede ("Three analyses a month cost nothing") and the stat tile
("$0 · To start · 3 / month free"). Both sat on the page for weeks, telling
visitors they get less than a third of what the server actually gives them,
next to cards saying 10.

Nothing caught it because nothing looked. The token was introduced without a
test that the literals were gone, so "fixed" meant "fixed in the places
somebody happened to grep that afternoon".

The number itself lives in the tier table and is asserted nowhere here: this
file is about the pages never spelling it out themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services import pricing

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
PAGES = ["landing.html", "index.html"]

# Spellings of a hard-coded allowance. Deliberately not a bare "3": the page is
# full of legitimate threes (three capture rules, three plans, a $39 add-on).
HARDCODED = re.compile(
    r"(?:three|\b\d+)\s+analyses\b"          # "three analyses", "3 analyses"
    r"|\b\d+\s*/\s*month free"               # "3 / month free"
    r"|\b\d+\s+analyses\s*(?:/|per)\s*month",
    re.IGNORECASE,
)


def _copy_only(text: str) -> str:
    """The page with its comments removed.

    A comment explaining the bug ("read 'Free account required · 10 analyses'
    off their own uploader") is not copy, and failing on it would teach the next
    person to weaken the rule instead of obeying it. Comments go first, then the
    sanctioned token.
    """
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)   # HTML, incl. the token
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)    # CSS + JS block
    text = re.sub(r"^\s*//.*$", " ", text, flags=re.M)    # JS line comment
    return text


@pytest.mark.parametrize("page", PAGES)
def test_no_page_spells_out_the_free_allowance(page):
    text = (STATIC / page).read_text(encoding="utf-8")
    found = HARDCODED.findall(_copy_only(text))
    assert not found, (
        f"{page} spells out the free allowance: {found}. "
        f"Write {pricing.FREE_LIMIT_TOKEN} instead -- the server renders the "
        f"real number from the tier table."
    )


def test_the_landing_page_actually_uses_the_token():
    """A page with no token cannot be wrong, and cannot be right either."""
    text = (STATIC / "landing.html").read_text(encoding="utf-8")
    # Hero trust line, FAQ answer, CTA band. (A fourth, in the count-up stat
    # tile, left with the tile: design system v3 retired the stats row.)
    assert text.count(pricing.FREE_LIMIT_TOKEN) >= 3


def test_rendering_replaces_every_token_with_the_enforced_number():
    text = (STATIC / "landing.html").read_text(encoding="utf-8")
    rendered = text.replace(
        pricing.FREE_LIMIT_TOKEN, str(pricing.starter_monthly_limit()),
    )
    assert pricing.FREE_LIMIT_TOKEN not in rendered
    assert f"{pricing.starter_monthly_limit()} analyses a month" in rendered
