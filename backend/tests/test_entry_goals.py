"""The landing's "What brings you here?" -- three ways in, followed to the sale.

The point of the three buttons is a number: which problem brings the people
who end up paying. That only works if the label survives the whole way:

1. the landing links carry ``?goal=`` with the keys the app knows;
2. the app remembers it, registers it on every later browser event, and puts
   it on the person at identify -- purchases are recorded by the server, under
   the person, so a browser-only property would never reach them;
3. the definitions sit above ``trackIdentify``: a ``const`` read before its
   line throws, ``trackIdentify`` swallows errors, and identify would stop
   for everyone without a sound.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


@pytest.fixture(scope="module")
def landing() -> str:
    return (STATIC / "landing.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _app_goals(app_html: str) -> set[str]:
    block = app_html[app_html.index("const GOALS={"):]
    block = block[:block.index("};")]
    return set(re.findall(r"^\s*([a-z]+):\{sport:", block, flags=re.MULTILINE))


def test_the_hero_offers_three_ways_in_and_a_plain_one(landing):
    hero = landing[landing.index('<header class="hero">'):landing.index("</header>")]
    goals = re.findall(r'class="goal-btn" id="\w+" href="/app\?goal=([a-z]+)"', hero)
    assert goals == ["knee", "faster", "race"]
    assert 'id="heroGo" href="/app"' in hero


def test_every_landing_goal_is_one_the_app_knows(landing, app_html):
    sent = set(re.findall(r'href="/app\?goal=([a-z]+)"', landing))
    assert sent and sent <= _app_goals(app_html)


def test_each_goal_opens_a_real_sport_or_none(app_html):
    block = app_html[app_html.index("const GOALS={"):]
    block = block[:block.index("};")]
    for sport in re.findall(r"sport:([^,]+),", block):
        assert sport in ("'bike'", "'run'", "null"), sport


def test_the_goal_reaches_every_event_and_the_person(app_html):
    assert "register({entry_goal:g})" in app_html
    ident = app_html[app_html.index("function trackIdentify(){"):]
    ident = ident[:ident.index("\n}\n")]
    assert "props.entry_goal=g" in ident


def test_the_definitions_come_before_their_first_reader(app_html):
    assert app_html.index("const GOALS={") < app_html.index("function trackIdentify(){")


def test_the_url_is_cleaned_and_the_note_has_a_home(app_html):
    handler = app_html[app_html.index("const q=new URLSearchParams(location.search).get('goal')"):]
    handler = handler[:handler.index("})();")]
    assert "searchParams.delete('goal')" in handler
    assert "track('entry_goal_landed'" in handler
    assert 'id="goalNote"' in app_html
