"""Examples, Academy and What's new: the site's menu for visitors, the app's
sidebar for signed-in athletes.

Artur, 2026-10-24, after the menu changed under him a second time: a visitor
who clicks "See 4 real reports" on the landing page should still see the
landing's menu, not an app sidebar of empty Progress / History links.

What breaks this is a refactor that "unifies the menus" again, so the
contract is pinned here: the guest bar carries the landing's items in the
landing's order, the shell is chosen before first paint, and an expired token
falls back to the guest bar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

LANDING = Path(__file__).resolve().parents[1] / "app" / "static" / "landing.html"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _landing_nav_labels() -> list[str]:
    html = LANDING.read_text(encoding="utf-8")
    nav = html[html.index('<div class="nav-links">'):]
    nav = nav[:nav.index("</div>")]
    return re.findall(r">([^<>]+)</a>", nav)


@pytest.mark.parametrize("path", ["/examples", "/academy", "/changelog"])
def test_visitors_get_the_landing_menu(client, path):
    html = client.get(path).text
    bar = html[html.index('<nav class="gnav"'):]
    bar = bar[:bar.index("</nav>")]
    links = re.findall(r'<div class="gnav-links">(.*?)</div>', bar, flags=re.DOTALL)[0]
    assert re.findall(r">([^<>]+)</a>", links) == _landing_nav_labels()
    assert 'href="/app#login"' in bar and 'href="/app">Analyze free' in bar


@pytest.mark.parametrize("path", ["/examples", "/academy", "/changelog"])
def test_the_shell_is_picked_before_first_paint(client, path):
    html = client.get(path).text
    head = html[:html.index("</head>")]
    assert "classList.add('signed-in')" in head
    # The app sidebar is still on the page for signed-in athletes.
    assert 'id="sidebar"' in html
    # Hidden for visitors by CSS, shown for signed-in.
    assert "html:not(.signed-in) .sidebar" in head and ".signed-in .gnav{display:none}" in head


def test_an_expired_session_falls_back_to_the_guest_menu(client):
    html = client.get("/examples").text
    assert "classList.remove('signed-in')" in html
