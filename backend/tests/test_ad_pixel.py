"""The Meta ad pixel: where it is, what it waits for, and what the policy says.

Four promises, each one a sentence somebody reads on the site:

1. **Public pages only.** The landing page, the Academy, the sample reports,
   What's new and the privacy policy carry it; /app never does -- the upload
   screen says "No ad trackers in the app".
2. **Nothing before consent.** Meta's script URL appears only inside the
   function the Allow button calls. The ``<noscript><img>`` fallback Meta hands
   out fires without asking anyone, so it must never be on a page.
3. **No automatic collection.** ``autoConfig`` is set off before ``init``.
4. **The policy matches the switch.** With the pixel on it names Meta and
   explains the banner; with it off it says there are no ad trackers at all.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import _render_shell, app
from app.services import ad_pixel

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
LOADER = "connect.facebook.net/en_US/fbevents.js"


@pytest.fixture
def client():
    _render_shell.cache_clear()
    with TestClient(app) as c:
        yield c
    _render_shell.cache_clear()


@pytest.fixture
def pixel_off(monkeypatch):
    monkeypatch.setattr(ad_pixel, "settings", dataclasses.replace(settings, meta_pixel_id=None))
    _render_shell.cache_clear()
    yield
    _render_shell.cache_clear()


# --------------------------------------------------------------------------
# where it is
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/", "/academy", "/examples", "/changelog", "/privacy"])
def test_public_pages_carry_the_consent_gated_pixel(client, path):
    html = client.get(path).text
    assert "w.flappAds={" in html, path
    assert settings.meta_pixel_id in html, path


def test_the_app_never_does(client):
    html = client.get("/app").text
    assert "flappAds" not in html
    assert LOADER not in html
    assert "fbq(" not in html


def test_an_academy_article_carries_it_too(client):
    hub = client.get("/academy").text
    import re
    slug = re.search(r'href="/academy/([a-z0-9-]+)"', hub).group(1)
    assert "w.flappAds={" in client.get(f"/academy/{slug}").text


# --------------------------------------------------------------------------
# nothing before consent, nothing automatic
# --------------------------------------------------------------------------
def test_the_loader_is_requested_only_from_the_consent_path():
    js = ad_pixel.snippet_html()
    assert js.count(LOADER) == 1
    load_fn = js[js.index("function load()"):js.index("var bar=null;")]
    assert LOADER in load_fn, "Meta's script must be requested from load() and nowhere else"
    # load() runs from allow(), or from a stored yes -- never unconditionally.
    tail = js[js.index("var dnt="):]
    assert "if(c==='yes' && !dnt){ load(); return; }" in tail


def test_do_not_track_is_a_no():
    js = ad_pixel.snippet_html()
    assert "if(c==='no' || dnt) return;" in js


def test_no_noscript_beacon_anywhere(client):
    for path in ("/", "/academy", "/examples", "/changelog", "/privacy", "/app"):
        html = client.get(path).text
        assert "facebook.com/tr" not in html, path


def test_automatic_collection_is_off_before_init():
    js = ad_pixel.snippet_html()
    assert js.index("'autoConfig',false") < js.index("fbq('init'")


def test_a_no_after_a_yes_revokes():
    assert "fbq('consent','revoke')" in ad_pixel.snippet_html()


# --------------------------------------------------------------------------
# the switch, and the policy
# --------------------------------------------------------------------------
def test_off_means_no_banner_and_no_script(client, pixel_off):
    for path in ("/", "/academy", "/privacy"):
        html = client.get(path).text
        assert "flappAds" not in html and LOADER not in html, path


def test_the_real_policy_has_balanced_branches():
    doc = (STATIC / "privacy.html").read_text(encoding="utf-8")
    assert doc.count("<!--ADPIXEL:ON-->") == doc.count("<!--/ADPIXEL:ON-->") > 0
    assert doc.count("<!--ADPIXEL:OFF-->") == doc.count("<!--/ADPIXEL:OFF-->") > 0


def test_with_the_pixel_on_the_policy_names_meta_and_the_banner(client):
    html = client.get("/privacy").text
    assert "Meta Platforms" in html
    assert 'id="ads"' in html
    assert "Nothing loads until you choose" in html
    assert "never</b> loaded inside the app" in html
    assert "ADPIXEL" not in html


def test_with_the_pixel_off_the_policy_says_no_ad_trackers(client, pixel_off):
    html = client.get("/privacy").text
    assert "Meta" not in html
    assert "we do not use it for third-party advertising" in html
    assert "ADPIXEL" not in html


@pytest.mark.parametrize("page", ["landing.html", "index.html"])
def test_the_trust_line_is_scoped_to_the_app(page):
    doc = (STATIC / page).read_text(encoding="utf-8").lower()
    assert "no ad trackers in the app" in doc
    assert "no ad trackers &mdash;" not in doc and "no ad trackers, nothing sold" not in doc


def test_the_id_can_be_switched_off_from_the_environment(monkeypatch):
    from app.core import config

    monkeypatch.setenv("META_PIXEL_ID", "off")
    assert config._pixel_id_env() is None
    monkeypatch.setenv("META_PIXEL_ID", "123")
    assert config._pixel_id_env() == "123"
    monkeypatch.delenv("META_PIXEL_ID")
    assert config._pixel_id_env() == config.Settings.meta_pixel_id
