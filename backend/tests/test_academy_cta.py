"""Academy call to action (2026-11-24): mid-article block, end block, phone bar.

Session recordings showed readers going through a study to the end while 2%
started an analysis, with the only link at the bottom and the generic end
block pointing at the landing. An article can now say what to offer and where.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.services.academy.parser import Article, ArticleMeta, _app_href, markdown_to_html
from app.services.academy.renderer import CTA_DEFAULT_BUTTON, render_article

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def _article(body_md: str, **meta) -> Article:
    m = ArticleMeta(slug="t-cta", title="T", description="d", category="bike-position",
                    category_label="Bike Position", sport="bike", read_time=3, published="2026-11-24",
                    featured=False, **meta)
    return Article(meta=m, body_html=markdown_to_html(body_md))


def test_a_cta_only_ever_goes_into_the_app():
    assert _app_href("/app?goal=crank&what=bike-photo") == "/app?goal=crank&what=bike-photo"
    assert _app_href("/app") == "/app"
    for bad in ("https://example.com/app", "/", "/app#signup", "//evil.com", "/app?x=<b>", None, ""):
        assert _app_href(bad) == "/app"


def test_the_widget_line_puts_the_articles_own_cta_mid_article():
    a = _article("Intro paragraph.\n\nThe key table.\n\n[[widget:cta]]\n\nLater section.",
                 cta="Got a side-on photo? Start there", cta_button="Analyze my photo",
                 cta_href="/app?goal=crank&what=bike-photo", cta_bar="Have a bike photo?")
    html = render_article(a, "https://getflapp.com")
    inline = re.search(r'<div class="try-cta" data-cta="inline">(.*?)</div>', html, re.S)
    assert inline and "Got a side-on photo? Start there" in inline.group(1)
    assert 'href="/app?goal=crank&amp;what=bike-photo"' in inline.group(1)
    assert html.index('data-cta="inline"') < html.index("Later section")
    assert 'data-widget="cta"' not in html
    # the same offer closes the article and rides in the phone bar
    end = re.search(r'data-cta="end">(.*?)</div>', html, re.S).group(1)
    assert "Analyze my photo" in end and "what=bike-photo" in end
    bar = re.search(r'<div class="cta-bar" id="ctaBar"[^>]*hidden>(.*?)</div>', html, re.S)
    assert bar and "Have a bike photo?" in bar.group(1) and 'aria-label="Close"' in bar.group(1)


def test_an_article_without_its_own_cta_ends_in_the_app_not_the_landing():
    html = render_article(_article("Just text."), "https://getflapp.com")
    end = re.search(r'data-cta="end">.*?href="([^"]+)"', html, re.S)
    assert end and end.group(1) == "/app"
    assert CTA_DEFAULT_BUTTON in html
    assert 'data-cta="inline"' not in html          # no widget line, no mid-article block
    assert 'href="/">Analyze' not in html


def test_cta_copy_follows_the_house_rules():
    html = render_article(_article("x"), "https://getflapp.com")
    for txt in re.findall(r'data-cta="end"><p>(.*?)</p>', html) + re.findall(r'id="ctaBar"[^>]*><span>(.*?)</span>', html):
        assert not txt.endswith("."), txt            # short statements carry no full stop
        assert "—" not in txt                    # no em dash


def test_the_app_opens_the_tile_a_link_asks_for_without_logging_a_step():
    spa = SPA.read_text(encoding="utf-8")
    assert "params.get('what')" in spa
    block = spa[spa.index("params.get('what')"):spa.index("params.get('what')") + 1500]
    assert "autoPick(()=>t.click())" in block           # not counted as analyze_step_done
    assert "state.userSet.what=true" in block           # and nothing auto-picks over it
    assert "w!=='run-photo'" in block                   # the run photo stays behind its flag
