"""Academy covers: every article has its own picture, and every picture is ours to show.

Artur, 2026-10-26: one drawn template on every card read as dry and the
articles as interchangeable. Each guide now has its own cover -- our footage,
or Marey's chronophotographs from the 1880s, which are public domain (one,
Wellcome's, is CC BY 4.0 and must carry its credit). The same picture is the
page's og:image and Article image, which the head never had before.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services.academy import get_all_articles, get_article, invalidate_cache
from app.services.academy.parser import _own_media
from app.services.academy.renderer import HERO, HERO_CREDIT, render_article, render_hub, render_sitemap

MEDIA = Path(__file__).resolve().parents[1] / "app" / "static" / "media"
BASE = "https://getflapp.com"


@pytest.fixture(scope="module")
def metas():
    invalidate_cache()
    return get_all_articles()


def test_every_article_has_a_cover_it_can_show(metas):
    for m in metas:
        assert m.cover, f"{m.slug} has no cover"
        assert (MEDIA / m.cover.removeprefix("/media/")).is_file(), m.cover
        assert m.cover_alt and m.cover_credit, f"{m.slug}: alt and credit are not optional"


def test_no_two_guides_share_a_cover(metas):
    covers = [m.cover for m in metas]
    assert len(covers) == len(set(covers))


def test_a_cover_is_our_own_media_or_nothing():
    assert _own_media("/media/academy/covers/foot-strike.webp")
    assert _own_media("https://example.com/photo.jpg") is None
    assert _own_media("/media/../secrets/x.png") is None
    assert _own_media('/media/a.webp" onerror="x') is None


def test_the_hub_shows_every_cover_and_credits_the_archive(metas):
    page = render_hub(metas, BASE)
    for m in metas:
        assert f'src="{m.cover}"' in page, m.slug
    assert HERO_CREDIT in page and "CC BY 4.0" in HERO_CREDIT, "Wellcome's licence needs its credit"
    assert f'<meta property="og:image" content="{BASE}{HERO}">' in page
    for key in ("research", "running", "cycling"):
        assert f'id="{key}"' in page and f'href="#{key}"' in page


def test_an_article_carries_its_picture_in_the_head_and_the_schema(metas):
    for m in metas:
        page = render_article(get_article(m.slug), BASE)
        assert f'<meta property="og:image" content="{BASE}{m.cover}">' in page
        assert 'name="robots" content="max-image-preview:large"' in page
        ld = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', page).group(1))
        art = ld[0]
        assert art["image"] == [BASE + m.cover]
        assert art["author"]["@type"] == "Person"
        assert page.count(f'src="{m.cover}"') == 1, f"{m.slug} shows its cover twice"


def test_a_study_shows_its_cover_figure_once_with_its_caption():
    invalidate_cache()
    page = render_article(get_article("kona-2026-hip-over-crank"), BASE)
    cover = re.search(r'<figure class="cover[^"]*">.*?</figure>', page, flags=re.S).group(0)
    assert "88°" in cover, "the caption moved up with the figure"


def test_the_sitemap_lists_each_cover(metas):
    xml = render_sitemap(metas, BASE)
    assert 'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"' in xml
    for m in metas:
        assert f"<image:loc>{BASE}{m.cover}</image:loc>" in xml
