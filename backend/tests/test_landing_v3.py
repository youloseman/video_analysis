"""The scroll-driven landing, previewed at /v3 until it replaces the live page.

String assertions on the file, like test_landing_order.py: what breaks here is
a number copied from the backend, a credit, or a link.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.video_analysis.biomechanics.aero_estimator import (
    _TT_REL_POINTS,
    PHYSICS,
    _rel_for_trunk,
    time_saved,
)

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
CREDIT = "Scott Flathouse"


@pytest.fixture(scope="module")
def html() -> str:
    return (STATIC / "landing-v3.html").read_text(encoding="utf-8")


def test_the_route_renders_it_like_the_live_page():
    with TestClient(app) as client:
        page = client.get("/v3").text
    assert "Faster and pain-free" in page
    for token in ("<!--PRICING-->", "<!--FREE-LIMIT-->", "<!--NOTES-PRICE-->", "<!--EXPERT-PRICE-->", "ADPIXEL"):
        assert token not in page, token
    assert 'id="pr-swim"' in page  # the hero's news link lands somewhere


def test_the_preview_is_not_indexed(html):
    assert '<meta name="robots" content="noindex">' in html


def test_the_whole_product_then_prices_then_the_extras(html):
    """Artur's order, 2026-11-06: the prices come straight after the product."""
    ids = re.findall(r'<section id="([a-z]+)"', html)
    assert ids == ["figures", "outcomes", "report", "proof", "bike", "run", "how", "pricing",
                   "kona", "coach", "swim", "whynow", "faq"]


def test_no_video_downloads_before_it_is_near(html):
    """autoplay or a src in the markup fetches the whole file on open."""
    assert not re.search(r"<video[^>]*\sautoplay", html)
    assert not re.search(r"<video[^>]*\ssrc=", html)
    assert len(re.findall(r"<video[^>]*\sdata-src=", html)) == 3


def test_every_clip_on_the_page_reaches_the_deploy(html):
    """`railway up` honours .gitignore, which ignores every .mp4 but the ones it
    names, so a clip added with `git add -f` passes CI and is a 404 in production."""
    ignore = (STATIC.parents[2] / ".gitignore").read_text(encoding="utf-8")
    clips = set(re.findall(r'data-src="(media/[^"?]+\.mp4)', html))
    assert clips
    for clip in clips:
        assert f"!backend/app/static/{clip}" in ignore, clip


def test_the_hero_keeps_its_line_and_every_way_in(html):
    hero = html[html.index('<header class="hero">'):html.index("</header>")]
    assert "Faster and pain-free" in hero
    assert "Film again" not in hero
    goals = re.findall(r'class="goal-btn" id="\w+" href="/app\?goal=([a-z]+)"', hero)
    assert goals == ["knee", "faster", "race", "run"]


def test_no_marketing_link_opens_the_sign_up_form(html):
    assert not re.search(r'<a [^>]*href="/app#signup"', html)


def test_speed_comes_before_pain_and_every_claim_has_a_source(html):
    sec = html[html.index('<section id="outcomes"'):]
    sec = sec[:sec.index("</section>")]
    tags = re.findall(r'class="out-tag (\w+)"', sec)
    assert tags == ["speed", "both", "pain"]
    for source in ("Heiderscheit", "Bini", "Fintelman", "Kona 2026"):
        assert source in sec, source
    low = sec.lower()
    for promise in ("no more pain", "pain-free guaranteed", "prevent injur", "injury-free"):
        assert promise not in low, promise


def test_every_kona_frame_on_the_wall_is_credited(html):
    wall = html[html.index('id="konaWall"'):]
    wall = wall[:wall.index("</div>")]
    frames = re.findall(r'<img [^>]*position-(\d\d)\.webp[^>]*>', wall)
    assert len(frames) == 12
    assert wall.count(CREDIT) == 12
    for n in frames:
        assert (STATIC / "media" / "academy" / "kona-2026" / f"position-{n}.webp").is_file(), n


def test_scott_is_credited_wherever_his_frames_are_full_size(html):
    sections = re.findall(r"<(?:section|header)[^>]*>.*?</(?:section|header)>", html, flags=re.S)
    using = [s for s in sections if "media/v3/kona-" in s]
    assert len(using) == 3    # hero, the speed beat, the bike chapter
    for sec in using:
        assert CREDIT in sec
    hero = next(s for s in using if "kona-hero" in s)
    assert "scottflathousephoto.com/events" in hero


def test_every_local_media_file_exists(html):
    for src in re.findall(r'(?:src|poster)="(/?media/[^"?]+)', html):
        assert (STATIC / src.lstrip("/")).is_file(), src


def test_the_aero_tool_mirrors_the_backend(html):
    tool = html[html.index("/* The aero tool."):]
    assert "PTS=[[8,.80*.302/.335],[16,.80*.318/.335],[24,.80],[33,.87]]" in tool
    assert [a for a, _ in _TT_REL_POINTS] == [8.0, 16.0, 24.0, 33.0]
    assert _TT_REL_POINTS[0][1] == pytest.approx(0.80 * 0.302 / 0.335)
    assert _TT_REL_POINTS[1][1] == pytest.approx(0.80 * 0.318 / 0.335)
    assert (_TT_REL_POINTS[2][1], _TT_REL_POINTS[3][1]) == (0.80, 0.87)
    assert PHYSICS == {"rho": 1.20, "crr": 0.004, "mass_kg": 84.0, "eta": 0.975, "cda": 0.26}
    assert "P={rho:1.20,crr:.004,mass:84,eta:.975,cda:.26}, SPEED=35, TARGET=20" in tool


def test_the_headline_minutes_are_the_model_s_own(html):
    drop = 1 - _rel_for_trunk(20) / _rel_for_trunk(30)
    minutes = time_saved(drop, 35, 90)["minutes"]
    assert f'<span id="aeroMin">{minutes}</span>' in html
    assert f'<div class="num">{minutes}<small' in html


def test_motion_is_opt_in_and_never_takes_the_wheel(html):
    """No .m means plain blocks, and native scrolling stays native."""
    assert "prefers-reduced-motion: reduce" in html
    assert ".m .scene{" in html
    assert "html:not(.m)" in html
    for hijack in ("addEventListener('wheel'", 'addEventListener("wheel"', "preventDefault();scrollTo"):
        assert hijack not in html


def test_no_edited_race_photo_is_left(html):
    """Their rights stayed open since October; generated images replaced them."""
    assert "media/race-" not in html
    assert "Race photo" not in html


def test_every_generated_runner_says_what_it_is(html):
    for sec in re.findall(r"<(?:section|article|figure)[^>]*>.*?</(?:section|article|figure)>", html, flags=re.S):
        if "gen-run-ov.webp" in sec and "<article" in sec[:10]:
            assert "Generated image" in sec
    how = html[html.index('<section id="how"'):]
    assert "Generated scene" in how[:how.index("</section>")]
    run = html[html.index('<section id="run"'):]
    run = run[:run.index("</section>")]
    assert run.count("<span>Illustration</span>") == 2
    assert "Lieberman" in run


def test_swim_is_one_chapter_with_its_prices(html):
    """Twice on one page was a defect; the price list's block moves to the picture."""
    sec = html[html.index('<section id="swim"'):]
    sec = sec[:sec.index("</section>")]
    assert 'id="swimSlot"' in sec
    assert "Generated image" in sec
    assert "slot.appendChild(block)" in html


def test_the_report_table_is_the_sample_report(html):
    from app.services import examples

    data = examples.get_sample("bike-road").data
    assert data["technique_score"] == 70
    table = html[html.index('class="rp-table"'):]
    table = table[:table.index("</figure>")]
    assert table.count("<li ") == 8
    diagnostics = {d["metric_name"]: d["current_value"] for d in data["fit_plan"]["diagnostics"]}
    assert f'{diagnostics["knee_at_bdc"]}' in table
    assert f'{diagnostics["trunk_angle"]}' in table


def test_every_full_size_scott_frame_links_to_his_events_page(html):
    """His terms: credit and the link, on each. The phone must not clip it."""
    for sec in re.findall(r"<(?:header|article|section)[^>]*>.*?</(?:header|article|section)>", html, flags=re.S):
        if "media/v3/kona-" in sec and "<article" in sec[:10]:
            assert "scottflathousephoto.com/events" in sec
    assert html.count('href="https://www.scottflathousephoto.com/events/"') >= 4


def test_pricing_is_dark_here_and_only_here(html):
    """The block is shared with the app. The dark version is its tokens restated
    in this page's own CSS, so the shared stylesheet stays light."""
    assert ".pr,.go,.pr-swim{--c-bg:#101318" in html
    shared = (STATIC.parents[0] / "services" / "pricing.py").read_text(encoding="utf-8")
    assert "#101318" not in shared


def test_one_ground_to_the_foot_of_the_page(html):
    for rule in (".pricing{background:var(--g)", ".faq{background:var(--g)", "footer{background:var(--g)}"):
        assert rule in html, rule
