"""The triathlon helmet article and the app point at each other and say the same numbers."""
import math
from pathlib import Path

from app.services.academy.parser import get_all_articles, invalidate_cache
from app.services.video_analysis.biomechanics import aero_estimator as ae

ROOT = Path(__file__).resolve().parents[1]
SLUG = "triathlon-helmets-2026"
TEXT = (ROOT / "content" / "academy" / f"{SLUG}.md").read_text(encoding="utf-8")
SPA = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_it_is_published():
    invalidate_cache()
    assert SLUG in {m.slug for m in get_all_articles()}


def test_the_article_opens_the_helmet_goal_and_the_goal_exists():
    assert "(/app?goal=helmet)" in TEXT
    assert "helmet:{sport:'bike'" in SPA


def test_the_aero_panel_links_to_the_article():
    assert f'href="/academy/{SLUG}"' in SPA


def test_the_head_numbers_are_the_estimators():
    # "Dropping yours from +30% to +18% is worth about 3-4 W: 1 to 1.4 minutes over 90 km at 35 km/h,
    # 2 to 2.8 minutes over 180 km" -- the same model the report's aero panel runs.
    assert ae.HEAD_TARGET == 18.0 and "a median 18% of torso length" in TEXT
    lo, hi = ae.head_drag_reduction(30, ae.HEAD_TARGET)
    a90, b90 = ae.time_saved(lo / 100, 35, 90), ae.time_saved(hi / 100, 35, 90)
    a180, b180 = ae.time_saved(lo / 100, 35, 180), ae.time_saved(hi / 100, 35, 180)
    assert f"about {math.floor(a90['watts'])}-{math.floor(b90['watts'])} W" in TEXT
    assert f"{a90['minutes']:g} to {b90['minutes']:g} minutes over 90 km" in TEXT
    assert f"{a180['minutes']:g} to {b180['minutes']:g} minutes over 180 km" in TEXT


def test_house_copy_rules():
    assert "—" not in TEXT
    assert "/app#signup" not in TEXT
