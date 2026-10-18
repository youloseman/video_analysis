"""The Kona pro field, beside the bands: served, shown, never graded.

Artur, 2026-10-12: a rider on aero bars should see their own number next to
what the pros rode -- in the joint table (a "Kona pros" column), on the metric
tiles (the pros' median as a tick on the range bar, with the middle half
written under it), on the photo table, in the coach's brief and in the AI
export. The field lives in one place (kona_field.py) and is pinned to the
article's table, so the app and the study cannot quote two different Konas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.academy.parser import CONTENT_DIR
from app.services.video_analysis.biomechanics import kona_field as KF

BACKEND = Path(__file__).resolve().parents[1]
STATIC = BACKEND / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
ARTICLE = (CONTENT_DIR / "kona-2026-pro-bike-positions.md").read_text(encoding="utf-8")


# --- one field, pinned to the article -----------------------------------

def _article_row(label: str) -> tuple[int, int, int]:
    """(median, q1, q3) as the article prints them."""
    m = re.search(rf"^\| {re.escape(label)} \| (\d+)[°%] \| (\d+)–(\d+)[°%] \|", ARTICLE, re.M)
    assert m, label
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


@pytest.mark.parametrize("field,label", [
    ("trunk_angle_avg", "Torso, from horizontal"),
    ("shoulder_angle_avg", "Shoulder, torso to upper arm"),
    ("elbow_angle_avg", "Elbow"),
    ("forearm_tilt_avg", "Forearm tilt, above horizontal"),
    ("helmet_above_back", "Helmet above the back, % of torso length"),
])
def test_the_app_quotes_the_same_kona_as_the_article(field, label):
    v = KF.KONA_PRO_FIELD[field]
    assert (round(v["median"]), round(v["q1"]), round(v["q3"])) == _article_row(label)


def test_the_count_is_the_article_s():
    assert f"What {KF.KONA_PRO_N} Pro Bike Positions" in ARTICLE
    assert KF.KONA_PRO_MEN + KF.KONA_PRO_WOMEN == KF.KONA_PRO_N


def test_the_helmet_reference_is_the_same_field():
    from app.services.video_analysis.biomechanics.head_silhouette import (
        KONA_PRO_MEDIAN,
        KONA_PRO_RANGE,
    )
    h = KF.KONA_PRO_FIELD["helmet_above_back"]
    assert KONA_PRO_MEDIAN == h["median"]
    assert KONA_PRO_RANGE == (round(h["lo"]), round(h["hi"]))


# --- served on aero positions only ----------------------------------------

def test_served_on_the_aero_bars_and_nowhere_else():
    for pos in ("tt_aero", "triathlon"):
        f = KF.pro_field(pos)
        assert f and f["n"] == KF.KONA_PRO_N and set(f["metrics"]) == set(KF.KONA_PRO_FIELD)
        assert f["article"].startswith("/academy/")
    for pos in ("road_hoods", "road_drops", "casual", None):
        assert KF.pro_field(pos) is None


def test_a_video_and_a_photo_result_carry_it():
    runner = (BACKEND / "app" / "services" / "video_analysis" / "runner.py").read_text(encoding="utf-8")
    photo = (BACKEND / "app" / "services" / "video_analysis" / "photo_analyzer.py").read_text(encoding="utf-8")
    assert '"pro_field": pro_field(cycling_position) if is_bike else None' in runner
    assert 'response["pro_field"] = pro_field(cycling_position)' in photo


# --- the coach and the export get it as context, not as a rubric ----------

def test_the_coach_is_told_the_field_and_that_it_is_not_a_band():
    from app.services.video_analysis.llm_recommendations import _build_photo_prompt, _build_prompt
    summary = {"trunk_angle_avg": 24.0, "elbow_angle_avg": 90.0, "shoulder_angle_avg": 88.0}
    tri = _build_prompt("bike", 80, "B", "triathlon", [], {}, summary)
    assert "Kona 2026 pro field (n=32" in tri and "not a band" in tri
    assert "trunk angle 20°" in tri and "forearm tilt 26°" in tri
    road = _build_prompt("bike", 80, "B", "road_hoods", [], {}, summary)
    assert "Kona 2026 pro field" not in road
    res = {"score": {"overall_score": 80, "grade": "B"}, "cycling_position": "tt_aero",
           "angles_with_context": {"trunk": {"value": 22.0, "label": "Trunk", "optimal_min": 15,
                                             "optimal_max": 30, "status": "optimal"}}}
    assert "Kona 2026 pro field" in _build_photo_prompt("bike", res)


def test_the_export_carries_it_beside_the_reference_ranges():
    from app.services.export import ai_export
    aero = ai_export.build_json({"sport_type": "bike", "cycling_position": "triathlon",
                                    "sport_specific_metrics": {}, "angle_statistics": {}}, job_id="j")
    assert aero["kona_pro_field"]["n"] == KF.KONA_PRO_N
    road = ai_export.build_json({"sport_type": "bike", "cycling_position": "road_hoods",
                                    "sport_specific_metrics": {}, "angle_statistics": {}}, job_id="j")
    assert road["kona_pro_field"] is None


# --- the page shows it ------------------------------------------------------

def test_the_joint_table_and_the_photo_table_have_the_column():
    assert SPA.count('class="num col-pro" title="Kona 2026 pros: median, and the middle half of the field under it">Kona 2026</th>') == 2
    assert "state.pro = r.pro_field || null" in SPA
    assert "angleTableRows(merged.stats, sport, r.cycling_position||null, srcMap, pro)" in SPA
    assert "classList.toggle('no-pro', !pro)" in SPA
    assert "table.data.no-pro .col-pro{display:none}" in CSS


def test_the_measurements_table_draws_the_pros_median_on_the_range_bar():
    bar = SPA[SPA.index("function measBar("):SPA.index("function measRow(")]
    assert "mbar-pro" in bar and ".mbar-pro{" in CSS
    assert "Kona pros ${num(pro.median,0)}${u} <small>· middle half" in bar
    # The figures follow the angle convention like every printed angle.
    assert "rg.pro=Object.assign({},range.pro,{median:d(range.pro.median)" in SPA


def test_the_forearm_gets_a_tile_on_aero_bars():
    i = SPA.index("t.push(tile('Forearm tilt'")
    assert "position==='tt_aero'||position==='triathlon'" in SPA[i - 200:i]
