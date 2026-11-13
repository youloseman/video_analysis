"""An A has nothing red (technique_scorer.cycling_red_components, 2026-11-13)."""
from __future__ import annotations

from pathlib import Path

from app.services.report_pdf import _score_notes
from app.services.video_analysis.biomechanics.technique_scorer import score_cycling

# Gona's tri clip after the held arms: everything in its band but the knee at
# the top, 78.5 against 60-72 -- red. It averaged 96 and read 91/A.
GONA = {"near_side": "left", "camera_side": "left", "frames_analyzed": 279,
        "left_knee_at_bdc": 143.99, "left_knee_at_tdc": 78.55, "trunk_angle_avg": 13.91,
        "elbow_angle_avg": 79.91, "shoulder_angle_avg": 79.03, "forearm_tilt_avg": 16.9}


def test_a_red_row_holds_the_grade_at_b():
    out = score_cycling(GONA, {}, "triathlon")
    assert out["overall_score"] == 89 and out["letter_grade"] == "B"
    assert out["coverage"]["capped_by"] == ["knee_tdc"]


def test_without_the_red_row_it_is_an_a():
    out = score_cycling(dict(GONA, left_knee_at_tdc=70.0), {}, "triathlon")
    assert out["letter_grade"] == "A" and "capped_by" not in out["coverage"]


def test_an_aero_choice_is_amber_and_does_not_cap():
    # Torso 14 and shoulder 70 on aero bars: flatter / closed past the band,
    # the rider's choice -- amber, never red, so no cap.
    out = score_cycling(dict(GONA, left_knee_at_tdc=70.0, shoulder_angle_avg=70.0), {}, "triathlon")
    assert "capped_by" not in out["coverage"] and out["letter_grade"] == "A"


def test_a_borderline_miss_is_not_red():
    # 74 against 60-72 is inside the 3 deg warning margin: amber.
    out = score_cycling(dict(GONA, left_knee_at_tdc=74.0), {}, "triathlon")
    assert "capped_by" not in out["coverage"]


def test_the_report_says_why():
    out = score_cycling(GONA, {}, "triathlon")
    res = {"sport_type": "bike", "score_breakdown": out["component_scores"], "score_coverage": out["coverage"]}
    assert any("Knee at top is outside its range, so the grade stops at B" in n
               for n in _score_notes(res, out["overall_score"]))
    spa = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Grade held at B" in spa and "capped_by" in spa
