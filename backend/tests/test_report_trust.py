"""A report that does not argue with itself (Gona's PDF, 2026-11-10).

The reader saw one forearm written as 15 and 15.5 deg, a weight column adding
to 101 %, two list points run into one line, "5.4 cm" with nothing to scale
it, and "Pelvis rocking (ratio) 3.55" with no unit.
"""
from __future__ import annotations

from app.services.report_pdf import _tile, hob_gap_text, whole_percents
from app.services.review_pdf import HOW_MEASURED, metric_rows
from app.services.video_analysis.llm_recommendations import split_glued_bullets


def test_the_weight_column_adds_to_exactly_100():
    gona = [0.22, 0.18, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05]   # helmet left out
    assert sum(round(w / sum(gona) * 100) for w in gona) == 101, "the old rounding"
    shares = whole_percents(gona)
    assert sum(shares) == 100 and shares[:5] == [24, 20, 17, 11, 11]
    assert whole_percents([1, 1, 1]) == [34, 33, 33]
    assert whole_percents([]) == [] and whole_percents([0, 0]) == [0, 0]


def test_points_run_into_one_line_become_a_list():
    text = ("**Overall** — Clean — really.\n"
            "**What's working** — Knee extension sits at 141°. — Trunk angle sits in the band.\n"
            "**Fix first** — Raise the stack 10 mm — this opens the hip.")
    out = split_glued_bullets(text).splitlines()
    assert out[1:4] == ["**What's working**", "- Knee extension sits at 141°.", "- Trunk angle sits in the band."]
    assert out[0] == "**Overall** — Clean — really.", "not a list section"
    assert out[4] == "**Fix first** — Raise the stack 10 mm — this opens the hip.", "a dash inside a sentence stays"
    listed = "**What's working**\n- A\n- B"
    assert split_glued_bullets(listed) == listed


def test_the_centimetres_say_what_they_are_scaled_from():
    typical = {"gap_pro_cm": -5.4, "gap_pro_deg": -3.7, "scale_source": "typical"}
    t = hob_gap_text(typical)
    assert t.startswith("About 5 cm behind") and "typical 83 cm" in t and "3.7°" in t
    crank = {"gap_pro_cm": -5.1, "gap_pro_deg": -3.7, "scale_source": "crank_given", "crank_to_hip_cm": 79.0}
    assert hob_gap_text(crank).startswith("5.1 cm behind") and "your 79 cm" in hob_gap_text(crank)
    assert hob_gap_text({"gap_pro_cm": None}) == ""


def test_every_bike_row_says_what_it_measures():
    result = {"sport_specific_metrics": {"knee_at_bdc": 141.1, "pelvic_ratio": 3.55, "hip_at_tdc": 54.0},
              "reference_bands": {"knee_at_bdc": {"lo": 138, "hi": 145}, "pelvic_ratio": {"lo": 2, "hi": 4}}}
    rows = {r["field"]: r for r in metric_rows(result)}
    assert "rocking" not in rows["pelvic_ratio"]["label"].lower()
    assert "divided by the torso angle" in rows["pelvic_ratio"]["how"]
    assert "Inside angle between thigh and shin" in rows["knee_at_bdc"]["how"]
    assert "top of the stroke" in rows["hip_at_tdc"]["how"]
    assert "tile-how" in _tile(rows["knee_at_bdc"])
    assert set(HOW_MEASURED) >= {"knee_at_bdc", "knee_at_tdc", "hip_at_tdc", "trunk_angle_avg",
                                 "shoulder_angle_avg", "elbow_angle_avg", "forearm_tilt_avg",
                                 "pelvic_ratio", "helmet_above_back"}
