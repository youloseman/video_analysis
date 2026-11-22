"""Road bands from measured riders (Artur, 2026-11-22;
docs/research/road-fit-bands-2026-11-22.md): trunk, shoulder and elbow moved
to what studies measured, and the drops elbow and shoulder -- never measured
anywhere -- are shown, not graded, on every surface."""
from __future__ import annotations

from app.services.video_analysis.biomechanics import technique_scorer as TS
from app.services.video_analysis.biomechanics.cycling_positions import (
    CYCLING_POSITIONS,
    UNGRADED_NOTE,
    detect_position_archetype,
    get_cycling_reference,
    get_position_meta,
    is_graded,
)
from app.services.video_analysis.biomechanics.sport_configs import reference_bands

CLIP = {"near_side": "left", "left_knee_at_bdc": 141.0, "knee_at_bdc": 141.0,
        "left_knee_at_tdc": 70.0, "trunk_angle_avg": 30.0,
        "elbow_angle_avg": 120.0, "shoulder_angle_avg": 110.0}


def test_the_bands_are_the_measured_ones():
    h, d = CYCLING_POSITIONS["road_hoods"], CYCLING_POSITIONS["road_drops"]
    assert h["trunk_angle"] == (32, 50) and d["trunk_angle"] == (22, 38)
    assert h["shoulder_angle"] == (70, 90) and h["elbow_angle"] == (145, 170)
    assert d["shoulder_angle"] is None and d["elbow_angle"] is None
    # the knee waits for the marker study; it did not move
    assert h["knee_at_bdc"] == d["knee_at_bdc"] == (135, 145)
    assert "Skovereng 2020" in get_position_meta("road_hoods", "trunk_angle")["source"]
    assert "Burt" in get_position_meta("road_hoods", "hip_angle_max")["source"]


def test_the_drops_elbow_and_shoulder_leave_the_clip_score():
    res = TS.score_cycling(CLIP, {}, cycling_position="road_drops")
    assert "elbow_angle" not in res["component_scores"]
    assert "shoulder_angle" not in res["component_scores"]
    # left the rubric, not "missing"
    assert "elbow_angle" not in res["coverage"]["missing"]
    assert "shoulder_angle" not in res["coverage"]["missing"]
    hoods = TS.score_cycling(CLIP, {}, cycling_position="road_hoods")
    assert hoods["component_scores"]["elbow_angle"] < 100
    assert hoods["component_scores"]["shoulder_angle"] < 100


def test_no_verdict_anywhere_for_them():
    sev = TS.build_severity_map("road_drops", CLIP)
    assert "elbow_angle" not in sev and "shoulder_angle" not in sev
    assert "trunk_angle" in sev
    bands = reference_bands("bike", "road_drops")
    assert "elbow_angle_avg" not in bands and "shoulder_angle_avg" not in bands
    assert bands["trunk_angle_avg"]["lo"] == 22
    assert not TS.cycling_red_components(CLIP, "road_drops", {"elbow_angle": 0.0, "shoulder_angle": 0.0})
    assert not is_graded("road_drops", "elbow_angle") and is_graded("road_hoods", "elbow_angle")
    assert detect_position_archetype(80.0, 150.0, 30.0, 60.0, "road_drops") is None


def test_a_drops_photo_shows_them_and_does_not_score_them():
    from app.services.video_analysis import photo_analyzer as PA

    ranges = PA._get_cycling_optimal_ranges("road_drops", "tdc")
    assert "shoulder" not in ranges and "elbow" not in ranges and ranges["trunk"] == (22, 38)
    r = PA._score_photo_angles({"knee": 70.0, "hip": 62.0, "trunk": 30.0, "shoulder": 110.0, "elbow": 120.0},
                               ranges, "bike", PA._BIKE_PHASE_WEIGHTS["tdc"], "road_drops", "tdc")
    assert "shoulder" not in r["per_angle"] and "elbow" not in r["per_angle"]


def test_the_overlay_draws_them_neutral_and_never_calls_them_out():
    from app.services.video_analysis.video_visualizer import _status_for_cfg

    assert _status_for_cfg({"optimal": None}, 120.0) == "muted"
    assert _status_for_cfg({"optimal": (145, 170)}, 120.0) == "bad"


def test_the_pdf_says_why():
    from app.services.report_pdf import _score_notes
    from app.services.review_pdf import metric_rows

    res = {"sport_type": "bike", "cycling_position": "road_drops",
           "score_breakdown": {"knee_bdc": 100.0, "knee_tdc": 100.0, "trunk_angle": 100.0},
           "sport_specific_metrics": {"elbow_angle_avg": 120.0, "shoulder_angle_avg": 80.0},
           "reference_bands": reference_bands("bike", "road_drops")}
    notes = _score_notes(res, 100)
    assert any("No study has measured the elbow and shoulder angles in the drops" in n for n in notes)
    assert not any(n.startswith("Not measured") and "Elbow" in n for n in notes)
    rows = {r["field"]: r for r in metric_rows(res)}
    assert rows["elbow_angle_avg"]["band"] is None and rows["elbow_angle_avg"]["noband"] == UNGRADED_NOTE


def test_get_cycling_reference_serves_none_not_a_made_up_band():
    ref = get_cycling_reference("road_drops")
    assert ref["elbow_angle"] is None and ref["shoulder_angle"] is None
