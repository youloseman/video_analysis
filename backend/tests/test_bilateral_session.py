"""Two single-side analyses -> one session result.

The product rule this file guards: a two-sided session shows ONE score. Two
scores for one body is what sent the rider looking for an asymmetry that was
never there -- his right-side clip scored 100 and his left 89, from the same
fit, ten minutes apart.
"""

import pytest

from app.services.video_analysis.bilateral_session import build_pair_result

# Geometry as measured from the real clips (crank-radius units).
GEOM_LEFT = {
    "camera_side": "left", "thigh": 2.322, "shin": 2.308, "torso": 2.54,
    "chord_bdc": 4.520, "chord_sd": 0.026, "revolutions": 11,
    "measured_frames": 387, "crank_radius_px": 145.6,
}
GEOM_RIGHT = {
    "camera_side": "right", "thigh": 2.442, "shin": 2.276, "torso": 2.65,
    "chord_bdc": 4.510, "chord_sd": 0.030, "revolutions": 5,
    "measured_frames": 209, "crank_radius_px": 152.3,
}
# The 24 Aug pair, whose crank-radius ruler is ~11% out.
GEOM_BAD_RIGHT = {
    "camera_side": "right", "thigh": 2.593, "shin": 2.480, "torso": 2.82,
    "chord_bdc": 4.925, "chord_sd": 0.032, "revolutions": 13,
    "measured_frames": 239, "crank_radius_px": 184.0,
}


def _result(side, knee, score, *, geom, trunk=68.0, frames=560, warn=None):
    return {
        "status": "completed", "sport_type": "bike",
        "cycling_position": "road_hoods", "camera_side": side,
        "frames_analyzed": frames,
        "technique_score": score, "letter_grade": "A",
        "score_breakdown": {"knee_bdc": 100.0},
        "keyframe_base64": f"data:image/jpeg;base64,{side}",
        "overlay_video_path": f"/jobs/x/{side}",
        "bilateral_geometry": geom,
        "angle_statistics": {"knee_angle": {"mean": knee}},
        "detected_issues": [],
        "quality_warnings": list(warn or []),
        "ai_recommendations": {"report": f"about the {side} leg only"},
        "sport_specific_metrics": {
            "camera_side": side, "near_side": side,
            "knee_at_bdc": knee, f"{side}_knee_at_bdc": knee,
            "knee_at_tdc": 64.0, "trunk_angle_avg": trunk,
            "hip_angle_avg": 25.0, "elbow_angle_avg": 150.0,
            "shoulder_angle_avg": 88.0, "pelvic_ratio": 0.37,
            "saddle_height_assessment": "optimal", "frames_analyzed": frames,
            "bilateral_geometry": geom,
        },
    }


LEFT = _result("left", 154.7, 89, geom=GEOM_LEFT, trunk=67.3)
RIGHT = _result("right", 145.8, 100, geom=GEOM_RIGHT, trunk=68.3)


class TestAGoodPair:
    def test_it_combines(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        assert out["bilateral"]["combined"] is True

    def test_there_is_exactly_one_score(self):
        """The whole point. Two scores for one body is the confusion."""
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        assert isinstance(out["technique_score"], (int, float))
        for card in out["bilateral"]["sides"]:
            assert "technique_score" not in card
            assert "score" not in card

    def test_the_score_is_not_just_one_clips_score_carried_over(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        # 100 was the flattering side, 89 the other; the merged ride is scored
        # from merged metrics, so it need not equal either -- but it must not
        # silently BE the better one.
        assert out["technique_score"] != 100 or out["sport_specific_metrics"][
            "knee_at_bdc"] != 145.8

    def test_the_knee_shown_is_the_merged_one(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        merged = out["sport_specific_metrics"]["knee_at_bdc"]
        # The block is a display payload (rounded); the summary keeps full
        # precision because the scorer reads it.
        assert merged == pytest.approx(out["bilateral"]["knee_at_bdc"], abs=0.1)
        assert 145.8 < merged < 154.7

    def test_it_reports_the_session_as_two_sided(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        assert out["camera_side"] == "both"
        assert out["frames_analyzed"] == 1120
        assert {c["camera_side"] for c in out["bilateral"]["sides"]} == {"left", "right"}

    def test_the_asymmetry_question_is_answered_not_dodged(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        assert out["bilateral"]["asymmetry_significant"] is False
        assert abs(out["bilateral"]["asymmetry_deg"]) < 2.0

    def test_the_two_clips_agreement_is_reported(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        agree = out["bilateral"]["agreement"]
        assert agree["agree"] is True
        assert agree["gaps"]["trunk_angle_avg"] == pytest.approx(1.0, abs=0.01)

    def test_one_clips_coaching_prose_is_not_inherited(self):
        """A report written about one leg, printed under a merged score, is
        the same contradiction in words instead of digits."""
        out = build_pair_result(LEFT, RIGHT, "road_hoods", recommendations=False)
        assert "leg only" not in str(out.get("ai_recommendations") or "")

    def test_per_clip_artifacts_do_not_masquerade_as_the_sessions(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        # The overlay belongs to one clip; the session must not present it as
        # a recording of both.
        assert "overlay_video_path" not in out

    def test_warnings_from_both_clips_survive(self):
        left = _result("left", 154.7, 89, geom=GEOM_LEFT, warn=["left is dim"])
        right = _result("right", 145.8, 100, geom=GEOM_RIGHT, warn=["drive side"])
        out = build_pair_result(left, right, "road_hoods")
        assert set(out["quality_warnings"]) == {"left is dim", "drive side"}

    def test_argument_order_does_not_matter(self):
        a = build_pair_result(LEFT, RIGHT, "road_hoods")
        b = build_pair_result(RIGHT, LEFT, "road_hoods")
        assert a["technique_score"] == b["technique_score"]
        assert a["sport_specific_metrics"]["knee_at_bdc"] == pytest.approx(
            b["sport_specific_metrics"]["knee_at_bdc"])


class TestAPairItCannotMerge:
    def test_a_broken_ruler_refuses_and_says_so(self):
        bad = _result("right", 152.3, 95, geom=GEOM_BAD_RIGHT)
        out = build_pair_result(LEFT, bad, "road_hoods")
        assert out["bilateral"]["combined"] is False
        assert out["bilateral"]["reason"] == "scale_mismatch"

    def test_a_refusal_still_returns_a_readable_result(self):
        bad = _result("right", 152.3, 95, geom=GEOM_BAD_RIGHT)
        out = build_pair_result(LEFT, bad, "road_hoods")
        # The rider filmed two clips and is owed an answer about them.
        assert out["status"] == "completed"
        assert out["technique_score"] is not None
        assert len(out["bilateral"]["sides"]) == 2

    def test_a_refusal_publishes_no_merged_number(self):
        bad = _result("right", 152.3, 95, geom=GEOM_BAD_RIGHT)
        out = build_pair_result(LEFT, bad, "road_hoods")
        assert out["bilateral"].get("knee_at_bdc") is None
        assert out["bilateral"].get("asymmetry_deg") is None

    def test_missing_geometry_refuses(self):
        naked = _result("right", 145.8, 100, geom=None)
        out = build_pair_result(LEFT, naked, "road_hoods")
        assert out["bilateral"]["combined"] is False
        assert out["bilateral"]["reason"] == "geometry_unavailable"

    def test_two_clips_of_one_side_are_not_a_session(self):
        out = build_pair_result(LEFT, _result("left", 151.0, 90, geom=GEOM_LEFT),
                                "road_hoods")
        assert out["bilateral"]["combined"] is False
        assert out["bilateral"]["reason"] == "sides_not_identified"

    def test_an_unidentified_side_refuses(self):
        blind = _result("left", 150.0, 90, geom=GEOM_LEFT)
        blind["camera_side"] = None
        out = build_pair_result(blind, RIGHT, "road_hoods")
        assert out["bilateral"]["combined"] is False
