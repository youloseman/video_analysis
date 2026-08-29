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
    "camera_side": "left", "thigh": 2.3226, "shin": 2.3100, "torso": 2.5432,
    "chord_bdc": 4.5202, "chord_sd": 0.0263, "revolutions": 11,
    "measured_frames": 387, "crank_radius_px": 0.051721,
}
GEOM_RIGHT = {
    "camera_side": "right", "thigh": 2.4423, "shin": 2.2756, "torso": 2.6495,
    "chord_bdc": 4.5102, "chord_sd": 0.0301, "revolutions": 5,
    "measured_frames": 209, "crank_radius_px": 0.054067,
}
# The 24 Aug right clip, whose ankle orbit the chainring corrupted: its crank
# ruler is ~11% out, and the merge now falls back to the torso rather than
# refusing the pair.
GEOM_BAD_RIGHT = {
    "camera_side": "right", "thigh": 2.5926, "shin": 2.4800, "torso": 2.8189,
    "chord_bdc": 4.9253, "chord_sd": 0.0325, "revolutions": 13,
    "measured_frames": 239, "crank_radius_px": 0.065334,
}
# A leg no ruler can reconcile with the left clip's: not one rider.
GEOM_NOT_A_PAIR = {
    "camera_side": "right", "thigh": 2.79, "shin": 2.77, "torso": 2.5432,
    "chord_bdc": 4.5202, "chord_sd": 0.026, "revolutions": 9,
    "measured_frames": 300, "crank_radius_px": 0.051721,
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

        "ai_recommendations": {"report": f"about the {side} leg only"},
        "sport_specific_metrics": {
            "camera_side": side, "near_side": side,
            "knee_at_bdc": knee, f"{side}_knee_at_bdc": knee,
            "knee_at_tdc": 64.0, "trunk_angle_avg": trunk,
            "hip_angle_avg": 25.0, "elbow_angle_avg": 150.0,
            "shoulder_angle_avg": 88.0, "pelvic_ratio": 0.37,
            "saddle_height_assessment": "optimal", "frames_analyzed": frames,
            "bilateral_geometry": geom,
            # Where runner.py actually writes them.
            "quality_warnings": list(warn or []),
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

    def test_no_asymmetry_number_is_published(self):
        """Left-right difference and scale error are degenerate here, so the
        session offers each clip's reading and no verdict about the legs."""
        b = build_pair_result(LEFT, RIGHT, "road_hoods")["bilateral"]
        for banned in ("asymmetry_deg", "asymmetry_significant",
                       "asymmetry_floor_deg", "per_side"):
            assert banned not in b

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
        # Read from where runner.py writes them and the SPA reads them --
        # a top-level key here looks right and renders nowhere.
        warns = out["sport_specific_metrics"]["quality_warnings"]
        assert set(warns) == {"left is dim", "drive side"}

    def test_argument_order_does_not_matter(self):
        a = build_pair_result(LEFT, RIGHT, "road_hoods")
        b = build_pair_result(RIGHT, LEFT, "road_hoods")
        assert a["technique_score"] == b["technique_score"]
        assert a["sport_specific_metrics"]["knee_at_bdc"] == pytest.approx(
            b["sport_specific_metrics"]["knee_at_bdc"])


class TestWhatEachSideCardShows:
    """Reported from production 2026-08-27, then re-decided 2026-08-29.

    The rider saw "LEFT SIDE 153.2, RIGHT SIDE 143" and read the old
    contradiction back into a merged result. The first fix replaced those with
    per-side values reconciled against the shared body, which landed a degree
    apart and looked far better. It was an artifact -- that split is the scale
    choice restated (see TestWhyThereIsNoAsymmetryNumber), so it agreed by
    construction and would have agreed for any pair whatsoever.

    So the cards are back to each clip's own reading, which is at least a true
    statement about a clip, and the panel's job is to say that the difference
    between them is the instrument rather than the athlete.
    """

    def test_a_card_reports_what_that_clip_measured_alone(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        by_side = {c["camera_side"]: c for c in out["bilateral"]["sides"]}
        assert by_side["left"]["knee_at_bdc"] == pytest.approx(154.7)
        assert by_side["right"]["knee_at_bdc"] == pytest.approx(145.8)

    def test_no_reconciled_per_leg_value_is_offered(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        assert "per_side" not in out["bilateral"]
        for card in out["bilateral"]["sides"]:
            assert "knee_at_bdc_alone" not in card

    def test_the_session_answer_is_not_either_card(self):
        """The combined number is the verdict; the cards only show where it
        came from."""
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        vals = sorted(c["knee_at_bdc"] for c in out["bilateral"]["sides"])
        assert vals[0] < out["bilateral"]["knee_at_bdc"] < vals[1]

    def test_both_clips_hand_over_their_keyframe(self):
        """A two-sided session that shows one photo looks half-done."""
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        frames = [c["keyframe_base64"] for c in out["bilateral"]["sides"]]
        assert all(frames)
        assert frames[0] != frames[1]


class TestARefusalIsLoud:
    """A refusal changes what every number on the page means -- from "your
    fit" to "one side of your fit". That cannot live only in a panel below
    the metrics."""

    def _refused(self):
        bad = _result("right", 152.3, 95, geom=GEOM_NOT_A_PAIR)
        return build_pair_result(LEFT, bad, "road_hoods")

    def test_it_reaches_the_quality_warnings(self):
        out = self._refused()
        warns = out["sport_specific_metrics"]["quality_warnings"]
        assert any("could not be merged" in w for w in warns)

    def test_it_names_which_clip_the_numbers_came_from(self):
        out = self._refused()
        assert out["bilateral"]["metrics_side"] in ("left", "right")
        side = out["bilateral"]["metrics_side"]
        warns = out["sport_specific_metrics"]["quality_warnings"]
        assert any(f"{side}-side clip alone" in w for w in warns)

    def test_the_clips_own_warnings_are_not_displaced_by_it(self):
        left = _result("left", 154.7, 89, geom=GEOM_LEFT, warn=["left is dim"])
        bad = _result("right", 152.3, 95, geom=GEOM_NOT_A_PAIR, warn=["drive side"])
        out = build_pair_result(left, bad, "road_hoods")
        warns = out["sport_specific_metrics"]["quality_warnings"]
        assert any("left is dim" in w for w in warns)

    def test_a_merged_session_carries_no_such_warning(self):
        out = build_pair_result(LEFT, RIGHT, "road_hoods")
        warns = out["sport_specific_metrics"].get("quality_warnings") or []
        assert not any("could not be merged" in w for w in warns)
        assert "metrics_side" not in out["bilateral"]


class TestAPairItCannotMerge:
    def test_a_corrupted_crank_ruler_no_longer_sinks_the_session(self):
        """This pair used to be refused outright -- and it was two of the
        three pairs ever filmed in the wild. The torso ruler carries it."""
        bad = _result("right", 152.3, 95, geom=GEOM_BAD_RIGHT)
        out = build_pair_result(LEFT, bad, "road_hoods")
        assert out["bilateral"]["combined"] is True
        assert out["bilateral"]["scale_anchor"] == "torso"

    def test_a_pair_that_is_not_one_rider_still_refuses(self):
        bad = _result("right", 152.3, 95, geom=GEOM_NOT_A_PAIR)
        out = build_pair_result(LEFT, bad, "road_hoods")
        assert out["bilateral"]["combined"] is False
        assert out["bilateral"]["reason"] == "leg_mismatch"

    def test_a_refusal_still_returns_a_readable_result(self):
        bad = _result("right", 152.3, 95, geom=GEOM_NOT_A_PAIR)
        out = build_pair_result(LEFT, bad, "road_hoods")
        # The rider filmed two clips and is owed an answer about them.
        assert out["status"] == "completed"
        assert out["technique_score"] is not None
        assert len(out["bilateral"]["sides"]) == 2

    def test_a_refusal_publishes_no_merged_number(self):
        bad = _result("right", 152.3, 95, geom=GEOM_NOT_A_PAIR)
        out = build_pair_result(LEFT, bad, "road_hoods")
        assert out["bilateral"].get("knee_at_bdc") is None
        assert out["bilateral"].get("uncertainty_deg") is None

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
