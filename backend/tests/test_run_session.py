"""Two run clips into one verdict -- and the two gates that stop it.

A run pair is not a bike pair. There is no shared rigid object to pool
geometry against; what the two clips share is the athlete. So cadence, trunk
lean, ground contact, flight time and oscillation are ONE quantity measured
twice, and the gap between the clips on those IS this session's error bar,
measured on the day rather than assumed.

Measured on the first real pair (IMG_4258 / IMG_4262): those whole-body
metrics disagreed by 6% on cadence, 36% on ground contact and 68% on trunk
lean, while the two legs' mean knee angles differed by 4%. The error on
quantities that cannot differ was larger than the difference between the legs
-- so a left-versus-right verdict there would have been invented, and the
bike feature already had to withdraw one such number once.
"""

from __future__ import annotations

from app.services.video_analysis.run_session import (
    AGREEMENT_LIMIT_PCT,
    build_run_session,
    compare_legs,
)


def _result(side, *, cadence=180.0, trunk=3.0, contact=200.0, flight=100.0,
            osc=0.07, stance=0.65, knee_min=90.0, knee_mean=135.0,
            instability=0.05, unstable=False, frames=550):
    return {
        "status": "completed", "sport_type": "run", "camera_side": side,
        "frames_analyzed": frames,
        "technique_score": 93, "letter_grade": "A",
        "keyframe_base64": f"data:image/jpeg;base64,{side}",
        "sport_specific_metrics": {
            "camera_side": side, "near_side": side,
            "cadence_spm": cadence, "trunk_lean_avg": trunk,
            "ground_contact_ms": contact, "flight_time_ms": flight,
            "vertical_oscillation_m": osc, "stance_fraction": stance,
            "knee_min": knee_min, "knee_max": 163.0, "knee_mean": knee_mean,
            "knee_flexion_velocity_dps": 450.0,
            "knee_extension_velocity_dps": 430.0,
            "foot_strike_angle_deg": 9.0, "overstride_ratio": 0.38,
            "tracking_stability": {
                "leg_identity": {
                    "stability": {"instability": instability,
                                  "unstable": unstable},
                },
            },
        },
    }


LEFT = _result("left")
RIGHT = _result("right")


class TestAGoodPair:
    def test_it_combines(self):
        out = build_run_session(LEFT, RIGHT)["session"]
        assert out["combined"] is True

    def test_whole_body_metrics_are_averaged(self):
        out = build_run_session(_result("left", cadence=180.0),
                                _result("right", cadence=176.0))
        assert out["merged_summary"]["cadence_spm"] == 178.0
        assert out["session"]["merged_whole_body"]["cadence_spm"] == 178.0

    def test_the_error_bar_comes_from_the_clips_themselves(self):
        """Not a constant: the largest gap on a quantity that cannot differ."""
        out = build_run_session(_result("left", trunk=3.0),
                                _result("right", trunk=3.3))["session"]
        agree = out["agreement"]
        assert agree["worst_metric"] == "trunk_lean_avg"
        assert agree["worst"] > 0
        assert agree["agree"] is True

    def test_both_legs_are_reported_separately(self):
        out = build_run_session(LEFT, RIGHT)["session"]
        assert set(out["legs"]) == {"left", "right"}
        assert out["legs"]["left"]["metrics"]["knee_mean"] == 135.0

    def test_no_leg_card_carries_a_score(self):
        """One body, one score -- two scores is what confused the rider on the
        bike side."""
        out = build_run_session(LEFT, RIGHT)["session"]
        for card in out["legs"].values():
            assert "technique_score" not in card
            assert "score" not in card


class TestJudgingTheLegsAgainstTheError:
    def test_a_difference_inside_the_error_bar_is_not_readable(self):
        left = _result("left", knee_mean=135.0, trunk=3.0)
        right = _result("right", knee_mean=132.0, trunk=3.6)   # 20% trunk gap
        out = build_run_session(left, right)["session"]
        knee = next(r for r in out["leg_comparison"] if r["metric"] == "knee_mean")
        assert knee["difference_pct"] < out["agreement"]["worst"]
        assert knee["readable"] is False

    def test_a_difference_clear_of_the_error_bar_is_readable(self):
        left = _result("left", knee_min=100.0, trunk=3.0)
        right = _result("right", knee_min=70.0, trunk=3.05)    # tiny trunk gap
        out = build_run_session(left, right)["session"]
        knee = next(r for r in out["leg_comparison"] if r["metric"] == "knee_min")
        assert knee["difference_pct"] > out["agreement"]["worst"]
        assert knee["readable"] is True

    def test_without_an_error_bar_nothing_is_declared_significant(self):
        """Unknown is not the same as significant."""
        rows = compare_legs({"knee_mean": 130.0}, {"knee_mean": 120.0},
                            {"worst": None})
        assert rows[0]["readable"] is None


class TestTheGates:
    def test_an_unstable_clip_blocks_the_merge_and_is_named(self):
        """A clip whose legs traded places would merge the swap into the
        answer, and its own per-leg numbers are already spliced from both."""
        out = build_run_session(
            LEFT, _result("right", instability=0.21, unstable=True))["session"]
        assert out["combined"] is False
        assert out["reason"] == "identity_unstable"
        assert out["unstable_sides"] == ["right"]

    def test_clips_that_disagree_about_the_run_do_not_merge(self):
        out = build_run_session(
            _result("left", contact=220.0),
            _result("right", contact=140.0))["session"]
        assert out["combined"] is False
        assert out["reason"] == "clips_disagree"
        assert out["agreement"]["worst"] > AGREEMENT_LIMIT_PCT

    def test_two_clips_of_one_side_are_not_a_session(self):
        out = build_run_session(LEFT, _result("left"))["session"]
        assert out["combined"] is False
        assert out["reason"] == "sides_not_identified"

    def test_a_refusal_still_shows_both_clips_and_the_evidence(self):
        """The athlete filmed two clips and is owed an answer about them."""
        out = build_run_session(
            LEFT, _result("right", instability=0.21, unstable=True))["session"]
        assert set(out["legs"]) == {"left", "right"}
        assert out["agreement"]["gaps"]

    def test_a_refusal_publishes_no_merged_numbers(self):
        out = build_run_session(
            LEFT, _result("right", instability=0.21, unstable=True))
        assert out["merged_summary"] is None
        assert "merged_whole_body" not in out["session"]
        assert "leg_comparison" not in out["session"]


class TestTheRealPairFromProduction:
    """IMG_4258 + IMG_4262, as they actually measured on 2026-08-29."""

    def test_that_pair_is_refused_and_for_the_right_reason(self):
        left = _result("left", cadence=186.8, trunk=2.1, contact=223.4,
                       flight=103.1, osc=0.06, stance=0.66,
                       instability=0.116, unstable=False)
        right = _result("right", cadence=175.5, trunk=6.6, contact=143.1,
                        flight=92.6, osc=0.08, stance=0.63,
                        instability=0.205, unstable=True)
        out = build_run_session(left, right)["session"]
        assert out["combined"] is False
        assert out["reason"] == "identity_unstable"
        # And the agreement it reports is the reason a merge would have lied:
        # trunk lean is a midline structure and the clips are 68% apart on it.
        assert out["agreement"]["worst_metric"] == "trunk_lean_avg"
        assert out["agreement"]["worst"] > 60
