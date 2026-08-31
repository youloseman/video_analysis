"""The golden record: what it captures, and what it lets through.

These run everywhere -- no clips, no MediaPipe. The guard in
test_golden_clips.py can only run where the footage lives, so the machinery it
depends on is tested here instead, or the whole thing would be unverified on
every machine but one.

Two ways this could be useless without ever failing. Too tight, and it cries
about float noise until someone deletes it. Too loose, and it sits green
through the exact bugs it was written for -- which is what the rest of the
suite already did.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis.golden import (
    FLOAT_ATOL,
    FLOAT_RTOL,
    build_record,
    compare,
)


def _result(**over):
    """A minimal analysis result, shaped like the real thing."""
    base = {
        "status": "completed",
        "sport_type": "bike",
        "cycling_position": "tt_aero",
        "camera_side": "right",
        "technique_score": 74,
        "letter_grade": "C",
        "frames_analyzed": 503,
        "quality_gate_triggered": False,
        "score_breakdown": {"knee_bdc": 88.0, "trunk_angle": 100.0},
        "score_coverage": {
            "scored": ["knee_bdc", "trunk_angle"], "missing": [], "excluded": {},
            "measures_scored": 2, "measures_total": 2, "weight_covered": 1.0,
        },
        "angle_statistics": {
            "right_knee": {
                "mean": 120.5, "min": 70.0, "max": 155.4, "range": 85.4,
                "std": 25.1, "nan_pct": 20.1, "valid_frames": 402,
            },
        },
        "detected_issues": [{"issue": "saddle_too_high"}],
        "fit_plan": {"diagnostics": [{
            "component": "saddle_height", "metric_name": "knee_at_bdc",
            "action": "lower_saddle", "status": "needs_adjustment",
            "current_value": 155.4,
        }]},
        "sport_specific_metrics": {
            "knee_at_bdc": 155.4,
            "trunk_angle_avg": 22.3,
            "saddle_height_assessment": "too_high",
            "near_side": "right",
            "quality_warnings": ["small in frame"],
            "capture_report": {"verdict": "fair"},
            "analysis_confidence": {"level": "medium"},
            "tracking_stability": {"leg_swap_pct": 2.5},
            "sampling": {"sample_rate": 1},
            "aero_estimate": {"zone": "b"},
        },
        "keyframe_base64": "data:image/jpeg;base64,AAAA",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# what the record keeps
# --------------------------------------------------------------------------

def test_an_unchanged_result_produces_no_differences():
    rec = build_record(_result())
    assert compare(rec, build_record(_result())) == []


def test_conclusions_land_in_the_exact_half():
    rec = build_record(_result())
    e = rec["exact"]
    assert e["letter_grade"] == "C"
    assert e["camera_side"] == "right"
    assert e["capture_verdict"] == "fair"
    assert e["confidence_level"] == "medium"
    assert e["summary.saddle_height_assessment"] == "too_high"
    assert e["issues"] == ["saddle_too_high"]


def test_measurements_land_in_the_approximate_half():
    a = build_record(_result())["approx"]
    assert a["technique_score"] == 74
    assert a["summary.knee_at_bdc"] == 155.4
    assert a["angles"]["right_knee"]["mean"] == 120.5


def test_the_recommended_action_is_recorded():
    """The inverted hip sign changed nothing but this."""
    plan = build_record(_result())["exact"]["fit_plan"]
    assert plan == [{
        "component": "saddle_height", "metric": "knee_at_bdc",
        "action": "lower_saddle", "status": "needs_adjustment", "current": 155.4,
    }]


def test_images_and_prose_are_not_recorded():
    """A base64 frame would make the baseline megabytes, and the LLM writes
    different words every call -- a baseline containing either fails for
    reasons that are not regressions."""
    rec = build_record(_result(ai_recommendations={"report": "nice riding"}))
    flat = repr(rec)
    assert "AAAA" not in flat
    assert "nice riding" not in flat
    assert rec["exact"]["has_keyframe"] is True   # presence, not content


# --------------------------------------------------------------------------
# the three bugs that got through a green suite
# --------------------------------------------------------------------------

def test_a_flipped_action_is_caught():
    """Same measurement, same score, opposite instruction. Nothing numeric
    moves -- if the action were not recorded, nothing at all would."""
    before = build_record(_result())
    after = _result()
    after["fit_plan"]["diagnostics"][0]["action"] = "raise_saddle"
    diffs = compare(before, build_record(after))
    assert diffs == ["exact.fit_plan[0].action: 'lower_saddle' -> 'raise_saddle'"]


def test_a_component_vanishing_from_the_rubric_is_caught():
    """The dead waveform component. The score still computes -- the weighted
    average just renormalises over what is left, so the number alone looks
    fine."""
    after = _result()
    after["score_coverage"] = {
        "scored": ["knee_bdc"], "missing": ["trunk_angle"], "excluded": {},
        "measures_scored": 1, "measures_total": 2, "weight_covered": 0.6,
    }
    diffs = compare(build_record(_result()), build_record(after))
    assert any("coverage.scored" in d for d in diffs)
    assert any("coverage.missing" in d for d in diffs)


def test_an_angle_shifting_by_degrees_is_caught():
    """The world-vs-image landmark bug moved bike angles by ~8 deg per view."""
    after = _result()
    after["angle_statistics"]["right_knee"]["mean"] = 128.5
    diffs = compare(build_record(_result()), build_record(after))
    assert diffs == ["approx.angles.right_knee.mean: 120.5 -> 128.5"]


# --------------------------------------------------------------------------
# tolerance: tight enough to matter, loose enough to live with
# --------------------------------------------------------------------------

def test_float_noise_does_not_fail():
    after = _result()
    after["angle_statistics"]["right_knee"]["mean"] = 120.5 + FLOAT_ATOL / 2
    assert compare(build_record(_result()), build_record(after)) == []


def test_a_fifth_of_a_degree_does_fail():
    """The bar for "a real change". The fit plan's own deadband is about 3 deg
    -- a tolerance that swallowed a fifth of one could swallow a change that
    flips a recommendation."""
    after = _result()
    after["angle_statistics"]["right_knee"]["mean"] = 120.5 + 0.2
    assert compare(build_record(_result()), build_record(after)) != []


def test_the_tolerance_is_not_quietly_generous():
    """The measured run-to-run difference on a real clip is zero across 54
    numeric fields, so there is no noise budget to spend. A percent here would
    be 1.5 deg on a knee -- half the deadband that decides whether the rider is
    told to move their saddle."""
    assert FLOAT_RTOL <= 0.001
    assert FLOAT_ATOL <= 0.05
    # The tolerance on the largest angle we record must stay under the fit
    # plan's own deadband, or a change that alters advice could pass silently.
    assert 180 * FLOAT_RTOL + FLOAT_ATOL < 1.0


def test_exact_fields_have_no_tolerance():
    """A grade is a letter. There is no such thing as nearly a C."""
    after = _result(letter_grade="B")
    assert compare(build_record(_result()), build_record(after)) == [
        "exact.letter_grade: 'C' -> 'B'"
    ]


# --------------------------------------------------------------------------
# missing data
# --------------------------------------------------------------------------

def test_nan_and_none_are_the_same_fact():
    """Both mean "not measured". Distinguishing them would fail the baseline
    whenever a missing value changed which KIND of missing it was."""
    after = _result()
    after["angle_statistics"]["right_knee"]["mean"] = float("nan")
    rec = build_record(after)
    assert rec["approx"]["angles"]["right_knee"]["mean"] is None


def test_a_field_appearing_or_disappearing_is_reported():
    """Not just changed values -- a metric the pipeline stops emitting is a
    regression, and one it starts emitting is a baseline that needs re-recording."""
    thin = _result()
    del thin["sport_specific_metrics"]["trunk_angle_avg"]
    diffs = compare(build_record(_result()), build_record(thin))
    assert any("disappeared" in d for d in diffs)
    diffs_back = compare(build_record(thin), build_record(_result()))
    assert any("appeared" in d for d in diffs_back)


def test_a_run_result_records_its_own_metrics():
    run = _result(
        sport_type="run", cycling_position=None,
        sport_specific_metrics={
            "cadence_spm": 168.7, "trunk_lean_avg": 5.9,
            "vertical_oscillation_m": 0.0939, "slow_motion_factor": 8,
            "foot_strike": "midfoot", "stance_fraction": 0.727,
        },
        score_coverage={
            "scored": ["trunk_lean"], "missing": ["cadence"],
            "excluded": {"cadence": "reads as 8x slow motion"},
            "measures_scored": 1, "measures_total": 2, "weight_covered": 0.76,
        },
    )
    rec = build_record(run)
    assert rec["approx"]["summary.cadence_spm"] == 168.7
    assert rec["exact"]["summary.foot_strike"] == "midfoot"
    assert rec["exact"]["summary.slow_motion_factor"] == 8
    # The exclusion is a conclusion about our own confidence -- pin it exactly.
    assert rec["exact"]["coverage"]["excluded"] == ["cadence"]


def test_an_empty_result_does_not_explode():
    """A failed analysis still has to produce a comparable record rather than
    raising inside the guard."""
    rec = build_record({})
    assert rec["exact"]["status"] is None
    assert rec["approx"]["angles"] == {}


@pytest.mark.parametrize("bad", [None, "", "n/a", float("inf")])
def test_unparseable_numbers_become_none(bad):
    after = _result()
    after["sport_specific_metrics"]["knee_at_bdc"] = bad
    assert build_record(after)["approx"]["summary.knee_at_bdc"] is None
