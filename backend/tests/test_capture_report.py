"""The capture report has to name the fix, in the order worth fixing.

Its whole reason to exist is that a controlled experiment showed framing --
not the pose model, not the filtering -- is what decides whether a running
analysis means anything. So the tests care about two things: that the numbers
land in the right band, and that the thing worth re-filming for comes first.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis.capture_report import (
    FRAMING_GOOD_PX,
    FRAMING_WARN_PX,
    build_capture_report,
)


def report(**kw):
    base = dict(
        sport_type="run",
        duration_s=8.0,
        frame_width=1080,
        frame_height=1920,
        framing={"subject_height_px": 700, "subject_height_frac": 0.55},
        tracking_stability={"leg_swap_pct": 2.0},
    )
    base.update(kw)
    return build_capture_report(**base)


def check(rep, check_id):
    return next((c for c in rep["checks"] if c["id"] == check_id), None)


# --- framing: the one that matters -----------------------------------------

def test_a_well_filled_frame_passes():
    rep = report()
    assert check(rep, "framing")["status"] == "good"
    assert rep["verdict"] == "good"
    assert rep["problem_count"] == 0


def test_the_clip_that_fooled_the_old_threshold_now_fails():
    """255 px was rated OK while the pose model swapped the legs on 46% of
    frames. A false all-clear on the measurement everything rests on."""
    rep = report(framing={"subject_height_px": 255, "subject_height_frac": 0.199})
    assert check(rep, "framing")["status"] == "bad"


def test_the_cropped_version_of_that_same_clip_passes():
    rep = report(framing={"subject_height_px": 452, "subject_height_frac": 0.437})
    assert check(rep, "framing")["status"] == "good"


@pytest.mark.parametrize("px,expected", [
    (100, "bad"), (FRAMING_WARN_PX - 1, "bad"),
    (FRAMING_WARN_PX, "warn"), (FRAMING_GOOD_PX - 1, "warn"),
    (FRAMING_GOOD_PX, "good"), (900, "good"),
])
def test_framing_bands(px, expected):
    assert check(report(framing={"subject_height_px": px}), "framing")["status"] == expected


def test_framing_says_how_much_to_zoom_rather_than_just_saying_too_small():
    c = check(report(framing={"subject_height_px": 275}), "framing")
    assert "2.0x" in c["action"]           # 550 target / 275 measured
    assert "275 px" in c["measured"]


def test_framing_that_could_not_be_measured_says_so_instead_of_guessing():
    c = check(report(framing={}), "framing")
    assert c["status"] == "unknown"
    assert not c["action"]


# --- orientation: the free win ---------------------------------------------

def test_portrait_passes():
    assert check(report(frame_width=1080, frame_height=1920), "orientation")["status"] == "good"


def test_landscape_with_a_small_athlete_is_flagged():
    c = check(report(frame_width=1920, frame_height=1080,
                     framing={"subject_height_px": 255}), "orientation")
    assert c["status"] == "warn"
    assert "upright" in c["action"]


def test_landscape_is_not_nagged_about_when_the_frame_was_filled_anyway():
    """Landscape is wasteful, not wrong. Somebody who filled it regardless has
    nothing to fix, and telling them otherwise is noise."""
    c = check(report(frame_width=1920, frame_height=1080,
                     framing={"subject_height_px": 700}), "orientation")
    assert c["status"] == "good"
    assert not c["action"]


# --- leg identity: the symptom that decides the metrics --------------------

def test_swapped_legs_are_reported_as_a_symptom_of_framing():
    c = check(report(tracking_stability={"leg_swap_pct": 45.6}), "leg_identity")
    assert c["status"] == "bad"
    assert "45.6%" in c["measured"]
    assert "symptom of the framing" in c["action"]


def test_clean_leg_tracking_passes():
    assert check(report(tracking_stability={"leg_swap_pct": 1.8}),
                 "leg_identity")["status"] == "good"


def test_bike_has_no_leg_identity_row():
    """Bike locks the side up front and never runs the leg pass, so there is
    no measurement to report."""
    assert check(report(sport_type="bike", tracking_stability={"leg_swap_pct": 40.0}),
                 "leg_identity") is None


def test_the_bar_for_swapped_legs_is_the_confidence_scorers():
    from app.services.video_analysis.biomechanics.confidence_scorer import THRESHOLDS

    below = THRESHOLDS["leg_swap_pct_medium"] - 0.1
    at_low = THRESHOLDS["leg_swap_pct_low"]
    assert check(report(tracking_stability={"leg_swap_pct": below}),
                 "leg_identity")["status"] == "good"
    assert check(report(tracking_stability={"leg_swap_pct": at_low}),
                 "leg_identity")["status"] == "bad"


# --- duration ---------------------------------------------------------------

@pytest.mark.parametrize("secs,expected", [
    (1.1, "bad"), (2.9, "bad"), (3.0, "warn"), (4.9, "warn"),
    (5.0, "good"), (12.0, "good"), (25.0, "warn"),
])
def test_duration_bands(secs, expected):
    assert check(report(duration_s=secs), "duration")["status"] == expected


def test_a_decimated_clip_is_flagged_even_inside_the_good_band():
    c = check(report(duration_s=18.0, sampling_degraded="every 4th frame"), "duration")
    assert c["status"] == "warn"
    assert "decimated" in c["action"]


# --- slow motion ------------------------------------------------------------

def test_slow_motion_only_appears_when_it_happened():
    assert check(report(), "time_base") is None
    c = check(report(time_base_uncertain=True), "time_base")
    assert c["status"] == "bad"
    assert "slow-motion off" in c["action"]


# --- ordering: the point of the whole thing --------------------------------

def test_the_worst_problem_leads():
    rep = report(
        duration_s=4.0,                                    # warn
        framing={"subject_height_px": 180},                # bad
        tracking_stability={"leg_swap_pct": 3.0},          # good
    )
    assert rep["checks"][0]["id"] == "framing"
    assert rep["verdict"] == "poor"


def test_high_impact_wins_ties_within_the_same_status():
    rep = report(
        duration_s=4.0,                                    # warn, medium impact
        framing={"subject_height_px": 300},                # warn, high impact
        tracking_stability={"leg_swap_pct": 2.0},
    )
    assert rep["checks"][0]["id"] == "framing"
    assert rep["verdict"] == "fair"


def test_the_headline_is_the_first_actionable_fix():
    rep = report(framing={"subject_height_px": 200})
    assert rep["headline"] == check(rep, "framing")["action"]


def test_a_clean_clip_has_no_headline_to_give():
    assert report()["headline"] == ""


def test_every_check_carries_what_was_measured_and_what_to_aim_for():
    for c in report(framing={"subject_height_px": 120}, duration_s=1.0,
                    tracking_stability={"leg_swap_pct": 40.0},
                    time_base_uncertain=True)["checks"]:
        assert c["measured"] and c["target"]
        assert c["status"] in ("good", "warn", "bad", "unknown")
        assert c["impact"] in ("high", "medium", "low")


def test_the_report_is_json_serializable():
    import json

    json.dumps(report(framing={"subject_height_px": 200}, time_base_uncertain=True))


def test_missing_everything_degrades_instead_of_raising():
    rep = build_capture_report(sport_type="run")
    assert rep["verdict"] in ("unknown", "good", "fair", "poor")
    assert rep["checks"]


# --- it reaches the people who most need it --------------------------------

def test_a_free_caller_still_gets_the_capture_report():
    """The one part of a free result that tells them how to get a better one.
    A paywall in front of the instructions would be charging for the manual."""
    from app.services.result_gating import quality_block

    result = {
        "status": "completed", "sport_type": "run", "technique_score": 61,
        "sport_specific_metrics": {
            "quality_warnings": [],
            "capture_report": report(framing={"subject_height_px": 200}),
        },
    }
    block = quality_block(result)
    assert block["capture_report"]["checks"][0]["id"] == "framing"
    assert block["capture_report"]["headline"]


def test_the_export_tells_an_outside_model_the_clip_was_badly_framed():
    """A model handed joint angles has no way to know the athlete was 200 px
    tall -- and without that it diagnoses technique from a signal that was
    never there."""
    from app.services.export import ai_export

    md = ai_export.build_markdown({
        "status": "completed", "sport_type": "run", "technique_score": 61,
        "letter_grade": "D",
        "sport_specific_metrics": {
            "capture_report": report(framing={"subject_height_px": 200},
                                     tracking_stability={"leg_swap_pct": 40.0}),
        },
    })
    assert "Capture: poor" in md
    assert "200 px" in md
    assert "40.0% of frames" in md


def test_a_clean_capture_adds_no_noise_to_the_export():
    from app.services.export import ai_export

    md = ai_export.build_markdown({
        "status": "completed", "sport_type": "run", "technique_score": 88,
        "letter_grade": "A",
        "sport_specific_metrics": {"capture_report": report()},
    })
    assert "Capture:" not in md
