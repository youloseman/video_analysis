"""A clip whose skeleton kept flipping legs must not report high confidence.

Leg-identity swaps keep MediaPipe visibility high and produce no NaN
angles, so every pre-existing confidence factor stays green -- a user with
a backlit silhouette clip saw garbage angles under a HIGH badge and the
green 'Full confidence' chip. The tracking_stability factor is the channel
that finally sees it: severe instability forces LOW, mild forces MEDIUM,
and unmeasured signals (bike) change nothing.
"""

from __future__ import annotations

from app.services.video_analysis.biomechanics.confidence_scorer import (
    assess_tracking_stability,
    compute_analysis_confidence,
)

CLEAN_STATS = {"knee": {"nan_pct": 2.0, "valid_frames": 400}}


def confidence(tracking=None, warnings=()):
    return compute_analysis_confidence(
        angle_statistics=CLEAN_STATS,
        frames_processed=400,
        butterworth_meta=None,
        analysis_warnings=list(warnings),
        landmark_quality={"overall_pct": 95.0, "confidence": "high"},
        tracking_stability=tracking,
    )


# ---------------------------------------------------------------------------
# assess_tracking_stability: the shared severity gauge
# ---------------------------------------------------------------------------
def test_no_signals_reads_ok():
    assert assess_tracking_stability(None) == "ok"
    assert assess_tracking_stability({}) == "ok"


def test_unmeasured_none_values_are_not_evidence_of_stability():
    # Bike: side pre-locked, leg pass off -- everything None/0.
    assert assess_tracking_stability({
        "side_vote_disagreement_pct": None,
        "flip_pct": 0.0,
        "leg_swap_pct": None,
        "leg_collapse_pct": None,
    }) == "ok"


def test_each_signal_alone_can_flag_severe():
    assert assess_tracking_stability({"leg_swap_pct": 16.0}) == "severe"
    assert assess_tracking_stability({"leg_collapse_pct": 30.0}) == "severe"
    assert assess_tracking_stability({"side_vote_disagreement_pct": 35.0}) == "severe"
    assert assess_tracking_stability({"flip_pct": 22.0}) == "severe"


def test_mild_band_sits_between_clean_and_severe():
    assert assess_tracking_stability({"leg_swap_pct": 8.0}) == "mild"
    # A healthy 60fps run clip probes ~5% from scissor crossings -- below mild.
    assert assess_tracking_stability({"leg_swap_pct": 5.0}) == "ok"


def test_booleans_and_nan_do_not_trip_thresholds():
    assert assess_tracking_stability({"leg_swap_pct": True}) == "ok"
    assert assess_tracking_stability({"leg_swap_pct": float("nan")}) == "ok"


# ---------------------------------------------------------------------------
# The confidence level itself
# ---------------------------------------------------------------------------
def test_severe_instability_forces_low_and_names_the_cause():
    result = confidence({"leg_swap_pct": 25.0, "leg_collapse_pct": 40.0})
    assert result["level"] == "low"
    text = result["explanation"].lower()
    assert "left and right" in text
    assert "backlit" in text or "silhouette" in text
    assert result["factors"]["tracking_severity"] == "severe"
    assert result["factors"]["leg_swap_pct"] == 25.0


def test_mild_instability_reads_medium():
    result = confidence({"leg_swap_pct": 8.0})
    assert result["level"] == "medium"
    assert result["factors"]["tracking_severity"] == "mild"


def test_clean_tracking_leaves_high_untouched():
    result = confidence({"leg_swap_pct": 2.0, "flip_pct": 0.0,
                         "side_vote_disagreement_pct": 1.0,
                         "leg_collapse_pct": 4.0})
    assert result["level"] == "high"
    assert result["factors"]["tracking_severity"] == "ok"


def test_omitting_the_param_changes_nothing_for_legacy_callers():
    result = compute_analysis_confidence(
        angle_statistics=CLEAN_STATS,
        frames_processed=400,
        butterworth_meta=None,
        analysis_warnings=[],
        landmark_quality={"overall_pct": 95.0, "confidence": "high"},
    )
    assert result["level"] == "high"
    assert result["factors"]["tracking_severity"] == "ok"


# ---------------------------------------------------------------------------
# The AI coach must not contradict the quality verdict on the same page
# ---------------------------------------------------------------------------
def test_llm_prompt_carries_a_caveat_on_unstable_tracking():
    from app.services.video_analysis.llm_recommendations import _build_prompt

    severe = _build_prompt(
        "run", 54, "D", None, [], {},
        {"tracking_stability": {"leg_swap_pct": 25.0}},
    )
    assert "unreliable" in severe and "do NOT build" in severe

    mild = _build_prompt(
        "run", 70, "B", None, [], {},
        {"tracking_stability": {"leg_swap_pct": 8.0}},
    )
    assert "approximate" in mild

    clean = _build_prompt(
        "run", 80, "A", None, [], {},
        {"tracking_stability": {"leg_swap_pct": 1.0}},
    )
    assert "NOTE: pose tracking" not in clean
