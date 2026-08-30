"""The coach must not say "nothing to fix" over a table full of red rows.

Materiality was stated outright in the photo prompt and merely hoped for in the
video prompt, which got the numbers and their bands but no verdict on any of
them. It does not reliably apply the floor itself: a real clip whose table
showed knee-at-TDC 76 (band 60-72) and shoulder 70 (band 80-105) both out of
range came back with "Nothing to fix -- your position is solid". The rule-based
issue list could not save it either, since for cycling it only covers saddle
height and trunk angle.

So the prompt now states which measurements cleared the floor, using the same
classifier the overlay chips and the results table read.
"""
from __future__ import annotations

from app.services.video_analysis.llm_recommendations import (
    _build_prompt,
    _materiality_block,
    _metric_statuses,
)

POSITION = "triathlon"

# Triathlon bands: knee_at_bdc 138-145, knee_at_tdc 60-72, trunk 20-30,
# elbow 75-110, shoulder 80-105, hip 40-55, forearm 5-25, pelvic 2.0-4.0.
CLEAN = {
    "knee_at_bdc": 141.0,
    "knee_at_tdc": 66.0,
    "trunk_angle_avg": 25.0,
    "elbow_angle_avg": 90.0,
    "shoulder_angle_avg": 92.0,
    "hip_angle_avg": 50.0,
    "forearm_tilt_avg": 15.0,
    "pelvic_ratio": 3.0,
    "head_alignment_avg": 95.0,
}


def _summary(**over) -> dict:
    return {**CLEAN, **over}


def test_a_clean_position_says_there_is_nothing_to_fix():
    block = _materiality_block("bike", POSITION, _summary())
    assert "MATERIAL PROBLEMS: NONE" in block
    assert "Nothing to fix" in block


def test_the_clip_that_started_this_lists_both_red_rows():
    """FB-01: knee TDC 76.1 and shoulder 70.3, scored 96/100 grade A."""
    block = _materiality_block(
        "bike", POSITION,
        _summary(knee_at_tdc=76.1, shoulder_angle_avg=70.3, hip_angle_avg=72.9),
    )
    assert "MATERIAL PROBLEMS (" in block
    assert "Knee angle at top of stroke (TDC)" in block
    assert "Shoulder angle" in block
    assert "NONE" not in block


def test_a_saddle_clearly_too_high_is_material():
    """FB-02: knee at BDC 156.7 against a 138-145 band."""
    block = _materiality_block("bike", POSITION, _summary(knee_at_bdc=156.7))
    assert "Knee angle at bottom of stroke (BDC)" in block


def test_a_miss_inside_the_instrument_s_own_spread_is_not_a_finding():
    """2 deg outside a band is not a fault this method can assert."""
    block = _materiality_block("bike", POSITION, _summary(trunk_angle_avg=18.0))
    assert "MATERIAL PROBLEMS: NONE" in block
    assert "WITHIN TOLERANCE" in block
    assert "Trunk angle" in block.split("WITHIN TOLERANCE")[1]


def test_an_open_hip_is_never_a_problem():
    """Closing the hip is the risk; opening it is a comfort trade-off. The
    results table stretches the upper bound to the measurement, and the coach
    has to agree with the page the athlete is reading."""
    statuses = _metric_statuses("bike", POSITION, _summary(hip_angle_avg=72.9))
    assert statuses["hip_angle_avg"][0] == "good"
    assert "Hip angle" not in _materiality_block(
        "bike", POSITION, _summary(hip_angle_avg=72.9),
    )


def test_a_closing_hip_still_is_a_problem():
    block = _materiality_block("bike", POSITION, _summary(hip_angle_avg=30.0))
    assert "Hip angle" in block.split("may appear in Fix first):")[1]


def test_the_metric_lines_carry_the_same_verdict_as_the_list():
    """One classifier, so the prose cannot argue with the table beside it."""
    prompt = _build_prompt(
        "bike", 96, "A", POSITION, [], {},
        _summary(knee_at_tdc=76.1, shoulder_angle_avg=70.3),
    )
    knee_line = next(
        ln for ln in prompt.splitlines() if ln.startswith("- Knee angle at top")
    )
    assert "OUT OF RANGE" in knee_line
    bdc_line = next(
        ln for ln in prompt.splitlines() if ln.startswith("- Knee angle at bottom")
    )
    assert "in range" in bdc_line


def test_the_block_survives_a_summary_with_nothing_measured():
    block = _materiality_block("bike", POSITION, {})
    assert "MATERIAL PROBLEMS: NONE" in block


def test_running_clips_get_the_statement_too():
    """Run had the same gap; reference_bands covers both sports."""
    block = _materiality_block("run", None, {"cadence_spm": 170.0})
    assert "MATERIAL PROBLEMS" in block
