"""A bike fit is as good as its worst contact point, not as its average.

The overall used to be a plain weighted mean of nine components with a heaviest
weight of 0.22, so one badly wrong measurement could only cost its own weight.
Measured on a real outdoor TT clip: a rider whose knee at BDC read 156.7 deg
against a 138-145 band -- a saddle roughly 12 deg too high, a power and injury
problem -- scored 81/100 and a grade of B, because the rest of the position was
excellent.
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics.technique_scorer import (
    CYCLING_DERIVED_COMPONENTS,
    CYCLING_WEIGHTS,
    WORST_COMPONENT_PULL,
    assign_grade,
    compute_weighted_score,
)


def _bike(**over) -> dict[str, float]:
    return {**{k: 100.0 for k in CYCLING_WEIGHTS}, **over}


def _scored(components: dict[str, float]) -> int:
    return compute_weighted_score(
        components, CYCLING_WEIGHTS,
        worst_pull=WORST_COMPONENT_PULL,
        exclude_from_worst=CYCLING_DERIVED_COMPONENTS,
    )


# Component scores as the two clips actually produced them.
FB01 = {
    "knee_bdc": 100.0, "knee_tdc": 91.42, "trunk_angle": 97.35,
    "saddle_fit": 100.0, "elbow_angle": 100.0, "shoulder_angle": 84.68,
    "forearm_tilt": 100.0, "head_alignment": 99.91, "pelvic_ratio": 100.0,
}
FB02 = {
    "knee_bdc": 30.5, "knee_tdc": 100.0, "trunk_angle": 100.0,
    "saddle_fit": 40.0, "elbow_angle": 100.0, "shoulder_angle": 100.0,
    "forearm_tilt": 100.0, "head_alignment": 100.0, "pelvic_ratio": 100.0,
}


def test_a_saddle_clearly_too_high_no_longer_reads_as_a_b():
    assert compute_weighted_score(FB02, CYCLING_WEIGHTS) == 81   # the old mean
    assert _scored(FB02) == 63
    assert assign_grade(_scored(FB02)) == "C"


def test_a_strong_position_with_one_real_fault_stays_an_a():
    """FB-01's worst component is 85/100 -- nothing about it is badly wrong."""
    assert compute_weighted_score(FB01, CYCLING_WEIGHTS) == 96
    assert _scored(FB01) == 92
    assert assign_grade(_scored(FB01)) == "A"


def test_a_clean_position_is_left_alone():
    assert _scored(_bike()) == 100


def test_one_wrecked_component_can_no_longer_hide_behind_eight_good_ones():
    wrecked = _bike(knee_bdc=0.0, saddle_fit=40.0)
    assert compute_weighted_score(wrecked, CYCLING_WEIGHTS) == 75
    assert _scored(wrecked) == 48


def test_a_merely_acceptable_saddle_verdict_does_not_set_the_floor():
    """``saddle_fit`` is derived from knee-at-BDC and is categorical: its
    "acceptable" maps to 75. Letting that define the worst case would count one
    measurement twice and dock an otherwise perfect fit for a non-fault."""
    assert _scored(_bike(saddle_fit=75.0)) == 99


def test_an_unweighted_extra_score_cannot_define_the_worst_case():
    """Anything outside the weight table is not part of the average, so it must
    not be allowed to set the floor of it either."""
    assert _scored({**_bike(), "some_diagnostic": 3.0}) == 100


def test_the_plain_mean_is_still_the_default():
    """Running and swimming keep the old aggregation until there is a fixture
    to measure the change against."""
    assert compute_weighted_score(FB02, CYCLING_WEIGHTS) == 81


def test_no_measured_component_still_returns_the_neutral_default():
    assert _scored({}) == 50
