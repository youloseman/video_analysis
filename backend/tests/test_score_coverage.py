"""The score must say what it was computed from.

``compute_weighted_score`` renormalises over whichever components are present.
That is the right arithmetic -- averaging over absent measures would drag every
partial clip toward zero -- but on its own it is a misleading presentation: a
score built from five of nine measures is indistinguishable from one built from
all nine. The clip audit made that concrete. ``vid1.MOV`` scored 92/100, grade
A, on a cadence the analyzer reached by multiplying an observed ~21 spm by a
GUESSED 8x slow-motion factor. Cadence carries 16% of the running weight, so an
assumption about the frame rate was being graded as a measurement.

Two rules, tested here:
  1. every scorer reports its coverage;
  2. a guessed time base disqualifies cadence from the score, with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.video_analysis.biomechanics.technique_scorer import (
    CYCLING_WEIGHTS,
    RUNNING_WEIGHTS,
    score_coverage,
    score_cycling,
    score_running,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _run_summary(**over):
    """A running clip with everything measurable present."""
    base = {
        "trunk_lean_avg": 6.0,
        "cadence_spm": 176.0,
        "vertical_oscillation_m": 0.08,
        "overstride_ratio": 0.10,
        "biomechanics": {
            "phase_portraits": {"overall_stability_score": 82.0},
            "waveform": {"overall_similarity_score": 79.0},
        },
    }
    base.update(over)
    return base


_RUN_ANGLES = {
    "knee": {"p95": 165.0, "p05": 75.0},
    "elbow": {"mean": 95.0},
}


def _bike_summary(**over):
    base = {
        "near_side": "left",
        "left_knee_at_bdc": 145.0,
        "left_knee_at_tdc": 71.0,
        "trunk_angle_avg": 45.0,
        "elbow_angle_avg": 150.0,
        "shoulder_angle_avg": 88.0,
        "head_alignment_avg": 80.0,
        "pelvic_ratio": 0.5,
        "forearm_tilt_avg": 5.0,
        "saddle_height_assessment": "optimal",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# score_coverage itself
# --------------------------------------------------------------------------

def test_coverage_counts_scored_against_the_whole_rubric():
    cov = score_coverage({"a": 90.0}, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert cov["measures_scored"] == 1
    assert cov["measures_total"] == 3
    assert cov["scored"] == ["a"]
    assert cov["missing"] == ["b", "c"]


def test_coverage_reports_weight_not_just_count():
    """Three of four measures can still be most of the score, or barely any.

    A bare count hides that: dropping one 40%-weighted measure and dropping one
    5%-weighted measure both read as "3 of 4".
    """
    weights = {"heavy": 0.7, "a": 0.1, "b": 0.1, "c": 0.1}
    lost_heavy = score_coverage({"a": 1, "b": 1, "c": 1}, weights)
    lost_light = score_coverage({"heavy": 1, "a": 1, "b": 1}, weights)
    assert lost_heavy["measures_scored"] == lost_light["measures_scored"] == 3
    assert lost_heavy["weight_covered"] == pytest.approx(0.3)
    assert lost_light["weight_covered"] == pytest.approx(0.9)


def test_a_full_house_covers_all_the_weight():
    cov = score_coverage({"a": 1, "b": 1}, {"a": 0.4, "b": 0.6})
    assert cov["missing"] == []
    assert cov["excluded"] == {}
    assert cov["weight_covered"] == pytest.approx(1.0)


def test_excluded_is_separate_from_missing():
    """"We could not measure it" and "we refuse to grade it" are different facts.

    Both leave the component out of the average; only one of them is a
    statement about our own confidence, and the reader deserves to tell them
    apart.
    """
    cov = score_coverage(
        {"a": 90.0}, {"a": 0.5, "b": 0.5}, {"b": "the time base was guessed"},
    )
    assert cov["missing"] == ["b"]
    assert cov["excluded"]["b"] == "the time base was guessed"


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def test_running_score_carries_coverage():
    result = score_running(_run_summary(), _RUN_ANGLES)
    cov = result["coverage"]
    assert cov["measures_total"] == len(RUNNING_WEIGHTS)
    assert cov["measures_scored"] == len(RUNNING_WEIGHTS)
    assert cov["weight_covered"] == pytest.approx(1.0)


def test_slow_motion_clip_does_not_get_its_cadence_graded():
    """The vid1.MOV case: 8x slow motion, cadence rescaled, grade A.

    The multiplier is a guess the analyzer cannot verify -- it cannot tell 8x
    slow motion of a runner from normal speed of somebody jogging away in the
    distance. Grading it lends a guess the authority of a measurement.
    """
    result = score_running(_run_summary(slow_motion_factor=8), _RUN_ANGLES)
    cov = result["coverage"]
    assert "cadence" not in result["component_scores"]
    assert "cadence" in cov["excluded"]
    assert "8x slow motion" in cov["excluded"]["cadence"]
    assert cov["measures_scored"] == len(RUNNING_WEIGHTS) - 1


def test_the_cadence_number_itself_survives_exclusion():
    """Not graded is not withheld. The athlete still sees the value and the
    warning that comes with it -- we only decline to fold it into a verdict."""
    summary = _run_summary(slow_motion_factor=8)
    score_running(summary, _RUN_ANGLES)
    assert summary["cadence_spm"] == 176.0


def test_normal_speed_cadence_is_still_scored():
    result = score_running(_run_summary(), _RUN_ANGLES)
    assert "cadence" in result["component_scores"]
    assert result["coverage"]["excluded"] == {}


def test_unmeasurable_cadence_is_missing_not_excluded():
    """A clip where cadence never resolved is a different story from one where
    we refused to grade it -- nothing was assumed, the measure simply is not
    there."""
    cov = score_running(_run_summary(cadence_spm=0), _RUN_ANGLES)["coverage"]
    assert "cadence" in cov["missing"]
    assert "cadence" not in cov["excluded"]


def test_a_sparse_running_clip_reports_how_sparse():
    """Trunk lean alone still produces a number. It must not look complete."""
    result = score_running({"trunk_lean_avg": 6.0}, {})
    cov = result["coverage"]
    assert result["overall_score"] > 0
    assert cov["measures_scored"] == 1
    assert cov["weight_covered"] == pytest.approx(RUNNING_WEIGHTS["trunk_lean"], abs=1e-3)


# --------------------------------------------------------------------------
# cycling
# --------------------------------------------------------------------------

def test_cycling_score_carries_coverage():
    cov = score_cycling(_bike_summary(), {})["coverage"]
    assert cov["measures_total"] == len(CYCLING_WEIGHTS)
    assert cov["measures_scored"] == len(CYCLING_WEIGHTS)


def test_cycling_missing_measures_are_named():
    summary = _bike_summary()
    del summary["head_alignment_avg"]
    del summary["pelvic_ratio"]
    cov = score_cycling(summary, {})["coverage"]
    assert set(cov["missing"]) == {"head_alignment", "pelvic_ratio"}
    assert cov["measures_scored"] == len(CYCLING_WEIGHTS) - 2


def test_every_coverage_key_is_a_real_weight_key():
    """Coverage names components; the UI maps those names to English. A typo
    here would print a raw key at the reader."""
    for weights, cov in (
        (RUNNING_WEIGHTS, score_running(_run_summary(), _RUN_ANGLES)["coverage"]),
        (CYCLING_WEIGHTS, score_cycling(_bike_summary(), {})["coverage"]),
    ):
        assert set(cov["scored"]) | set(cov["missing"]) == set(weights)


# --------------------------------------------------------------------------
# the SPA side: the reader has to be told, in English
# --------------------------------------------------------------------------

_SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def spa() -> str:
    return _SPA.read_text(encoding="utf-8")


def test_the_score_card_renders_coverage(spa: str):
    assert "coverageLine(r.score_coverage)" in spa


def test_every_component_has_an_english_name(spa: str):
    """The score card prints component names at a person. A weight key with no
    entry in the SPA's map falls through to a de-underscored raw key
    ("knee bdc"), which is our schema leaking into the product."""
    block = spa[spa.index("const COVERAGE_NAMES={"):]
    block = block[:block.index("};")]
    named = set(re.findall(r"(\w+):'", block))
    for weights, label in ((RUNNING_WEIGHTS, "running"), (CYCLING_WEIGHTS, "cycling")):
        missing = set(weights) - named
        assert not missing, f"{label} components with no English name: {sorted(missing)}"
