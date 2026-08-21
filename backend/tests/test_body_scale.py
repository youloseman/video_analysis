"""Centimetres need one known real length. This is the test that it is the right one.

A side view measures everything as a fraction of the picture. Every metric
reported in centimetres rests on a single assumed length, and until now that
assumption was a 0.45 m torso for everyone -- which is a person about 156 cm
tall. These tests pin the new behaviour: told a height, model the torso from it
(Winter 2009); not told, fall back and SAY so, so the reading can be read with
the right amount of trust.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics.running_analyzer import (
    DEFAULT_TORSO_M,
    PLAUSIBLE_HEIGHT_CM,
    TORSO_FRACTION_OF_HEIGHT,
    RunningAnalyzer,
)

# The torso as MediaPipe would see it on a well-framed clip: shoulders and hips
# a fixed fraction of the frame apart.
TORSO_NORM = 0.20


def analyzer_with(height_cm=None, torso_norm=TORSO_NORM, frames=30):
    an = RunningAnalyzer(fps=50.0, height_cm=height_cm)
    for _ in range(frames):
        an._estimate_body_scale(_landmarks(torso_norm))
    return an


def _landmarks(torso_norm: float):
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]
    for i in (11, 12):
        lm[i] = SimpleNamespace(x=0.5, y=0.40, z=0.0, visibility=0.9)
    for i in (23, 24):
        lm[i] = SimpleNamespace(x=0.5, y=0.40 + torso_norm, z=0.0, visibility=0.9)
    return lm


# --- the scale itself ------------------------------------------------------

def test_a_stated_height_models_the_torso_from_it():
    an = analyzer_with(height_cm=180)
    an._lock_body_scale()
    expected_torso = TORSO_FRACTION_OF_HEIGHT * 1.80
    assert an._pixel_to_meter == pytest.approx(expected_torso / TORSO_NORM)
    assert an._body_scale_source == "athlete_height"


def test_no_height_falls_back_to_the_population_torso_and_says_so():
    an = analyzer_with(height_cm=None)
    an._lock_body_scale()
    assert an._pixel_to_meter == pytest.approx(DEFAULT_TORSO_M / TORSO_NORM)
    assert an._body_scale_source == "population_average"


def test_the_old_constant_is_a_156_cm_person():
    """Worth pinning, because it is the size of the bias everyone carried:
    the fallback is not neutral, it is short."""
    implied_height_m = DEFAULT_TORSO_M / TORSO_FRACTION_OF_HEIGHT
    assert implied_height_m == pytest.approx(1.56, abs=0.02)


def test_a_taller_athlete_reads_bigger_centimetres_than_the_average_assumed():
    """The whole point. Same clip, same pixels -- the assumption was costing a
    180 cm runner about 13% of every centimetre reading."""
    told = analyzer_with(height_cm=180)
    told._lock_body_scale()
    assumed = analyzer_with(height_cm=None)
    assumed._lock_body_scale()
    ratio = told._pixel_to_meter / assumed._pixel_to_meter
    assert ratio == pytest.approx(1.152, abs=0.01)


def test_the_crossover_is_at_the_implied_height_not_at_a_typical_one():
    """The fallback is not a neutral average, and this is where that bites:
    an athlete has to be under ~156 cm before the old assumption was reading
    them HIGH. Every adult above that was being under-reported."""
    assumed = analyzer_with(height_cm=None)
    assumed._lock_body_scale()

    shorter = analyzer_with(height_cm=145)
    shorter._lock_body_scale()
    assert shorter._pixel_to_meter < assumed._pixel_to_meter

    taller = analyzer_with(height_cm=165)
    taller._lock_body_scale()
    assert taller._pixel_to_meter > assumed._pixel_to_meter


@pytest.mark.parametrize("bad", [0, 5.9, 71, 1.8, 119, 231, 400, -180, None])
def test_a_height_in_the_wrong_units_is_treated_as_not_told(bad):
    """5.9 (feet), 71 (inches), 1.8 (metres) all look like numbers and would
    each silently rescale every reading."""
    an = RunningAnalyzer(fps=50.0, height_cm=bad)
    assert an.height_cm is None


@pytest.mark.parametrize("ok", [120, 155.5, 180, 230])
def test_a_plausible_height_is_kept(ok):
    assert RunningAnalyzer(fps=50.0, height_cm=ok).height_cm == float(ok)


def test_the_bounds_are_the_ones_the_api_advertises():
    assert PLAUSIBLE_HEIGHT_CM == (120.0, 230.0)


def test_locking_is_idempotent():
    an = analyzer_with(height_cm=180)
    an._lock_body_scale()
    first = an._pixel_to_meter
    an.height_cm = 120.0                 # a later change must not re-scale
    an._lock_body_scale()
    assert an._pixel_to_meter == first


def test_a_clip_with_no_usable_torso_says_so_rather_than_guessing_quietly():
    an = RunningAnalyzer(fps=50.0, height_cm=180)
    an._lock_body_scale()
    assert an._body_scale_source == "no_samples"


def test_a_degenerate_torso_sample_is_not_collected():
    """A shoulder and hip on top of each other is broken tracking, and would
    divide the scale by nearly zero."""
    an = RunningAnalyzer(fps=50.0)
    an._estimate_body_scale(_landmarks(0.001))
    an._estimate_body_scale(_landmarks(float("nan")))
    assert an._torso_norm_samples == []


# --- reaching the report ---------------------------------------------------

def summary_of(height_cm):
    """Drive the real vertical-oscillation path with a bouncing hip."""
    an = RunningAnalyzer(fps=50.0, height_cm=height_cm)
    for i in range(120):
        an._estimate_body_scale(_landmarks(TORSO_NORM))
        an.norm_hip_y_history.append(0.5 + 0.02 * math.sin(i / 4.0))
        an.norm_hip_y_timestamps.append(i / 50.0 * 1000.0)
    return an


def test_the_reading_carries_which_length_it_was_scaled_from():
    an = summary_of(180)
    osc = an.compute_vertical_oscillation()
    assert osc > 0
    assert an._body_scale_source == "athlete_height"


def test_the_same_bounce_reads_higher_for_a_taller_athlete():
    tall = summary_of(190).compute_vertical_oscillation()
    short = summary_of(160).compute_vertical_oscillation()
    assert tall > short


def test_an_analyzer_built_the_old_way_still_works():
    """Every existing caller passes no height; none of them may break."""
    an = analyzer_with()
    an._lock_body_scale()
    assert an._pixel_to_meter > 0
