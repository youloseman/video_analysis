"""Pedalling style, read from the ankle angle's distribution.

The ankle is the noisiest landmark on a side view - reading crank position off
it was measured and abandoned - so the guard rails matter more than the
classifier. A style label is a specific-sounding claim, and the worst version
of it is a confident one drawn from a foot the model could barely see.
"""
from __future__ import annotations

import math

from app.services.video_analysis.biomechanics.cycling_analyzer import (
    ANKLE_HEEL_DOWN_MAX_DEG,
    ANKLE_TOE_DOWN_MIN_DEG,
    CyclingAnalyzer,
    _shares_to_percent,
)


def _analyzer(series: list[float], side: str = "left") -> CyclingAnalyzer:
    """An analyzer carrying nothing but one ankle series."""
    a = object.__new__(CyclingAnalyzer)
    a.angle_history = {f"{side}_ankle": series}
    return a


def test_a_steady_neutral_foot_is_called_neutral():
    st = _analyzer([108.0 + (i % 5) for i in range(120)])._pedaling_style("left")
    assert st["style"] == "neutral"
    assert st["time_in_zone"]["neutral"] == 100
    assert st["readable_pct"] == 100


def test_a_pointed_toe_is_called_toe_down():
    st = _analyzer([135.0 + (i % 4) for i in range(120)])._pedaling_style("left")
    assert st["style"] == "toe_down"
    assert st["time_in_zone"]["toe_down"] >= 90
    assert st["median_deg"] > ANKLE_TOE_DOWN_MIN_DEG


def test_a_dropped_heel_is_called_heel_down():
    st = _analyzer([85.0 + (i % 4) for i in range(120)])._pedaling_style("left")
    assert st["style"] == "heel_down"
    assert st["median_deg"] < ANKLE_HEEL_DOWN_MAX_DEG


def test_a_foot_that_works_through_the_stroke_is_ankling():
    """Neutral on average, but swinging 40 deg across the stroke."""
    series = [109.0 + 20.0 * math.sin(i / 6.0) for i in range(240)]
    st = _analyzer(series)._pedaling_style("left")
    assert st["style"] == "ankling"
    assert st["spread_deg"] >= 25


def test_an_unreadable_foot_gets_no_label_at_all():
    """Mostly NaN: the honest output is nothing, not a guess."""
    series = [float("nan")] * 90 + [108.0] * 30
    assert _analyzer(series)._pedaling_style("left") is None


def test_implausible_angles_do_not_count_as_readable():
    """Values outside the physical envelope are tracking noise, not a foot."""
    series = [400.0] * 90 + [108.0] * 30        # 400 deg is not an ankle
    assert _analyzer(series)._pedaling_style("left") is None


def test_a_short_clip_gets_no_label():
    assert _analyzer([108.0] * 10)._pedaling_style("left") is None


def test_nothing_at_all_is_not_an_error():
    assert _analyzer([])._pedaling_style("left") is None
    assert _analyzer([108.0] * 120)._pedaling_style("right") is None   # other side


def test_the_zone_split_always_adds_to_a_hundred():
    """Three independently rounded shares read 99 often enough to be noticed."""
    for shares in (
        {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
        {"a": 0.155, "b": 0.155, "c": 0.69},
        {"a": 0.005, "b": 0.005, "c": 0.99},
    ):
        pct = _shares_to_percent(shares)
        assert sum(pct.values()) == 100, pct


def test_the_split_of_a_real_classification_also_adds_up():
    st = _analyzer([109.0 + 18.0 * math.sin(i / 5.0) for i in range(240)])._pedaling_style("left")
    assert sum(st["time_in_zone"].values()) == 100
