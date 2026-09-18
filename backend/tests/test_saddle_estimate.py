"""The saddle change in millimetres (``saddle_estimate``).

Pinned: the sign follows the knee (bent = raise, stretched = lower); the
magnitude is the fitters' rule of thumb, a few millimetres per degree; in
band it is "none"; the interval carries the 3 deg floor and widens on a
drive-side clip; it refuses without a ruler; the default crank is named as
assumed and the rider's crank scales the answer.
"""
from __future__ import annotations

import pytest

from app.services.video_analysis.biomechanics.saddle_estimate import (
    DEFAULT_CRANK_MM,
    estimate_saddle_change,
    format_amount,
)

# An adult leg in crank radii: thigh 430 mm, shin 400 mm on 172.5 mm cranks.
GEOM = {"camera_side": "left", "thigh": 2.49, "shin": 2.32, "torso": 3.0,
        "chord_bdc": 4.5, "chord_sd": 0.02, "revolutions": 6, "measured_frames": 300}
BAND = (138.0, 145.0)


def test_a_bent_knee_means_raise_and_a_stretched_one_lower():
    low = estimate_saddle_change(GEOM, 128.0, BAND)
    high = estimate_saddle_change(GEOM, 152.0, BAND)
    assert low["direction"] == "raise" and low["delta_mm"] > 0
    assert high["direction"] == "lower" and high["delta_mm"] < 0
    assert low["crank_source"] == "assumed" and low["crank_length_mm"] == DEFAULT_CRANK_MM


def test_the_magnitude_is_a_few_millimetres_per_degree():
    # 13.5 deg short of mid-band (141.5): fitters say ~2.5-3.5 mm/deg here.
    est = estimate_saddle_change(GEOM, 128.0, BAND)
    assert 2.0 <= est["mm_per_deg"] <= 4.0
    assert 25 <= est["delta_mm"] <= 50
    assert est["ci_mm"] >= 3 * 2.0          # the 3 deg floor, through the lever


def test_in_band_is_none_and_the_interval_still_reports():
    est = estimate_saddle_change(GEOM, 141.0, BAND)
    assert est["direction"] == "none" and est["in_band"] is True
    assert abs(est["delta_mm"]) < 3 and est["to_band_mm"] == 0.0


def test_inside_the_verdicts_margin_there_is_no_direction():
    # 136 is out of band but within the 5 deg the verdict calls "acceptable":
    # the distances are reported, the direction is not.
    est = estimate_saddle_change(GEOM, 136.0, BAND)
    assert est["direction"] == "none" and est["in_band"] is False
    assert 0 < est["to_band_mm"] < est["delta_mm"]


def test_the_riders_crank_scales_the_answer():
    a = estimate_saddle_change(GEOM, 128.0, BAND)
    b = estimate_saddle_change(GEOM, 128.0, BAND, crank_length_mm=165.0)
    assert b["crank_source"] == "given"
    assert b["delta_mm"] == pytest.approx(a["delta_mm"] * 165.0 / DEFAULT_CRANK_MM, rel=0.02)


def test_a_drive_side_clip_widens_the_interval():
    non = estimate_saddle_change(GEOM, 128.0, BAND, camera_side="left")
    drv = estimate_saddle_change(GEOM, 128.0, BAND, camera_side="right")
    assert drv["drive_side"] and drv["ci_mm"] > non["ci_mm"]


def test_it_refuses_without_a_ruler_or_a_knee():
    assert estimate_saddle_change(None, 128.0, BAND) is None
    assert estimate_saddle_change({**GEOM, "revolutions": 1}, 128.0, BAND) is None
    assert estimate_saddle_change(GEOM, None, BAND) is None


def test_the_report_carries_the_estimate_and_scales_it_to_the_crank():
    from tests.test_analyze_from_frames import _analyze, _frames

    base = _analyze(_frames())["sport_specific_metrics"]["saddle_estimate"]
    assert base["crank_source"] == "assumed" and base["revolutions"] >= 2
    given = _analyze(_frames(), crank_length_mm=165.0)["sport_specific_metrics"]["saddle_estimate"]
    assert given["crank_source"] == "given"
    assert given["delta_mm"] == pytest.approx(base["delta_mm"] * 165.0 / DEFAULT_CRANK_MM, abs=0.2)


def test_the_plan_amount_names_the_direction_it_backs():
    est = estimate_saddle_change(GEOM, 128.0, BAND)
    assert format_amount(est, "raise").startswith("≈")
    assert format_amount(est, "lower") is None
    assert format_amount(None, "raise") is None
