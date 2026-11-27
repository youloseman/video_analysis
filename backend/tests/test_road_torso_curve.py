"""The road torso is a slope, not a cliff, and the head cue says lower the head, not the eyes.

2026-11-26, from the QEF Cycling Science CFD videos (upload/research/qef):
- the zones jumped from 0.87 to 1.00 at 38.0 deg, so a rider at 39 deg on the hoods, inside the
  measured 32-50 band, was told 33 deg was worth 13 % of drag and ~25 W. QEF's one rider on the
  hoods: 30 -> 38 deg cost 13.4 W at 39.6 km/h, about 0.9 % a degree;
- looking at the ground with the neck relaxed adds drag (~1.2 %), so the cue to drop the helmet
  must not read as "look down".
"""
from pathlib import Path

import pytest

from app.services.video_analysis import concern_hints
from app.services.video_analysis import photo_analyzer
from app.services.video_analysis.biomechanics import aero_estimator as AE

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_no_cliff_at_38_degrees():
    assert abs(AE._rel_for_trunk(37.9) - AE._rel_for_trunk(38.1)) < 0.005


def test_the_road_slope_is_under_a_percent_a_degree():
    per_degree = (AE._rel_for_trunk(48.0) - AE._rel_for_trunk(33.0)) / 15.0
    assert per_degree == pytest.approx(0.13 / 15.0)
    assert AE._rel_for_trunk(48.0) == AE._rel_for_trunk(70.0) == 1.0


def test_a_hoods_rider_at_39_is_not_promised_13_percent():
    r = AE.estimate_aero(39.0, "road_hoods", optimal_trunk_band=(32, 50))
    nz = r["next_zone"]
    assert nz["target_trunk_angle"] == 33.0
    assert 4.0 < nz["drag_reduction_pct"] < 7.0, nz
    assert nz["approx_watts_saved"] <= 15
    assert "QEF" in r["source"]


def test_the_measured_tt_points_are_unchanged():
    assert AE._rel_for_trunk(24.0) == pytest.approx(0.80)
    assert AE._rel_for_trunk(33.0) == pytest.approx(0.87)


def test_the_app_fallback_curve_matches_the_server():
    assert [list(p) for p in AE._TT_REL_POINTS][-1] == [48.0, 1.0]
    assert "[33,0.87],[48,1]]" in SPA


def test_lower_the_head_not_the_eyes():
    lever = AE.helmet_lever(28.0, False, "triathlon")
    assert lever["status"] == "available"
    assert "not the eyes" in lever["cue"]
    assert "only the eyes" in concern_hints._HINTS[("helmet_above_back", "high")]
    assert "not the eyes" in photo_analyzer._HELMET_HIGH_NOTE
