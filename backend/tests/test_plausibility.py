"""Numbers no body can produce leave the report; a sat-up clip is not graded as aero.

From the DB audit of 2026-10-15: knee at the bottom 172, at the top 34,
shoulder 150, run cadence 118 and 220 -- almost all on clips framed "tiny".
And on the triathlon position, elbows of 150-172 with a torso of 61: measured
right, the rider on the base bar, graded against aero bands to a 30.
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics import plausibility as P


def _bike(**over):
    s = {"knee_at_bdc": 146.0, "knee_at_tdc": 70.0, "hip_at_tdc": 52.0, "shoulder_angle_avg": 85.0,
         "elbow_angle_avg": 92.0, "trunk_angle_avg": 21.0, "forearm_tilt_avg": 17.0,
         "saddle_height_assessment": "optimal"}
    s.update(over)
    return s


def test_a_clean_clip_loses_nothing():
    s, stats, hist = _bike(), {"right_shoulder": {}}, {"right_shoulder": [1]}
    assert P.drop_implausible("bike", "triathlon", s, stats, hist, "right") == []
    assert s == _bike() and "right_shoulder" in stats
    assert P.position_mismatch("triathlon", s) is None


def test_impossible_readings_leave_everywhere_they_would_show():
    s = _bike(knee_at_bdc=172.7, knee_at_tdc=33.9, shoulder_angle_avg=149.6)
    stats = {"left_shoulder": {"mean": 149.6}, "left_knee": {"mean": 100}}
    hist = {"left_shoulder": [149.6], "left_knee": [100]}
    dropped = P.drop_implausible("bike", "triathlon", s, stats, hist, "left")
    assert {d["field"] for d in dropped} == {"knee_at_bdc", "knee_at_tdc", "shoulder_angle_avg"}
    assert s["knee_at_bdc"] is None and s["knee_at_tdc"] is None and s["shoulder_angle_avg"] is None
    assert "saddle_height_assessment" not in s, "the saddle verdict was built on the dropped knee"
    assert "left_shoulder" not in stats and "left_shoulder" not in hist, "no table row, no label"
    assert "left_knee" in hist, "the knee series stays: the keyframe and kinogram time off it"
    w = P.implausible_warning(dropped)
    assert "knee at the bottom of the stroke 172.7" in w and "Film closer" in w


def test_a_straight_road_arm_is_not_impossible():
    """Elbows of 150-165 are what straight arms on the hoods measure."""
    s = _bike(elbow_angle_avg=165.1, trunk_angle_avg=41.0, forearm_tilt_avg=-25.5)
    assert P.drop_implausible("bike", "road_hoods", s, {}, {}, "left") == []
    assert P.position_mismatch("road_hoods", s) is None


def test_sitting_up_on_aero_bars_withholds_the_score_not_the_numbers():
    s = _bike(elbow_angle_avg=172.2, trunk_angle_avg=61.0)
    assert P.drop_implausible("bike", "triathlon", s, {}, {}, "left") == []
    msg = P.position_mismatch("triathlon", s)
    assert msg and "torso at 61°" in msg and "elbows at 172°" in msg and "base bar" in msg
    assert s["elbow_angle_avg"] == 172.2, "measured right: kept"


def test_run_cadence_and_lean_have_limits_too():
    s = {"cadence_spm": 118.4, "trunk_lean_avg": 61.5, "ground_contact_ms": 101.0}
    hist = {"trunk_lean": [61.5]}
    dropped = P.drop_implausible("run", None, s, {"trunk_lean": {}}, hist, "right")
    assert {d["field"] for d in dropped} == {"cadence_spm", "trunk_lean_avg"}
    assert s["ground_contact_ms"] == 101.0 and "trunk_lean" not in hist
    assert P.drop_implausible("run", None, {"cadence_spm": 182.0}, {}, {}, "right") == []


def test_a_small_rider_is_told_at_the_top_free_or_paid():
    """70 % of bike clips in the audit were framed small; the banner by the
    score says so, and the free result carries what it needs."""
    from pathlib import Path

    from app.services import result_gating as RG
    res = {"sport_specific_metrics": {"tracking_stability": {"framing": {
        "verdict": "small", "subject_height_frac": 0.2757, "subject_height_px": 353}}}}
    assert RG.quality_block(res)["framing"] == {"verdict": "small", "subject_height_frac": 0.2757}
    spa = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="framingWarn"' in spa and "Accuracy reduced: you\u2019re small in the frame" in spa
    assert "(r.quality&&r.quality.framing)" in spa
