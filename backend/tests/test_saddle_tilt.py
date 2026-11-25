"""Saddle tilt from two points on the saddle, levelled on the wheel axles.

Fitters start level and stop at 2-3 deg nose down; more nose down usually
makes up for a saddle that is too high (Burt 2014). The athlete marks the nose
and the back of the saddle in the same editor as the crank axle.
"""

import math

import pytest

from app.services.video_analysis.biomechanics.hip_over_bb import (
    hip_over_bb,
    normalize_bike_points,
    report_says_knee_high,
    saddle_tilt,
)


def _line(deg, facing=1, cx=0.5, cy=0.4, half=0.08):
    """Saddle back -> nose at ``deg`` (+ nose up), in a square picture."""
    r = math.radians(deg)
    dx, dy = facing * half * math.cos(r), half * math.sin(r)
    return {"ST": (cx - dx, cy + dy), "SN": (cx + dx, cy - dy)}


@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("deg", [-6.0, -2.0, 0.0, 3.0])
def test_reads_the_angle_whichever_way_the_bike_faces(deg, facing):
    t = saddle_tilt(_line(deg, facing), aspect=1.0)
    assert t["deg"] == pytest.approx(deg, abs=0.05)
    assert t["ref"] == "frame"


def test_the_wheel_axles_are_the_horizontal():
    # The phone (or the trainer) is tipped 3 deg nose down; the saddle is level
    # on the bike, so it reads level against the axles and -3 against the frame.
    pts = _line(-3.0)
    tipped = math.radians(-3.0)
    fa = (0.5 + 0.3 * math.cos(tipped), 0.6 - 0.3 * math.sin(tipped))
    ra = (0.5 - 0.3 * math.cos(tipped), 0.6 + 0.3 * math.sin(tipped))
    t = saddle_tilt({**pts, "FA": fa, "RA": ra}, aspect=1.0)
    assert t["ref"] == "axles"
    assert t["deg"] == pytest.approx(0.0, abs=0.05)
    assert saddle_tilt(pts, aspect=1.0)["deg"] == pytest.approx(-3.0, abs=0.05)


def test_aspect_is_applied():
    # A 16:9 picture: normalized x is wider than it looks.
    pts = {"ST": (0.40, 0.40), "SN": (0.50, 0.40 + 0.1 * 16 / 9 * math.tan(math.radians(5)))}
    assert saddle_tilt(pts, aspect=16 / 9)["deg"] == pytest.approx(-5.0, abs=0.05)


def test_verdicts_and_notes():
    down = saddle_tilt(_line(-6.0), aspect=1.0)
    assert down["verdict"] == "nose_down" and "Nose down 6°" in down["note"]
    assert "too high" in down["note"]
    linked = saddle_tilt(_line(-6.0), aspect=1.0, knee_bdc_high=True)
    assert "Lower the saddle first" in linked["note"]
    assert saddle_tilt(_line(-3.0), aspect=1.0)["verdict"] == "level"
    up = saddle_tilt(_line(3.0), aspect=1.0)
    assert up["verdict"] == "nose_up" and "Nose up 3°" in up["note"]
    for t in (down, linked, up):
        assert "—" not in t["note"] and not t["note"].endswith(".")


def test_precision_follows_the_saddle_length():
    short = saddle_tilt(_line(0.0, half=0.03), aspect=1.0)["plus_minus_deg"]
    long = saddle_tilt(_line(0.0, half=0.12), aspect=1.0)["plus_minus_deg"]
    assert short > long >= 1.0


def test_no_saddle_points_no_reading():
    assert saddle_tilt({"C": (0.5, 0.7)}, aspect=1.0) is None


def test_a_saddle_steeper_than_any_saddle_is_refused():
    with pytest.raises(ValueError, match="saddle"):
        saddle_tilt(_line(-40.0), aspect=1.0)


def test_points_are_validated_as_a_pair():
    with pytest.raises(ValueError, match="both ends of the saddle"):
        normalize_bike_points({"C": [0.5, 0.7], "SN": [0.55, 0.4]})
    with pytest.raises(ValueError, match="on top of each other"):
        normalize_bike_points({"C": [0.5, 0.7], "SN": [0.5, 0.4], "ST": [0.505, 0.4]})
    pts = normalize_bike_points({"C": [0.5, 0.7], "SN": [0.56, 0.41], "ST": [0.44, 0.40]})
    assert set(pts) == {"C", "SN", "ST"}


def test_rides_on_the_crank_reading():
    pts = normalize_bike_points({"C": [0.5, 0.75], "SN": [0.56, 0.42], "ST": [0.44, 0.40]})
    block = hip_over_bb((0.47, 0.38), pts, aspect=1.0, shoulder=(0.62, 0.2))
    assert block["saddle_tilt"]["verdict"] == "nose_down"
    assert hip_over_bb((0.47, 0.38), {"C": (0.5, 0.75)}, aspect=1.0)["saddle_tilt"] is None


def test_the_report_says_the_knee_is_high():
    assert report_says_knee_high({"fit_plan": {"diagnostics": [{"action": "lower_saddle"}]}})
    assert not report_says_knee_high({"fit_plan": {"diagnostics": [{"action": "raise_saddle"}]}})
    photo = {"pedal_phase": "bdc", "angles_with_context": {"knee": {
        "value": 158, "optimal_min": 140, "optimal_max": 150, "status": "needs_work"}}}
    assert report_says_knee_high(photo)
    assert not report_says_knee_high({**photo, "pedal_phase": "tdc"})
    assert not report_says_knee_high(None)


def test_the_pdf_prints_the_saddle_tilt():
    from app.services.report_pdf import render_report_html

    tilt = saddle_tilt(_line(-6.0), aspect=1.0, knee_bdc_high=True)
    hob = {"angle_deg": 82.0, "aero": False, "c_source": "user", "saddle_tilt": tilt}
    res = {"sport_type": "bike", "technique_score": 80, "cycling_position": "road_hoods",
           "camera_side": "right", "hip_over_bb": hob, "fit_plan": {"diagnostics": []},
           "sport_specific_metrics": {"capture_report": {"checks": []}, "hip_over_bb": hob}}
    html = render_report_html(res, {})
    assert "Saddle tilt: nose down 6°" in html and "Lower the saddle first" in html
    no_tilt = {**hob, "saddle_tilt": None}
    html = render_report_html({**res, "hip_over_bb": no_tilt,
                               "sport_specific_metrics": {"capture_report": {"checks": []},
                                                          "hip_over_bb": no_tilt}}, {})
    assert "Saddle tilt" not in html


def test_the_editor_marks_and_sends_the_saddle():
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "HOB_SN=204, HOB_ST=205" in html and "204:'SN'" in html and "205:'ST'" in html
    # Both or neither go to the server, like the wheel axles.
    assert "if(out.SN&&!out.ST) delete out.SN; if(out.ST&&!out.SN) delete out.ST;" in html
    # Every list of the bike points is the one list.
    assert "[HOB_C,HOB_P,HOB_FA,HOB_RA]" not in html
    assert "function hobSaddleHtml(" in html and "hobSaddleHtml(h.saddle_tilt)" in html
