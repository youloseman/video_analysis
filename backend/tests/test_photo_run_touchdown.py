"""A running photo at touchdown, as the athlete and the coach get it.

The readings themselves are pinned in test_photo_run_contact.py; this is what
is done with them: the knee flagged where the two knees cross, and the
contact block reaching the coach's prompt and the "worth a look" checklist
the same way an out-of-band angle does.
"""

from __future__ import annotations

from app.services.video_analysis.llm_recommendations import (
    _build_photo_prompt,
    _photo_concerns,
    _run_contact_prompt_line,
)
from app.services.video_analysis.photo_analyzer import _RUN_KNEES_CROSS, _run_knees_cross


def test_knees_closer_than_the_threshold_cross():
    assert _run_knees_cross({"knee_gap": _RUN_KNEES_CROSS - 0.01})
    assert not _run_knees_cross({"knee_gap": _RUN_KNEES_CROSS + 0.01})


def test_unread_legs_are_not_called_crossed():
    assert not _run_knees_cross({"knee_gap": None})
    assert not _run_knees_cross(None)


def test_the_threshold_splits_the_study_frames():
    """runphoto2026: midstance frames 0.09-0.19, touchdowns 0.27 and up."""
    assert 0.19 < _RUN_KNEES_CROSS <= 0.27


def _touchdown(status="needs_work", shin=-19.1):
    return {
        "measured": True, "overstride_ratio": 0.39, "overstride_note": "Shown, not scored",
        "shin_angle": shin, "shin_band": [-10.0, 20.0], "shin_status": status,
        "foot_strike": "heel",
    }


def _result(contact, knee=None):
    return {
        "sport": "run", "gait_phase": "initial_contact",
        "score": {"overall_score": 70, "grade": "Fair"},
        "angles_with_context": {
            "knee": knee or {"value": 165.3, "optimal_min": 80, "optimal_max": 175,
                             "status": "optimal", "label": "Knee angle"},
        },
        "contact": contact,
    }


def test_a_reaching_landing_is_a_material_problem_for_the_coach():
    prompt = _build_photo_prompt("run", _result(_touchdown()))
    assert "At touchdown" in prompt
    material = next(ln for ln in prompt.splitlines() if ln.startswith("Material problems"))
    assert "Shin at touchdown (-19" in material and "NONE" not in material
    assert "Overstride" not in material


def test_a_landing_just_past_the_band_is_borderline_not_material():
    prompt = _build_photo_prompt("run", _result(_touchdown("acceptable", -11.7)))
    assert "Material problems: NONE" in prompt
    assert "Shin at touchdown (-12" in next(ln for ln in prompt.splitlines() if ln.startswith("Just outside"))


def test_the_shin_joins_the_checklist_like_an_angle():
    got = _photo_concerns("run", _result(_touchdown()))
    shin = [c for c in got if c["field"] == "shin_angle"]
    assert len(shin) == 1 and shin[0]["material"] and shin[0]["side"] == "low"
    assert "in front of the knee" in shin[0]["hint"]
    assert not [c for c in got if c["field"] == "overstride_ratio"]


def test_the_coach_is_told_the_reach_is_not_a_fault():
    line = _run_contact_prompt_line(_touchdown())
    assert "NOT scored" in line and "scored)" in line.split("overstride")[0]


def test_a_landing_in_range_adds_nothing():
    assert _photo_concerns("run", _result(_touchdown("optimal", -4.0))) == []


def test_a_frame_away_from_touchdown_says_nothing_about_it():
    assert _run_contact_prompt_line({"measured": False, "note": "x"}) is None
    prompt = _build_photo_prompt("run", _result({"measured": False, "note": "x"}))
    assert "At touchdown" not in prompt


def test_overstride_is_not_called_degrees():
    assert "NOT degrees" in _run_contact_prompt_line(_touchdown())


def test_a_crossed_knee_is_named_to_the_coach():
    knee = {"value": 119.3, "optimal_min": 150, "optimal_max": 180, "status": "needs_work",
            "label": "Knee angle", "check": True}
    prompt = _build_photo_prompt("run", _result({"measured": False}, knee))
    assert "knees overlap" in prompt


def test_the_checklist_writes_a_signed_band_as_one():
    """"-11.7° vs -10-20°" read as a typo in the Kiptum report."""
    from app.services.video_analysis.concern_hints import worth_a_look

    text = worth_a_look(_photo_concerns("run", _result(_touchdown("acceptable", -11.7))))
    assert "-11.7° vs -10° to +20°" in text
    assert "-10-20" not in text


def test_an_edge_shin_is_not_a_braking_free_touchdown():
    """Kiptum: "a low-braking touchdown ... Nothing to fix" over an amber shin
    whose own line says the foot lands in front of the knee and brakes."""
    prompt = _build_photo_prompt("run", _result(_touchdown("acceptable", -11.7)))
    assert "never call this touchdown low-braking" in prompt
    assert "Do NOT describe these as ideal" in prompt
    clean = _build_photo_prompt("run", _result(_touchdown("optimal", -4.0)))
    assert "low-braking" not in clean and "Do NOT describe these" not in clean


def test_the_shin_is_on_the_picture_in_its_rows_colour(monkeypatch):
    import pytest

    cv2 = pytest.importorskip("cv2")
    from types import SimpleNamespace

    import numpy as np

    from app.services.video_analysis import overlay_style
    from app.services.video_analysis import photo_analyzer as pa

    staged = []
    real = overlay_style.ChipLayer.metric_chip

    def spy(self, anchor, label, value, status, **kw):
        staged.append((label, value, status))
        return real(self, anchor, label, value, status, **kw)

    monkeypatch.setattr(overlay_style.ChipLayer, "metric_chip", spy)
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.2) for _ in range(33)]
    for i, (x, y) in {
        0: (.40, .09), 12: (.47, .24), 14: (.50, .36), 16: (.42, .42), 24: (.50, .50),
        26: (.44, .66), 28: (.40, .82), 30: (.42, .84), 32: (.36, .85),
    }.items():
        lms[i] = SimpleNamespace(x=x, y=y, z=0.0, visibility=0.95)
    angles = {"knee": 158.0, "hip": 140.0, "trunk": 6.0}
    ranges = pa._get_running_optimal_ranges("initial_contact")
    photo = np.full((1400, 900, 3), 120, np.uint8)

    def draw(**kw):
        staged.clear()
        pa._generate_photo_thumbnail(
            cv2, photo, lms, angles, pa._score_photo_angles(angles, ranges, "run"),
            "right", "run", pa._build_arc_triplets("run", "right"),
            optimal_ranges=ranges, **kw)
        return [s for s in staged if s[0] == "SHIN"]

    assert draw(shin=_touchdown("acceptable", -11.7)) == [("SHIN", "-12°", "warn")]
    assert draw(shin=_touchdown("acceptable", -11.7), hide_angle_values=True)[0][1] == "LOCKED"
    assert draw() == []
