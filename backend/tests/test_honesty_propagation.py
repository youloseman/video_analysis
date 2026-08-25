"""Wave 2 of the biomechanics audit: honesty rules the family already keeps,
enforced in the places that had quietly opted out.

Each test here failed before its fix:
- one guessed cadence was judged by three modules and only the scorer
  abstained;
- zero detected cycles produced a phantom neutral 50 worth 12% of the rubric;
- the aero card told riders to get lower next to a mobility card saying the
  range is not there;
- NaN-blind votes pushed camera-side and anti-flip calibration to one side;
- the all-contacts fallback fabricated overstride and fed the kinogram.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer
from app.services.video_analysis.biomechanics.running_action_plan_builder import (
    build_running_action_plan,
)
from app.services.video_analysis.runner import _withhold_aero_next_zone


# --- one number, one judge (time base) -------------------------------------

def test_an_inferred_time_base_gets_no_cadence_diagnosis_or_warning():
    result = {
        "score": 70,
        "summary": {
            "cadence_spm": 150.0,           # would be "low" and "risky"
            "time_base_inferred": True,
        },
    }
    plan = build_running_action_plan(result)
    assert not any(d.metric_name == "cadence" for d in plan.diagnostics)
    assert not any("cadence" in w.lower() for w in plan.warnings)


def test_a_trusted_time_base_still_diagnoses_cadence():
    result = {"score": 70, "summary": {"cadence_spm": 150.0}}
    plan = build_running_action_plan(result)
    assert any(d.metric_name == "cadence" for d in plan.diagnostics)
    assert any("cadence" in w.lower() for w in plan.warnings)


# --- phantom neutral 50 ----------------------------------------------------

def test_zero_cycles_reports_no_stability_not_a_neutral_50():
    from app.services.video_analysis.biomechanics.phase_portrait import (
        compute_phase_portraits,
    )

    out = compute_phase_portraits({}, [], "run", camera_side="left")
    assert out["overall_stability_score"] is None


def test_the_scorer_skips_an_absent_stability_score():
    from app.services.video_analysis.biomechanics.technique_scorer import (
        score_analysis,
    )

    scoring = score_analysis(
        sport_type="run",
        summary={"biomechanics": {
            "phase_portraits": {"overall_stability_score": None},
        }},
        angle_stats={},
    )
    assert "phase_stability" not in (scoring.get("score_breakdown") or {})


# --- aero vs mobility ------------------------------------------------------

def test_next_zone_is_withheld_when_mobility_says_no_range():
    summary = {"aero_estimate": {"zone_label": "Z3", "next_zone": {"x": 1}}}
    _withhold_aero_next_zone(summary, {"within": False})
    assert summary["aero_estimate"]["next_zone"] is None
    assert summary["aero_estimate"]["next_zone_withheld"] == "mobility_limited"


def test_next_zone_survives_when_mobility_is_fine_or_absent():
    for fit in ({"within": True}, None, {}):
        summary = {"aero_estimate": {"next_zone": {"x": 1}}}
        _withhold_aero_next_zone(summary, fit)
        assert summary["aero_estimate"]["next_zone"] == {"x": 1}


# --- NaN-blind votes -------------------------------------------------------

def _lm(z=0.0):
    return SimpleNamespace(x=0.5, y=0.5, z=z, visibility=0.9)


def test_a_frame_that_cannot_see_the_body_does_not_vote():
    an = CyclingAnalyzer(fps=30.0)
    wl = [_lm() for _ in range(33)]
    wl[11] = _lm(z=math.nan)
    assert an.detect_camera_side(wl) is None
    wl_ok = [_lm(z=0.1) for _ in range(33)]
    for i in (11, 23):
        wl_ok[i] = _lm(z=-0.1)
    assert an.detect_camera_side(wl_ok) == "left"


def test_flip_calibration_needs_a_quorum_of_readable_frames():
    from app.services.video_analysis.biomechanics.landmark_stabilizer import (
        _fix_flips,
    )

    def frame(zl, zr):
        lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9)
               for _ in range(33)]
        lms[23] = SimpleNamespace(x=0.5, y=0.5, z=zl, visibility=0.9)
        lms[24] = SimpleNamespace(x=0.5, y=0.5, z=zr, visibility=0.9)
        return {"normalized_landmarks": list(lms), "world_landmarks": lms}

    # Calibration window: 5 unreadable frames + 5 unanimous "left closer".
    # The old denominator (the whole window) read that as 5 > 5 -> False and
    # then "corrected" every left-closer frame with a full-body swap.
    frames = [frame(math.nan, math.nan) for _ in range(5)]
    frames += [frame(-0.2, 0.2) for _ in range(45)]
    flips = _fix_flips(frames, "run")
    assert flips == 0, "a unanimous readable majority must not be overturned"

    # Zero readable calibration frames: no evidence, no flipping at all.
    frames = [frame(math.nan, math.nan) for _ in range(10)]
    frames += [frame(-0.2, 0.2) for _ in range(40)]
    assert _fix_flips(frames, "run") == 0


# --- the all-contacts fallback says so and pays for it ---------------------

def _running_analyzer_with_depths(depths):
    from app.services.video_analysis.biomechanics.running_analyzer import (
        RunningAnalyzer,
    )

    an = RunningAnalyzer(fps=30.0)
    an.frame_results = [
        SimpleNamespace(extra_metrics={"_near_foot_depth": d}) for d in depths
    ]
    an.stance_runs = lambda min_run=3: [(0, 4), (10, 14)]
    return an


def test_unreadable_depths_raise_the_unfiltered_flag():
    an = _running_analyzer_with_depths([float("nan")] * 20)
    starts = an._contact_frame_indices()
    assert starts == [0, 10], "the fallback itself is unchanged"
    assert an._contacts_unfiltered is True


def test_readable_depths_do_not_raise_the_flag():
    depths = [0.0] * 20
    for k in (0, 1, 2, 3, 4):
        depths[k] = 1.0                   # first stance clearly deepest
    an = _running_analyzer_with_depths(depths)
    an._contact_frame_indices()
    assert not getattr(an, "_contacts_unfiltered", False)


def test_kinogram_refuses_a_clip_whose_contacts_were_unfiltered():
    import tests.test_kinogram as tk
    from app.services.video_analysis import kinogram

    analyzer = tk.build_analyzer(cycles=3)
    analyzer._contacts_unfiltered = True
    assert kinogram.select_run_kinogram(analyzer) is None


# --- photo and video read the same plane -----------------------------------

def test_photo_plane_matches_the_video_analyzers_knee():
    """The photo bike branch builds image-plane landmarks the same way the
    video analyzer does; on one synthetic pose the two must read one knee."""
    from app.services.video_analysis.biomechanics.angle_calculator import (
        calculate_angle_2d,
    )

    aspect = 9 / 16
    nl = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9)
          for _ in range(33)]
    nl[24] = SimpleNamespace(x=0.52, y=0.45, z=0.0, visibility=0.9)
    nl[26] = SimpleNamespace(x=0.60, y=0.60, z=0.0, visibility=0.9)
    nl[28] = SimpleNamespace(x=0.55, y=0.78, z=0.0, visibility=0.9)

    an = CyclingAnalyzer(fps=30.0, frame_aspect=aspect)
    an._near_side = "right"
    video_knee = an.analyze_frame(nl, nl, 0.0).angles["right_knee"]

    pl = [SimpleNamespace(x=lm.x * aspect, y=lm.y, z=0.0,
                          visibility=lm.visibility) for lm in nl]
    photo_knee, _ = calculate_angle_2d(pl, 24, 26, 28)

    assert abs(video_knee - photo_knee) < 1e-6
    assert not np.isnan(video_knee)


# --- scoring does not convict on instrument spread -------------------------

def test_score_in_range_forgives_the_instruments_own_spread():
    """The same fit filmed twice landed 99 vs 86 because one clip sat 3 deg
    outside a 7-deg band and the slope charged from the very first degree.
    Within +/-2 deg of the band -- the honest stroke-to-stroke spread on
    clean fixtures -- the score stays full; beyond it the slope is
    unchanged, so real faults still cost what they cost."""
    from app.services.video_analysis.biomechanics.technique_scorer import (
        score_in_range,
    )

    assert score_in_range(146.9, 138, 145) == 100.0     # within tolerance
    assert score_in_range(143.0, 138, 145) == 100.0     # in band
    beyond = score_in_range(150.0, 138, 145)            # 3 deg past tolerance
    assert 75.0 < beyond < 85.0
    far = score_in_range(160.0, 138, 145)
    assert far < beyond, "the slope beyond the tolerance is intact"
