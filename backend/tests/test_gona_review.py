"""Gona's second review (IMG_3088, 2026-11-11): what each fix promised."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.services.review_pdf import _lenient, metric_rows
from app.services.video_analysis.biomechanics.landmark_stabilizer import hold_bike_arms
from app.services.video_analysis.biomechanics.sport_configs import reference_bands
from app.services.video_analysis.video_visualizer import VideoVisualizer

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def _lm(x, y, vis=0.9):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


def _clip(n=60):
    """A rider whose arm does not move; at the top of every stroke the knee
    comes up under the elbow and drags the wrist up with it."""
    frames = []
    for f in range(n):
        phase = 2 * math.pi * f / 20
        knee = (0.47 - 0.05 * math.cos(phase), 0.45 + 0.12 * math.cos(phase))
        near_top = math.cos(phase) < -0.6
        wrist = (0.30, 0.20 - (0.06 if near_top else 0.0))
        lms = [_lm(0.5, 0.5) for _ in range(33)]
        lms[11], lms[13], lms[15] = _lm(0.42, 0.10), _lm(0.43, 0.32), _lm(*wrist)
        lms[25] = _lm(*knee)
        frames.append({"normalized_landmarks": lms,
                       "world_landmarks": [_lm(p.x, p.y) for p in lms],
                       "frame_width": 1920, "frame_height": 1080})
    return frames


def _elbow(fr):
    lm = fr["normalized_landmarks"]
    s, e, w = (np.array([lm[j].x * 16 / 9, lm[j].y]) for j in (11, 13, 15))
    u, v = s - e, w - e
    return math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))


def test_an_arm_on_the_bars_is_held_where_the_clean_frames_put_it():
    frames = _clip()
    raw = [_elbow(f) for f in frames]
    assert max(raw) - min(raw) > 10          # the knee drags the wrist
    diag = hold_bike_arms(frames)
    assert diag["held"] and diag["left"]["held"]
    held = [_elbow(f) for f in frames]
    assert max(held) - min(held) < 0.01
    # ...and held at the clean reading (frame 0: knee at the bottom, far from
    # the arm), not at an average of clean and dragged.
    assert abs(held[0] - raw[0]) < 0.5


def test_too_few_clean_frames_leaves_the_arm_alone():
    frames = _clip()
    for fr in frames:   # the knee sits on the elbow all clip long
        fr["normalized_landmarks"][25] = _lm(0.43, 0.30)
    before = [f["normalized_landmarks"][15].y for f in frames]
    assert not hold_bike_arms(frames)["left"]["held"]
    assert [f["normalized_landmarks"][15].y for f in frames] == before


class _Vis(VideoVisualizer):
    def __init__(self, knee, bdc, tdc):  # noqa: D107 -- just the fields the colour reads
        self.sport_type = "bike"
        self.cycling_position = "triathlon"
        self.analyzer = SimpleNamespace(angle_history={"left_knee": knee},
                                        _bdc_tdc_diag={"left": {"bdc_indices": bdc, "tdc_indices": tdc}})


def test_the_knee_is_coloured_only_at_the_ends_of_the_stroke():
    knee = [110.0] * 60
    for i in (0, 24, 48):
        knee[i] = 143.0
    for i in (12, 36):
        knee[i] = 80.0
    v = _Vis(knee, [0, 24, 48], [12, 36])
    cfg = {"key": "left_knee", "optimal": (138, 145)}
    assert v._knee_phase_status(cfg, 24) == "good"       # bottom, in its band
    assert v._knee_phase_status(cfg, 25) == "good"       # same stroke's reading
    assert v._knee_phase_status(cfg, 12) == "bad"        # top, 80 against 60-72
    assert v._knee_phase_status(cfg, 18) == "plain"      # on the way down: no verdict
    assert v._knee_phase_status({"key": "left_elbow", "optimal": (75, 110)}, 18) is None


def test_the_pelvic_ratio_has_no_band_anywhere():
    assert "pelvic_ratio" not in reference_bands("bike", "triathlon")
    # A stored report still carries the old band; the PDF ignores it.
    res = {"cycling_position": "triathlon", "sport_specific_metrics": {"pelvic_ratio": 5.0},
           "reference_bands": {"pelvic_ratio": {"lo": 2, "hi": 4}}}
    row = next(r for r in metric_rows(res) if r["field"] == "pelvic_ratio")
    assert row["band"] is None and row["status"] == "info"


def test_a_choice_the_score_forgives_is_never_red_in_the_pdf():
    assert _lenient("trunk_angle_avg", 14.0, (20, 25), "triathlon", "bad") == "warn"
    assert _lenient("shoulder_angle_avg", 70.0, (80, 105), "triathlon", "bad") == "warn"
    assert _lenient("shoulder_angle_avg", 70.0, (80, 100), "road_hoods", "bad") == "bad"
    assert _lenient("hip_at_tdc", 70.0, (45, 62), "triathlon", "bad") == "good"
    assert _lenient("elbow_angle_avg", 60.0, (75, 110), "triathlon", "bad") == "bad"


def test_the_spa_tells_a_small_file_from_a_small_rider():
    assert "Resolution of the file" in SPA and "the file is low-resolution" in SPA


def test_review_comes_before_the_coach_notes_and_the_table_notes_fold():
    assert SPA.index('id="reviewBlock"') < SPA.index('id="coachNoteBlock"')
    notes = SPA[SPA.index('id="angleNotes"'):]
    assert notes.index('id="angleStatNote"') < notes.index("</details>")


def test_every_bike_row_has_its_drawn_icon():
    from app.services.report_pdf import angle_icon
    from app.services.review_pdf import _BIKE_ROWS

    for field, *_ in _BIKE_ROWS:
        assert angle_icon(field).startswith("<img class='mt-fig' src='data:image/png;base64,"), field
