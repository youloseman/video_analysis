"""The kinogram must land on the right five frames, not five plausible ones.

Every position is defined by a geometric event -- the foot passing under the
hip, the pelvis reaching its apex, the far thigh crossing vertical -- so a
synthetic stride with those events placed at known offsets is enough to check
the finders actually find them rather than returning something in the
neighbourhood.

The stride below runs left to right, near side left, with:

    frames 0-7    ground contact   (foot passes under the hip at frame 4)
    frames 8-19   flight           (pelvis apex at 14, far thigh vertical at 17)

repeated every 20 frames.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.video_analysis import kinogram
from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis
from app.services.video_analysis.biomechanics.running_analyzer import RunningAnalyzer

FPS = 30.0
PERIOD = 20          # frames per stride
STANCE = 8           # frames 0..7 on the ground
FULL_SUPPORT_P = 4   # ankle passes under the hip here
MVP_P = 14           # pelvis apex
STRIKE_P = 17        # far thigh crosses vertical

HIP_X = 0.50
HIP_Y = 0.55


def _lm(x: float, y: float, vis: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


def _landmarks(p: int, *, far_vis: float = 0.85) -> list[SimpleNamespace]:
    """One frame of the synthetic stride, as 33 normalized landmarks."""
    # Pelvis: flat through stance, a parabolic apex through flight.
    if p < STANCE:
        hip_y = HIP_Y
    else:
        # 0 at the ends of the flight, 1 at MVP_P -> subtract to raise the hip.
        span = max(1, (PERIOD - 1) - STANCE)
        t = (p - STANCE) / span
        peak = (MVP_P - STANCE) / span
        hip_y = HIP_Y - 0.06 * max(0.0, 1.0 - ((t - peak) / max(peak, 1e-6)) ** 2)

    # Near ankle: planted through stance, so the BODY passes over it -- the
    # horizontal offset crosses zero at FULL_SUPPORT_P. Swung forward after.
    if p < STANCE:
        near_ankle_x = HIP_X + 0.06 * (FULL_SUPPORT_P - p) / FULL_SUPPORT_P
    else:
        near_ankle_x = HIP_X - 0.06 + 0.12 * (p - STANCE) / (PERIOD - STANCE)
    near_ankle_y = hip_y + 0.30 if p < STANCE else hip_y + 0.22

    # Far thigh: knee offset crosses zero (thigh vertical) at STRIKE_P.
    far_knee_dx = 0.09 * (p - STRIKE_P) / 6.0

    lms = [_lm(HIP_X, hip_y - 0.35) for _ in range(33)]     # head-ish default
    lms[0] = _lm(HIP_X, hip_y - 0.34)                        # nose
    lms[11] = _lm(HIP_X, hip_y - 0.20)                       # L shoulder (near)
    lms[12] = _lm(HIP_X, hip_y - 0.20, far_vis)              # R shoulder (far)
    lms[13] = _lm(HIP_X - 0.05, hip_y - 0.10)
    lms[14] = _lm(HIP_X + 0.05, hip_y - 0.10, far_vis)
    lms[15] = _lm(HIP_X - 0.08, hip_y - 0.02)
    lms[16] = _lm(HIP_X + 0.08, hip_y - 0.02, far_vis)
    lms[23] = _lm(HIP_X, hip_y)                              # L hip (near)
    lms[24] = _lm(HIP_X, hip_y, far_vis)                     # R hip (far)
    lms[25] = _lm((HIP_X + near_ankle_x) / 2, hip_y + 0.15)  # L knee (near)
    lms[26] = _lm(HIP_X + far_knee_dx, hip_y + 0.15, far_vis)   # R knee (far)
    lms[27] = _lm(near_ankle_x, near_ankle_y)                # L ankle
    lms[28] = _lm(HIP_X + far_knee_dx, near_ankle_y, far_vis)
    lms[29] = _lm(near_ankle_x - 0.02, near_ankle_y)         # L heel
    lms[30] = _lm(HIP_X + far_knee_dx - 0.02, near_ankle_y, far_vis)
    lms[31] = _lm(near_ankle_x + 0.03, near_ankle_y)         # L toe (ahead)
    lms[32] = _lm(HIP_X + far_knee_dx + 0.03, near_ankle_y, far_vis)
    return lms


def build_analyzer(cycles: int = 3, *, far_vis: float = 0.85) -> RunningAnalyzer:
    analyzer = RunningAnalyzer(fps=FPS)
    analyzer.camera_side = "left"
    n = cycles * PERIOD
    knee_series: list[float] = []
    trunk_series: list[float] = []
    for i in range(n):
        p = i % PERIOD
        in_stance = p < STANCE
        # Knee: near-straight at contact, flexed through swing.
        knee = 166.0 - 24.0 * (p / STANCE) if in_stance else 95.0
        fr = FrameAnalysis(timestamp_ms=i / FPS * 1000.0)
        fr.angles["knee"] = knee
        fr.angles["trunk"] = 6.0
        fr.extra_metrics["gait_phase"] = "midstance" if in_stance else "mid_swing"
        fr.extra_metrics["_norm_landmarks"] = _landmarks(p, far_vis=far_vis)
        analyzer.frame_results.append(fr)
        knee_series.append(knee)
        trunk_series.append(6.0)
    analyzer.angle_history["knee"] = knee_series
    analyzer.angle_history["trunk"] = trunk_series
    return analyzer


def phase_of(sel) -> dict[str, int]:
    """Position key -> its offset within the stride."""
    return {p.key: p.analyzed_idx % PERIOD for p in sel.positions}


# --- stance runs (the shared foundation) -----------------------------------

def test_stance_runs_report_both_ends_of_every_contact():
    analyzer = build_analyzer(cycles=3)
    runs = analyzer.stance_runs()
    assert runs == [(0, 7), (20, 27), (40, 47)]


def test_contact_indices_still_read_the_starts_of_those_runs():
    """The refactor must not move the frames overstride/foot-strike sample."""
    analyzer = build_analyzer(cycles=3)
    assert analyzer._contact_frame_indices() == [
        start for start, _ in analyzer.stance_runs()
    ]


def test_a_flickering_single_stance_frame_is_not_a_contact():
    analyzer = build_analyzer(cycles=2)
    analyzer.frame_results[12].extra_metrics["gait_phase"] = "midstance"
    assert analyzer.stance_runs() == [(0, 7), (20, 27)]


# --- position finding ------------------------------------------------------

def test_the_five_positions_land_on_their_geometric_events():
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    assert sel is not None
    assert sel.complete
    assert phase_of(sel) == {
        "touch_down": 0,
        "full_support": FULL_SUPPORT_P,
        "toe_off": STANCE - 1,
        "mvp": MVP_P,
        "strike": STRIKE_P,
    }


def test_positions_come_back_in_cycle_order():
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    keys = [p.key for p in sel.positions]
    assert keys == list(kinogram.POSITION_ORDER)
    idxs = [p.analyzed_idx for p in sel.positions]
    assert idxs == sorted(idxs)


def test_a_well_tracked_far_leg_uses_the_real_altis_definition():
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3, far_vis=0.85))
    assert sel.strike_source == "far_thigh_vertical"


def test_an_untracked_far_leg_falls_back_to_the_near_thigh():
    """A side view tracks the far leg worst -- the substitute must engage, and
    must say so rather than passing itself off as the ALTIS reading."""
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3, far_vis=0.30))
    assert sel is not None
    assert sel.strike_source == "near_max_hip_flexion"
    assert sel.complete
    assert "strike" in phase_of(sel)


def test_a_clip_with_one_contact_yields_no_kinogram():
    analyzer = build_analyzer(cycles=3)
    analyzer.frame_results = analyzer.frame_results[:14]
    assert kinogram.select_run_kinogram(analyzer) is None


def test_no_gait_phases_at_all_yields_no_kinogram():
    analyzer = build_analyzer(cycles=3)
    for fr in analyzer.frame_results:
        fr.extra_metrics["gait_phase"] = "unknown"
    assert kinogram.select_run_kinogram(analyzer) is None


def test_a_stride_with_no_flight_degrades_to_the_three_stance_positions():
    """Coarse sampling can leave no room between contacts. Three real
    positions beat none -- but the result must not claim to be complete."""
    analyzer = build_analyzer(cycles=3)
    for fr in analyzer.frame_results:
        p = analyzer.frame_results.index(fr) % PERIOD
        fr.extra_metrics["gait_phase"] = "mid_swing" if p in (18, 19) else "midstance"
    sel = kinogram.select_run_kinogram(analyzer)
    assert sel is not None
    assert not sel.complete
    assert [p.key for p in sel.positions] == list(kinogram.STANCE_POSITIONS)
    assert any("flight phase" in w for w in sel.warnings)


def test_a_cycling_analyzer_is_refused_outright():
    from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer

    assert kinogram.select_run_kinogram(CyclingAnalyzer(fps=FPS)) is None


# --- readings on the tiles -------------------------------------------------

def test_touch_down_grades_its_knee_against_the_contact_band():
    """166 deg at contact sits inside knee_at_initial_contact (160-175)."""
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    td = next(p for p in sel.positions if p.key == "touch_down")
    knee = next(m for m in td.metrics if m.label == "KNEE")
    assert knee.value == "166°"
    assert knee.status == "good"


def test_a_locked_out_knee_at_contact_grades_bad():
    analyzer = build_analyzer(cycles=3)
    analyzer.angle_history["knee"] = [179.0] * len(analyzer.frame_results)
    sel = kinogram.select_run_kinogram(analyzer)
    td = next(p for p in sel.positions if p.key == "touch_down")
    knee = next(m for m in td.metrics if m.label == "KNEE")
    assert knee.status == "bad"


def test_a_reading_with_no_published_band_is_shown_but_not_graded():
    """The strike thigh angle has no reference in the literature we cite --
    inventing a colour for it would be inventing authority."""
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    strike = next(p for p in sel.positions if p.key == "strike")
    thigh = next(m for m in strike.metrics if m.label == "THIGH")
    assert thigh.status == "muted"
    assert thigh.value != "--"


def test_an_unmeasured_angle_reads_as_a_dash_not_a_zero():
    analyzer = build_analyzer(cycles=3)
    analyzer.angle_history.pop("knee")
    sel = kinogram.select_run_kinogram(analyzer)
    td = next(p for p in sel.positions if p.key == "touch_down")
    knee = next(m for m in td.metrics if m.label == "KNEE")
    assert knee.value == "--" and knee.status == "muted"


def test_a_summary_that_withheld_trunk_lean_keeps_it_off_the_tiles():
    """compute_summary drops trunk lean when it fails its plausibility gate.
    A tile printing a confident red angle beside a table that says the trunk
    could not be measured is the report arguing with itself."""
    analyzer = build_analyzer(cycles=3)
    sel = kinogram.select_run_kinogram(analyzer, summary={"trunk_lean": None})
    fs = next(p for p in sel.positions if p.key == "full_support")
    trunk = next(m for m in fs.metrics if m.label == "TRUNK")
    assert trunk.value == "--" and trunk.status == "muted"


def test_a_published_trunk_lean_does_reach_the_tiles():
    analyzer = build_analyzer(cycles=3)
    sel = kinogram.select_run_kinogram(analyzer, summary={"trunk_lean": 6.0})
    fs = next(p for p in sel.positions if p.key == "full_support")
    trunk = next(m for m in fs.metrics if m.label == "TRUNK")
    assert trunk.value == "6°" and trunk.status == "good"


def test_complete_is_counted_after_duplicates_are_dropped(monkeypatch):
    """Five events found and one lost to a frame collision is a four-tile
    kinogram -- calling it complete is how the caller promises five."""
    monkeypatch.setattr(
        kinogram, "_find_strike",
        lambda analyzer, side, start, end, forward: (start - 1, "far_thigh_vertical"),
    )
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    assert sel is not None
    assert len(sel.positions) == 4
    assert not sel.complete
    assert any("same frame" in w for w in sel.warnings)


# --- refusing to publish a picture the report does not stand behind --------

CLEAN_SUMMARY = {"cadence_spm": 172.0}
CLEAN_TRACKING = {
    "leg_swap_pct": 4.0, "flip_pct": 3.0, "side_vote_disagreement_pct": 2.0,
    "framing": {"verdict": "ok", "subject_height_px": 620},
}


def test_a_cleanly_tracked_clip_is_allowed_a_kinogram():
    assert kinogram.stride_trust_block(CLEAN_SUMMARY, CLEAN_TRACKING) is None


def test_a_stance_phase_covering_most_of_the_cycle_means_no_kinogram():
    """Every position is cut out of the stance/swing split, so a clip where
    that split is nonsense yields five frames that LOOK like a kinogram and are
    not one. Real case: a cleanly tracked clip whose detector called 87% of the
    cycle stance, so all five positions came from a 10-frame window of a
    170-frame stride."""
    blocked = kinogram.stride_trust_block(
        {**CLEAN_SUMMARY, "stance_fraction": 0.87}, CLEAN_TRACKING)
    assert blocked.startswith("implausible_stance_fraction")


def test_a_stance_phase_nobody_could_stand_on_means_no_kinogram_either():
    assert kinogram.stride_trust_block(
        {**CLEAN_SUMMARY, "stance_fraction": 0.02},
        CLEAN_TRACKING).startswith("implausible_stance_fraction")


@pytest.mark.parametrize("fraction", [0.44, 0.60, 0.70, 0.80])
def test_a_believable_stance_share_is_allowed_through(fraction):
    """These are shares of the clip with SOME foot down, which is what the
    phases have meant since 2026-08-23 -- roughly twice the per-leg duty
    factor. Sprinting near 0.22 per leg is 0.44 here; easy distance running at
    0.40 per leg is 0.80. The band is not measuring technique, only asking
    whether this could be running at all."""
    assert kinogram.stride_trust_block(
        {**CLEAN_SUMMARY, "stance_fraction": fraction}, CLEAN_TRACKING) is None


def test_an_analyzer_that_never_reported_a_stance_share_is_not_punished():
    assert kinogram.stride_trust_block(CLEAN_SUMMARY, CLEAN_TRACKING) is None


def test_no_cadence_means_no_kinogram():
    """Cadence is the analyzer's own verdict on whether it could time the
    stride -- and the five positions ARE stride timing."""
    assert kinogram.stride_trust_block(
        {"cadence_spm": None}, CLEAN_TRACKING) == "cadence_unmeasured"


def test_a_small_but_cleanly_tracked_athlete_still_gets_a_kinogram():
    """Framing was a criterion here until it was caught measuring the wrong
    thing: the bar had been calibrated on clips where the athlete was small AND
    backlit, so pixel height took the blame for what the light was doing. A
    well-lit 171 px clip tracked better than anything else tested (legs
    confused on 0.8% of frames) and was refused for being 'too small'."""
    tracking = {**CLEAN_TRACKING,
                "leg_swap_pct": 0.8,
                "framing": {"verdict": "tiny", "subject_height_px": 171}}
    assert kinogram.stride_trust_block(CLEAN_SUMMARY, tracking) is None


def test_a_small_athlete_the_model_ALSO_lost_is_still_refused():
    """Dropping the framing criterion must not drop the protection: what the
    gate exists for is a stride nobody could follow, and that is still caught
    -- by the outcome rather than by a guess at its cause."""
    tracking = {**CLEAN_TRACKING,
                "leg_swap_pct": 45.6,
                "framing": {"verdict": "tiny", "subject_height_px": 116}}
    assert kinogram.stride_trust_block(
        CLEAN_SUMMARY, tracking) == "unstable_tracking:leg_swap_pct"


def test_legs_the_model_could_not_tell_apart_means_no_kinogram():
    """Every position is defined relative to the NEAR leg. Past the confidence
    scorer's 'low' bar, which leg is which is a coin flip."""
    tracking = {**CLEAN_TRACKING, "leg_swap_pct": 42.7}
    assert kinogram.stride_trust_block(
        CLEAN_SUMMARY, tracking) == "unstable_tracking:leg_swap_pct"


def test_the_trust_bar_is_the_confidence_scorers_not_a_new_one():
    from app.services.video_analysis.biomechanics.confidence_scorer import THRESHOLDS

    limit = THRESHOLDS["leg_swap_pct_low"]
    assert kinogram.stride_trust_block(
        CLEAN_SUMMARY, {**CLEAN_TRACKING, "leg_swap_pct": limit - 0.1}) is None
    assert kinogram.stride_trust_block(
        CLEAN_SUMMARY, {**CLEAN_TRACKING, "leg_swap_pct": limit}) is not None


def test_missing_diagnostics_do_not_by_themselves_block():
    """A caller with no tracking_stability (the CLI path) still gets a picture;
    absence of evidence is not evidence of bad tracking."""
    assert kinogram.stride_trust_block({"cadence_spm": 168.0}, None) is None


def test_a_refused_clip_renders_nothing_but_says_why():
    analyzer = build_analyzer(cycles=3)
    uri, meta = kinogram.build_run_kinogram(
        "unused.mp4", analyzer, [],
        summary={"cadence_spm": None}, tracking_stability=CLEAN_TRACKING,
    )
    assert uri is None
    assert meta == {"refused": "cadence_unmeasured"}


def test_meta_is_json_serializable_and_names_its_method():
    import json

    sel = kinogram.select_run_kinogram(build_analyzer(cycles=3))
    meta = sel.to_meta()
    json.dumps(meta)
    assert meta["strike_source"] == "far_thigh_vertical"
    assert len(meta["positions"]) == 5
    assert "ALTIS" in meta["method"]


# --- direction of travel ---------------------------------------------------

def test_direction_of_travel_is_voted_not_read_off_one_frame():
    analyzer = build_analyzer(cycles=3)
    # Blind the feet on a handful of frames: one bad frame must not decide.
    for i in (0, 1, 2, 20, 21):
        for idx in (29, 30, 31, 32):
            analyzer.frame_results[i].extra_metrics["_norm_landmarks"][idx].visibility = 0.1
    assert kinogram._median_forward_sign(analyzer) == 1.0


def test_running_the_other_way_flips_the_sign():
    analyzer = build_analyzer(cycles=3)
    for fr in analyzer.frame_results:
        lms = fr.extra_metrics["_norm_landmarks"]
        for heel, toe in ((29, 31), (30, 32)):
            lms[heel].x, lms[toe].x = lms[toe].x, lms[heel].x
    assert kinogram._median_forward_sign(analyzer) == -1.0


# --- gating ----------------------------------------------------------------

def test_the_kinogram_never_reaches_a_free_caller():
    from app.services.result_gating import gate_free_result, gate_preview_result

    result = {
        "status": "completed", "sport_type": "run", "technique_score": 71,
        "letter_grade": "C", "keyframe_base64": "data:image/jpeg;base64,AAAA",
        "kinogram_base64": "data:image/jpeg;base64," + "B" * 400,
        "kinogram": {"positions": [], "complete": True},
        "sport_specific_metrics": {},
    }
    for gated in (gate_free_result(result), gate_preview_result(result)):
        assert gated.get("kinogram_base64") is None
        assert gated.get("kinogram") is None
        assert "kinogram" in gated["locked"]["unlocks"]


@pytest.mark.parametrize("n", [2, 3, 5])
def test_selection_is_stable_across_clip_lengths(n):
    sel = kinogram.select_run_kinogram(build_analyzer(cycles=n))
    assert sel is not None and sel.complete
    assert phase_of(sel)["mvp"] == MVP_P


# --- composition -----------------------------------------------------------
#
# Guarded by an import skip: the renderer needs opencv, which the rest of the
# suite deliberately avoids (see conftest).

def _write_clip(path, frames: int, w: int = 640, h: int = 480):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Something that changes per frame, so a mis-seek would be visible.
        frame[:, (i * 3) % w:((i * 3) % w) + 4] = 200
        writer.write(frame)
    writer.release()
    return str(path)


def test_the_composite_is_five_tiles_wide_and_decodes(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="needs the analysis stack (opencv)")
    import base64

    import numpy as np

    analyzer = build_analyzer(cycles=3)
    n = len(analyzer.frame_results)
    video = _write_clip(tmp_path / "clip.mp4", n)
    frame_data = [
        {"frame_idx": i,
         "normalized_landmarks": fr.extra_metrics["_norm_landmarks"]}
        for i, fr in enumerate(analyzer.frame_results)
    ]

    uri, meta = kinogram.build_run_kinogram(
        video, analyzer, frame_data,
        summary=CLEAN_SUMMARY, tracking_stability=CLEAN_TRACKING,
        technique_score=81, letter_grade="B",
    )
    assert uri is not None and uri.startswith("data:image/jpeg;base64,")
    assert len(meta["positions"]) == 5

    raw = base64.b64decode(uri.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    tile_w = round(kinogram.TILE_H * kinogram.TILE_ASPECT)
    assert img.shape[1] == (
        kinogram.OUTER_PAD * 2 + tile_w * 5 + kinogram.TILE_GAP * 4
    )
    assert img.shape[0] == (
        kinogram.OUTER_PAD * 2 + kinogram.HEADER_H
        + kinogram.TILE_H + kinogram.FOOTER_H
    )


def test_every_tile_is_cropped_to_the_same_scale(tmp_path):
    """A kinogram whose tiles are differently zoomed cannot be read as a
    sequence -- the athlete must not change size from frame to frame."""
    pytest.importorskip("cv2", reason="needs the analysis stack (opencv)")

    analyzer = build_analyzer(cycles=3)
    sel = kinogram.select_run_kinogram(analyzer, summary=CLEAN_SUMMARY)
    boxes = [
        kinogram._athlete_box(
            kinogram._norm_landmarks(analyzer, p.analyzed_idx), 640, 480)
        for p in sel.positions
    ]
    crop_h = max(b[3] for b in boxes) * kinogram.CROP_MARGIN
    windows = [kinogram._crop_window(b, crop_h, 640, 480) for b in boxes]
    assert len({(w[2], w[3]) for w in windows}) == 1


def test_a_crop_running_off_the_frame_slides_back_in_rather_than_shrinking():
    box = (10.0, 240.0, 60.0, 300.0)     # athlete hard against the left edge
    x0, y0, cw, ch = kinogram._crop_window(box, 420.0, 640, 480)
    assert x0 >= 0 and y0 >= 0
    assert x0 + cw <= 640 and y0 + ch <= 480
    assert ch == round(420.0)
