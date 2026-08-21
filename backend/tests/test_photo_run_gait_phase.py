"""A running photo is one instant of a cycle, and the bands move with it.

The hip is ~123 deg at peak knee drive and ~165 deg at toe-off. Scored against
one always-on band -- (150, 180), which is the toe-off band -- a deliberately
chosen knee-drive frame came back 69/100 with "hip too closed, restricts
propulsion, do high-knee drills". Textbook peak hip flexion, reported as the
athlete's biggest fault, with a drill attached that asks for more of what was
already there.

The bike path had solved this (``_estimate_pedal_phase``, per-phase hip bands,
hip excluded when the crank position is unknown). These tests hold the running
path to the same contract: resolve the stride instant first, score the hip only
where its band means something, and say so plainly everywhere else.

Pure functions and numpy -- no mediapipe, no opencv (see conftest note 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis import overlay_style
from app.services.video_analysis.photo_analyzer import (
    _estimate_run_gait_phase,
    _get_running_optimal_ranges,
    _running_forward_sign,
    _score_photo_angles,
    _thigh_deviation_deg,
)


class LM:
    """Minimal stand-in for a MediaPipe landmark."""

    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _skeleton(
    *, toe_dx: float = 0.08, hip=(0.50, 0.50), knee=(0.50, 0.75),
    foot_vis: float = 1.0,
) -> list[LM]:
    """A landmark list long enough to index feet (29-32), hip (23) and knee (25).

    Only the landmarks the phase detector reads are meaningful; the rest are
    filler so the indices line up.
    """
    lms = [LM(0.5, 0.5) for _ in range(33)]
    lms[23] = LM(*hip)
    lms[25] = LM(*knee)
    for heel_i, toe_i in ((29, 31), (30, 32)):
        lms[heel_i] = LM(0.50, 0.95, foot_vis)
        lms[toe_i] = LM(0.50 + toe_dx, 0.95, foot_vis)
    return lms


# --------------------------------------------------------------------------
# Direction of travel
# --------------------------------------------------------------------------

def test_forward_sign_reads_the_feet_not_the_limbs():
    """Toe ahead of heel is the one orientation that never flips mid-stride."""
    assert _running_forward_sign(_skeleton(toe_dx=0.08)) == 1.0
    assert _running_forward_sign(_skeleton(toe_dx=-0.08)) == -1.0


def test_forward_sign_unknown_when_feet_are_not_visible():
    """No direction -> no phase, rather than a coin flip on which band applies."""
    assert _running_forward_sign(_skeleton(foot_vis=0.1)) == 0.0


# --------------------------------------------------------------------------
# Signed thigh deviation -- the quantity the phase is read from
# --------------------------------------------------------------------------

def test_thigh_deviation_is_signed_where_the_hip_angle_folds():
    """Ahead and behind must not read alike.

    The measured hip angle is an unsigned vertex angle, so a thigh 10 deg past
    vertical reads about the same whichever side it is on -- with opposite
    meanings. That fold is exactly what a phase detector cannot afford.
    """
    ahead = _thigh_deviation_deg(_skeleton(knee=(0.60, 0.75)), 23, 25, 1.0)
    behind = _thigh_deviation_deg(_skeleton(knee=(0.40, 0.75)), 23, 25, 1.0)
    assert ahead > 0 and behind < 0
    assert ahead == pytest.approx(-behind, abs=0.2)


def test_thigh_deviation_is_mirror_symmetric():
    """Filmed from the other side, the same stride reads the same."""
    right = _thigh_deviation_deg(_skeleton(knee=(0.60, 0.75)), 23, 25, 1.0)
    left = _thigh_deviation_deg(_skeleton(knee=(0.40, 0.75)), 23, 25, -1.0)
    assert right == pytest.approx(left, abs=0.2)


def test_thigh_deviation_needs_a_direction():
    assert np.isnan(_thigh_deviation_deg(_skeleton(), 23, 25, 0.0))


# --------------------------------------------------------------------------
# Phase classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "thigh_dev, knee, expected",
    [
        # Thigh well forward, heel tucked: mid-swing/knee drive. The frame the
        # athlete hand-picks when asked for "the top of the knee drive".
        (56.5, 73.8, "swing"),
        # Thigh forward on a near-straight leg: reaching out to land.
        (28.0, 162.0, "initial_contact"),
        # Thigh behind the body: the instant hip extension is defined at.
        (-18.0, 150.0, "toe_off"),
        # Thigh under the body: hip reads ~180 whatever the athlete does.
        (4.0, 120.0, "midstance"),
        # Thigh forward but the knee is between the two bands -- do not guess,
        # because guessing picks a band and the band is the whole verdict.
        (30.0, 128.0, "uncertain"),
        (float("nan"), 90.0, "uncertain"),
        (30.0, float("nan"), "uncertain"),
    ],
)
def test_gait_phase_classification(thigh_dev, knee, expected):
    assert _estimate_run_gait_phase(thigh_dev, knee) == expected


# --------------------------------------------------------------------------
# Bands per phase
# --------------------------------------------------------------------------

def test_hip_is_unscored_where_a_still_cannot_support_a_verdict():
    """Swing and midstance carry no hip band at all.

    Absent from the dict is the contract: ``_score_photo_angles`` skips a key
    it has no band for and renormalises the remaining weights, and
    ``analyze_photo`` reports it as ``phase_dependent`` in the table.
    """
    for phase in ("swing", "midstance", "uncertain"):
        assert "hip" not in _get_running_optimal_ranges(phase)


def test_hip_keeps_its_band_where_the_phase_defines_one():
    assert _get_running_optimal_ranges("toe_off")["hip"] == (150.0, 180.0)
    assert _get_running_optimal_ranges("initial_contact")["hip"] == (138.0, 165.0)


def test_phase_never_removes_the_phase_independent_joints():
    """Only the hip is phase-gated; nothing else may disappear with it."""
    for phase in ("swing", "midstance", "toe_off", "initial_contact", "uncertain"):
        ranges = _get_running_optimal_ranges(phase)
        assert {"knee", "trunk", "elbow", "ankle"} <= set(ranges)


def test_elbow_band_covers_the_whole_arm_swing():
    """The elbow opens through the backswing; ~90 deg is a mid-cycle average.

    Widened rather than phase-split: the swing is small, the band does not
    invert across the cycle, and the fault worth naming (arms carried straight)
    is visible without knowing the phase.
    """
    lo, hi = _get_running_optimal_ranges("swing")["elbow"]
    assert lo <= 78 and hi >= 125      # front of the swing .. backswing
    assert hi < 150                    # straight arms are still a fault


# --------------------------------------------------------------------------
# The reported frame, end to end through the scorer
# --------------------------------------------------------------------------

# Measured off the athlete's own knee-drive frame, which scored 69/100.
_KNEE_DRIVE_FRAME = {
    "knee": 73.8, "hip": 123.5, "elbow": 117.4, "ankle": 88.3, "trunk": 0.0,
}


def test_knee_drive_frame_is_no_longer_faulted():
    """69/100 was two phase-mismatched bands, not two defects.

    The hip drops out (no band at this instant) and the elbow lands inside the
    full-cycle band, leaving only the genuinely borderline knee.
    """
    ranges = _get_running_optimal_ranges("swing")
    result = _score_photo_angles(_KNEE_DRIVE_FRAME, ranges, "run")

    assert "hip" not in result["per_angle"], "hip must not be scored mid-swing"
    assert result["per_angle"]["elbow"]["score"] == 100
    assert result["overall_score"] >= 90
    assert result["grade"] == "Excellent"


def test_dropping_the_hip_renormalises_rather_than_donating_points():
    """An unscored joint must not quietly count as a perfect one."""
    ranges = _get_running_optimal_ranges("swing")
    result = _score_photo_angles(_KNEE_DRIVE_FRAME, ranges, "run")
    scored = set(result["per_angle"])
    assert scored == {"knee", "trunk", "elbow", "ankle"}
    assert sum(a["weight"] for a in result["per_angle"].values()) < 1.0


def test_a_closed_hip_at_toe_off_is_still_a_fault():
    """The fix is a correct band, not a disabled check."""
    angles = dict(_KNEE_DRIVE_FRAME, hip=120.0)
    ranges = _get_running_optimal_ranges("toe_off")
    result = _score_photo_angles(angles, ranges, "run")
    assert result["per_angle"]["hip"]["score"] < 60
    assert result["overall_score"] < 80


def test_straight_arms_are_still_a_fault():
    angles = dict(_KNEE_DRIVE_FRAME, elbow=165.0)
    ranges = _get_running_optimal_ranges("swing")
    result = _score_photo_angles(angles, ranges, "run")
    assert result["per_angle"]["elbow"]["score"] < 60


# --------------------------------------------------------------------------
# Overlay: the annotation has to fit the picture it is drawn on
# --------------------------------------------------------------------------

class FakeCV2:
    """Just enough cv2 for ``fit_render_canvas`` -- keeps opencv out of the suite."""

    INTER_LANCZOS4 = 4
    INTER_AREA = 3

    @staticmethod
    def resize(img, size, interpolation=None):
        w, h = size
        return np.zeros((h, w, 3), dtype=img.dtype)


def _img(w: int, h: int):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_small_frame_is_upscaled_to_the_design_size():
    """Chip type has a hard minimum size, so the canvas must come up to meet it.

    A frame cropped out of a video is ~300 px across, where one chip spans half
    the picture and the rest run off the edge.
    """
    out = overlay_style.fit_render_canvas(FakeCV2, _img(284, 453))
    h, w = out.shape[:2]
    assert w >= overlay_style.RENDER_MIN_WIDTH
    assert w / h == pytest.approx(284 / 453, rel=0.01)


def test_upscale_is_capped():
    """A thumbnail must not be blown up without limit."""
    out = overlay_style.fit_render_canvas(FakeCV2, _img(100, 100))
    assert out.shape[1] == 100 * overlay_style.RENDER_MAX_UPSCALE


@pytest.mark.parametrize("size", [(1600, 900), (1080, 1920), (900, 1200)])
def test_frames_already_in_the_band_are_left_alone(size):
    """A photo the overlay already fits is not touched -- no resample, no cost."""
    src = _img(*size)
    assert overlay_style.fit_render_canvas(FakeCV2, src) is src


def test_oversized_frames_are_capped():
    """The annotated render ships inline as base64 on a synchronous request.

    A full-size phone photo encodes to a ~15 MB data URI, which is a broken
    experience on a phone regardless of how good the annotation looks.
    Measurement is unaffected -- landmarks come from the original pixels.
    """
    out = overlay_style.fit_render_canvas(FakeCV2, _img(3024, 4032))
    h, w = out.shape[:2]
    assert max(w, h) <= overlay_style.RENDER_MAX_LONG_SIDE
    assert w / h == pytest.approx(3024 / 4032, rel=0.01)


def test_annotated_render_is_a_jpeg_the_size_of_a_web_image():
    """The annotated image ships inline as base64 on a synchronous request.

    PNG on a photograph is pathological -- a full-size phone upload encoded to
    a ~15 MB data URI, which is broken on a phone however good the annotation
    looks. Needs the real opencv, so it skips on a slim install (conftest
    note 2).
    """
    cv2 = pytest.importorskip("cv2")
    from app.services.video_analysis import photo_analyzer as pa

    rng = np.random.default_rng(1)
    photo = cv2.GaussianBlur(
        rng.integers(0, 255, (4032, 3024, 3)).astype(np.uint8), (0, 0), 3,
    )
    lms = [LM(0.5, 0.5, 0.2) for _ in range(33)]
    for i, (x, y) in {
        0: (.52, .09), 11: (.44, .24), 12: (.53, .24), 13: (.40, .36),
        14: (.62, .34), 15: (.46, .44), 16: (.58, .45), 23: (.47, .50),
        24: (.54, .50), 25: (.44, .66), 26: (.63, .57), 27: (.49, .80),
        28: (.55, .70), 29: (.46, .83), 30: (.51, .72), 31: (.55, .84),
        32: (.60, .74),
    }.items():
        lms[i] = LM(x, y, 0.95)

    ranges = pa._get_running_optimal_ranges("swing")
    blob = pa._generate_photo_thumbnail(
        cv2, photo, lms, _KNEE_DRIVE_FRAME,
        pa._score_photo_angles(_KNEE_DRIVE_FRAME, ranges, "run"),
        "right", "run", pa._build_arc_triplets("run", "right"),
        optimal_ranges=ranges,
    )

    assert blob[:2] == b"\xff\xd8", "JPEG SOI marker -- not a PNG"
    assert pa._THUMB_MIME == "image/jpeg", "data URI must match what is encoded"

    base64_bytes = len(blob) * 4 / 3
    assert base64_bytes < 2_000_000, f"data URI too heavy: {base64_bytes/1e6:.1f} MB"

    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= overlay_style.RENDER_MAX_LONG_SIDE


def test_chips_are_clamped_inside_the_frame():
    """Callers clamp the anchor, which is not the same as clamping the chip.

    A left-aligned chip anchored near the right edge still runs off it -- which
    is how "KNEE ANGLE 148" rendered as "KNEE" with the number outside.
    """
    frame = _img(900, 1400)
    layer = overlay_style.ChipLayer(frame)
    anchors = [(895, 40), (2, 1395), (880, 1390), (450, 700)]
    for i, at in enumerate(anchors):
        x1, y1, x2, y2 = layer.metric_chip(at, "ELBOW ANGLE", "117°", "good")
        assert 0 <= x1 < x2 <= 900, f"anchor {at} escaped horizontally"
        assert 0 <= y1 < y2 <= 1400, f"anchor {at} escaped vertically"
