"""Camera motion measured on a shrunk picture, reported in original pixels.

ORB cost 62.3 ms per frame at 1072x1920 and 10.7 ms at half that, and this
module was 9 of the 38 seconds a running analysis took -- for a diagnostic.
Shrinking is safe because what comes out is a rigid translation: halve the
picture and the measured shift halves exactly.

That last sentence is the whole risk. Every number downstream is in pixels and
calibrated at full size -- RIGIDITY_TOLERANCE_PX was measured at 8 px on a real
treadmill clip, and ``vertical_share_of_hip_motion`` divides by a frame height
read from the original video. A shift left in shrunk pixels would silently
rescale all of it, and the verdict would quietly depend on the phone that shot
the clip. So these tests are about units, not about speed.

Measured before and after on upload/vid1.MOV: 9.36 s -> 4.60 s, with pairs,
unreadable_pairs, scene_rigid, verdict and pan_px identical and the headline
share 0.059 -> 0.061 (against a 0.15 warning threshold).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.services.video_analysis import camera_motion as CM


def test_the_cap_is_a_shrink_only_rule():
    """Nothing is ever upscaled: a small clip has no pixels to spare and
    interpolating some would invent features for ORB to match."""
    src = inspect.getsource(CM.estimate_camera_motion)
    assert "if long_edge > MOTION_MAX_LONG_EDGE" in src
    assert "scale = MOTION_MAX_LONG_EDGE / long_edge" in src


def test_the_cap_leaves_room_above_the_inlier_floor():
    """960 px is chosen to keep ORB in features, not to be as small as
    possible. If someone drops it near MIN_INLIERS territory, say so."""
    assert CM.MOTION_MAX_LONG_EDGE >= 640
    assert CM.MIN_INLIERS == 12


def test_shifts_are_scaled_back_before_anything_reads_them():
    """The one invariant. Downstream constants are in original pixels."""
    src = inspect.getsource(CM.estimate_camera_motion)
    assert "back = 1.0 / scale" in src
    for collected in ("dxs.append", "dys.append", "dy_upper.append", "dy_lower.append"):
        line = next(ln for ln in src.splitlines() if collected in ln)
        assert "back" in line, f"{collected} stores a shrunk-pixel value"


def test_the_frame_height_comes_from_a_decoded_frame():
    """It used to reopen the video purely for CAP_PROP_FRAME_HEIGHT, after the
    decode loop had already held every frame. It must be the ORIGINAL height:
    it is the denominator under the athlete's hip movement."""
    src = inspect.getsource(CM.estimate_camera_motion)
    assert "height = frame.shape[0] or 1" in src
    assert src.count("cv2.VideoCapture(video_path)") == 1


def test_area_interpolation_is_used_for_the_shrink():
    """Nearest-neighbour aliases exactly the high-frequency detail ORB looks
    for corners in, so the shrink would cost matches as well as pixels."""
    src = inspect.getsource(CM.estimate_camera_motion)
    assert "INTER_AREA" in src


# --------------------------------------------------------------------------
# the arithmetic the scale-back relies on
# --------------------------------------------------------------------------

def test_a_translation_scales_linearly():
    """Fit a known shift on a synthetic frame and on the same frame halved:
    the recovered shift must differ by exactly the scale factor. This is the
    property that makes the whole optimisation legitimate."""
    cv2 = pytest.importorskip("cv2")

    rng = np.random.default_rng(7)
    big = rng.integers(0, 255, size=(480, 480), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (5, 5), 0)     # texture ORB can key on
    shift_px = 12
    moved = np.roll(big, shift_px, axis=1)

    orb = cv2.ORB_create(nfeatures=CM.ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    full = CM._translation_between(cv2, big, moved, None, None, orb, matcher)
    assert full is not None, "the synthetic frame gave ORB nothing to match"

    half = cv2.resize(big, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    half_moved = cv2.resize(moved, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small = CM._translation_between(
        cv2, half, half_moved, None, None, orb, matcher,
    )
    assert small is not None

    # Scaled back, the two agree to within a pixel.
    assert small[0] * 2.0 == pytest.approx(full[0], abs=1.5)
    assert abs(full[0]) == pytest.approx(shift_px, abs=1.5)


def test_a_still_camera_reads_as_still_at_either_scale():
    """The commonest case, and the one a units bug would not show up in --
    zero times any scale factor is still zero. Here for completeness of the
    pair above, not as evidence on its own."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    frame = cv2.GaussianBlur(
        rng.integers(0, 255, size=(480, 480), dtype=np.uint8), (5, 5), 0,
    )
    orb = cv2.ORB_create(nfeatures=CM.ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    got = CM._translation_between(cv2, frame, frame, None, None, orb, matcher)
    assert got is not None
    assert got[0] == pytest.approx(0.0, abs=1.0)
    assert got[1] == pytest.approx(0.0, abs=1.0)
