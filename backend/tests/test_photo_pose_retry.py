"""Finding a pose in a photo the detector almost refuses — without inventing one.

A quarter of photo analyses were coming back "No body pose detected", which
costs the athlete their attempt and tells them to go and take a better
picture. Measured on the stored photos, a lower detection threshold recovers
several of them.

It also invents poses. The detector is deterministic -- the same image twice
at 0.5 agrees to four decimal places -- and at 0.2 it can put joints a quarter
of the frame from where 0.5 put them. On one race photo the low pass drew a
correct skeleton on the rider; on another it draped one over the frame and the
saddle while the rider's own legs, plainly visible, got nothing. Average
landmark visibility does not separate those two cases (0.99 against 0.83).

So the low pass is quarantined behind two rules, and this file is about them:

* it is reached only when every attempt at the normal threshold has failed,
  so no photo that works today can have its numbers changed by it;
* what it finds is believed only if the pose holds still when the photo is fed
  at other sizes -- a body does, an artefact assembled out of frame tubes does
  not.

And whatever survives both is still marked, warned about, and left to the
score gate. "We only just made it out" is a thing the athlete is told, not a
thing hidden behind a confident-looking number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.services.video_analysis import photo_analyzer as pa


class _LM:
    """One landmark, as MediaPipe hands it over."""

    def __init__(self, x: float, y: float, visibility: float = 0.9):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, visibility


def _pose(offset: float = 0.0, n: int = 33) -> list[_LM]:
    """A plausible skeleton, optionally displaced by ``offset`` of the frame."""
    return [_LM(0.4 + offset + i * 0.001, 0.3 + offset + i * 0.002) for i in range(n)]


def _result(landmarks):
    return SimpleNamespace(
        pose_landmarks=[landmarks] if landmarks else [],
        pose_world_landmarks=[landmarks] if landmarks else [],
        segmentation_masks=None,
    )


def _image(side: int = 1600):
    return np.zeros((side, side, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# the thresholds themselves
# --------------------------------------------------------------------------

def test_the_last_resort_is_lower_than_the_normal_threshold():
    assert pa._POSE_CONFIDENCE_LAST_RESORT < pa._POSE_CONFIDENCE


def test_several_sizes_are_checked():
    """One comparison is a coin flip: a skeleton on a bicycle frame agreed
    with itself between 1252 px and 900 px and disagreed at 1000 px."""
    assert len(pa._LOW_CONFIDENCE_VERIFY_SIDES) >= 3


def test_the_agreement_cut_sits_in_the_measured_gap():
    """Correct skeletons scored 0.010 and 0.020; wrong ones 0.062, 0.166,
    0.286. Nothing was measured in between, and the cut belongs there."""
    assert 0.020 < pa._LOW_CONFIDENCE_AGREEMENT < 0.062


# --------------------------------------------------------------------------
# the stability guard
# --------------------------------------------------------------------------

def test_a_pose_that_holds_still_is_believed():
    found = _pose()
    calls = []

    def attempt(rgb, conf):
        calls.append(rgb.shape[0])
        return _result(_pose())          # identical at every size

    assert pa._agrees_across_scale(found, _image(), 0.2, attempt) is True
    assert len(calls) == len(pa._LOW_CONFIDENCE_VERIFY_SIDES)


def test_a_pose_that_moves_is_refused():
    """The bicycle-frame case. Displaced by a tenth of the frame, far past
    anything a real body does between two renderings of the same photo."""
    found = _pose()

    def attempt(rgb, conf):
        return _result(_pose(offset=0.10))

    assert pa._agrees_across_scale(found, _image(), 0.2, attempt) is False


def test_one_steady_size_cannot_rescue_an_otherwise_moving_pose():
    """The verdict is the WORST of the sizes, not the best -- an artefact only
    has to hold still once to look convincing."""
    found = _pose()
    seen = {"n": 0}

    def attempt(rgb, conf):
        seen["n"] += 1
        # Agrees on the first size, wanders on the rest.
        return _result(_pose(offset=0.0 if seen["n"] == 1 else 0.10))

    assert pa._agrees_across_scale(found, _image(), 0.2, attempt) is False


def test_a_single_wandering_joint_does_not_veto_a_solid_pose():
    """Compared on the median, so a limb the model is genuinely unsure about
    cannot throw away a reading that is right everywhere else."""
    found = _pose()

    def attempt(rgb, conf):
        other = _pose()
        other[27] = _LM(0.9, 0.9)        # one ankle, far away
        return _result(other)

    assert pa._agrees_across_scale(found, _image(), 0.2, attempt) is True


def test_a_photo_too_small_to_verify_is_not_refused_for_it():
    """A guard that cannot run must not silently become a rejection. The
    reading is flagged and gated either way."""
    found = _pose()

    def attempt(rgb, conf):
        raise AssertionError("should not be called for a small photo")

    small = _image(min(pa._LOW_CONFIDENCE_VERIFY_SIDES) - 50)
    assert pa._agrees_across_scale(found, small, 0.2, attempt) is True


def test_a_verification_that_finds_nothing_does_not_refuse_either():
    found = _pose()

    def attempt(rgb, conf):
        return _result(None)

    assert pa._agrees_across_scale(found, _image(), 0.2, attempt) is True


# --------------------------------------------------------------------------
# what the athlete is told
# --------------------------------------------------------------------------

def test_the_warning_says_the_reading_is_weaker_than_usual():
    """The whole risk of the low pass is a marginal reading that looks exactly
    like a confident one. The sentence is what stops that, so it is pinned."""
    import inspect

    src = inspect.getsource(pa.detect_pose_in_image)
    assert "low_confidence" in src
    assert "could only just make out" in src


def test_the_result_carries_the_flag():
    """Downstream -- the report, the gate, the analytics -- has to be able to
    tell a marginal detection from a confident one."""
    import inspect

    src = inspect.getsource(pa.detect_pose_in_image)
    assert '"low_confidence": low_confidence' in src


# --------------------------------------------------------------------------
# the ladder's shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src_line", [
    # The normal threshold is tried first, at full size, exactly as before.
    'ladder = [("full", image_rgb, _POSE_CONFIDENCE)]',
    # One low-confidence attempt, after them.
    'ladder.append(("low-confidence", image_rgb, _POSE_CONFIDENCE_LAST_RESORT))',
])
def test_the_normal_threshold_comes_first(src_line):
    """A photo that works today must never reach the low pass -- its numbers
    would change. Order is the guarantee, so it is pinned in the source."""
    import inspect

    assert src_line in inspect.getsource(pa.detect_pose_in_image)


def test_an_unstable_low_confidence_reading_ends_the_search():
    """It must not be retried on a resized copy. That is asking the same
    unreliable question until it answers the way we wanted, and it is how a
    skeleton on a bicycle frame got through the first version of this."""
    import inspect

    src = inspect.getsource(pa.detect_pose_in_image)
    unstable = src.index('logger.info("PHOTO_POSE_UNSTABLE"')
    tail = src[unstable:unstable + 220]
    assert "break" in tail and "continue" not in tail
