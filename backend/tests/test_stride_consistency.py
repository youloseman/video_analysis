"""Does the clip's leg identity hold? The gait answers, so the report can too.

Two rounds of work on the run resolver cut leg swaps a long way and did not
end them, and no further fix could be VALIDATED: a structural repair would
only ever be scored by a structural metric, and appearance proved far too weak
to arbitrate (twenty deliberately injected swaps moved it 2%, less than the
noise between two runs). So the remaining honesty is to detect the bad clips
rather than to keep tuning blind.

The measurement leans on one thing running guarantees: the two ankles swap
vertical order exactly twice per stride. Everything past that is a swap.

Calibrated on the run fixtures: the clips that read clean sit at 0.00-0.11,
and IMG_4262 -- the one Artur called unusable -- sits at 0.21.
"""

from __future__ import annotations

import math

from app.services.video_analysis.biomechanics.stride_consistency import (
    INSTABILITY_WARN,
    measure_leg_identity_stability,
)


class _LM:
    def __init__(self, y):
        self.x, self.y = 0.5, y
        self.z, self.visibility = 0.0, 1.0


def _clip(n=360, cycle=40, swaps=(), fps=60.0):
    """A clean run: the ankles cross twice a cycle. `swaps` flips identity
    from each given frame onward, the way a resolver error does."""
    frames = []
    flip = 0
    for k in range(n):
        if k in swaps:
            flip ^= 1
        phase = 2 * math.pi * k / cycle
        a = 0.60 + 0.10 * math.sin(phase)
        b = 0.60 - 0.10 * math.sin(phase)
        left, right = (b, a) if flip else (a, b)
        lms = [_LM(0.5) for _ in range(33)]
        lms[27], lms[28] = _LM(left), _LM(right)
        frames.append({"normalized_landmarks": lms, "frame_idx": k,
                       "timestamp_ms": k * 1000.0 / fps})
    return frames


class TestACleanClip:
    def test_it_finds_the_stride(self):
        m = measure_leg_identity_stability(_clip(), 60.0)
        assert m is not None
        assert m["cycles"] == 9.0
        assert m["expected_crossings"] == 18.0

    def test_no_excess_and_no_flag(self):
        m = measure_leg_identity_stability(_clip(), 60.0)
        assert m["excess_crossings"] == 0.0
        assert m["instability"] == 0.0
        assert m["unstable"] is False


class TestSwapsShowUp:
    # Frames where the ankles are far apart (cycle 40, so k % 40 == 10 or 30
    # are the extremes). A swap there is exactly the one the athlete sees.
    VISIBLE = (70, 110, 150, 190, 230, 270, 310)

    def test_each_swap_adds_a_crossing(self):
        clean = measure_leg_identity_stability(_clip(), 60.0)
        for k in (1, 2, 3):
            m = measure_leg_identity_stability(
                _clip(swaps=self.VISIBLE[:k]), 60.0)
            gained = m["observed_crossings"] - clean["observed_crossings"]
            assert gained == k, f"{k} swaps should add {k} crossings, added {gained}"

    def test_a_swap_at_a_crossing_is_invisible_and_that_is_correct(self):
        """When the two ankles are level the legs occupy the same height, so
        exchanging their labels there changes nothing the eye can see -- and
        nothing this metric should count. Such a swap even cancels the natural
        crossing it lands on, which is why an earlier version of this test
        (swaps at 70/140/210/280, two of them on crossings) measured no change
        at all and read as a broken counter."""
        clean = measure_leg_identity_stability(_clip(), 60.0)
        at_crossing = measure_leg_identity_stability(_clip(swaps=(140, 280)), 60.0)
        assert at_crossing["observed_crossings"] <= clean["observed_crossings"]

    def test_enough_swaps_raise_the_flag(self):
        m = measure_leg_identity_stability(_clip(swaps=self.VISIBLE), 60.0)
        assert m["instability"] > INSTABILITY_WARN
        assert m["unstable"] is True

    def test_one_swap_does_not_cry_wolf(self):
        """A single glitch in six seconds is not a clip worth warning about --
        the warning has to mean something when it appears."""
        m = measure_leg_identity_stability(_clip(swaps=(190,)), 60.0)
        assert m["unstable"] is False


class TestItRefusesRatherThanReassures:
    def test_a_short_clip_is_not_measurable(self):
        """Fewer frames than a couple of strides cannot support the count --
        None, never a comforting zero."""
        assert measure_leg_identity_stability(_clip(n=40), 60.0) is None

    def test_a_still_athlete_is_not_measurable(self):
        frames = _clip(n=200)
        for fr in frames:
            fr["normalized_landmarks"][27] = _LM(0.6)
            fr["normalized_landmarks"][28] = _LM(0.6)
        assert measure_leg_identity_stability(frames, 60.0) is None

    def test_empty_input(self):
        assert measure_leg_identity_stability([], 60.0) is None
        assert measure_leg_identity_stability(None, 60.0) is None

    def test_missing_ankles_are_not_counted_as_agreement(self):
        frames = _clip(n=200)
        for fr in frames:
            fr["normalized_landmarks"] = []
        assert measure_leg_identity_stability(frames, 60.0) is None


class TestTheDeadband:
    def test_level_ankle_jitter_is_not_a_swap(self):
        """Where the two ankles are level, a hair of noise flips their order
        many times a second. Counting that would flag every clip."""
        import random
        rng = random.Random(4)
        frames = _clip(n=360)
        for fr in frames:
            for idx in (27, 28):
                fr["normalized_landmarks"][idx].y += rng.uniform(-0.002, 0.002)
        m = measure_leg_identity_stability(frames, 60.0)
        assert m["unstable"] is False
