"""The two-sided merge: does it end the contradiction, and does it refuse?

These tests carry the 2026-08-26 measurements as fixtures. The geometry
numbers below are the real ones from Artur's clips, in crank-radius units:

  IMG_4148 (right view)  thigh 2.442  shin 2.276  chord 4.510  (5 revolutions)
  IMG_4149 (left view)   thigh 2.322  shin 2.308  chord 4.520  (11 revolutions)
  IMG_4090 (right view)  thigh 2.593  shin 2.480  chord 4.925  <- corrupted orbit
  IMG_4088 (left view)   thigh 2.302  shin 2.186  chord 4.312

The first pair is what the feature is for: measured separately the two sides
read 9 deg apart, and the app told the rider his saddle was perfect from one
side and too high from the other. The second pair is what it must refuse: the
chainring wrecked 4090's ankle orbit, so the crank-radius ruler is 11% wrong
and pooling across it produced a 51 deg gap -- worse than not combining.
"""

import math

import pytest

from app.services.video_analysis.biomechanics.bilateral import (
    BilateralFit,
    SideGeometry,
    combine_sides,
    knee_from_chord,
    leg_length_sensitivity_deg,
    summarize_side,
)


def _side(camera_side, thigh, shin, chord, *, chord_sd=0.028, revs=8, torso=2.6):
    return SideGeometry(
        camera_side=camera_side, thigh=thigh, shin=shin, torso=torso,
        chord_bdc=chord, chord_sd=chord_sd, revolutions=revs,
        measured_frames=300, crank_radius_px=150.0,
    )


# The real pairs, as measured.
RIGHT_4148 = _side("right", 2.442, 2.276, 4.510, chord_sd=0.030, revs=5)
LEFT_4149 = _side("left", 2.322, 2.308, 4.520, chord_sd=0.026, revs=11)
RIGHT_4090 = _side("right", 2.593, 2.480, 4.925, chord_sd=0.032, revs=13)
LEFT_4088 = _side("left", 2.302, 2.186, 4.312, chord_sd=0.026, revs=13)


class TestKneeFromChord:
    def test_straight_leg_reads_180(self):
        assert knee_from_chord(2.0, 2.0, 4.0) == pytest.approx(180.0, abs=0.01)

    def test_folded_leg_reads_zero(self):
        assert knee_from_chord(2.0, 2.0, 0.0) == pytest.approx(0.0, abs=0.01)

    def test_right_angle(self):
        assert knee_from_chord(3.0, 4.0, 5.0) == pytest.approx(90.0, abs=0.01)

    def test_scale_free(self):
        # Only ratios matter -- which is why two clips at different camera
        # distances can still be compared once each is in its own crank units.
        a = knee_from_chord(2.4, 2.3, 4.5)
        b = knee_from_chord(24.0, 23.0, 45.0)
        assert a == pytest.approx(b, abs=1e-9)

    def test_rejects_impossible_geometry(self):
        assert knee_from_chord(0.0, 2.0, 3.0) is None
        assert knee_from_chord(2.0, 2.0, float("nan")) is None


class TestSensitivity:
    def test_the_lever_is_worse_near_extension(self):
        """~4 deg per 1% at 150 deg, and much gentler at 90 -- the reason the
        two views only ever disagreed at the bottom of the stroke."""
        near_straight = leg_length_sensitivity_deg(2.382, 2.292, 4.515)
        mid_stroke = leg_length_sensitivity_deg(2.382, 2.292, 3.30)
        assert 3.5 < near_straight < 5.0
        assert mid_stroke < near_straight / 2


class TestCombineTheRealPair:
    """IMG_4148 + IMG_4149 -- the pair that started this."""

    def test_measured_separately_the_sides_contradict_each_other(self):
        # This is what shipped before the merge existed, and what the rider
        # rightly refused to believe.
        right = knee_from_chord(RIGHT_4148.thigh, RIGHT_4148.shin, RIGHT_4148.chord_bdc)
        left = knee_from_chord(LEFT_4149.thigh, LEFT_4149.shin, LEFT_4149.chord_bdc)
        assert abs(left - right) > 8.0

    def test_one_body_collapses_the_gap(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert fit.combined
        gap = abs(fit.per_side["left"] - fit.per_side["right"])
        assert gap < 2.0, f"sides still disagree by {gap:.1f} deg"

    def test_the_combined_answer_is_one_number_with_an_error_bar(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert fit.knee_at_bdc == pytest.approx(150.0, abs=1.5)
        # Never bare: pooling ends the disagreement, not the shared bias.
        assert fit.uncertainty_deg >= 3.0
        assert fit.uncertainty_deg < 8.0

    def test_the_answer_lands_between_the_two_raw_readings(self):
        # Not "the left clip was right": the merge is not picking a winner. It
        # lands between the two raw values, above the band top (145) but with
        # an interval that still touches it -- so the honest verdict is "at or
        # above the top of the range", not a confident order to drop the
        # saddle. Anything stronger would be inventing precision again.
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        lo = min(fit.raw_per_side.values())
        hi = max(fit.raw_per_side.values())
        assert lo < fit.knee_at_bdc < hi
        assert fit.knee_at_bdc > 145.0
        assert fit.knee_at_bdc - fit.uncertainty_deg < 145.0

    def test_asymmetry_comes_back_as_none_worth_reporting(self):
        # The whole scare was a 9 deg "asymmetry". With one body imposed the
        # legs are within a degree of each other.
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert abs(fit.asymmetry_deg) < 2.0
        assert fit.asymmetry_significant is False

    def test_argument_order_does_not_change_the_answer(self):
        a = combine_sides(LEFT_4149, RIGHT_4148)
        b = combine_sides(RIGHT_4148, LEFT_4149)
        assert a.knee_at_bdc == pytest.approx(b.knee_at_bdc, abs=1e-9)
        assert a.per_side == b.per_side


class TestTheErrorBarIsHonest:
    """A reflection cannot change a real angle. Whatever it DOES change is
    error, so it is the one available audit of the stated uncertainty.

    Re-detecting both clips with the pixels mirrored and merging that pair
    gives a different answer -- pooling ends the disagreement BETWEEN the two
    sides, it does not remove a bias they share. The error bar has to be big
    enough to cover that, or it is decoration.
    """

    # The same two clips, re-detected mirrored (measured 2026-08-26).
    MIRRORED_LEFT = _side("left", 2.3624, 2.2352, 4.4633, chord_sd=0.0508, revs=4)
    MIRRORED_RIGHT = _side("right", 2.3654, 2.2108, 4.5063, chord_sd=0.011, revs=4)

    def test_reflection_still_moves_the_combined_answer(self):
        upright = combine_sides(LEFT_4149, RIGHT_4148)
        mirrored = combine_sides(self.MIRRORED_LEFT, self.MIRRORED_RIGHT)
        assert upright.combined and mirrored.combined
        assert abs(upright.knee_at_bdc - mirrored.knee_at_bdc) > 3.0

    def test_but_the_two_answers_agree_inside_their_error_bars(self):
        upright = combine_sides(LEFT_4149, RIGHT_4148)
        mirrored = combine_sides(self.MIRRORED_LEFT, self.MIRRORED_RIGHT)
        shift = abs(upright.knee_at_bdc - mirrored.knee_at_bdc)
        combined_bar = math.hypot(upright.uncertainty_deg, mirrored.uncertainty_deg)
        assert shift <= combined_bar, (
            f"reflection moved the answer {shift:.1f} deg but we only claim "
            f"+/-{combined_bar:.1f} -- the uncertainty is understated"
        )


class TestItRefuses:
    def test_a_broken_ruler_is_refused_not_averaged(self):
        """The 24 Aug pair: 4090's ankle orbit was chainring-corrupted, so its
        crank radius is ~11% wrong. Forcing a shared body there gave a 51 deg
        gap. Refusing is the only honest move."""
        fit = combine_sides(LEFT_4088, RIGHT_4090)
        assert not fit.combined
        assert fit.reason == "scale_mismatch"
        assert fit.knee_at_bdc is None
        assert fit.scale_disagreement_pct > 2.0

    def test_two_clips_of_the_same_side_are_not_a_pair(self):
        fit = combine_sides(LEFT_4149, _side("left", 2.32, 2.31, 4.52))
        assert not fit.combined
        assert fit.reason == "same_side"

    def test_a_missing_side_refuses(self):
        assert combine_sides(LEFT_4149, None).reason == "missing_side"
        assert combine_sides(None, None).reason == "missing_side"

    def test_wildly_different_legs_refuse(self):
        # Same chord (passes the scale gate) but one leg 15% longer: not two
        # views of one rider.
        other = _side("right", 2.322 * 1.15, 2.308 * 1.15, 4.520)
        fit = combine_sides(LEFT_4149, other)
        assert not fit.combined
        assert fit.reason == "leg_mismatch"

    def test_a_refusal_carries_no_number_anywhere(self):
        fit = combine_sides(LEFT_4088, RIGHT_4090)
        d = fit.as_dict()
        assert d["knee_at_bdc"] is None
        assert d["uncertainty_deg"] is None
        assert d["per_side"] == {}
        assert d["asymmetry_deg"] is None


class TestAsymmetryReporting:
    def test_a_genuinely_asymmetric_pair_is_reported(self):
        # One leg reaching 3% further at the bottom is a real difference, not
        # the model: same body, different chords.
        left = _side("left", 2.35, 2.30, 4.60, chord_sd=0.010)
        right = _side("right", 2.35, 2.30, 4.52, chord_sd=0.010)
        fit = combine_sides(left, right)
        assert fit.combined
        assert fit.asymmetry_significant is True
        assert abs(fit.asymmetry_deg) > fit.asymmetry_floor_deg

    def test_a_noisy_pair_raises_its_own_bar(self):
        # Same difference, but the chord wanders three times as much between
        # revolutions -- the floor rises with it and the claim is withdrawn.
        left = _side("left", 2.35, 2.30, 4.60, chord_sd=0.075)
        right = _side("right", 2.35, 2.30, 4.52, chord_sd=0.075)
        fit = combine_sides(left, right)
        assert fit.combined
        assert fit.asymmetry_significant is False

    def test_asymmetry_sign_is_left_minus_right(self):
        left = _side("left", 2.35, 2.30, 4.60, chord_sd=0.010)
        right = _side("right", 2.35, 2.30, 4.52, chord_sd=0.010)
        assert combine_sides(left, right).asymmetry_deg > 0


class TestSummarizeSide:
    """The reducer that turns stabilized frames into a SideGeometry."""

    @staticmethod
    def _clip(n=240, *, fill_side=None, radius=0.06, hip=(0.5, 0.25)):
        """A synthetic rider: ankle on a circle, knee on the hip-ankle line
        offset sideways, so the geometry is known exactly."""
        import types
        frames = []
        for i in range(n):
            phase = 2 * math.pi * i / 40.0
            ax = hip[0] + radius * math.sin(phase)
            ay = hip[1] + 0.60 + radius * math.cos(phase)
            kx = (hip[0] + ax) / 2 + 0.05
            ky = (hip[1] + ay) / 2
            lms = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
                   for _ in range(33)]
            lms[23] = types.SimpleNamespace(x=hip[0], y=hip[1], z=0.0, visibility=1.0)
            lms[25] = types.SimpleNamespace(x=kx, y=ky, z=0.0, visibility=1.0)
            lms[27] = types.SimpleNamespace(x=ax, y=ay, z=0.0, visibility=1.0)
            lms[11] = types.SimpleNamespace(x=hip[0], y=hip[1] - 0.2, z=0.0, visibility=1.0)
            fr = {"normalized_landmarks": lms}
            if fill_side:
                fr["leg_gate_filled"] = {fill_side}
            frames.append(fr)
        return frames

    def test_reduces_a_clean_clip(self):
        g = summarize_side(self._clip(), "left", 1.0)
        assert g is not None
        assert g.camera_side == "left"
        assert g.revolutions >= 3
        assert g.thigh > 0 and g.shin > 0
        # The ankle orbit was drawn at radius 0.06, and lengths are reported in
        # units of it, so the leg (~0.60 of frame) lands near 10 radii.
        assert 8.0 < g.leg < 12.0

    def test_gate_filled_frames_are_not_measured(self):
        """A filled frame is a drawing, not a measurement -- letting one into a
        length estimate would be measuring our own reconstruction."""
        assert summarize_side(self._clip(fill_side="left"), "left", 1.0) is None

    def test_a_far_leg_gate_also_disqualifies_the_frame(self):
        """Stricter than the analyzer's per-side rule, and deliberately so.

        On the 26 Aug pair, admitting frames whose FAR leg was gated moved the
        crank-radius estimate 3.9% and took the two clips' agreement on
        hip-to-ankle from 0.22% to 2.72% -- which is the two sides landing 12
        deg apart instead of 0.9. When the gate fires anywhere in the lower
        body, the tracker is losing the leg region, not just one side of it.
        """
        assert summarize_side(self._clip(fill_side="right"), "left", 1.0) is None

    def test_a_short_clip_refuses(self):
        assert summarize_side(self._clip(n=20), "left", 1.0) is None

    def test_empty_and_nonsense_input(self):
        assert summarize_side([], "left", 1.0) is None
        assert summarize_side(self._clip(), "front", 1.0) is None

    def test_aspect_is_applied(self):
        """Portrait clips stretch x; a geometry read without the aspect is a
        different body. Two aspects must not give the same leg length."""
        a = summarize_side(self._clip(), "left", 1.0)
        b = summarize_side(self._clip(), "left", 0.5625)
        assert a is not None and b is not None
        assert abs(a.leg - b.leg) > 0.05

    def test_round_trip_through_combine(self):
        left = summarize_side(self._clip(), "left", 1.0)
        right = summarize_side(self._clip(), "right", 1.0)
        assert right is None  # the synthetic clip only carries left landmarks
        assert isinstance(combine_sides(left, None), BilateralFit)
