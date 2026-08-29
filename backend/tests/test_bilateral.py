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
    merge_summaries,
    midline_agreement,
    summarize_side,
)


def _side(camera_side, thigh, shin, chord, *, chord_sd=0.028, revs=8,
          torso=2.6, radius=0.055):
    # `radius` matters: every length above is expressed in crank radii, so a
    # second ruler (the torso) can only be formed once the clip's own radius
    # is known. Fixtures that shared one radius silently made the two rulers
    # identical and hid the whole fallback.
    return SideGeometry(
        camera_side=camera_side, thigh=thigh, shin=shin, torso=torso,
        chord_bdc=chord, chord_sd=chord_sd, revolutions=revs,
        measured_frames=300, crank_radius_px=radius,
    )


# The real pairs, exactly as summarize_side reduces them.
RIGHT_4148 = _side("right", 2.4423, 2.2756, 4.5102, chord_sd=0.0301, revs=5,
                   torso=2.6495, radius=0.054067)
LEFT_4149 = _side("left", 2.3226, 2.3100, 4.5202, chord_sd=0.0263, revs=11,
                  torso=2.5432, radius=0.051721)
RIGHT_4090 = _side("right", 2.5926, 2.4800, 4.9253, chord_sd=0.0325, revs=13,
                   torso=2.8189, radius=0.065334)
LEFT_4088 = _side("left", 2.3023, 2.1893, 4.3118, chord_sd=0.0260, revs=13,
                  torso=2.5083, radius=0.079713)


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

    def test_it_combines(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert fit.combined
        assert fit.scale_anchor in ("crank", "torso")

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

    def test_no_asymmetry_is_published_at_all(self):
        """See TestWhyThereIsNoAsymmetryNumber. The per-side split turned out
        to be the scale choice restated, so it is not offered in any form."""
        d = combine_sides(LEFT_4149, RIGHT_4148).as_dict()
        assert "asymmetry_deg" not in d
        assert "per_side" not in d

    def test_argument_order_barely_moves_the_answer(self):
        # Not bit-identical any more: the scale ratio is formed from the first
        # argument's units, so swapping them re-expresses the pool. The ANSWER
        # must not care.
        a = combine_sides(LEFT_4149, RIGHT_4148)
        b = combine_sides(RIGHT_4148, LEFT_4149)
        assert a.knee_at_bdc == pytest.approx(b.knee_at_bdc, abs=0.5)
        assert a.raw_per_side == b.raw_per_side


class TestTheErrorBarIsHonest:
    """A reflection cannot change a real angle. Whatever it DOES change is
    error, so it is the one available audit of the stated uncertainty.

    Re-detecting both clips with the pixels mirrored and merging that pair
    gives a different answer -- pooling ends the disagreement BETWEEN the two
    sides, it does not remove a bias they share. The error bar has to be big
    enough to cover that, or it is decoration.
    """

    # The same two clips, re-detected mirrored (measured 2026-08-26).
    MIRRORED_LEFT = _side("left", 2.3624, 2.2352, 4.4633, chord_sd=0.0508,
                          revs=4, torso=2.5891, radius=0.055083)
    MIRRORED_RIGHT = _side("right", 2.3654, 2.2108, 4.5063, chord_sd=0.0110,
                           revs=4, torso=2.6138, radius=0.052035)

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


class TestWhyThereIsNoAsymmetryNumber:
    """Measured 2026-08-29, and it cost the feature its headline claim.

    Pooling needs the ratio of the two clips' image scales. Whatever ratio is
    used, the resulting difference between the two sides came out as the
    clips' disagreement about the hip-ankle chord times the sensitivity
    constant -- across four pairs and two anchors the ratio was 4.1, 4.6, 3.6,
    4.3, 5.4, i.e. the ~4 deg per 1% lever, every time.

    So the per-side split carries no information the chord disagreement did
    not already carry, and the chord disagreement is exactly what choosing a
    scale controls. Picking the scale that makes the bike look unchanged --
    which is the physically right thing to do -- therefore FORCES the legs to
    look equal. Asymmetry and scale error are degenerate; the honest move is
    to publish neither a per-side leg nor an asymmetry.
    """

    def test_the_split_is_proportional_to_the_chord_disagreement(self):
        """One body, chords a little apart: the "asymmetry" that would fall
        out grows in step with the chord gap and with nothing else.

        Proportionality is the claim, not a particular constant -- the lever
        steepens as the leg straightens, which is why the measured ratios
        across the real pairs ranged 3.6 to 5.4 rather than sitting on one
        number. What matters is that the split has no other source.
        """
        t, sh, base = 2.40, 2.35, 4.55
        ratios = []
        for pct in (0.5, 1.0, 2.0):
            gap = abs(knee_from_chord(t, sh, base)
                      - knee_from_chord(t, sh, base * (1 + pct / 100)))
            ratios.append(gap / pct)
        assert max(ratios) / min(ratios) < 1.35, (
            f"gap-per-percent should be near constant, got {ratios}"
        )
        assert min(ratios) > 1.0

    def test_the_payload_offers_no_per_leg_number(self):
        d = combine_sides(LEFT_4149, RIGHT_4148).as_dict()
        for banned in ("per_side", "asymmetry_deg", "asymmetry_significant",
                       "asymmetry_floor_deg"):
            assert banned not in d, f"{banned} is back in the payload"

    def test_what_each_clip_measured_alone_is_still_reported(self):
        """That is a fact about a clip, and it is how the reader sees where
        the combined number came from."""
        d = combine_sides(LEFT_4149, RIGHT_4148).as_dict()
        assert set(d["raw_per_side"]) == {"left", "right"}


class TestItPicksARuler:
    """Two clips of one session must agree about the hip-ankle chord, because
    the bike did not change between them. Whichever scale makes that true is
    the one to pool with."""

    def test_a_chainring_corrupted_orbit_falls_back_to_the_body(self):
        """The 24 Aug pair: 4090's ankle orbit was chainring-corrupted, so its
        crank radius is ~11% out. The old build refused the pair outright --
        two of the three pairs ever filmed. The torso ruler rescues it."""
        fit = combine_sides(LEFT_4088, RIGHT_4090)
        assert fit.combined
        assert fit.scale_anchor == "torso"
        assert fit.scale_chord_disagreement_pct < 5.0

    def test_a_clean_orbit_keeps_the_crank(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert fit.scale_anchor == "crank"
        assert fit.scale_chord_disagreement_pct < 1.0

    def test_the_ruler_barely_changes_the_answer(self):
        """Which is why refusing over it was overcautious: scaling a whole
        triangle does not change its shape."""
        for fit in (combine_sides(LEFT_4149, RIGHT_4148),
                    combine_sides(LEFT_4088, RIGHT_4090)):
            assert fit.combined
            assert fit.scale_anchor_spread_deg < 1.5

    def test_the_other_rulers_disagreement_is_inside_the_error_bar(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        assert fit.uncertainty_deg >= fit.scale_anchor_spread_deg


class TestItRefuses:

    def test_two_clips_of_the_same_side_are_not_a_pair(self):
        fit = combine_sides(LEFT_4149, _side("left", 2.32, 2.31, 4.52))
        assert not fit.combined
        assert fit.reason == "same_side"

    def test_a_missing_side_refuses(self):
        assert combine_sides(LEFT_4149, None).reason == "missing_side"
        assert combine_sides(None, None).reason == "missing_side"

    def test_wildly_different_legs_refuse(self):
        # Same chord, so no ruler can reconcile a leg 20% longer: this is not
        # two views of one rider.
        other = _side("right", 2.322 * 1.20, 2.308 * 1.20, 4.520)
        fit = combine_sides(LEFT_4149, other)
        assert not fit.combined
        assert fit.reason == "leg_mismatch"

    def test_a_refusal_carries_no_number_anywhere(self):
        other = _side("right", 2.322 * 1.20, 2.308 * 1.20, 4.520)
        d = combine_sides(LEFT_4149, other).as_dict()
        assert d["combined"] is False
        assert d["knee_at_bdc"] is None
        assert d["uncertainty_deg"] is None
        assert d["raw_per_side"] == {}


class TestMergeSummaries:
    """One set of metrics for the rider, from two one-legged clips."""

    @staticmethod
    def _summary(side, knee_bdc, *, trunk=68.0, revs=8, tdc=64.0):
        return {
            "camera_side": side, "near_side": side,
            "camera_side_label": side.capitalize(),
            "knee_at_bdc": knee_bdc, f"{side}_knee_at_bdc": knee_bdc,
            "knee_at_tdc": tdc, "trunk_angle_avg": trunk,
            "hip_angle_avg": 25.0, "elbow_angle_avg": 150.0,
            "shoulder_angle_avg": 88.0, "pelvic_ratio": 0.37,
            "saddle_height_assessment": "optimal", "frames_analyzed": 400,
            "bilateral_geometry": {"revolutions": revs},
        }

    def _merged(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        return fit, merge_summaries(
            self._summary("left", 154.7, revs=11),
            self._summary("right", 145.8, revs=5),
            fit, "road_hoods",
        )

    def test_the_scored_knee_is_the_merged_one(self):
        fit, merged = self._merged()
        assert merged["knee_at_bdc"] == pytest.approx(fit.knee_at_bdc)

    def test_both_per_side_keys_get_the_merged_value(self):
        """score_cycling reads `{near}_knee_at_bdc` BEFORE the plain key, so
        leaving them alone would score the merged ride on one unmerged leg.
        They get the SAME number because the method cannot tell the legs
        apart from its own scale error -- see
        TestWhyThereIsNoAsymmetryNumber."""
        fit, merged = self._merged()
        assert merged["left_knee_at_bdc"] == pytest.approx(fit.knee_at_bdc)
        assert merged["right_knee_at_bdc"] == pytest.approx(fit.knee_at_bdc)
        assert merged["left_knee_at_bdc"] != 154.7

    def test_the_saddle_verdict_is_re_derived_not_inherited(self):
        # Both inputs claimed "optimal"; the merged 150 is above the road band
        # (138-145) by more than the 5 deg slack, so it must not stay optimal.
        _, merged = self._merged()
        assert merged["saddle_height_assessment"] != "optimal"

    def test_midline_metrics_are_averaged(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        merged = merge_summaries(
            self._summary("left", 154.7, trunk=70.0, revs=11),
            self._summary("right", 145.8, trunk=66.0, revs=5),
            fit, "road_hoods",
        )
        assert merged["trunk_angle_avg"] == pytest.approx(68.0)

    def test_it_builds_on_the_better_supported_clip(self):
        fit = combine_sides(LEFT_4149, RIGHT_4148)
        merged = merge_summaries(
            self._summary("left", 154.7, revs=11),
            self._summary("right", 145.8, revs=2),
            fit, "road_hoods",
        )
        assert merged["frames_analyzed"] == 800
        assert merged["camera_side"] == "both"

    def test_a_refusal_cannot_be_merged(self):
        # A leg 20% longer on one side: no ruler reconciles that.
        refusal = combine_sides(
            LEFT_4149, _side("right", 2.79, 2.77, 4.5202, radius=0.051721))
        with pytest.raises(ValueError):
            merge_summaries(self._summary("left", 149.0),
                            self._summary("right", 152.0), refusal, "road_hoods")

    def test_the_merge_carries_its_own_provenance(self):
        _, merged = self._merged()
        assert merged["bilateral"]["combined"] is True
        assert merged["bilateral"]["uncertainty_deg"] is not None


class TestMidlineAgreement:
    """Metrics that cannot differ by side become the session's error gauge."""

    def test_close_clips_agree(self):
        a = TestMergeSummaries._summary("left", 150.0, trunk=68.3)
        b = TestMergeSummaries._summary("right", 143.0, trunk=67.3)
        out = midline_agreement(a, b)
        assert out["agree"] is True
        assert out["worst"] < 2.0

    def test_a_badly_framed_clip_shows_up_here(self):
        a = TestMergeSummaries._summary("left", 150.0, trunk=68.0)
        b = TestMergeSummaries._summary("right", 143.0, trunk=50.0)
        out = midline_agreement(a, b)
        assert out["agree"] is False
        assert out["worst_metric"] == "trunk_angle_avg"

    def test_no_shared_metrics_is_not_an_answer(self):
        assert midline_agreement({}, {})["agree"] is None


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
