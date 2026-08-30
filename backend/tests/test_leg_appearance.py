"""Pixels as identity evidence, and the ways that can go wrong quietly.

The run resolver decides each link by how far the joints moved, and geometry
is blind at exactly the moment identity is decided: when the legs cross, both
matchings cost nearly the same. Appearance is not blind there, so the link
cost gained a term built from small patches down each shin.

What this file guards is mostly the failure modes that would not raise:

* a descriptor that encodes BRIGHTNESS rather than texture -- the near leg is
  usually the brighter one, so such a descriptor would "identify" legs by
  lighting that changes with the stride and be confidently wrong;
* a descriptor whose size depends on how much of the frame the athlete fills,
  which would make the evidence mean different things on different clips;
* costs that depend on the parity the resolver is still choosing, which would
  make the two branches incomparable;
* silent breakage when a clip has no patches at all (cached fixtures, the
  photo path, bike) -- that has to degrade to geometry, not to an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis.biomechanics.leg_appearance import (
    PATCH_N,
    describe_legs,
    link_costs,
    patch_distance,
)


def _frame(h=400, w=300, seed=0):
    """A frame with texture, so patches are not degenerate."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, size=(h, w, 3))).astype(np.uint8)


class _LM:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.z, self.visibility = 0.0, 1.0


def _landmarks(left=((0.4, 0.5), (0.4, 0.8)), right=((0.6, 0.5), (0.6, 0.8))):
    lms = [_LM(0.5, 0.5) for _ in range(33)]
    lms[25], lms[27] = _LM(*left[0]), _LM(*left[1])
    lms[26], lms[28] = _LM(*right[0]), _LM(*right[1])
    return lms


class TestDescriptor:
    def test_it_describes_both_legs(self):
        d = describe_legs(_frame(), _landmarks())
        assert set(d) == {"left", "right"}
        assert all(p.shape == (PATCH_N, PATCH_N) for p in d["left"])

    def test_a_missing_shin_yields_no_descriptor_for_that_leg(self):
        lms = _landmarks()
        lms[27] = _LM(float("nan"), float("nan"))
        d = describe_legs(_frame(), lms)
        assert d["left"] is None
        assert d["right"] is not None

    def test_it_is_blind_to_brightness(self):
        """The near leg is usually the brighter one. A descriptor that noticed
        would separate the legs by lighting -- which changes through the
        stride -- and be confidently wrong at every crossing."""
        img = _frame(seed=1)
        dark = describe_legs(img, _landmarks())
        bright = describe_legs(np.clip(img.astype(int) + 40, 0, 255).astype(np.uint8),
                               _landmarks())
        d = patch_distance(dark["left"], bright["left"])
        assert d is not None
        assert d < 0.35, f"brightness moved the descriptor by {d:.3f}"

    def test_the_patch_scales_with_the_athlete(self):
        """Sampled at a fraction of the shin, so a rider filling the frame and
        one in the corner produce comparable evidence rather than one
        describing a calf and the other a knee."""
        big = describe_legs(_frame(seed=2),
                            _landmarks(left=((0.4, 0.2), (0.4, 0.9))))
        small = describe_legs(_frame(seed=2),
                              _landmarks(left=((0.4, 0.55), (0.4, 0.70))))
        assert big["left"] is not None and small["left"] is not None
        assert big["left"][0].shape == small["left"][0].shape

    def test_a_flat_patch_is_refused(self):
        """No texture means nothing to match on -- usually the leg has left
        the frame and this is a crop of blown-out wall."""
        flat = np.full((400, 300, 3), 200, dtype=np.uint8)
        assert describe_legs(flat, _landmarks()) is None

    def test_no_frame_no_descriptor(self):
        assert describe_legs(None, _landmarks()) is None
        assert describe_legs(_frame(), None) is None


class TestDistance:
    def test_a_patch_matches_itself(self):
        d = describe_legs(_frame(seed=3), _landmarks())
        assert patch_distance(d["left"], d["left"]) == pytest.approx(0.0, abs=1e-6)

    def test_different_places_look_different(self):
        d = describe_legs(_frame(seed=4), _landmarks())
        assert patch_distance(d["left"], d["right"]) > 0.5

    def test_missing_input_is_none_not_zero(self):
        """Zero would read as a perfect match and silently win every link."""
        d = describe_legs(_frame(), _landmarks())
        assert patch_distance(None, d["left"]) is None
        assert patch_distance(d["left"], None) is None


class TestLinkCosts:
    def test_it_scores_both_hypotheses(self):
        a = describe_legs(_frame(seed=5), _landmarks())
        b = describe_legs(_frame(seed=6), _landmarks())
        stay, cross = link_costs(a, b)
        assert stay >= 0 and cross >= 0

    def test_a_clean_continuation_prefers_staying(self):
        a = describe_legs(_frame(seed=7), _landmarks())
        stay, cross = link_costs(a, a)
        assert stay < cross

    def test_swapped_legs_prefer_crossing(self):
        """The whole point: when the two legs' appearances have exchanged
        places, the pixels say so even if the joints are on top of each other.
        """
        a = describe_legs(_frame(seed=8), _landmarks())
        b = {"left": a["right"], "right": a["left"]}
        stay, cross = link_costs(a, b)
        assert cross < stay

    def test_it_does_not_depend_on_which_branch_the_path_is_on(self):
        """A parity flip relabels BOTH frames, so the raw-label pairing is
        unchanged. If this cost tracked parity, the two DP branches would be
        scored on different evidence and could not be compared."""
        a = describe_legs(_frame(seed=9), _landmarks())
        b = describe_legs(_frame(seed=10), _landmarks())
        stay, cross = link_costs(a, b)
        flip_a = {"left": a["right"], "right": a["left"]}
        flip_b = {"left": b["right"], "right": b["left"]}
        stay2, cross2 = link_costs(flip_a, flip_b)
        assert stay2 == pytest.approx(stay)
        assert cross2 == pytest.approx(cross)

    def test_absent_patches_yield_no_evidence(self):
        """Cached fixtures, the photo path and bike carry no patches. That has
        to fall back to geometry, not raise and not score zero."""
        a = describe_legs(_frame(), _landmarks())
        assert link_costs(None, a) is None
        assert link_costs(a, None) is None
        assert link_costs({"left": None, "right": None}, a) is None


class TestTheResolverUsesIt:
    """The wiring, not the maths: a term that is computed and then quietly
    ignored would pass every test above and change nothing."""

    @staticmethod
    def _clip(n=120, with_patches=True):
        """A synthetic run whose RAW labels swap halfway, with appearance that
        does not: the pixels stay with their leg, which is the whole premise.
        """
        import numpy as np
        rng = np.random.default_rng(11)
        left_look = [rng.normal(size=(4, 4)).astype("float32") for _ in range(2)]
        right_look = [rng.normal(size=(4, 4)).astype("float32") for _ in range(2)]

        def look(base):
            # A leg looks like itself plus a little frame-to-frame noise.
            # Identical arrays would make every link a perfect match, the
            # median link cost zero, and the term switch itself off -- which
            # is right in production and useless in a fixture.
            return [b + 0.15 * rng.normal(size=b.shape).astype("float32")
                    for b in base]

        frames = []
        for k in range(n):
            phase = 2 * np.pi * k / 30.0
            ly, ry = 0.6 + 0.1 * np.sin(phase), 0.6 - 0.1 * np.sin(phase)
            lms = [_LM(0.5, 0.5) for _ in range(33)]
            swapped = k >= n // 2
            a, b = (ry, ly) if swapped else (ly, ry)
            lms[25], lms[27] = _LM(0.45, a - 0.15), _LM(0.45, a)
            lms[26], lms[28] = _LM(0.55, b - 0.15), _LM(0.55, b)
            fr = {"normalized_landmarks": lms, "world_landmarks": lms,
                  "timestamp_ms": k * 16.7, "frame_idx": k,
                  "frame_width": 300, "frame_height": 400}
            if with_patches:
                la, ra = (right_look, left_look) if swapped else (left_look, right_look)
                fr["leg_patches"] = {"left": look(la), "right": look(ra)}
            frames.append(fr)
        return frames

    def test_the_diagnostic_reports_links_that_used_pixels(self):
        from app.services.video_analysis.biomechanics.leg_identity import (
            resolve_run_leg_identity,
        )
        _, _, _, diag = resolve_run_leg_identity(self._clip())
        assert diag.get("appearance_links", 0) > 0

    def test_without_patches_it_reports_none_and_still_runs(self):
        """Cached fixtures, the photo path and bike carry no patches. The
        resolver must fall back to geometry rather than raise."""
        from app.services.video_analysis.biomechanics.leg_identity import (
            resolve_run_leg_identity,
        )
        _, _, _, diag = resolve_run_leg_identity(self._clip(with_patches=False))
        assert diag.get("appearance_links", 0) == 0

    def test_bike_frames_never_carry_patches(self):
        """Descriptors are taken for run only -- the bike resolver runs in a
        blank-don't-correct mode where the far leg is half invented, and was
        measured getting worse when asked to trust both legs."""
        import inspect
        from app.services.video_analysis import runner
        src = inspect.getsource(runner.extract_frames)
        assert src.index('sport_type == "run"') < src.index("describe_legs")
