"""A two-sided session whose knee cannot be pooled, but whose rider can.

The numbers are the real pair's (IMG_4527 left / IMG_4525 right, 17 Sep):
the drive-side clip has no pedal circle -- its far leg was occluded on 138 of
180 frames -- while the two clips agree on the trunk to 4.8 deg, the hip to
1.5, the shoulder to 4.1 and the elbow to 1.2. Refusing the whole session
over the knee threw away four measurements the pair had made twice.

What is pinned:

* those four are pooled, and the knee (both ends of the stroke), the saddle
  verdict and the score's near side stay the knee clip's own;
* the reason for the knee is said, and the drive side is named as such;
* the SPA has a branch for it that prints one knee, labelled by clip;
* a pair that disagrees on the midline is refused for THAT, whatever the
  knee did; a pair refused for not being one rider stays refused.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis.bilateral_session import build_pair_result
from app.services.video_analysis.biomechanics.bilateral import (
    _MIDLINE_METRICS,
    merge_summaries_partial,
)

GEOM_LEFT = {
    "camera_side": "left", "thigh": 2.475, "shin": 2.376, "leg": 4.852,
    "torso": 2.636, "chord_bdc": 4.717, "chord_sd": 0.012, "revolutions": 8,
    "measured_frames": 140, "crank_radius_px": 0.0633,
}


def _clip(side, *, geom, knee_bdc, knee_tdc, trunk, hip, shoulder, elbow, frames, warn=()):
    return {
        "status": "completed", "sport_type": "bike", "cycling_position": "road_hoods",
        "camera_side": side, "frames_analyzed": frames,
        "technique_score": 90, "letter_grade": "A",
        "keyframe_base64": f"data:image/jpeg;base64,{side}",
        "bilateral_geometry": geom,
        "angle_statistics": {f"{side}_knee": {"mean": 110.0, "valid_frames": 100}},
        "detected_issues": [],
        "sport_specific_metrics": {
            "camera_side": side, "near_side": side,
            "knee_at_bdc": knee_bdc, f"{side}_knee_at_bdc": knee_bdc,
            "knee_at_tdc": knee_tdc, f"{side}_knee_at_tdc": knee_tdc,
            "trunk_angle_avg": trunk, "hip_angle_avg": hip,
            "shoulder_angle_avg": shoulder, "elbow_angle_avg": elbow,
            "pelvic_ratio": 2.2, "frames_analyzed": frames,
            "saddle_height_assessment": "acceptable" if side == "left" else "optimal",
            "bilateral_geometry": geom, "quality_warnings": list(warn),
        },
    }


LEFT = _clip("left", geom=GEOM_LEFT, knee_bdc=148.07, knee_tdc=73.59,
             trunk=39.74, hip=88.57, shoulder=77.34, elbow=159.7, frames=197)
RIGHT_DRIVE = _clip("right", geom=None, knee_bdc=139.2, knee_tdc=66.4,
                    trunk=44.51, hip=87.11, shoulder=73.23, elbow=160.92, frames=180,
                    warn=["This clip was filmed from the drive side (chainring toward the camera)."])


@pytest.fixture(scope="module")
def out():
    return build_pair_result(LEFT, RIGHT_DRIVE, "road_hoods")


class TestWhatIsPooled:
    def test_it_is_partial_not_a_refusal(self, out):
        b = out["bilateral"]
        assert b["combined"] is False and b["partial"] is True
        assert b["reason"] == "geometry_unavailable"
        assert b["knee_side"] == "left" and b["base_side"] == "left"
        assert "metrics_side" not in b, "not a one-clip refusal"

    def test_the_midline_is_averaged(self, out):
        sm = out["sport_specific_metrics"]
        assert sm["trunk_angle_avg"] == pytest.approx((39.74 + 44.51) / 2)
        assert sm["hip_angle_avg"] == pytest.approx((88.57 + 87.11) / 2)
        assert sm["shoulder_angle_avg"] == pytest.approx((77.34 + 73.23) / 2)
        assert sm["elbow_angle_avg"] == pytest.approx((159.7 + 160.92) / 2)

    def test_the_knee_is_one_clips_at_both_ends_of_the_stroke(self, out):
        sm = out["sport_specific_metrics"]
        assert sm["knee_at_bdc"] == 148.07 and sm["knee_at_tdc"] == 73.59
        assert sm["left_knee_at_bdc"] == 148.07
        assert "right_knee_at_bdc" not in sm, "no invented second leg"
        assert sm["near_side"] == "left", "the scorer reads the knee clip's leg"
        assert sm["saddle_height_assessment"] == "acceptable", "the knee clip's own verdict"
        assert "knee_at_tdc" not in _MIDLINE_METRICS

    def test_the_session_is_two_sided_where_it_can_be(self, out):
        assert out["camera_side"] == "both"
        assert out["frames_analyzed"] == 377
        assert out["sport_specific_metrics"]["frames_analyzed"] == 377
        assert out["keyframe_base64"].endswith("left")

    def test_there_is_one_score_and_it_is_rescored(self, out):
        assert isinstance(out["technique_score"], int)
        assert out["score_breakdown"]

    def test_the_knee_is_offered_as_one_clips_reading(self, out):
        ks = out["bilateral"]["knee_single"]
        assert ks == {"side": "left", "value": 148.07, "revolutions": 8}
        assert out["bilateral"].get("knee_at_bdc") is None, "no merged number is claimed"

    def test_the_agreement_is_the_error_bar(self, out):
        a = out["bilateral"]["agreement"]
        assert a["agree"] is True and a["worst"] == 4.8
        assert a["worst_metric"] == "trunk_angle_avg"


class TestWhatIsSaid:
    def test_the_warning_leads_and_names_the_drive_side(self, out):
        w = out["sport_specific_metrics"]["quality_warnings"]
        assert w[0].startswith("Merged on everything both clips see")
        assert "4.8" in w[0]
        assert "left-side clip alone" in w[0]
        assert "drive side" in w[0]
        assert "chainring" in w[0]

    def test_both_clips_own_warnings_survive(self, out):
        w = out["sport_specific_metrics"]["quality_warnings"]
        assert any("filmed from the drive side (chainring toward" in x for x in w)

    def test_both_side_cards_are_there(self, out):
        assert {c["camera_side"] for c in out["bilateral"]["sides"]} == {"left", "right"}


class TestWhenItMustNotHappen:
    def test_a_midline_disagreement_is_refused_for_that(self):
        far = _clip("right", geom=None, knee_bdc=139.2, knee_tdc=66.4,
                    trunk=55.0, hip=87.11, shoulder=73.23, elbow=160.92, frames=180)
        out = build_pair_result(LEFT, far, "road_hoods")
        b = out["bilateral"]
        assert b["combined"] is False and not b.get("partial")
        assert b["reason"] == "midline_disagree"
        assert b["agreement"]["agree"] is False

    def test_nothing_measured_twice_means_no_merge(self):
        mute = _clip("right", geom=None, knee_bdc=139.2, knee_tdc=66.4,
                     trunk=None, hip=None, shoulder=None, elbow=None, frames=180)
        for k in ("trunk_angle_avg", "hip_angle_avg", "shoulder_angle_avg", "elbow_angle_avg"):
            mute["sport_specific_metrics"].pop(k)
        out = build_pair_result(LEFT, mute, "road_hoods")
        assert out["bilateral"]["combined"] is False
        assert not out["bilateral"].get("partial")
        assert out["bilateral"]["reason"] == "geometry_unavailable"

    def test_the_knee_clip_is_the_one_with_a_circle(self):
        # Same pair the other way round: the circle is on the right this time.
        left_drive = dict(LEFT, camera_side="left", bilateral_geometry=None)
        left_drive["sport_specific_metrics"] = dict(LEFT["sport_specific_metrics"], bilateral_geometry=None)
        right_ok = dict(RIGHT_DRIVE, bilateral_geometry=dict(GEOM_LEFT, camera_side="right"))
        right_ok["sport_specific_metrics"] = dict(RIGHT_DRIVE["sport_specific_metrics"],
                                                  bilateral_geometry=right_ok["bilateral_geometry"])
        out = build_pair_result(left_drive, right_ok, "road_hoods")
        assert out["bilateral"]["partial"] is True
        assert out["bilateral"]["knee_side"] == "right"
        assert out["sport_specific_metrics"]["knee_at_bdc"] == 139.2
        assert "drive side" not in out["sport_specific_metrics"]["quality_warnings"][0]


class TestTheHelper:
    def test_partial_merge_leaves_the_knee_and_pools_the_rest(self):
        m = merge_summaries_partial(
            {"knee_at_bdc": 148.0, "knee_at_tdc": 73.0, "trunk_angle_avg": 40.0,
             "near_side": "left", "frames_analyzed": 10},
            {"knee_at_bdc": 139.0, "knee_at_tdc": 66.0, "trunk_angle_avg": 44.0,
             "pelvic_ratio": 2.0, "frames_analyzed": 5},
        )
        assert m["knee_at_bdc"] == 148.0 and m["knee_at_tdc"] == 73.0
        assert m["trunk_angle_avg"] == 42.0
        assert m["pelvic_ratio"] == 2.0, "a metric only the other clip has is still one reading"
        assert m["near_side"] == "left"
        assert m["camera_side"] == "both" and m["frames_analyzed"] == 15


class TestThePanel:
    @pytest.fixture(scope="class")
    def fn(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("function renderBilateral(")
        return html[i:html.index("\nfunction ", i + 10)]

    def test_there_is_a_partial_branch_that_prints_one_labelled_knee(self, fn):
        assert "if(b.partial){" in fn
        branch = fn[fn.index("if(b.partial){"):fn.index("if(!b.combined){")]
        assert "knee_single" in branch
        assert "clip only" in branch
        assert "merged, except the knee" in branch
        assert "drive side" in branch

    def test_the_partial_branch_shows_the_videos_before_the_stills(self, fn):
        branch = fn[fn.index("if(b.partial){"):fn.index("if(!b.combined){")]
        assert branch.index("${vids}") < branch.index('class="bil-sides"')

    def test_the_new_refusal_reason_has_copy(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("const BIL_REASONS={")
        assert "midline_disagree:" in html[i:html.index("};", i)]
