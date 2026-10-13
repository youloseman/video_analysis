"""Three aero-bar rules, changed on sources after measuring the Kona 2026 field.

Fourteen professional positions from the Ironman World Championship
(2026-10-10) disagreed with three of our aero-bar rules. Each was changed only
as far as a source supports (Artur's rule: a band moves on evidence, not to
fit the data):

- Forearm tilt ceiling 25 -> 30. UCI raised the forearm-support inclination
  cap from 15 to 30 deg (in force 1 Jan 2023); Maddocks & Walker 2025 (CFD,
  one pro) found drag still falling at ~35 deg. Seven of fourteen pros sat
  above 25; the median WAS 25.
- Shoulder: no source names a floor below 80, so the band stays. Empfield
  (Slowtwitch, 2007): below 90 deg "is the personal choice of the athlete",
  above it "can become an issue". So on aero bars a closed shoulder costs a
  third as much and never goes past amber; an open one costs full rate. Five
  of fourteen pros rode 70-77.
- Aero scale: one bucket for every trunk angle under 28 deg, and ten of
  fourteen pros were in it. Fintelman's thesis measured frontal area at
  24 / 16 / 8 deg (n=19), so the bucket splits into three zones centred on
  those, and the drag figure inside the TT range is interpolated on them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.video_analysis.biomechanics import aero_estimator as AE
from app.services.video_analysis.biomechanics import technique_scorer as TS
from app.services.video_analysis.biomechanics.cycling_positions import (
    AERO_POSITIONS,
    CYCLING_POSITIONS,
    SHOULDER_CLOSED_FORGIVENESS,
    SHOULDER_CLOSED_OK_POSITIONS,
    get_position_meta,
)

HTML = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


class TestForearmTilt:
    @pytest.mark.parametrize("pos", ["tt_aero", "triathlon"])
    def test_the_ceiling_is_the_uci_pad_cap(self, pos):
        assert CYCLING_POSITIONS[pos]["forearm_tilt"] == (5, 30)
        src = get_position_meta(pos, "forearm_tilt")["source"]
        assert "30 deg" in src and "Maddocks" in src

    def test_road_positions_are_untouched(self):
        assert CYCLING_POSITIONS["road_hoods"]["forearm_tilt"] == (5, 20)
        assert CYCLING_POSITIONS["road_drops"]["forearm_tilt"] == (0, 15)

    def test_the_client_mirrors_it(self):
        for pos in ("tt_aero", "triathlon"):
            i = HTML.index(f"  {pos}:{{trunk_angle:")
            line = HTML[i:HTML.index("\n", i)]
            assert "forearm_tilt:[5,30]" in line, line


class TestTheClosedShoulder:
    def test_it_applies_on_aero_bars_only(self):
        assert SHOULDER_CLOSED_OK_POSITIONS == AERO_POSITIONS
        assert SHOULDER_CLOSED_FORGIVENESS == 3.0
        assert CYCLING_POSITIONS["tt_aero"]["shoulder_angle"] == (80, 100), "the band itself did not move"

    def test_video_severity_closed_is_acceptable_open_is_not(self):
        c = TS.classify_metric_severity
        assert c("triathlon", "shoulder_angle", 70.0) == "ACCEPTABLE"
        assert c("tt_aero", "shoulder_angle", 72.0) == "ACCEPTABLE"
        assert c("tt_aero", "shoulder_angle", 120.0) in ("WARNING", "CRITICAL")
        assert c("road_hoods", "shoulder_angle", 65.0) in ("WARNING", "CRITICAL"), "road keeps the symmetric rule"

    def test_video_score_closed_costs_a_third(self):
        base = {"near_side": "left", "left_knee_at_bdc": 141.0, "knee_at_bdc": 141.0,
                "trunk_angle_avg": 22.0, "elbow_angle_avg": 90.0, "frames_analyzed": 300,
                "forearm_tilt_avg": 15.0, "head_alignment_avg": 95.0}
        closed = TS.score_cycling({**base, "shoulder_angle_avg": 70.0}, {}, cycling_position="triathlon")
        road = TS.score_cycling({**base, "shoulder_angle_avg": 70.0}, {}, cycling_position="road_hoods")
        comp = lambda r: (r.get("component_scores") or r.get("components") or {}).get("shoulder_angle")  # noqa: E731
        assert comp(closed) is not None and comp(road) is not None
        assert comp(closed) > comp(road)
        # 10 deg closed on aero bars grades like ~3.3 deg past the band
        assert comp(closed) == pytest.approx(TS.score_in_range(80 - 10 / 3, 80, 105), abs=0.5)

    def test_photo_closed_shoulder_is_amber_with_its_reason(self):
        import math
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from test_photo_pedal_phase import _bike_pose

        from app.services.video_analysis import photo_analyzer as PA
        world, norm = _bike_pose(142.0)
        res = PA.analyze_from_pose(np.full((1280, 720, 3), 128, np.uint8), world, norm,
                                   "right", [], "bike", "triathlon")
        sh = res["angles"]["shoulder"]
        assert not math.isnan(sh)
        row = res["angles_with_context"]["shoulder"]
        if sh < 80:
            assert row["status"] == "acceptable" and "choice" in row["note"]
        # the photo scorer carries the same forgiveness
        assert PA._AERO_TOLERANCES["shoulder"] == (SHOULDER_CLOSED_FORGIVENESS, 1.0)

    def test_the_overlay_never_paints_it_red(self):
        from app.services.video_analysis.video_visualizer import _status_for_cfg
        cfg = {"optimal": (80, 100), "closed_ok": True}
        assert _status_for_cfg(cfg, 65.0) == "warn"
        assert _status_for_cfg(cfg, 125.0) == "bad"

    def test_the_overlay_marks_it_on_aero_positions(self):
        from app.services.video_analysis.pipeline import VideoAnalysisPipeline
        p = VideoAnalysisPipeline.__new__(VideoAnalysisPipeline)
        for pos, expect in (("triathlon", True), ("road_hoods", False)):
            cfgs = p._get_angle_display_config("bike", {"camera_side": "left"}, cycling_position=pos)
            sh = next(c for c in cfgs if c["key"].endswith("shoulder"))
            assert bool(sh.get("closed_ok")) is expect, pos

    def test_the_report_tile_caps_it_at_amber(self):
        i = HTML.index("function grZone(")
        fn = HTML[i:HTML.index("\n}", i)]
        assert "metric==='shoulder_closed_ok'" in fn and "v<lo) return 'warning'" in fn
        assert "shR.metric='shoulder_closed_ok'" in HTML


class TestTheAeroScale:
    def test_the_bucket_under_28_is_split_on_fintelmans_angles(self):
        tt = [z for z in AE.AERO_ZONES if z["trunk_band"][1] <= 28]
        assert [z["trunk_band"] for z in tt] == [(20, 28), (12, 20), (0, 12)]
        rels = [z["rel"] for z in tt]
        assert rels == sorted(rels, reverse=True)
        assert rels[1] == pytest.approx(0.80 * 0.318 / 0.335, abs=1e-3)
        assert rels[2] == pytest.approx(0.80 * 0.302 / 0.335, abs=1e-3)

    def test_the_kona_field_spreads_across_the_zones(self):
        kona = [11.9, 14.6, 15.4, 16.4, 18.8, 18.9, 20.0, 20.8, 21.4, 21.6, 24.7, 30.0, 31.9, 39.1]
        labels = {AE._zone_for_trunk(t)["key"] for t in kona if t < 28}
        assert labels == {"tt_tuck", "tt_low", "tt_flat"}

    def test_two_degrees_are_worth_two_degrees(self):
        # 22 -> 20 on the measured slope is ~1.3 %, under the 2 % noise floor:
        # no suggestion, where the zone step would have claimed 5 %.
        r = AE.estimate_aero(22.0, "triathlon", optimal_trunk_band=(20, 30))
        assert r["next_zone"] is None
        # The saving is the slope between the two angles, nothing more.
        r = AE.estimate_aero(26.0, "triathlon", optimal_trunk_band=(20, 30))
        nz = r["next_zone"]
        expect = (AE._rel_for_trunk(26.0) - AE._rel_for_trunk(nz["target_trunk_angle"])) / AE._rel_for_trunk(26.0)
        assert nz["drag_reduction_pct"] == pytest.approx(expect * 100, abs=0.1)

    def test_a_degree_over_the_tt_zone_is_not_worth_a_zone(self):
        """A $4 report (2026-10-13), torso 28.9 on a tri bike: told to go
        'toward 28.0' for 8 % and ~15 W, the whole step between two postures.
        Now the target is the middle of the next zone and the saving is the
        slope between the measured postures."""
        r = AE.estimate_aero(28.9, "triathlon", optimal_trunk_band=(20, 30))
        assert r["next_zone"]["target_trunk_angle"] == 24.0
        assert r["next_zone"]["drag_reduction_pct"] < 5.0
        a, b = AE._rel_for_trunk(27.9), AE._rel_for_trunk(28.1)
        assert abs(a - b) < 0.005, "no step at 28 deg"

    def test_the_road_steps_are_unchanged(self):
        r = AE.estimate_aero(33.0, "triathlon", optimal_trunk_band=(20, 30))
        assert r["zone_key"] == "dropped_low"
        assert r["next_zone"]["zone_key"] == "tt_tuck"
        assert r["next_zone"]["drag_reduction_pct"] == pytest.approx(8.0, abs=0.1)

    def test_the_source_says_where_the_number_came_from(self):
        assert "Fintelman" in AE.estimate_aero(16.0, "tt_aero", optimal_trunk_band=(15, 30))["source"]
        assert "Fintelman" not in AE.estimate_aero(40.0, "road_hoods", optimal_trunk_band=(40, 55))["source"]


class TestTheHeadLever:
    """The aero head lever read the retired head score and told a rider with
    the head held high that it was 'well tucked' (2026-10-13). It reads the
    helmet now, on the same field and free band as the score."""

    def test_no_head_score_reaches_the_aero_read_out(self):
        import inspect
        assert "head_alignment" not in inspect.signature(AE.estimate_aero).parameters
        r = AE.estimate_aero(24.0, "triathlon", optimal_trunk_band=(20, 30))
        assert not any(lv["key"] == "head" for lv in r["levers"])

    def test_a_high_helmet_is_a_lever_a_low_one_is_fine(self):
        base = AE.estimate_aero(24.0, "triathlon", optimal_trunk_band=(20, 30))
        high = AE.with_helmet_lever(base, 27.0, False, "triathlon")
        head = next(lv for lv in high["levers"] if lv["key"] == "head")
        assert head["status"] == "available" and "Kona pros" in head["cue"]
        assert "drag_reduction_pct" not in head, "no invented watt figure"
        low = AE.with_helmet_lever(base, 15.0, False, "triathlon")
        assert next(lv for lv in low["levers"] if lv["key"] == "head")["status"] == "ok"

    def test_an_unreadable_helmet_asks_for_the_race_helmet(self):
        base = AE.estimate_aero(29.0, "triathlon", optimal_trunk_band=(20, 30))
        out = AE.with_helmet_lever(base, 25.9, True, "triathlon")
        head = next(lv for lv in out["levers"] if lv["key"] == "head")
        assert head["status"] == "unknown" and "race helmet" in head["cue"]
        assert AE.with_helmet_lever(base, 25.9, False, "road_hoods")["levers"] == [
            lv for lv in base["levers"] if lv["key"] != "head"]
