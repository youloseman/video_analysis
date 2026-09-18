"""The frame-by-frame record the player reads (``timeline``).

Pinned here: it is built off the same frames and the same filtered series the
report reads, so the curve under the video cannot disagree with the number on
it; its stroke markers are the analyzer's own BDC/TDC frames; the run events
are the overlay's stride counter, first frame of the streak; and a free
reader never receives it.
"""
from __future__ import annotations

from app.services.result_gating import gate_free_result, gate_preview_result
from app.services.video_analysis.timeline import _run_events
from tests.test_analyze_from_frames import (
    EXPECTED_BDC,
    EXPECTED_TDC,
    FRAMES_PER_REV,
    N_FRAMES,
    _analyze,
    _frames,
)


def test_the_record_is_the_report_frame_by_frame():
    res = _analyze(_frames())
    tl = res["timeline"]
    assert tl is not None and tl["version"] == 1
    assert tl["sport"] == "bike" and tl["camera_side"] == "right"
    assert tl["frames"] == list(range(N_FRAMES))
    assert tl["fps"] == 30.0

    # The near-side skeleton, every drawn joint, every frame.
    joints = {i for bone in tl["bones"] for i in bone}
    assert joints == {8, 12, 14, 16, 24, 26, 28, 30, 32}
    for j in joints:
        col = tl["points"][str(j)]
        assert len(col) == N_FRAMES
        assert all(p is not None and len(p) == 3 for p in col)

    # The knee curve is the series the summary read its extremes from.
    keys = {m["key"] for m in tl["metrics"]}
    assert "right_knee" in keys and "trunk_angle" in keys
    knee = [v for v in tl["series"]["right_knee"] if v is not None]
    assert len(knee) >= N_FRAMES - 10
    assert abs(max(knee) - EXPECTED_BDC) < 3.0
    assert abs(min(knee) - EXPECTED_TDC) < 3.0

    # Every metric carries the band the overlay coloured it against.
    for m in tl["metrics"]:
        assert len(m["band"]) == 2 and m["band"][0] < m["band"][1]
        assert isinstance(m["callout"], bool)


def test_bike_events_are_the_analyzers_own_stroke_frames():
    res = _analyze(_frames())
    tl = res["timeline"]
    kinds = [e["kind"] for e in tl["events"]]
    assert kinds.count("bdc") >= 2 and kinds.count("tdc") >= 2
    assert tl["events_index"] == "analyzed"
    # Consecutive BDCs are one revolution apart, give or take the peak finder.
    bdc = [e["k"] for e in tl["events"] if e["kind"] == "bdc"]
    gaps = [b - a for a, b in zip(bdc, bdc[1:])]
    assert all(abs(g - FRAMES_PER_REV) <= 3 for g in gaps), gaps
    # And they sit where the knee is most open.
    knee = tl["series"]["right_knee"]
    for k in bdc:
        assert knee[k] is not None and knee[k] > EXPECTED_BDC - 4


def test_run_events_land_on_the_first_frame_of_the_streak():
    swing, stance = "mid_swing", "midstance"
    # 4 swing, 1 stance blip, 3 swing, 5 stance, 4 swing
    phases = [swing] * 4 + [stance] + [swing] * 3 + [stance] * 5 + [swing] * 4
    ev = _run_events(phases)
    # The blip never counts; the real stance begins at index 8, the swing
    # after it at 13.
    assert ev == [{"k": 8, "kind": "contact"}, {"k": 13, "kind": "toe_off"}]


def test_a_free_reader_never_receives_the_record():
    res = _analyze(_frames())
    assert res["timeline"] is not None
    assert "timeline" not in gate_free_result(res)
    assert "timeline" not in gate_preview_result(res)
