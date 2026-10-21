"""Two guards on running video that came out of one treadmill session
(2026-10-20, memory flapp-60fps-run).

1. Ground contact + flight time is one step; cadence says how long a step is.
   When they disagree by more than RUN_TIMING_TOLERANCE, contact and flight
   are withheld -- on prod three 60 fps run clips missed by 38-46 % and two of
   them were shown.

2. A run clip analysed at 60 fps whose legs come out unstable is DETECTED
   again on every second frame, and kept that way if that holds: the same six
   clips scored 5/6 detected at 30 fps and 0/6 at 60. Re-detection, not a
   second look at the 60 fps detections -- taking every second one of those
   came out worse still.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis import runner


# --------------------------------------------------------------------------
# 1. timing that does not add up
# --------------------------------------------------------------------------

def _summary(cad, gct, flight):
    return {"cadence_spm": cad, "ground_contact_ms": gct, "flight_time_ms": flight}


def test_timing_that_adds_up_is_left_alone():
    s = _summary(170.0, 200.0, 153.0)          # step 353 ms
    issues = [{"type": "prolonged_ground_contact"}]
    assert runner._check_run_timing(s, issues) is None
    assert s["ground_contact_ms"] == 200.0 and s["flight_time_ms"] == 153.0
    assert issues == [{"type": "prolonged_ground_contact"}]


def test_timing_that_contradicts_the_cadence_is_withheld():
    """The 60 fps case: 133 + 83 = 216 ms against a 313 ms step."""
    s = _summary(191.5, 133.0, 83.0)
    issues = [{"type": "prolonged_ground_contact"}, {"type": "overstriding"}]
    w = runner._check_run_timing(s, issues)
    assert w and "216 ms" in w and "313 ms" in w
    assert s["ground_contact_ms"] is None and s["flight_time_ms"] is None
    assert s["timing_withheld"]["mismatch_pct"] == -31
    assert s["cadence_spm"] == 191.5                     # cadence is kept
    assert issues == [{"type": "overstriding"}]          # no verdict on a withheld number


def test_too_long_is_caught_as_well_as_too_short():
    s = _summary(220.0, 294.0, 84.0)                     # 378 vs 273 ms, a prod clip
    assert runner._check_run_timing(s, []) is not None


def test_the_line_is_twenty_percent():
    step = 60000.0 / 170.0
    ok = _summary(170.0, step * 0.6, step * 0.6 - 1)     # +19.7 %
    assert runner._check_run_timing(ok, []) is None
    bad = _summary(170.0, step * 0.61, step * 0.61)      # +22 %
    assert runner._check_run_timing(bad, []) is not None


@pytest.mark.parametrize("s", [
    _summary(None, 200.0, 150.0), _summary(170.0, None, 150.0),
    _summary(170.0, 200.0, None), _summary(0.0, 200.0, 150.0), {},
])
def test_nothing_to_compare_is_not_a_contradiction(s):
    assert runner._check_run_timing(s, []) is None


# --------------------------------------------------------------------------
# 2. the half-rate fallback
# --------------------------------------------------------------------------

def _wire(monkeypatch, *, video_fps, unstable_at):
    """Stub detection and measurement; record what stabilisation and the
    measuring half were handed. ``unstable_at(fps)`` decides whether the
    stabiliser reports the legs unstable at that rate."""
    calls = {"stabilize": [], "analyze": None, "extract": []}
    monkeypatch.setattr(runner, "get_video_info", lambda p: {"fps": video_fps, "frame_count": 20, "duration": 1.0})

    class _Det:
        def close(self):
            pass

    monkeypatch.setattr(runner, "build_detector", lambda *a, **k: _Det())

    def fake_extract(path, sport, fps, det, meta=None, progress=None, min_sample_rate=1):
        calls["extract"].append(min_sample_rate)
        k = min_sample_rate
        meta.update({"sample_rate": k, "base_sample_rate": k, "effective_fps": video_fps / k,
                     "video_fps": video_fps, "duration_s": 1.0, "downsampled": False})
        return [{"frame_idx": i, "timestamp_ms": i * 1000.0 / video_fps} for i in range(0, 20, k)]

    monkeypatch.setattr(runner, "extract_frames", fake_extract)

    def fake_stabilize(frames, sport, angle, fps, context, camera_view):
        calls["stabilize"].append((len(frames), fps))
        context["leg_identity"] = {"stability": {"unstable": unstable_at(fps), "instability": 1.0 if unstable_at(fps) else 0.1}}

    monkeypatch.setattr(runner, "stabilize_landmarks", fake_stabilize)

    def fake_analyze(frames, path, sport, pos, **kw):
        calls["analyze"] = {"n": len(frames), "sampling": kw["sampling_meta"], "ctx": kw["stabilizer_ctx"]}
        return {"status": "completed"}

    monkeypatch.setattr(runner, "analyze_from_frames", fake_analyze)
    return calls


def test_unstable_at_60_stable_at_30_detects_again_at_half_rate(monkeypatch):
    calls = _wire(monkeypatch, video_fps=60.0, unstable_at=lambda f: f > 45)
    runner.run_analysis("x.mov", "run", None, recommendations=False, kinogram=False)
    assert calls["extract"] == [1, 2]                    # a second detection pass
    assert calls["stabilize"] == [(20, 60.0), (10, 30.0)]
    a = calls["analyze"]
    assert a["n"] == 10
    assert a["sampling"]["sample_rate"] == 2 and a["sampling"]["effective_fps"] == 30.0
    assert a["sampling"]["downsampled"] is True
    assert a["sampling"]["half_rate_fallback"]["from_fps"] == 60.0
    assert a["ctx"]["leg_identity"]["stability"]["unstable"] is False


def test_unstable_at_both_rates_keeps_the_full_rate(monkeypatch):
    """Halving bought nothing, so the finer timing is not given up for it --
    the gate then withholds the score as it always did."""
    calls = _wire(monkeypatch, video_fps=60.0, unstable_at=lambda f: True)
    runner.run_analysis("x.mov", "run", None, recommendations=False, kinogram=False)
    assert calls["analyze"]["n"] == 20
    assert "half_rate_fallback" not in calls["analyze"]["sampling"]


def test_stable_at_60_is_left_at_60(monkeypatch):
    calls = _wire(monkeypatch, video_fps=60.0, unstable_at=lambda f: False)
    runner.run_analysis("x.mov", "run", None, recommendations=False, kinogram=False)
    assert calls["extract"] == [1]                       # no second pass when not needed
    assert calls["stabilize"] == [(20, 60.0)]
    assert calls["analyze"]["n"] == 20


def test_a_30_fps_clip_is_never_halved(monkeypatch):
    calls = _wire(monkeypatch, video_fps=30.0, unstable_at=lambda f: True)
    runner.run_analysis("x.mov", "run", None, recommendations=False, kinogram=False)
    assert calls["stabilize"] == [(20, 30.0)]


def test_the_bike_is_not_touched(monkeypatch):
    calls = _wire(monkeypatch, video_fps=60.0, unstable_at=lambda f: True)
    runner.run_analysis("x.mov", "bike", "road_hoods", recommendations=False, kinogram=False)
    assert calls["stabilize"] == [(20, 60.0)]
