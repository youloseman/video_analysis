"""The bike's leg labels are settled once per clip (rtmpose_refiner.finish).

IMG_5577 (chainring side, 2026-11-09): MediaPipe swapped the legs on 45 of 169
frames, the refined skeleton swapped with it 24 times, and the knee at the
bottom read 120-141 deg. RTMPose's own labels never swapped.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.video_analysis.detectors import rtmpose_refiner as RR
from app.services.video_analysis.detectors.base import DetectorLandmark

W, H = 1000, 1000
NEAR = {23: (500, 400), 25: (600, 600), 27: (550, 800), 29: (540, 820), 31: (600, 830)}
FAR = {24: (480, 410), 26: (560, 620), 28: (470, 780), 30: (460, 800), 32: (520, 810)}


def _refiner(sport: str = "bike") -> RR.RTMPoseRefiner:
    return RR.RTMPoseRefiner(SimpleNamespace(config=SimpleNamespace(sport=sport)), session=None)


def _frame(mp_swapped: bool, rtm_swapped: bool = False):
    own = np.full((33, 2), np.nan)
    for d in (NEAR, FAR):
        for i, xy in d.items():
            own[i] = xy
    mp_px = own.copy()
    L, R = RR.LEG["left"], RR.LEG["right"]
    if mp_swapped:
        mp_px[L + R] = own[R + L]
    if rtm_swapped:
        own[L + R] = own[R + L].copy()
    for b in (11, 12, 13, 14, 15, 16):
        own[b] = mp_px[b] = (300 + b, 300)
    mp_lms = [DetectorLandmark(float(x / W) if np.isfinite(x) else 0.5, float(y / H) if np.isfinite(y) else 0.5,
                               0.0, 0.9 if (i in L) != mp_swapped else 0.4)  # the near leg is the sure one
               for i, (x, y) in enumerate(mp_px)]
    sc = np.full(33, 0.9)
    return list(mp_lms), mp_lms, own, sc, mp_px, W, H


def _near_knee(out) -> tuple[float, float]:
    return round(out[25].x * W), round(out[25].y * H)


def test_mediapipe_swapping_some_frames_does_not_swap_the_legs():
    r = _refiner()
    r._kept = [_frame(mp_swapped=i % 4 == 1) for i in range(40)]
    outs = [k[0] for k in r._kept]
    stats = r.finish()
    assert stats["legs"] == {"flipped": False, "mediapipe_disagreed": 10}
    assert {_near_knee(o) for o in outs} == {NEAR[25]}
    # visibility follows the leg, not MediaPipe's label on a swapped frame
    assert {round(o[25].visibility, 1) for o in outs} == {0.9}


def test_rtmpose_labels_flip_as_a_whole_when_mediapipe_disagrees_with_them_mostly():
    r = _refiner()
    r._kept = [_frame(mp_swapped=False, rtm_swapped=True) for _ in range(10)]
    outs = [k[0] for k in r._kept]
    assert r.finish()["legs"]["flipped"] is True
    assert {_near_knee(o) for o in outs} == {NEAR[25]}


def test_an_unsure_joint_keeps_mediapipe_from_the_right_side():
    r = _refiner()
    f = _frame(mp_swapped=True)
    f[3][25] = 0.1  # RTMPose unsure of the near knee
    r._kept = [_frame(mp_swapped=False) for _ in range(5)] + [f]
    r.finish()
    assert _near_knee(f[0]) == NEAR[25]


def test_only_the_bike_keeps_frames_and_nothing_kept_does_nothing():
    assert _refiner("run")._clip_labels is False
    assert _refiner("bike").finish() is None
