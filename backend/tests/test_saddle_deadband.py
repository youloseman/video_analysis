"""The saddle verdict and its score near the edges (cycling_positions).

A clip read 150.0 on one machine and 149.75 on Railway: on a hard edge that
flipped "too high" to "acceptable", fired and dropped the issue, and moved the
score by 2 points. Here "too high/low" waits SADDLE_VERDICT_DEADBAND_DEG past
the edge and the score crosses each edge on a ramp.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis.biomechanics.cycling_positions import (
    SADDLE_VERDICT_DEADBAND_DEG as DB,
)
from app.services.video_analysis.biomechanics.cycling_positions import (
    saddle_fit_score,
    saddle_verdict,
)

LO, HI = 140.0, 145.0   # TT aero band; "too high" edge at 150, "too low" at 135


def test_too_high_waits_for_the_deadband_past_the_edge():
    assert saddle_verdict(150.0, LO, HI) == "acceptable"
    assert saddle_verdict(150.0 + DB, LO, HI) == "acceptable"
    assert saddle_verdict(150.0 + DB + 0.01, LO, HI) == "too_high"


def test_too_low_waits_for_the_deadband_too():
    assert saddle_verdict(135.0 - DB, LO, HI) == "acceptable"
    assert saddle_verdict(135.0 - DB - 0.01, LO, HI) == "too_low"


def test_the_band_itself_is_optimal_and_the_middle_acceptable():
    assert saddle_verdict(142.0, LO, HI) == "optimal"
    assert saddle_verdict(147.5, LO, HI) == "acceptable"


def test_the_non_drive_margin_moves_the_high_edge_and_keeps_the_deadband():
    # 9 deg margin for the non-drive side (MediaPipe alone): edge at 154.
    assert saddle_verdict(154.0 + DB, LO, HI, 9.0) == "acceptable"
    assert saddle_verdict(154.0 + DB + 0.01, LO, HI, 9.0) == "too_high"
    # A margin under 5 never narrows the old 5 deg edge.
    assert saddle_verdict(150.5, LO, HI, 0.0) == "acceptable"


def test_the_two_readings_of_the_railway_clip_now_agree():
    assert saddle_verdict(150.0, LO, HI) == saddle_verdict(149.75, LO, HI) == "acceptable"
    assert abs(saddle_fit_score(150.0, LO, HI) - saddle_fit_score(149.75, LO, HI)) < 5


@pytest.mark.parametrize("knee,score", [
    (142.0, 100.0),            # in the band
    (145.0, 100.0),            # on its edge, still optimal
    (147.5, 75.0),             # flat middle of "acceptable"
    (150.0 + DB, 40.0),        # where the verdict turns "too high"
    (160.0, 40.0),
    (135.0 - DB, 40.0),        # where it turns "too low"
    (137.5, 75.0),
])
def test_the_score_meets_the_old_steps_away_from_the_edges(knee, score):
    assert saddle_fit_score(knee, LO, HI) == pytest.approx(score)


def test_the_score_never_jumps():
    knees = [130 + i * 0.05 for i in range(600)]
    scores = [saddle_fit_score(k, LO, HI) for k in knees]
    # 0.05 deg moves it by at most 35 points over a 2*DB ramp, scaled.
    steepest = max(35.0, 25.0) / (2 * DB) * 0.05
    assert max(abs(a - b) for a, b in zip(scores, scores[1:])) <= steepest + 1e-9
    # And it agrees with the verdict: 100 only when optimal, 40 only when too far.
    for k, s in zip(knees, scores):
        v = saddle_verdict(k, LO, HI)
        assert (s == 100.0) == (v == "optimal") or LO <= k <= HI
        if v in ("too_high", "too_low"):
            assert s == pytest.approx(40.0)
