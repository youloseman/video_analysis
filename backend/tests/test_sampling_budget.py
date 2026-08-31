"""The adaptive sampling stride, which is a budget and a resolution at once.

``max_analysis_frames`` bounds detection CPU; the stride that enforces it also
decides the temporal resolution every timing metric is measured at. Integer
strides cannot hit the budget exactly, so the only question is which way to
miss, and both obvious answers are wrong:

``int()`` truncates, so everything under twice the budget got stride 1 and was
analysed whole -- 899 frames against a 450 budget, roughly 90 s of detection,
with the cap simply not applying. ``ceil()`` never exceeds the budget but pays
by halving the resolution the moment a clip passes 450 frames: a 30 fps run
clip of 17 s would drop to 15 fps effective, and ground contact from 33 ms to
67 ms, to save CPU we were 12% over.

Nearest keeps full resolution to 1.5x the budget and bounds the overshoot at
50%. These tests pin both halves of that, because a future "cleanup" to either
obvious form is exactly the change that would look harmless.
"""

from __future__ import annotations

import math

import pytest

MAX = 450  # settings.max_analysis_frames


def stride(total: int, base: int = 1, budget: int = MAX) -> int:
    """The formula in runner.extract_frames, isolated."""
    expected = total / base if base > 0 else total
    if expected > budget and total > 0:
        return max(base, math.floor(total / budget + 0.5))
    return base


def analysed(total: int, base: int = 1) -> int:
    return total // stride(total, base)


# --------------------------------------------------------------------------
# the bug this replaced
# --------------------------------------------------------------------------

def test_a_clip_just_under_twice_the_budget_is_no_longer_analysed_whole():
    """The truncating version returned stride 1 here and detected all 899
    frames -- 200% of the budget, and the cap did nothing at all."""
    assert stride(899) == 2
    assert analysed(899) <= MAX


@pytest.mark.parametrize("total", [451, 500, 600, 700, 899, 1000, 1349, 2000, 5000])
def test_the_overshoot_is_bounded_at_fifty_percent(total):
    """Not zero -- that is ceil, and it costs resolution. Bounded."""
    assert analysed(total) <= MAX * 1.5


# --------------------------------------------------------------------------
# the resolution this protects
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total", [450, 500, 600, 674])
def test_clips_up_to_one_and_a_half_budgets_keep_every_frame(total):
    """675 frames is 22 s at 30 fps -- well past the 5-15 s the capture guide
    asks for. Everything a user is told to film stays at full rate."""
    assert stride(total) == 1
    assert analysed(total) == total


def test_a_fifteen_second_clip_at_thirty_fps_is_untouched():
    """The exact shape the product asks for."""
    assert stride(15 * 30) == 1


def test_an_eight_second_clip_at_sixty_fps_is_untouched():
    """The bike clip in the repo: 503 frames. ceil() would have halved it to
    30 fps effective for a 12% overshoot."""
    assert stride(503) == 1
    assert analysed(503) == 503


# --------------------------------------------------------------------------
# the formula itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("total,expected", [
    (449, 1), (450, 1), (674, 1), (675, 2), (900, 2),
    (1124, 2), (1125, 3), (1350, 3), (1800, 4),
])
def test_the_stride_steps_where_it_should(total, expected):
    assert stride(total) == expected


def test_a_tie_rounds_to_the_coarser_stride():
    """floor(x + 0.5), not round(): round() is banker's rounding and sends
    exactly 2.5 down to 2, which is the wrong way for a budget."""
    assert stride(int(MAX * 2.5)) == 3


def test_the_stride_never_drops_below_the_sport_baseline():
    """Some sports sample sparsely by default; the budget may only ever make
    the stride coarser, never finer."""
    assert stride(10_000, base=3) >= 3
    assert stride(100, base=3) == 3


def test_a_clip_with_no_readable_length_is_left_alone():
    """A stream-written container reports 0 frames. Dividing by a budget on
    the strength of that would invent a stride from nothing."""
    assert stride(0) == 1


def test_the_runner_uses_this_formula():
    """The maths above is a copy. If the original changes shape, this catches
    the copy going stale rather than letting it pass a test of itself."""
    import inspect

    from app.services.video_analysis.runner import extract_frames

    src = inspect.getsource(extract_frames)
    assert "math.floor(total_video_frames / max_analysis_frames + 0.5)" in src, (
        "the sampling formula moved -- update stride() in this file to match"
    )
