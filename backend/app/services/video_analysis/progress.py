"""From the analysis's own phases to what the waiting athlete reads.

The runner reports ``(phase, fraction)`` (see ``runner.PHASES``); this turns
that into two fields on the job -- ``stage``, a sentence, and ``progress``,
0-100 -- which ``/jobs/{id}`` serves and the progress screen draws.

The percentages are a time budget, not a frame count: detection is the bulk
of the wall clock and the only phase with a countable unit, so it owns most
of the bar and moves smoothly through it; the four phases after it are
fixed-cost and each takes a fixed slice. What matters is that the bar keeps
moving in step with what is actually happening, not that 43% means 43% of
the seconds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# (start, end) of each phase on the 0-100 bar. Detection dominates: ~0.15 s
# a frame on the production box against a few seconds for everything else.
PHASE_SPAN: dict[str, tuple[int, int]] = {
    "detect": (0, 72),
    "stabilize": (72, 78),
    "measure": (78, 88),
    "visuals": (88, 95),
    "coach": (95, 99),
}

# One sentence per phase. Present tense, says what the machine is doing, in
# the words the report will use -- so the wait teaches the vocabulary.
PHASE_TEXT: dict[str, dict[str, str]] = {
    "detect": {
        "run": "Finding the runner in every frame…",
        "bike": "Finding the rider in every frame…",
    },
    "stabilize": {
        "run": "Telling the left leg from the right…",
        "bike": "Steadying the skeleton across frames…",
    },
    "measure": {
        "run": "Measuring joint angles, cadence and ground contact…",
        "bike": "Measuring joint angles through the pedal stroke…",
    },
    "visuals": {
        "run": "Drawing the annotated frames and the kinogram…",
        "bike": "Drawing the annotated frames…",
    },
    "coach": {
        "run": "Writing the coaching notes…",
        "bike": "Writing the fit notes…",
    },
}


def phase_to_progress(
    phase: str, fraction: float | None, span: tuple[float, float] = (0.0, 1.0),
) -> int:
    """Map a phase + in-phase fraction onto the bar. ``span`` narrows the whole
    bar to a window, for a job that runs several analyses in sequence."""
    lo, hi = PHASE_SPAN.get(phase, (0, 0))
    inside = lo + (hi - lo) * max(0.0, min(1.0, fraction or 0.0))
    if fraction is None:
        inside = lo
    start, end = span
    return int(round(start * 100 + (end - start) * inside))


def phase_sentence(phase: str, fraction: float | None, sport: str) -> str:
    text = PHASE_TEXT.get(phase, {}).get(sport) or PHASE_TEXT.get(phase, {}).get("run") or ""
    if phase == "detect" and fraction is not None and fraction > 0:
        text = text.rstrip("…") + f" {int(round(fraction * 100))}%…"
    return text


def job_progress_hook(
    job: dict[str, Any], sport: str, *, prefix: str = "",
    span: tuple[float, float] = (0.0, 1.0),
) -> Callable[[str, float | None], None]:
    """A ``runner.ProgressFn`` that writes ``stage`` and ``progress`` onto
    ``job``. Writes are plain dict assignments -- the same thing the status
    endpoint reads -- so nothing here can block the analysis thread."""

    def hook(phase: str, fraction: float | None) -> None:
        job["progress"] = phase_to_progress(phase, fraction, span)
        job["stage"] = prefix + phase_sentence(phase, fraction, sport)

    return hook
