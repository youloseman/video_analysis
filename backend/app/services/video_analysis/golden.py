"""What a real clip is supposed to measure, reduced to a comparable record.

Why this exists
---------------
The suite had 1218 tests and none of them looked at a number produced from
actual pixels. Every one of them checks logic against synthetic input, which is
the right way to test logic and no way at all to notice that the logic is being
fed -- or is producing -- the wrong values. Three HIGH bugs lived for months
inside a green suite: an inverted hip sign that recommended lowering a saddle
that needed raising, a dead waveform component that made 8% of the running
rubric unearnable, and photo-bike angles read off the wrong landmark space.

None of those three was a crash or a type error. Each one was a NUMBER that
changed, or failed to appear, while everything around it kept working. That is
the class of failure this module exists to catch, and the only way to catch it
is to run a real clip and compare the answer with the answer from last time.

What is captured, and why not everything
----------------------------------------
A whole result is megabytes of base64 frames and hundreds of floats that move
with any harmless refactor. Snapshotting all of it produces a test that fails
every week for no reason, which is a test everybody learns to ignore -- and an
ignored regression test is worse than none, because it looks like coverage.

So the record is a deliberate selection, split by how it should be compared:

* **Exact values** -- verdicts, grades, camera side, which components were
  scored, which fit actions were recommended, how many issues fired. These are
  categorical. A change here is never rounding; it is the analysis reaching a
  different conclusion, which is exactly what the three bugs above did.
* **Approximate values** -- the angles and derived measurements. These may
  drift a little with a MediaPipe version or a filter constant, so they carry a
  tolerance. It is deliberately tight: real bugs move these by degrees, not by
  fractions of a percent.

What is left out on purpose: anything with a timestamp, any base64 image, the
LLM prose (not deterministic, and not measurement), and the raw per-frame
series (huge, and already summarised by the statistics that are kept).
"""

from __future__ import annotations

import math
from typing import Any

# How far a float may move before it counts as a change.
#
# Set from a measurement, not from habit: the same clip run twice through the
# whole pipeline produced 54 numeric fields with ZERO differences, and identical
# exact halves. The pipeline is deterministic on a machine, so this tolerance is
# not absorbing run-to-run noise -- there is none. It exists only for drift from
# a library upgrade, and drift from a library upgrade is exactly the thing worth
# being told about rather than quietly absorbed.
#
# So it is tight on purpose. Together these allow ~0.16 deg on a 155 deg knee:
# below anything float representation will produce, well below the ~3 deg that
# separates "no advice" from "advice" in the fit plan, and far below the 8 deg
# the landmark-space bug moved every bike angle by. If a dependency bump trips
# this, that is the test working -- re-record and commit the new baseline with
# the bump, so the diff says what the upgrade did to a real measurement.
FLOAT_RTOL = 0.001
FLOAT_ATOL = 0.05


def _num(value: Any) -> float | None:
    """A finite float, or None. NaN and None are the same fact here: not
    measured. Recording them as distinct would make the baseline fail whenever
    a missing value changed which KIND of missing it was."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 4)


def _angle_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Per-joint statistics, the core of the measurement.

    ``nan_pct`` is in here and matters as much as the angles: a change in how
    much of a joint was measurable is a change in the analysis even when the
    surviving values look the same.
    """
    out: dict[str, Any] = {}
    for joint, s in sorted((stats or {}).items()):
        if not isinstance(s, dict):
            continue
        out[joint] = {
            k: _num(s.get(k)) for k in ("mean", "min", "max", "range", "std")
        }
        out[joint]["nan_pct"] = _num(s.get("nan_pct"))
        out[joint]["valid_frames"] = s.get("valid_frames")
    return out


def _coverage(cov: dict[str, Any] | None) -> dict[str, Any]:
    """What the score was computed from -- where the dead-waveform bug lived.

    The component LISTS are compared exactly. A component quietly dropping out
    of the rubric is precisely the failure that went unnoticed for months, and
    it is invisible in the score itself because the weighted average
    renormalises over whatever survived.
    """
    cov = cov or {}
    return {
        "scored": sorted(cov.get("scored") or []),
        "missing": sorted(cov.get("missing") or []),
        "excluded": sorted((cov.get("excluded") or {}).keys()),
        "measures_scored": cov.get("measures_scored"),
        "measures_total": cov.get("measures_total"),
        "weight_covered": _num(cov.get("weight_covered")),
    }


def _fit_plan(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The bike adjustments, as (part, action) pairs plus the value they fired on.

    The inverted hip sign showed up here and nowhere else: the measurement was
    right, the score was right, and the ACTION was the opposite of the correct
    one. Recording the action string is what makes that visible.
    """
    rows = []
    for d in (plan or {}).get("diagnostics") or []:
        rows.append({
            "component": d.get("component"),
            "metric": d.get("metric_name"),
            "action": d.get("action"),
            "status": d.get("status"),
            "current": _num(d.get("current_value")),
        })
    return rows


def build_record(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full analysis result to the record compared against a baseline."""
    sm = result.get("sport_specific_metrics") or {}
    quality = sm.get("quality_gate") or {}
    capture = sm.get("capture_report") or {}
    confidence = sm.get("analysis_confidence") or {}
    stability = sm.get("tracking_stability") or {}
    sampling = sm.get("sampling") or {}

    record: dict[str, Any] = {
        # --- exact: conclusions ---
        "exact": {
            "status": result.get("status"),
            "sport_type": result.get("sport_type"),
            "cycling_position": result.get("cycling_position"),
            "camera_side": result.get("camera_side"),
            "letter_grade": result.get("letter_grade"),
            "frames_analyzed": result.get("frames_analyzed"),
            "quality_gate_triggered": result.get("quality_gate_triggered"),
            "quality_gate_profile": quality.get("profile"),
            "capture_verdict": capture.get("verdict"),
            "confidence_level": confidence.get("level"),
            "framing_verdict": sm.get("framing"),
            "sample_rate": sampling.get("sample_rate"),
            "coverage": _coverage(result.get("score_coverage")),
            "fit_plan": _fit_plan(result.get("fit_plan")),
            # Issue TYPES, not their prose: the wording is copy and changes
            # freely; which issues fired is the analysis.
            "issues": sorted(
                str(i.get("issue") or i.get("type") or "?")
                for i in (result.get("detected_issues") or [])
            ),
            "score_components": sorted((result.get("score_breakdown") or {}).keys()),
            # Present-or-absent for the optional blocks. A block silently
            # disappearing is a regression the numbers alone would not show.
            "has_kinogram": bool(result.get("kinogram_base64")),
            "has_keyframe": bool(result.get("keyframe_base64")),
            "has_aero": sm.get("aero_estimate") is not None,
            "has_pedaling_style": sm.get("pedaling_style") is not None,
            "warning_count": len(sm.get("quality_warnings") or []),
        },
        # --- approximate: measurements ---
        "approx": {
            "technique_score": _num(result.get("technique_score")),
            "score_breakdown": {
                k: _num(v) for k, v in sorted((result.get("score_breakdown") or {}).items())
            },
            "angles": _angle_stats(result.get("angle_statistics")),
        },
    }

    # Sport-specific summary values, the ones a rider or runner actually reads.
    approx = record["approx"]
    for key in (
        # bike
        "knee_at_bdc", "knee_at_tdc", "trunk_angle_avg", "hip_angle_avg",
        "elbow_angle_avg", "shoulder_angle_avg", "head_alignment_avg",
        "pelvic_ratio", "forearm_tilt_avg", "ankle_at_bdc",
        "knee_extension_velocity_dps", "knee_flexion_velocity_dps",
        # run
        "cadence_spm", "trunk_lean_avg", "vertical_oscillation_m",
        "ground_contact_ms", "flight_time_ms", "overstride_ratio",
        "stance_fraction",
    ):
        if key in sm:
            approx[f"summary.{key}"] = _num(sm[key])

    # A handful of exact summary fields that are conclusions rather than
    # measurements -- a flipped saddle verdict or a changed foot strike is a
    # different answer, not a rounding difference.
    for key in (
        "saddle_height_assessment", "near_side", "foot_strike",
        "slow_motion_factor", "time_base_uncertain", "position_archetype",
    ):
        if key in sm:
            value = sm[key]
            record["exact"][f"summary.{key}"] = (
                value.get("type") if isinstance(value, dict) else value
            )

    # Tracking stability: how confident the skeleton itself was. These move
    # when the identity or gating logic changes, which is a change worth
    # failing on even if every angle survives it.
    for key in ("leg_swap_pct", "flip_pct", "leg_collapse_pct"):
        if key in stability:
            approx[f"stability.{key}"] = _num(stability[key])

    return record


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b or a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL)
    except (TypeError, ValueError):
        return a == b


def _walk(path: str, want: Any, got: Any, exact: bool, out: list[str]) -> None:
    if isinstance(want, dict):
        if not isinstance(got, dict):
            out.append(f"{path}: expected an object, got {type(got).__name__}")
            return
        for key in sorted(set(want) | set(got)):
            if key not in want:
                out.append(f"{path}.{key}: appeared (new value {got[key]!r})")
            elif key not in got:
                out.append(f"{path}.{key}: disappeared (was {want[key]!r})")
            else:
                _walk(f"{path}.{key}", want[key], got[key], exact, out)
        return
    if isinstance(want, list):
        got_list = list(got or [])
        if list(want) == got_list:
            return
        # A list of records (the fit plan) is walked element-wise, so a single
        # changed action reads as one line naming that action -- not as two
        # dumped lists for the reader to diff by eye at the moment they are
        # least able to. Lists of plain values stay atomic: for a set of
        # component names, "which set" IS the fact.
        if (
            len(want) == len(got_list)
            and all(isinstance(x, dict) for x in want)
            and all(isinstance(x, dict) for x in got_list)
        ):
            for i, (w, g) in enumerate(zip(want, got_list)):
                _walk(f"{path}[{i}]", w, g, exact, out)
            return
        out.append(f"{path}: {want!r} -> {got!r}")
        return
    if exact:
        if want != got:
            out.append(f"{path}: {want!r} -> {got!r}")
    elif not _close(want, got):
        out.append(f"{path}: {want!r} -> {got!r}")


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Every difference, as readable lines. Empty means the clip still
    measures what it measured when the baseline was taken."""
    diffs: list[str] = []
    _walk("exact", baseline.get("exact") or {}, current.get("exact") or {}, True, diffs)
    _walk("approx", baseline.get("approx") or {}, current.get("approx") or {}, False, diffs)
    return diffs
