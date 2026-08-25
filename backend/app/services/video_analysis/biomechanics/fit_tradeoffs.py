"""Which fit adjustment fixes which metric -- and what it costs elsewhere.

The coach used to see a flat list of metrics with statuses and nothing about how
they are coupled, so it would praise a trunk angle and then, two lines later,
prescribe raising the aerobar stack to open a hip -- an adjustment whose whole
effect is to raise that trunk. Every joint angle on a bike is downstream of the
same handful of contact points; moving one moves several.

This module encodes that coupling, so the ranking of "what should they actually
change" is computed here rather than guessed by a language model:

* which adjustments move a problem metric in the direction it needs to go
* what else each one moves, and whether the metric it disturbs has room left
* the costs a band cannot express -- raising the trunk is a drag penalty
  whether or not the result is still inside the optimal range

Adjustment names match the ``action`` vocabulary in action_plan_builder.

Sources: Retul fit windows; Burt 'Bike Fit' 2014; Steinmetz 2024 on aero
position archetypes; standard TT/tri fitting practice for the ordering
(pelvic rotation before front-end height when a hip is closed).
"""

from __future__ import annotations

from typing import Any

from app.services.video_analysis.biomechanics.cycling_positions import (
    AERO_POSITIONS,
)

# Direction convention: +1 = the adjustment increases the metric (opens the
# angle / raises the ratio), -1 = decreases it.
#
# ``aero_cost`` is the penalty a band cannot express: a trunk at 28 deg and a
# trunk at 24 deg are both "optimal", and one of them is measurably faster.
ADJUSTMENTS: dict[str, dict[str, Any]] = {
    "rotate_pelvis_forward": {
        "label": "rotate the pelvis forward -- saddle nose down 1-2 deg, "
                 "and/or saddle forward ~5mm",
        "moves": {"hip": +1, "pelvic_ratio": +1},
        "aero_cost": None,
        "note": "the aero-preserving way to open a hip: it changes how the "
                "rider sits ON the saddle, so the trunk does not move. Try "
                "this before touching the front end.",
    },
    "consider_shorter_cranks": {
        "label": "shorter cranks (-2.5 to -5mm)",
        "moves": {"hip": +1, "knee": +1},
        "aero_cost": None,
        "note": "opens the top of the pedal stroke without moving the front "
                "end at all. A workshop change, not a roadside one.",
    },
    # Saddle height moves hip and knee TOGETHER: raising extends the leg,
    # which opens the knee AND the hip (same direction as shorter cranks,
    # which open the top of the stroke the same way). The signs shipped
    # inverted for months -- a closed hip ranked "lower the saddle", the
    # exact opposite lever -- and the direction test pinned the bug.
    "lower_saddle": {
        "label": "lower the saddle ~5mm",
        "moves": {"hip": -1, "knee": -1},
        "aero_cost": None,
    },
    "raise_saddle": {
        "label": "raise the saddle ~5mm",
        "moves": {"hip": +1, "knee": +1},
        "aero_cost": None,
    },
    "raise_bars": {
        "label": "raise the aerobar stack 10-15mm",
        "moves": {"hip": +1, "shoulder": +1, "trunk": +1},
        "aero_cost": "raises the trunk, which is the largest single term in "
                     "frontal area -- a direct drag penalty even if the trunk "
                     "angle stays inside its band",
    },
    "lower_bars": {
        "label": "lower the aerobar stack 10-15mm",
        "moves": {"hip": -1, "shoulder": -1, "trunk": -1},
        "aero_cost": None,
        "note": "an aero gain, but it closes the hip. Never the answer when "
                "the hip is already at or below its band.",
    },
    "extend_reach": {
        "label": "lengthen the extensions / move the pads forward",
        "moves": {"elbow": +1, "shoulder": +1, "trunk": -1},
        "aero_cost": None,
    },
    "shorten_reach": {
        "label": "shorten the extensions / move the pads back",
        "moves": {"elbow": -1, "shoulder": -1, "trunk": +1},
        "aero_cost": "shortening reach sits the rider up slightly",
    },
    "tilt_extensions_up": {
        "label": "tilt the extensions up 10-20 deg",
        "moves": {"forearm_tilt": +1, "head_alignment": +1},
        "aero_cost": None,
        "note": "creates a pocket for the head (Steinmetz 2024).",
    },
}

# Below this share of its band remaining in the direction of travel, a metric
# that is currently in range is treated as about to leave it.
_HEADROOM_TIGHT = 0.25


def _needed_direction(entry: dict[str, Any]) -> int | None:
    """+1 if this metric must increase to reach its band, -1 if decrease."""
    value, lo, hi = entry.get("value"), entry.get("optimal_min"), entry.get("optimal_max")
    if value is None or lo is None or hi is None:
        return None
    if value < lo:
        return +1
    if value > hi:
        return -1
    return None


def _headroom_fraction(entry: dict[str, Any], direction: int) -> float | None:
    """How much of its band a currently-in-range metric has left, that way."""
    value, lo, hi = entry.get("value"), entry.get("optimal_min"), entry.get("optimal_max")
    if value is None or lo is None or hi is None:
        return None
    width = hi - lo
    if width <= 0:
        return None
    room = (hi - value) if direction > 0 else (value - lo)
    return max(0.0, room / width)


def _is_aero_gain(metric: str, direction: int, position: str | None) -> bool:
    """True when leaving the band this way is the point, not a side effect.

    A trunk flatter than its comfort band is the goal of an aero position --
    the same judgement _classify_angle_status makes when it returns
    "aero_optimized" instead of "needs_work". Without this, the ranking warns
    that a lever would "push the trunk out of its band" and steers the coach
    away from an adjustment that made the rider faster.
    """
    return metric == "trunk" and direction < 0 and position in AERO_POSITIONS


def rank_adjustments(
    angles: dict[str, dict[str, Any]],
    cycling_position: str | None = None,
) -> list[dict[str, Any]]:
    """Rank the adjustments that address the material problems in ``angles``.

    ``angles`` is a photo result's ``angles_with_context``. Returns dicts with
    ``label``, ``fixes``, ``costs`` and ``aero_cost``, best option first --
    "best" meaning it fixes the most while disturbing the least of what already
    works.
    """
    problems: dict[str, int] = {}
    for key, entry in angles.items():
        if entry.get("status") != "needs_work":
            continue
        direction = _needed_direction(entry)
        if direction is not None:
            problems[key] = direction
    if not problems:
        return []

    ranked: list[dict[str, Any]] = []
    for adj in ADJUSTMENTS.values():
        moves = adj["moves"]
        fixes = [
            angles[k].get("label", k)
            for k, need in problems.items()
            if moves.get(k) == need
        ]
        if not fixes:
            continue

        costs, breaks = [], 0
        for metric, direction in moves.items():
            if metric in problems or metric not in angles:
                continue
            entry = angles[metric]
            if entry.get("status") not in ("optimal", "acceptable"):
                continue
            room = _headroom_fraction(entry, direction)
            if room is None:
                continue
            label = entry.get("label", metric)
            if _is_aero_gain(metric, direction, cycling_position):
                costs.append(
                    f"flattens {label} ({entry.get('value')}) -- an aero gain, "
                    f"not a cost, as long as the rider can hold it"
                )
            elif room < _HEADROOM_TIGHT:
                breaks += 1
                costs.append(
                    f"pushes {label} ({entry.get('value')}) out of its band"
                )
            else:
                costs.append(
                    f"moves {label} ({entry.get('value')}, in range, room to spare)"
                )

        ranked.append({
            "label": adj["label"],
            "fixes": fixes,
            "costs": costs,
            "breaks": breaks,
            "aero_cost": adj.get("aero_cost"),
            "note": adj.get("note"),
        })

    ranked.sort(key=lambda r: (
        -len(r["fixes"]),                 # fix more of the problem
        r["breaks"],                      # without breaking what works
        1 if r["aero_cost"] else 0,       # and without paying drag for it
    ))
    return ranked


def build_tradeoff_block(
    angles: dict[str, dict[str, Any]],
    cycling_position: str | None = None,
) -> str:
    """Prompt text: the ranked levers and their costs. '' if nothing to fix."""
    ranked = rank_adjustments(angles, cycling_position)
    if not ranked:
        return ""

    lines = [
        "Adjustment options for those problems, RANKED (best first -- ranked by "
        "how much they fix against how much they disturb what already works). "
        "Recommend from the top of this list, and state the cost of whatever "
        "you recommend:",
    ]
    for i, r in enumerate(ranked[:5], 1):
        lines.append(f"{i}. {r['label']}")
        lines.append(f"   fixes: {', '.join(r['fixes'])}")
        detail = list(r["costs"])
        if r["aero_cost"]:
            detail.append(f"AERO COST: {r['aero_cost']}")
        lines.append(
            "   cost: " + ("; ".join(detail) if detail
                           else "nothing else currently in range")
        )
        if r.get("note"):
            lines.append(f"   note: {r['note']}")
    return "\n".join(lines)
