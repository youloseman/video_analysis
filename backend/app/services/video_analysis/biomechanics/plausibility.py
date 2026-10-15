"""Measurements a body cannot produce, and positions the rider was not in.

A DB-wide audit (2026-10-15) found the report printing numbers no rider can
hold: a knee at the bottom of the stroke of 172 deg (the leg locked straight
on the pedal), 34 deg at the top, a shoulder of 150 deg, a running cadence of
118 or 220. Almost all came from clips framed "tiny", where the pose model
puts joints on the wrong limb. One such number in a table is enough for a
reader to stop trusting the rest, so they leave the report: no score, no
finding, no label on the picture -- and a warning that says what was left
out and why.

A second, different case looked the same in the audit: on the triathlon
position, elbows of 150-172 deg and a torso of 61 deg. Those are measured
correctly -- the rider was sitting up on the base bar, not on the
extensions. Grading that against aero bands produced scores of 30, so the
score is withheld and the rider is asked to pick the position they rode;
every number stays.

The ranges are deliberately wide: they catch what cannot be, not what is
merely unusual. The per-frame envelope (base_analyzer.ANGLE_ENVELOPES) drops
single artefact frames; this looks at the clip's summary.
"""
from __future__ import annotations

import math
from typing import Any

from app.services.video_analysis.biomechanics.cycling_positions import AERO_POSITIONS

# field -> ((lo, hi), angle_history keys that print it, label)
_BIKE = {
    "knee_at_bdc": ((122.0, 162.0), (), "knee at the bottom of the stroke"),
    "knee_at_tdc": ((40.0, 100.0), (), "knee at the top of the stroke"),
    "hip_at_tdc": ((30.0, 90.0), (), "hip at the top of the stroke"),
    "shoulder_angle_avg": ((40.0, 135.0), ("{side}_shoulder",), "shoulder"),
}
_BIKE_AERO = {
    "forearm_tilt_avg": ((-20.0, 50.0), ("{side}_forearm_tilt",), "forearm tilt"),
}
_RUN = {
    "cadence_spm": ((135.0, 225.0), (), "cadence"),
    "trunk_lean_avg": ((-15.0, 30.0), ("trunk_lean",), "trunk lean"),
}
# Above these on aero bars the rider is on the base bar, not the extensions.
AERO_TRUNK_MAX = 45.0
AERO_ELBOW_MAX = 140.0
# A dropped knee reading takes the saddle verdict built on it with it.
_DEPENDENTS = {"knee_at_bdc": ("saddle_height_assessment",)}


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return None


def drop_implausible(
    sport: str, position: str | None, summary: dict[str, Any],
    angle_stats: dict[str, Any], angle_history: dict[str, Any], side: str | None,
) -> list[dict[str, Any]]:
    """Remove every summary value outside what a body can do; return them.

    Mutates ``summary`` (the value and anything built on it), ``angle_stats``
    (the table's row) and ``angle_history`` (the picture's label) in place.
    """
    table = dict(_RUN) if sport == "run" else {
        **_BIKE, **(_BIKE_AERO if position in AERO_POSITIONS else {})}
    dropped = []
    for field, ((lo, hi), keys, label) in table.items():
        v = _num(summary.get(field))
        if v is None or lo <= v <= hi:
            continue
        summary[field] = None
        for dep in _DEPENDENTS.get(field, ()):
            summary.pop(dep, None)
        for k in keys:
            name = k.format(side=side or "")
            angle_stats.pop(name, None)
            angle_history.pop(name, None)
        dropped.append({"field": field, "label": label, "value": round(v, 1), "lo": lo, "hi": hi})
    return dropped


def implausible_warning(dropped: list[dict[str, Any]]) -> str:
    parts = ", ".join(f"{d['label']} {d['value']:g}" for d in dropped)
    return (f"Some readings came out beyond what a body can do ({parts}) and were left "
            "out of the report rather than shown. That almost always means the pose model "
            "put a joint on the wrong limb, usually because the rider is small in the frame. "
            "Film closer, side-on, so you fill most of the picture.")


def position_mismatch(position: str | None, summary: dict[str, Any]) -> str | None:
    """A sentence when an aero-bar clip does not look like aero bars, else None."""
    if position not in AERO_POSITIONS:
        return None
    trunk = _num(summary.get("trunk_angle_avg"))
    elbow = _num(summary.get("elbow_angle_avg"))
    seen = []
    if trunk is not None and trunk > AERO_TRUNK_MAX:
        seen.append(f"torso at {trunk:.0f}°")
    if elbow is not None and elbow > AERO_ELBOW_MAX:
        seen.append(f"elbows at {elbow:.0f}°")
    if not seen:
        return None
    return (f"This doesn't look like riding on the aero bars ({' and '.join(seen)}): more "
            "like the base bar. The score is withheld rather than grading this against aero "
            "ranges. Run it again with the position you rode, or film on the extensions.")
