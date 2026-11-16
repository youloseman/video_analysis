"""What a difference between two measured CdA values is worth, and what to
test next.

Bogdan's case (2026-11-16): five field tests on one bike, the head in a
different place each time. The log kept five numbers and said nothing about
them. A rider wants "head up cost 0.015 m², about 18 W at 45 km/h", and then
"what next?".

**What changed.** Each measurement can say what it tested (``VARIABLES``).
A test is compared with the most recent BASELINE on the same setup, and only
when the two were measured alike (``comparable``): same tool, same protocol,
neither a single ordinary ride. One ride moves by more than most position
changes are worth (John Bye, 2026-11-08), and two tools do not read alike.

**Watts and seconds.** Watts are the aerodynamic power the difference costs at
a stated speed, at the pedals: 0.5 rho dCdA v^3 / eta. Seconds are the time over
a distance at the power that holds that speed on the baseline -- the same
constant-power model the aero panel's minutes use (aero_estimator.PHYSICS), so
the two screens never disagree about physics. Both are computed here, once, so
the app and the report chat quote the same numbers; the chat is told never to
compute its own.

**The ladder.** A fixed order of what to test, biggest typical gain per hour of
testing first. Fixed on purpose: a model inventing the next experiment would
be a different experiment for every rider who asked. Draft order, to be
reviewed with Bogdan and John Bye.
"""

from __future__ import annotations

from typing import Any

# Speeds the difference is quoted at: a strong age-group bike leg, and
# the speed Bogdan quotes his field results at.
QUOTE_SPEEDS_KMH = (40.0, 45.0)
SECONDS_SPEED_KMH = 40.0
SECONDS_DISTANCE_KM = 40.0

VARIABLES: dict[str, str] = {
    "baseline": "Baseline (my usual position)",
    "head": "Head position",
    "hands": "Hands and extensions (height, angle)",
    "elbows": "Elbow width and pad reach",
    "torso": "Torso and stack (pad height, saddle)",
    "helmet": "Helmet",
    "clothing": "Skinsuit and clothing",
    "wheels": "Wheels and tyres",
    "other": "Something else",
}
BASELINE = "baseline"

# (variable, what to try, why it comes here). Biggest typical gain for the
# least kit first: the rider's own body before anything bought.
LADDER: list[dict[str, str]] = [
    {"variable": "baseline", "title": "Your usual position, three runs",
     "why": "Everything after is measured against this, and three runs show how much the number moves on its own."},
    {"variable": "head", "title": "Head: tucked to the back vs your usual",
     "why": "Costs nothing to try and is often the biggest single change, but only if you can hold it and still see the road."},
    {"variable": "hands", "title": "Hands: higher or more tilted extensions",
     "why": "Hands up often brings the head and shoulders down with them. Needs spacers or a wedge you may already have."},
    {"variable": "elbows", "title": "Elbows: narrower pads, reach a step longer or shorter",
     "why": "Narrower pads cut frontal area; reach changes how flat the back sits. Check you can still steer."},
    {"variable": "torso", "title": "Torso: pad stack a step lower",
     "why": "Lower is not always faster, and it costs hip angle and power. Test it rather than assume it."},
    {"variable": "helmet", "title": "Helmet: the one you race in vs another you can borrow",
     "why": "The best helmet is the one that fits how you hold your head. Only a measurement can tell; a photo cannot."},
    {"variable": "clothing", "title": "Skinsuit or race kit vs training kit",
     "why": "Loose fabric is drag you can remove without changing the position at all."},
    {"variable": "wheels", "title": "Wheels and tyres",
     "why": "Usually smaller than the body changes above, and the most expensive to test. Last for that reason."},
]


def variable_or_none(v: str | None) -> str | None:
    """A known key or nothing. Unknown is dropped, as elsewhere on a profile."""
    if not v:
        return None
    v = v.strip().lower()
    return v if v in VARIABLES else None


def comparable(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same tool, same protocol, and neither on one ordinary ride."""
    return (
        a.get("source") == b.get("source")
        and a.get("protocol") == b.get("protocol")
        and a.get("protocol") != "single_ride"
    )


def watts(delta_cda: float, speed_kmh: float) -> float:
    """Aerodynamic power the difference costs at ``speed_kmh``, at the pedals.
    Positive = the second number needs more power."""
    from app.services.video_analysis.biomechanics.aero_estimator import PHYSICS

    v = speed_kmh / 3.6
    return 0.5 * PHYSICS["rho"] * delta_cda * v ** 3 / PHYSICS["eta"]


def seconds_over(cda_base: float, cda_test: float, speed_kmh: float, distance_km: float) -> float:
    """Seconds the test position gains (negative) or loses (positive) over the
    distance, at the power that holds ``speed_kmh`` in the baseline position."""
    from app.services.video_analysis.biomechanics.aero_estimator import _power

    v0 = speed_kmh / 3.6
    target = _power(v0, cda_base)
    lo, hi = v0 * 0.5, v0 * 1.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if _power(mid, cda_test) < target else (lo, mid)
    v1 = (lo + hi) / 2.0
    d = distance_km * 1000.0
    return d / v1 - d / v0


def difference(base: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    """The test against the baseline, in m², %, watts and seconds."""
    a, b = float(base["value_m2"]), float(test["value_m2"])
    d = b - a
    return {
        "baseline_id": base.get("id"),
        "baseline_on": base.get("measured_on"),
        "delta_m2": round(d, 4),
        "delta_pct": round(d / a * 100.0, 1),
        "watts": {f"{s:g}": round(watts(d, s), 1) for s in QUOTE_SPEEDS_KMH},
        "seconds": round(seconds_over(a, b, SECONDS_SPEED_KMH, SECONDS_DISTANCE_KM)),
        "seconds_basis": f"{SECONDS_DISTANCE_KM:g} km at {SECONDS_SPEED_KMH:g} km/h, same power",
    }


def annotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``vs_baseline`` to every tagged test on one setup.

    The baseline is the most recent baseline measured on or before the test's
    day that was measured alike. None when there is no such baseline -- the
    row then says why it has no difference rather than showing a wrong one.
    """
    baselines = sorted(
        (r for r in rows if r.get("variable") == BASELINE),
        key=lambda r: (r.get("measured_on") or "", r.get("created_at_ms") or 0),
        reverse=True,
    )
    for r in rows:
        r["vs_baseline"] = None
        if not r.get("variable") or r.get("variable") == BASELINE:
            continue
        same_way = [
            b for b in baselines
            if (b.get("measured_on") or "") <= (r.get("measured_on") or "") and comparable(b, r)
        ]
        if same_way:
            r["vs_baseline"] = difference(same_way[0], r)
        elif baselines:
            r["vs_baseline_missing"] = "measured_differently"
        else:
            r["vs_baseline_missing"] = "no_baseline"
    return rows


def ladder_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ladder with what this setup has already tested."""
    tested = {r.get("variable") for r in rows if r.get("variable")}
    out, nxt = [], None
    for step in LADDER:
        done = step["variable"] in tested
        if not done and nxt is None:
            nxt = step["variable"]
        out.append({**step, "done": done, "next": step["variable"] == nxt})
    return out
