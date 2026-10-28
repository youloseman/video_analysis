"""The Kona 2026 pro field: what the world's best actually ride, as numbers.

Thirty-two professional positions from the Ironman World Championship (race
photographs and broadcast stills, every joint point checked by hand -- the
Academy article kona-2026-pro-bike-positions, built by scripts/field_study).
Served next to the reference bands so an aero rider sees their own number
beside the pros' median and middle half: a band says where the literature
puts "good"; the field says where the fastest people on the day sat.

It is a FIELD, not a band. Nothing is graded against it, and a rider outside
it is not thereby wrong: the pros hold positions most age-groupers cannot.
The one place it sets a threshold is the helmet (head_silhouette.py), and
that is a penalty-free band up to where twenty-nine of the thirty-two sat.

Aero positions only: the field was measured on TT and triathlon bikes, and
a rider on the hoods has nothing to compare with it.

Re-measured whenever the study grows (backend/scripts/field_study/field.py
rebuild prints these); the article's table and this dict are pinned to each
other by a test.
"""
from __future__ import annotations

from typing import Any

KONA_PRO_N = 32
KONA_PRO_MEN = 21
KONA_PRO_WOMEN = 11
KONA_PRO_ARTICLE = "/academy/kona-2026-pro-bike-positions"
KONA_PRO_SOURCE = (
    f"Flapp field study: {KONA_PRO_N} professional positions at the Ironman "
    f"World Championship 2026 ({KONA_PRO_MEN} men, {KONA_PRO_WOMEN} women), "
    "every joint point checked by hand"
)

# Per summary field: median, middle half (q1-q3) and the full range. Re-read
# 2026-10-28 from the rebuild with Artur's leg-point corrections (hip moved on
# two positions): trunk q1, shoulder q3 and helmet q3 shifted by under 0.4.
KONA_PRO_FIELD: dict[str, dict[str, float]] = {
    "trunk_angle_avg":    {"median": 20.2, "q1": 18.7, "q3": 22.2, "lo": 12.2, "hi": 27.1},
    "shoulder_angle_avg": {"median": 87.7, "q1": 83.0, "q3": 91.7, "lo": 74.1, "hi": 99.6},
    "elbow_angle_avg":    {"median": 83.7, "q1": 76.4, "q3": 87.7, "lo": 66.4, "hi": 102.3},
    "forearm_tilt_avg":   {"median": 25.9, "q1": 23.4, "q3": 28.4, "lo": 18.8, "hi": 32.0},
    # n=31: one rider caught ducking the head is left out of the helmet only.
    "helmet_above_back":  {"median": 18.4, "q1": 14.0, "q3": 21.4, "lo": 6.7, "hi": 33.0},
}

# Units per field, for whoever prints these (the coach prompt, the export).
KONA_PRO_UNITS = {
    "trunk_angle_avg": "°", "shoulder_angle_avg": "°", "elbow_angle_avg": "°",
    "forearm_tilt_avg": "°", "helmet_above_back": "% of torso",
}

AERO_FIELD_POSITIONS = frozenset({"tt_aero", "triathlon"})


def pro_field(cycling_position: str | None) -> dict[str, Any] | None:
    """The field as served on a result, or None off the aero bars."""
    if cycling_position not in AERO_FIELD_POSITIONS:
        return None
    return {
        "n": KONA_PRO_N, "men": KONA_PRO_MEN, "women": KONA_PRO_WOMEN,
        "source": KONA_PRO_SOURCE, "article": KONA_PRO_ARTICLE,
        "metrics": {k: dict(v) for k, v in KONA_PRO_FIELD.items()},
    }


KONA_PRO_LABELS = {
    "trunk_angle_avg": "Trunk angle", "shoulder_angle_avg": "Shoulder angle",
    "elbow_angle_avg": "Elbow angle", "forearm_tilt_avg": "Forearm tilt",
    "helmet_above_back": "Helmet above the back",
}


def pro_gaps(cycling_position: str | None, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Readings outside the WHOLE Kona field, whatever their band says.

    A clip can sit inside every band and still outside where all 32 pros
    rode: Artur's own clip (2026-10-13) scored 100 with the forearms at 17 deg,
    below every pro (19-32). Not a fault -- the field is not a band -- but it
    is where the next gain is, and a report that scores 100 and says nothing
    about it has nothing to offer. Aero positions only; a helmet reading that
    flagged itself is skipped.
    """
    if cycling_position not in AERO_FIELD_POSITIONS:
        return []
    out = []
    for field, f in KONA_PRO_FIELD.items():
        v = summary.get(field)
        if not isinstance(v, (int, float)):
            continue
        if field == "helmet_above_back":
            from app.services.video_analysis.biomechanics.head_silhouette import graded_helmet
            v = graded_helmet(v, summary.get("helmet_check"))
            if v is None:
                continue
        if f["lo"] <= v <= f["hi"]:
            continue
        out.append({"field": field, "label": KONA_PRO_LABELS[field], "value": float(v),
                    "lo": f["lo"], "hi": f["hi"], "median": f["median"],
                    "side": "high" if v > f["hi"] else "low", "n": KONA_PRO_N})
    return out


def pro_field_prompt_block(cycling_position: str | None) -> str:
    """The field for the coach: context for an aero rider, never a rubric."""
    if cycling_position not in AERO_FIELD_POSITIONS:
        return ""
    rows = []
    for field, v in KONA_PRO_FIELD.items():
        name = field.replace("_avg", "").replace("_", " ")
        unit = "%" if field == "helmet_above_back" else "°"
        rows.append(f"{name} {v['median']:.0f}{unit} (middle half {v['q1']:.0f}-{v['q3']:.0f}{unit})")
    return (
        f"Kona 2026 pro field (n={KONA_PRO_N}, {KONA_PRO_MEN} men and "
        f"{KONA_PRO_WOMEN} women, aero positions, median and middle half): "
        + "; ".join(rows) + ". Use it as CONTEXT for an aero rider -- 'the pros sit "
        "at...' -- when it helps them place a number. It is a field, not a band: "
        "the pros hold positions most age-groupers cannot, so a rider outside it "
        "is not thereby wrong, and a rider inside it is not thereby fine."
    )
