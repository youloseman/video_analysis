"""The TT/tri bike families a profile can name.

The same seventeen families the bike-finder research covers
(docs/research/tt-bike-finder/SCHEMA.md), so a profile that says
``giant-trinity`` points at a file we hold data for. Anything else is
``other`` plus the free-text model line.

The family is a label, like the rest of a profile: it never changes a
measurement. It exists so that we learn what people actually ride, and so the
parts-compatibility data (docs/AERO_ADVISOR_PLAN_RU.md, stage 3) has a key to
hang off. The model line carries the generation and anything else the rider
knows ("Trinity Advanced Pro 2019", "Speedmax CF SLX 8 2022") -- generations
are where the compatibility question actually lives, and asking for them as a
second picker before we hold any generation data would be a picker of guesses.
"""

from __future__ import annotations

BIKE_FAMILIES: dict[str, str] = {
    "argon18-e119": "Argon 18 E-119 / E-118",
    "bmc-speedmachine": "BMC Speedmachine",
    "cadex-tri": "Cadex Tri",
    "canyon-speedmax": "Canyon Speedmax",
    "cervelo-p-series": "Cervélo P-Series",
    "cervelo-p5": "Cervélo P5",
    "cube-aerium": "Cube Aerium",
    "factor-hanzo": "Factor Hanzo",
    "felt-ia": "Felt IA",
    "giant-trinity": "Giant Trinity",
    "liv-avow": "Liv Avow",
    "orbea-ordu": "Orbea Ordu",
    "pinarello-bolide": "Pinarello Bolide",
    "quintana-roo": "Quintana Roo",
    "scott-plasma": "Scott Plasma",
    "specialized-shiv": "Specialized Shiv",
    "trek-speed-concept": "Trek Speed Concept",
    "ventum-one": "Ventum",
}
OTHER = "other"

MAX_MODEL_CHARS = 80
MAX_SIZE_CHARS = 12
YEAR_RANGE = (1990, 2030)


def family_or_none(v: str | None) -> str | None:
    """A known slug, ``other``, or nothing. An unknown slug is dropped rather
    than refused, for the same reason an unknown position is: a stale client
    must still be able to save the bike."""
    if not v:
        return None
    v = v.strip().lower()
    return v if v in BIKE_FAMILIES or v == OTHER else None


def family_label(slug: str | None) -> str | None:
    if not slug:
        return None
    if slug == OTHER:
        return "Other"
    return BIKE_FAMILIES.get(slug)


def describe(family: str | None, model: str | None, size: str | None,
             year: int | None) -> str | None:
    """One line for a person or a model: "Giant Trinity · Advanced Pro · size M · 2019"."""
    bits = [
        b for b in (
            family_label(family) if family != OTHER else None,
            (model or "").strip() or None,
            f"size {size}" if size else None,
            str(year) if year else None,
        ) if b
    ]
    return " · ".join(bits) or None
