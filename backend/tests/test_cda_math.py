"""What a measured CdA difference is worth, and what to test next.

Bogdan's example is the anchor: head up cost 0.015 m^2, "about 18 W at
45 km/h". The comparability rule is the one the two-setup compare already
keeps -- same tool, same protocol, never a single ordinary ride.
"""

from __future__ import annotations

import pytest

from app.api import cda as cda_api
from app.api import me
from app.models.user import TIER_FULL
from app.services import cda_math, report_chat


def row(value, *, variable=None, on="2026-11-10", source="notio", protocol="repeated_loops", rid=None):
    return {"id": rid, "value_m2": value, "variable": variable, "measured_on": on,
            "source": source, "protocol": protocol, "created_at_ms": 0}


def test_bogdans_example_is_about_eighteen_watts_at_45():
    w = cda_math.watts(0.015, 45.0)
    assert 17.5 <= w <= 18.5


def test_watts_follow_the_cube_of_speed():
    assert cda_math.watts(0.01, 45.0) / cda_math.watts(0.01, 40.0) == pytest.approx((45 / 40) ** 3)


def test_a_lower_cda_is_seconds_saved():
    assert cda_math.seconds_over(0.25, 0.24, 40.0, 40.0) < 0
    assert cda_math.seconds_over(0.24, 0.25, 40.0, 40.0) > 0
    assert cda_math.seconds_over(0.25, 0.25, 40.0, 40.0) == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize("a,b,ok", [
    (dict(), dict(), True),
    (dict(source="airo"), dict(), False),
    (dict(protocol="single_ride"), dict(protocol="single_ride"), False),
    (dict(protocol="not_a_ride", source="wind_tunnel"), dict(protocol="not_a_ride", source="wind_tunnel"), True),
])
def test_only_like_for_like_is_compared(a, b, ok):
    assert cda_math.comparable(row(0.23, **a), row(0.24, **b)) is ok


def test_a_test_is_compared_with_the_latest_baseline_before_it():
    rows = cda_math.annotate([
        row(0.240, variable="baseline", on="2026-10-01", rid=1),
        row(0.231, variable="baseline", on="2026-11-01", rid=2),
        row(0.246, variable="head", on="2026-11-01", rid=3),
        row(0.250, variable="baseline", on="2026-11-20", rid=4),   # after the test
    ])
    d = rows[2]["vs_baseline"]
    assert d["baseline_id"] == 2
    assert d["delta_m2"] == pytest.approx(0.015)
    assert d["delta_pct"] == pytest.approx(6.5)
    assert set(d["watts"]) == {"40", "45"}
    assert d["seconds"] > 0
    assert rows[0]["vs_baseline"] is None    # a baseline is not compared


def test_no_difference_is_shown_when_it_cannot_be_believed():
    rows = cda_math.annotate([
        row(0.231, variable="baseline", source="windsock"),
        row(0.246, variable="head", source="notio"),
        row(0.240, variable="hands"),
    ])
    assert rows[1]["vs_baseline"] is None
    assert rows[1]["vs_baseline_missing"] == "measured_differently"

    alone = cda_math.annotate([row(0.246, variable="head")])
    assert alone[0]["vs_baseline_missing"] == "no_baseline"

    untagged = cda_math.annotate([row(0.231, variable="baseline"), row(0.25)])
    assert untagged[1]["vs_baseline"] is None and "vs_baseline_missing" not in untagged[1]


def test_the_ladder_marks_what_was_tested_and_the_next_step():
    steps = cda_math.ladder_status([row(0.23, variable="baseline"), row(0.24, variable="hands")])
    by = {s["variable"]: s for s in steps}
    assert by["baseline"]["done"] and by["hands"]["done"]
    assert by["head"]["next"] and not by["elbows"]["next"]
    assert sum(s["next"] for s in steps) == 1
    assert steps[0]["variable"] == "baseline"


def test_every_ladder_step_is_a_tag_a_measurement_can_carry():
    assert {s["variable"] for s in cda_math.LADDER} <= set(cda_math.VARIABLES)


def test_an_unknown_tag_is_dropped():
    assert cda_api.CdaIn(value_m2=0.23, measured_on="2026-11-01", source="notio",
                         variable="vibes").variable is None
    assert cda_api.CdaIn(value_m2=0.23, measured_on="2026-11-01", source="notio",
                         variable="HEAD").variable == "head"


async def test_the_api_stores_the_tag_and_lists_the_difference(db, make_user):
    u = await make_user(tier=TIER_FULL)
    p = await me.create_profile(me.ProfileIn(name="TT", sport="bike", kind="tt"), u, db)
    for v, var in ((0.231, "baseline"), (0.246, "head")):
        await cda_api.add_cda(p["id"], cda_api.CdaIn(
            value_m2=v, measured_on="2026-11-01", source="notio", variable=var,
        ), None, u, db)
    listed = await me.list_profiles(u, db)
    prof = listed["profiles"][0]
    head = next(r for r in prof["cda"] if r["variable"] == "head")
    assert head["vs_baseline"]["delta_m2"] == pytest.approx(0.015)
    assert next(s for s in prof["ladder"] if s["next"])["variable"] == "hands"
    assert {"key": "head", "label": "Head position"} in listed["cda_variables"]


def test_the_chat_is_handed_the_difference_and_the_next_step():
    rows = cda_math.annotate([
        row(0.231, variable="baseline", on="2026-11-01"),
        row(0.246, variable="head", on="2026-11-01"),
    ])
    md = report_chat.athlete_block(sport="bike", profile={"name": "TT"}, cda=rows)
    assert "[tested: Head position]" in md
    assert "+0.015 m² (+6.5%)" in md
    assert "W at 45 km/h" in md
    assert "Next on the ladder: Hands" in md
    assert "Never compute" in report_chat.SYSTEM
