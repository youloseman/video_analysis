"""A stored road report read with today's trunk rule (Artur 2026-11-22): a trunk
lower than its band on the hoods or drops is a racing position, amber, not
"raise the bars" -- also in reports built before the rule (2026-11-13)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from app.services.video_analysis.biomechanics.action_plan_builder import current_road_trunk

ROOT = Path(__file__).resolve().parents[1]
SPA = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")

RAISE = {"component": "bar_position", "status": "needs_adjustment", "metric_name": "trunk_angle",
         "current_value": 35.0, "target_range": [40, 55], "action": "raise_bars", "amount": "",
         "reason": "Trunk angle is 35 deg (optimal 40-55 deg). Position is too aggressive -- raise the bars.",
         "priority": 3}


def _res(position: str) -> dict:
    return {"sport_type": "bike", "cycling_position": position,
            "fit_plan": {"diagnostics": [copy.deepcopy(RAISE)], "good_metrics": []}}


def test_a_stored_road_plan_no_longer_raises_the_bars():
    out = current_road_trunk(_res("road_hoods"))
    plan = out["fit_plan"]
    assert not plan["diagnostics"]
    trunk = [g for g in plan["good_metrics"] if g["metric"] == "trunk_angle"]
    assert trunk and trunk[0]["borderline"] and "racing position" in trunk[0]["note"]


def test_the_stored_row_is_not_touched_and_other_positions_keep_their_plan():
    res = _res("road_drops")
    current_road_trunk(res)
    assert res["fit_plan"]["diagnostics"][0]["action"] == "raise_bars"      # a copy was changed
    upright = _res("upright")
    assert current_road_trunk(upright) is upright


def test_the_road_sample_no_longer_says_raise_the_bars():
    from app.services.report_pdf import render_report_html

    data = json.loads((ROOT / "content" / "examples" / "bike-road.json").read_text(encoding="utf-8"))
    html = render_report_html(data, {})
    assert "Raise bars" not in html and "raise stem spacers" not in html.lower()


def test_what_to_fix_does_not_raise_the_bars_for_a_low_road_trunk():
    table = SPA[SPA.index("function fixHowTable(e){"):SPA.index("function fixHow(e, key, dir)")]
    assert "Lower than the range is a racing position" in table
    chips = SPA[SPA.index("const FIX_CHIP={"):SPA.index("function fixChip(")]
    bike = chips[chips.index("bike:{"):chips.index("aero:{")]
    assert "low:['↑','Bars'" not in bike
