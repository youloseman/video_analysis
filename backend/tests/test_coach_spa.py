"""The coach's screens, pinned where a server test cannot see.

The first version of the roster passed every server test and filed nothing: the
app handed the athlete to ``entryFromResult``, which built the entry without it.
These run the app's own function under node where they can, and read the source
where it is a matter of which call sits where.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def fn(name: str) -> str:
    """One top-level function's source, braces balanced."""
    m = re.search(rf"^(?:async )?function {name}\(", SPA, re.M)
    assert m, f"{name} not found"
    depth, i, started = 0, m.start(), False
    while True:
        ch = SPA[i]
        if ch == "{":
            depth, started = depth + 1, True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return SPA[m.start():i + 1]
        i += 1


node = shutil.which("node")


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_saved_entry_keeps_its_athlete_its_label_and_its_note():
    """Stubbed to the one thing asked: what survives the entry builder."""
    src = fn("entryFromResult")
    script = (
        "const metricsFromVideo=()=>[],metricsFromPhoto=()=>[],compactAngleStats=()=>({}),"
        "extractTrends=()=>({t:{},o:{},c:{}});\n" + src + "\n"
        "const e=entryFromResult('video',{sport_type:'bike',technique_score:80},"
        "{id:'h1',at:1,athleteId:7,linkedAthleteId:7,label:'May Race',coachNote:'n',coachName:'J',coachOrderId:3});\n"
        "console.log(JSON.stringify(e));"
    )
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out["athleteId"] == 7 and out["linkedAthleteId"] == 7
    assert out["label"] == "May Race" and out["coachNote"] == "n" and out["coachOrderId"] == 3


def test_the_choice_on_analyze_reaches_the_save():
    assert "state.athleteId?Number(state.athleteId):null" in fn("saveToHistory")


def test_a_linked_athletes_report_is_never_saved_into_the_coachs_account():
    src = fn("rehydrateEntry")
    guard, post = src.index("if(e.linkedAthleteId)"), src.index("fetch('/me/analyses',{method:'POST'")
    assert guard < post
    assert "return fresh" in src[guard:post]


def test_nothing_on_a_linked_entry_edits_deletes_or_sells():
    assert "e.linkedAthleteId?'':`<button class=\"hdel\"" in fn("renderHistList")
    assert "e.linkedAthleteId" in fn("renderHistoryHob")
    assert "e.linkedAthleteId" in fn("renderHistoryUpsell")
    assert "!e.linkedAthleteId&&" in fn("renderHistoryDetail")
    assert "putLinkedNote" in fn("renderCoachNote")


def test_every_report_url_goes_through_the_one_that_knows_about_the_link():
    left = [ln for ln in SPA.split("\n") if "'/me/analyses/'+encodeURIComponent" in ln]
    assert all("DELETE" in ln or "+'/'+suffix" in ln for ln in left), left
    assert "athlete_id=" in fn("meUrl")


def test_my_technique_and_a_trend_are_about_one_person():
    assert "filter(e=>e.athleteId==null)" in fn("renderDashboard")
    assert "state.histAthlete==null" in fn("renderProgress")
    assert "a.athleteId" in fn("cmpSame")


def test_somebody_elses_footage_is_asked_about_before_it_is_filed():
    assert "await confirmOthers()" in fn("batchGo")
    assert "others-consent" in fn("confirmOthers")
    assert "await confirmOthers()" in SPA[SPA.index("$('#athleteSel').addEventListener"):][:300]


def test_the_batch_size_matches_the_servers():
    from app.services.coach_lane import MAX_BATCH_FILES

    assert f"const BATCH_MAX={MAX_BATCH_FILES};" in SPA


def test_the_camera_reads_the_coachs_bars():
    src = fn("camLiveVerdict")
    assert "COACH.capture" in src
    assert "grid.tiny_px||(bike?CAM_LIVE.tinyPx:CAM_LIVE.runTinyPx)" in src
    assert "grid.small_px||(bike?CAM_LIVE.smallPx:CAM_LIVE.runSmallPx)" in src
