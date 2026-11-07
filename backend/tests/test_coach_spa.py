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


def test_send_takes_the_note_still_in_the_box():
    """Send reads what is saved. A note typed and not saved used to stay behind."""
    src = fn("renderCoachActions")
    assert src.index("saveCoachNoteIfTyped(e)") < src.index("base+'/send'")


def test_the_empty_squad_button_opens_the_form():
    assert 'onclick="openAthletes(true)">Add an athlete' in SPA
    assert "if(add===true) showAthEdit(null);" in fn("openAthletes")


def test_progress_does_not_open_on_an_empty_mine():
    assert "counts.own?chip(null,'Mine',counts.own):''" in fn("renderAthleteFilter")
    src = fn("renderProgress")
    assert src.index("state.histAthlete=String(top.id)") < src.index("renderAthleteFilter('progAthleteFilter'")


def test_the_camera_reads_the_coachs_bars():
    src = fn("camLiveVerdict")
    assert "COACH.capture" in src
    assert "grid.tiny_px||(bike?CAM_LIVE.tinyPx:CAM_LIVE.runTinyPx)" in src
    assert "grid.small_px||(bike?CAM_LIVE.smallPx:CAM_LIVE.runSmallPx)" in src


# --------------------------------------------------------------------------
# A change of side is not a change of rider
# --------------------------------------------------------------------------
def run_js(body: str) -> dict:
    out = subprocess.run([node, "-e", body], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(not node, reason="node is not installed")
def test_the_squad_reads_a_clip_against_one_from_the_same_side_only():
    script = fn("cmpSame") + fn("cmpPrevious") + fn("cmpGapWord") + """
    const base={sport:'bike',kind:'video',athleteId:1,position:'tt_aero',trends:{knee:150}};
    const right={...base,id:'r',at:30,cameraSide:'right'}, left={...base,id:'l',at:20,cameraSide:'left'},
          left0={...base,id:'l0',at:10,cameraSide:'left'};
    const es=[right,left,left0];
    console.log(JSON.stringify({afterSwitch:cmpPrevious(right,es),left:cmpPrevious(left,es).id,
      gap:cmpGapWord(right,es),none:cmpGapWord(left0,es)}));"""
    got = run_js(script)
    assert got["afterSwitch"] is None and got["left"] == "l0"
    assert "other side" in got["gap"] and got["none"] == ""


def test_no_screen_draws_a_line_or_a_delta_across_two_sides():
    assert "cmpPrevious(es[0],es)" in fn("renderSquadHome")
    assert "cmpPrevious(e,es)" in fn("openAthlete")
    assert "state.progressSide" in fn("renderProgress")
    assert "e.cameraSide===state.progressSide" in fn("renderProgress")
    assert "sameSide" in fn("sportCard")
    assert "cmpPrevious(cur,entries)" in fn("formTiles") and "cmpPrevious(cur,entries)" in fn("currentFormTable")
    assert "every[0].cameraSide" in fn("thenPair")


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_coachs_own_bars_silence_the_filming_notes():
    script = "const coachOn=()=>true;let COACH={capture:{small_px:300}};\n" + fn("filmNotesOf") + fn("filmLook") + """
    const notes=[{code:'framing',level:'warn',px:320,short:'small'},{code:'drive_side',level:'warn',short:'chain'},{code:'duration',level:'bad',short:'little'}];
    const quiet=filmNotesOf({filmNotes:notes}).map(n=>n.code);
    COACH={};
    const all=filmNotesOf({filmNotes:notes}).map(n=>n.code);
    const small={filmNotes:[notes[0]]};
    console.log(JSON.stringify({quiet,all,look:filmLook({filmNotes:notes}).code,smallOnly:filmLook(small)}));"""
    got = run_js(script)
    # A chainring note saved on an old clip stays quiet for everyone.
    assert got["quiet"] == ["duration"] and got["all"] == ["framing", "duration"]
    # Small in the frame alone is on the row but not under Needs attention.
    assert got["look"] == "duration" and got["smallOnly"] is None


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_rebuilt_entry_keeps_its_filming_notes():
    script = (
        "const metricsFromVideo=()=>[],metricsFromPhoto=()=>[],compactAngleStats=()=>({}),"
        "extractTrends=()=>({t:{},o:{},c:{}});\n" + fn("entryFromResult") + """
    const n=[{code:'drive_side',level:'warn',short:'s',text:'t'}];
    const full=entryFromResult('video',{sport_type:'bike',sport_specific_metrics:{capture_report:{notes:n}}},{id:'a',at:1});
    const free=entryFromResult('video',{sport_type:'bike',quality:{capture_report:{notes:n}}},{id:'b',at:1});
    const clean=entryFromResult('video',{sport_type:'bike',sport_specific_metrics:{capture_report:{notes:[]}}},{id:'c',at:1});
    console.log(JSON.stringify({full:full.filmNotes,free:free.filmNotes,clean:clean.filmNotes||null}));""")
    got = run_js(script)
    assert got["full"][0]["code"] == "drive_side" and got["free"][0]["code"] == "drive_side"
    assert got["clean"] is None


def test_the_lane_says_what_it_found_on_a_finished_clip():
    src = fn("renderLane")
    assert "i.status!=='done'" not in src and "Ready, read the note" in src


def test_a_coach_is_not_offered_a_way_to_wipe_the_squad():
    src = fn("renderHistList")
    assert "hint.hidden=!all.length||hasCoachPlan()" in src
    assert "clr.classList.toggle('hidden',hasCoachPlan())" in src
    assert "if(hasCoachPlan()) return;" in fn("clearAllHistory")



# --------------------------------------------------------------------------
# Two people are not a before and after
# --------------------------------------------------------------------------
@pytest.mark.skipif(not node, reason="node is not installed")
def test_two_people_read_as_a_gap_and_never_as_progress():
    script = "const esc=s=>s;\n" + fn("cmpGapRow") + """
    const m={key:'knee',label:'Knee',unit:' deg',noise:3};
    const A={trends:{knee:153},trendOpt:{knee:[138,145]}}, B={trends:{knee:145},trendOpt:{knee:[138,145]}};
    const close={trends:{knee:147},trendOpt:{}};
    console.log(JSON.stringify({row:cmpGapRow(m,A,B),near:cmpGapRow(m,A,{...close,trends:{knee:151}})}));"""
    got = run_js(script)
    assert "closer" not in got["row"] and "further" not in got["row"]
    assert "tdelta sta" in got["row"] and ">8 deg<" in got["row"]
    assert 'cmp-a cmp-off' in got["row"] and 'cmp-b cmp-off' not in got["row"]
    assert "within error" in got["near"]


def test_the_compare_page_says_two_athletes_when_it_is_two_athletes():
    src = fn("renderCompare")
    assert "people?'Two athletes':'Before / after'" in src
    assert "(setups||sidesMode||people) ? '' : baSliderHtml" in src
    assert "if(people) return" in src, "two people get no WHAT CHANGED headline"
    assert "Back to Squad" in src
    assert "different people" not in fn("cmpCaveats"), "the page says so itself; it is not a caveat on a before/after"


def test_the_menu_does_not_open_on_two_people_when_one_person_has_two_clips():
    src = fn("openCompare")
    assert "hasCoachPlan()&&list.find(e=>cmpPrevious(e,list))" in src
    assert "!cmpPeople(e,b)" in src
    assert "!cmpPeople(e,who)" in fn("cmpSides"), "left vs right is one person's two legs"


# --------------------------------------------------------------------------
# The batch's rows
# --------------------------------------------------------------------------
@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_row_is_not_filed_under_the_coach_by_default():
    script = "const esc=s=>s;let squad=[{id:7,name:'Maya'}];const liveAthletes=()=>squad;\n" + fn("batchAthleteOptions") + fn("batchWhose") + """
    const open=batchAthleteOptions('');
    const me=batchAthleteOptions('me');
    const maya=batchAthleteOptions(7);
    squad=[]; const alone=batchAthleteOptions('');
    console.log(JSON.stringify({open,me,maya,alone,w1:batchWhose({athleteId:'me'}),w2:batchWhose({athleteId:'7'}),w3:batchWhose({athleteId:''})}));"""
    got = run_js(script)
    assert got["open"].startswith('<option value="">Choose') and 'value="me" selected' not in got["open"]
    assert 'value="me" selected' in got["me"] and 'value="7" selected' in got["maya"]
    # No squad: there is nobody else it could be.
    assert 'value="me" selected' in got["alone"] and "Choose" not in got["alone"]
    assert got["w1"] is None and got["w2"] == "7" and got["w3"] is None


def test_the_batch_waits_for_an_answer_and_sends_only_real_ids():
    src = fn("renderBatch")
    assert "go.disabled=b.running||!todo||unnamed>0" in src and "first." in src
    assert "batchWhose(it)" in fn("batchSendVideo") and "filmed_at_ms" in fn("batchSendVideo")
    assert "batchWhose(it)" in fn("batchSendPhoto")
    assert "batchWhose(x)" in fn("batchGo")


def test_a_sent_clip_leaves_the_list_for_the_lane():
    assert "!(x.state==='sent'&&x.kind==='video')" in fn("batchGo")
    assert "it.state==='sent'?'<span></span>'" in fn("renderBatch")


def test_a_thumbnail_arriving_does_not_close_an_open_dropdown():
    src = fn("batchThumbs")
    assert "renderBatch" not in src and "el.outerHTML=batchThumbHtml(it)" in src


def test_the_date_a_clip_was_filmed_decides_the_order_and_survives_a_rebuild():
    assert "(b.filmedAt||b.at||0)-(a.filmedAt||a.at||0)" in fn("coachPool")
    assert "x.filmedAt||x.at" in fn("cmpPrevious")
    assert "'filmedAt','fileName'" in fn("entryFromResult")
    assert "fmtWhen(e.at)" not in SPA, "every date on a clip reads filmedAt first"


# --------------------------------------------------------------------------
# The coach's name
# --------------------------------------------------------------------------
@pytest.mark.skipif(not node, reason="node is not installed")
def test_the_squad_asks_for_the_name_only_until_there_is_one():
    script = "let on=true;const coachOn=()=>on;let COACH={loaded:true,name:''};\n" + fn("coachNameAsk") + """
    const ask=coachNameAsk();
    COACH.name='  '; const blank=coachNameAsk();
    COACH.name='Coach Mike'; const named=coachNameAsk();
    COACH={loaded:false,name:''}; const early=coachNameAsk();
    console.log(JSON.stringify({ask,blank,named,early}));"""
    got = run_js(script)
    assert "sqCoachName" in got["ask"] and "sqCoachName" in got["blank"]
    assert got["named"] == "" and got["early"] == "", "no flash of the question before the profile has loaded"


def test_both_squad_screens_carry_the_question_and_it_redraws_once_loaded():
    src = fn("renderSquadHome")
    assert src.count("${coachNameAsk()}") == 2 and src.count("wireCoachNameAsk()") == 2
    assert "renderSquadHome()" in fn("loadCoach")
