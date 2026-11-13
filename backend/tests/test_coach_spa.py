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
    assert src.index("saveCoachNoteIfTyped(e, W.note)") < src.index("base+'/send'")


def test_the_empty_squad_button_opens_the_form():
    assert 'onclick="openAthletes(true)">Add an athlete' in SPA
    assert "if(add===true) showAthEdit(null);" in fn("openAthletes")


def test_progress_does_not_open_on_an_empty_mine():
    assert "counts.own?chip(null,'Mine',counts.own):''" in fn("renderAthleteFilter")
    src = fn("renderProgress")
    assert src.index("state.histAthlete=String(top.id)") < src.index("renderAthleteFilter('progAthleteFilter'")


def test_several_clips_can_be_found_from_every_place_a_coach_starts():
    assert 'id="navBatch"' in SPA
    assert "$('#navBatch').classList.toggle('hidden', !hasCoachPlan())" in SPA
    home = fn("renderSquadHome")
    # With no athletes yet the page used to name a batch and offer no way to it.
    empty = home[home.index("if(!everyone.length)"):home.index("const now=Date.now()")]
    assert 'onclick="openBatch()"' in empty
    assert 'id="sqBatch"' in home
    assert "openBatch(a.id)" in fn("openAthlete")
    assert "openBatch(state.athleteId)" in SPA[SPA.index('id="upBatch"'):SPA.index('id="upBatch"') + 200]


def test_several_files_on_analyze_are_not_cut_to_the_first():
    assert "severalToBatch(ev.dataTransfer.files)" in SPA
    assert "severalToBatch(fileInput.files)" in SPA
    assert "fileInput.multiple=hasCoachPlan()" in SPA
    body = fn("severalToBatch")
    assert "files.length<2||!hasCoachPlan()" in body
    assert "batchAdd(files)" in body


def test_the_side_menu_does_not_file_a_batch_under_the_last_athlete():
    body = fn("openBatch")
    assert "batchAthleteOptions(id)" in body
    assert "state.athleteId" not in body
    # A click listener hands it an event, and an event is not somebody.
    assert "typeof who==='string'||typeof who==='number'" in body


def test_a_coach_is_not_sold_notes_on_their_own_athlete():
    assert "!(hasCoachPlan()&&e.athleteId!=null)&&(e.access==='full'" in fn("renderHistoryDetail")


def test_the_forearm_is_not_graded_off_the_aero_bars():
    assert "base==='forearm_tilt'&&!(res&&(res.cycling_position==='tt_aero'" in fn("angleOptimal")


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
    assert "go.disabled=!todo||unnamed>0" in src and "first." in src
    # While it runs the button counts, and after it never sits there grey.
    assert "Sending ${Math.min(finished+1,total)} of ${total}" in src
    assert "Add more clips or photos" in src
    assert "batchWhose(it)" in fn("batchSendVideo") and "filmed_at_ms" in fn("batchSendVideo")
    assert "batchWhose(it)" in fn("batchSendPhoto")
    assert "batchWhose(x)" in fn("batchGo")


def test_a_sent_clip_leaves_the_list_for_the_lane():
    assert "!(x.state==='sent'&&x.kind==='video')" in fn("batchGo")
    assert "it.state==='sent'||it.state==='sending'?'<span></span>'" in fn("renderBatch")
    # A photo is done on the spot: its row keeps an Open button instead.
    assert "data-openph" in fn("renderBatch") and "it.entryId=e.id" in fn("batchSendPhoto")


def test_a_batch_says_what_it_is_doing_and_how_it_ended():
    """Artur, 2026-11-13: four photos sent from a phone, a grey button, and no
    way to tell whether anything had happened."""
    rows = fn("batchStateHtml")
    assert "bat-dot" in rows and "#i-check" in rows and "#i-alert" in rows
    status = fn("renderBatchStatus")
    assert "done</b>" in status and "Keep this page open" in status
    assert "photos':'photo'" not in status and "ready.</b>" in status and "did not go through" in status
    go = fn("batchGo")
    assert "beforeunload" in go and "batch_finished" in go


def test_a_photo_check_that_never_answered_is_not_called_ambiguous():
    src = fn("batchSendPhoto")
    assert "if(!j){" in src and "Could not check this photo" in src
    assert src.index("if(!j){") < src.index("Could not tell run from ride")


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


def test_the_batch_starts_on_auto_and_the_roster_does_not_overrule_it():
    body = fn("openBatch")
    assert "s.value='auto'" in body and "batchSportOf" not in body
    assert "Tom runs" not in SPA
    assert '<option value="auto" selected>Auto, from the clip</option>' in SPA
    # A photo left on Auto asks the quick look, and the coach when it cannot tell.
    photo = fn("batchSendPhoto")
    assert "fetch('/prescan'" in photo and "sport=j&&j.sport;" in photo and "batchSportOf" not in photo


# --- Races, groups and seen marks on the Squad (api/squad.py) -----------------
@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_clip_is_unseen_only_from_the_day_the_mark_shipped():
    """Older history is not an inbox, a swim clip never counts, and the coach's
    own clips (no athlete) are nobody's to review."""
    m = re.search(r"^const UNSEEN_SINCE_MS=.*$", SPA, re.M)
    assert m
    script = (
        m.group(0) + "\n" + fn("clipUnseen") + "\n"
        "const t=UNSEEN_SINCE_MS;\n"
        "console.log(JSON.stringify(["
        "clipUnseen({athleteId:1,sport:'bike',at:t+5}),"
        "clipUnseen({athleteId:1,sport:'bike',at:t-5}),"
        "clipUnseen({athleteId:1,sport:'bike',at:t+5,reviewedAt:9}),"
        "clipUnseen({athleteId:1,sport:'swim',at:t+5}),"
        "clipUnseen({athleteId:null,sport:'bike',at:t+5}),"
        "t]));"
    )
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out[:5] == [True, False, False, False, False]
    # The same day the server counts from (api/squad.UNSEEN_SINCE_MS).
    from app.api.squad import UNSEEN_SINCE_MS

    assert out[5] == UNSEEN_SINCE_MS


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_race_countdown_reads_as_words():
    script = fn("raceWhen") + "\nconsole.log(JSON.stringify([-3,-1,0,1,9,21,56].map(d=>raceWhen({days_to_go:d}))));"
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out == ["3 days ago", "1 day ago", "today", "tomorrow", "in 9 days", "in 21 days", "in 8 weeks"]


def test_the_squad_reads_races_groups_and_seen_marks_from_the_server():
    home, one = fn("renderSquadHome"), fn("openAthlete")
    assert "state.squadGroup" in home and "To review" in home and "next_race" in home
    assert "/races" in one and "/groups" in one and "reviewed-all" in one
    assert "/reviewed'" in fn("clipSeen")
    assert "athlete-groups" in fn("loadGroups")
    # A report opened from the Squad is a report seen.
    assert "clipSeen(e,true,true)" in fn("openClipAsCoach")


def test_account_states_the_retention_the_policy_states():
    assert "deleted from our servers after processing" not in SPA
    assert "7 days on the free plan and 12 months on a paid plan" in SPA


def test_a_race_can_be_changed_in_place():
    one = fn("openAthlete")
    assert "data-raceedit" in one and "'PATCH','athletes/'+a.id+'/races/'+editR.id" in one
    # Editing is still possible with all twelve places taken.
    assert "races.length>=12&&!editR" in one
