"""The verdict in the score card and the Findings card say what the table says.

A ride with the knee 5 deg past its band and a "Lower the saddle" card opened
on "No major technique issues detected", with a Saddle height row reading
"Acceptable, within tolerance" beside it (2026-11-13). The three are now drawn
from one list of readings; these pin that, under node where the logic is, and
in the source where it is a matter of which call sits where.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
node = shutil.which("node")


def fn(name: str) -> str:
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


def const(name: str) -> str:
    m = re.search(rf"^const {name}=\{{.*?^\}};", SPA, re.M | re.S)
    assert m, f"{name} not found"
    return m.group(0)


STUBS = (
    "const esc=x=>String(x), pretty=s=>String(s).replace(/_/g,' '), num=(v,d)=>String(Math.round(v));\n"
    "const fitWords=s=>String(s||'').replace(/_/g,' ');\n"
    # the verdict's chip reads the fix table (index.html); not under test here
    "const VERDICT_FIX_KEY={}, ISSUE_FIX={}, DIAG_LABEL={trunk_angle:'Trunk'}, fixHow=()=>'', fixChip=()=>null;\n"
)
ROW = ("const row=(lbl,val,lo,hi,zone,unit)=>({lbl,val:String(val),unit:unit||'°',range:{value:val,lo,hi,unit:unit||'°'},"
       "zone,delta:val<lo?val-lo:(val>hi?val-hi:0),cls:zone==='optimal'?'good':zone==='warning'?'warn':'bad'});\n")


def run(case: str) -> dict:
    src = (STUBS + const("FIT_ACTION") + "\n"
           + "\n".join(fn(n) for n in ("measStatus", "measNum", "measUnit", "measSigned", "measRange", "resultVerdict"))
           + "\n" + ROW + case
           + "\nconsole.log(JSON.stringify({level:v.level,head:v.head,why:v.why,go:v.go,out:v.out.length,near:v.near.length,ok:v.ok.length}));")
    out = subprocess.run([node, "-e", src], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_knee_past_its_band_leads_with_the_plans_first_step():
    v = run(
        "const tiles=[row('Knee · bottom',150,138,145,'bad'),row('Trunk',20,20,25,'optimal'),"
        "Object.assign({lbl:'Saddle height',val:'Lower 5mm',isText:true,cls:'warn',range:null},{derived:true})];\n"
        "const v=resultVerdict({fit_plan:{diagnostics:[{component:'saddle_height',status:'needs_adjustment',action:'lower_saddle',amount:'5mm'}]}},'bike',tiles);"
    )
    assert v["level"] == "bad" and v["head"] == "Lower the saddle 5mm"
    assert "150°" in v["why"] and "138–145°" in v["why"]
    # The saddle row restates the knee: one finding, not two.
    assert (v["out"], v["near"], v["ok"]) == (1, 0, 1)
    assert v["go"][0] == "fitBlock"


@pytest.mark.skipif(not node, reason="node is not installed")
def test_nothing_out_of_range_says_so_and_only_then():
    v = run("const v=resultVerdict({},'bike',[row('Knee · bottom',142,138,145,'optimal'),row('Trunk',22,20,25,'optimal')]);")
    assert v["level"] == "good" and v["head"] == "Every graded reading is in range" and v["go"] is None
    assert "All 2 graded readings" in v["why"]


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_run_at_the_edge_points_at_its_drill():
    v = run(
        "const v=resultVerdict({training_plan:{top_3_priorities:[{drill:'Bounding Drill',weeks:4}]}},'run',"
        "[row('Vert. oscillation',12.7,6,10,'warning','cm'),row('Cadence',186,170,225,'optimal','spm')]);"
    )
    assert v["level"] == "warn" and v["head"] == "Vert. oscillation is at the edge of its range"
    assert "12.7 cm" in v["why"] and v["go"] == ["planBlock", "Drill: Bounding Drill, about 4 weeks"]


@pytest.mark.skipif(not node, reason="node is not installed")
def test_a_server_issue_with_a_clean_table_is_still_said():
    v = run("const v=resultVerdict({detected_issues:[{type:'overstriding',severity:'medium',recommendation:'Land closer.'}]},'run',"
            "[row('Cadence',186,170,225,'optimal','spm')]);")
    assert v["level"] == "warn" and v["head"] == "overstriding" and v["why"] == "Land closer."


def test_the_table_the_card_and_findings_read_one_list():
    render = fn("render")
    assert "const tiles=sport==='bike'?bikeTiles(s, r.cycling_position||null, r.fit_plan):runTiles(s);" in render
    assert "renderTiles(tiles);" in render and "renderFindings(verdict);" in render
    # A withheld score has no verdict to give; every new result starts without one.
    assert "renderVerdict(withheld?null:verdict);" in render and "renderVerdict(null);" in render
    # The all-clear is the last branch of Findings, after issues and after rows out of range.
    f = fn("renderFindings")
    assert f.index("v.issues.length") < f.index("v.out.length||v.near.length){") < f.index("No major technique issues detected")


def test_the_saddle_row_follows_the_plan():
    tiles = fn("bikeTiles")
    assert "d.component==='saddle_height'&&(d.action==='lower_saddle'||d.action==='raise_saddle')" in tiles
    assert "m.derived=true" in tiles


def test_the_video_caption_no_longer_says_results_are_not_stored():
    assert "Results aren’t stored, so download to keep it." not in SPA
    assert "A number on the video is that frame’s; the table reads the whole clip" in SPA


def test_a_bands_source_is_one_word_until_asked_for():
    row = fn("measRow")
    assert 'class="m-srcbtn"' in row and 'class="m-src hidden"' in row
    assert "e.target.closest('.t-what,.m-srcbtn')" in SPA



@pytest.mark.skipif(not node, reason="node is not installed")
def test_fix_first_line_is_about_its_own_reading():
    """'Raise the bars' over 'Elbow 131°' read as two unrelated things."""
    v = run(
        "const v=resultVerdict({fit_plan:{diagnostics:[{action:'raise_bars',status:'needs_adjustment',metric_name:'trunk_angle',"
        "current_value:35.5,target_range:[40,55]}]}},'bike',[row('Elbow',131,145,165,'critical')]);"
    )
    assert v["head"] == "Raise the bars"
    assert v["why"].startswith("Trunk") and "40" in v["why"] and "Elbow" not in v["why"]
