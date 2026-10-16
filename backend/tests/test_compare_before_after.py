"""Before / after: which clip a new one is read against, and what it says changed.

The comparison page used to open the two newest clips of a sport -- two bikes,
two sides of one bike, a photo and a video, all fine by it -- and colour every
difference green or red. Two things were wrong with that. A pair filmed from
opposite sides reports an "improvement" that is only the camera (the drive
side reads the knee at the bottom of the stroke 8-10 deg low), and two clips of
one unchanged position already differ by a few degrees, which it called
progress.

The pure functions are pulled out of the shipped index.html and run in node,
so what is pinned is the logic the browser runs, not a copy of it. Skipped
where node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is needed to run the SPA's own functions")


def _fn(html: str, name: str) -> str:
    i = html.index(f"function {name}(")
    j = html.index("\nfunction ", i + 10)
    return html[i:j]


def _const(html: str, name: str) -> str:
    i = html.index(f"const {name}=")
    return html[i:html.index("\n};", i) + 3]


def _run(expr: str):
    html = SPA.read_text(encoding="utf-8")
    src = "\n".join([
        "const esc=s=>String(s);",
        "const POSITION_NAMES={tt_aero:'TT aero',triathlon:'Triathlon'};",
        "const loadHist=()=>globalThis.HIST||[];",
        "const track=()=>{};",
        _const(html, "TREND_METRICS"),
        _fn(html, "distOut"), _fn(html, "metricDelta"),
        _fn(html, "cmpSame"), _fn(html, "cmpPrevious"),
        _fn(html, "cmpCaveats"), _fn(html, "cmpHeadline"),
        f"console.log(JSON.stringify({expr}));",
    ])
    out = subprocess.run([NODE, "-e", src], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _clip(i, at, **kw):
    base = {"id": i, "at": at, "sport": "bike", "kind": "video", "profileId": 7,
            "cameraSide": "left", "position": "triathlon", "score": 80,
            "trends": {"knee": 150.0}}
    base.update(kw)
    return json.dumps(base)


# --------------------------------------------------------------------------
# which clip is "before"
# --------------------------------------------------------------------------

def test_the_before_is_the_newest_earlier_clip_filmed_the_same_way():
    r = _run(f"""(()=>{{
      globalThis.HIST=[{_clip('new', 400)},
        {_clip('other-side', 300, cameraSide='right')},
        {_clip('other-bike', 250, profileId=9)},
        {_clip('photo', 220, kind='photo')},
        {_clip('match', 200)}, {_clip('older', 100)}];
      return cmpPrevious(HIST[0]).id; }})()""".replace("cameraSide='right'", "cameraSide:'right'")
              .replace("profileId=9", "profileId:9").replace("kind='photo'", "kind:'photo'"))
    assert r == "match"


def test_nothing_comparable_means_no_offer():
    r = _run(f"""(()=>{{ globalThis.HIST=[{_clip('new', 400)},
        {_clip('r', 300, cameraSide='x')}]; return cmpPrevious(HIST[0]); }})()""".replace(
        "cameraSide='x'", "cameraSide:'right'"))
    assert r is None


def test_a_clip_is_never_its_own_before_nor_a_later_one():
    r = _run(f"""(()=>{{ globalThis.HIST=[{_clip('a', 100)}, {_clip('b', 300)}];
      return [cmpPrevious(HIST[0]), cmpPrevious(HIST[1]).id]; }})()""")
    assert r == [None, "a"]


# --------------------------------------------------------------------------
# when the pair is not clean
# --------------------------------------------------------------------------

def _caveats(**b):
    bj = json.dumps({"kind": "video", "cameraSide": "left", "position": "triathlon",
                     "profileId": 7, **b})
    aj = json.dumps({"kind": "video", "cameraSide": "left", "position": "triathlon", "profileId": 7})
    return _run(f"cmpCaveats({aj},{bj})")


def test_a_clean_pair_has_no_warning():
    assert _caveats() == []


def test_opposite_sides_say_the_camera_may_be_the_difference():
    w = _caveats(cameraSide="right")
    assert len(w) == 1 and "opposite sides" in w[0] and "8–10" in w[0]


def test_a_two_sided_session_is_not_called_opposite():
    assert _caveats(cameraSide="both") == []


def test_photo_against_video_is_flagged():
    assert any("photo" in x for x in _caveats(kind="photo"))


def test_different_bikes_point_at_the_setups_view():
    assert any("Two setups" in x for x in _caveats(profileId=9))


# --------------------------------------------------------------------------
# what it says changed
# --------------------------------------------------------------------------

def _headline(a_trends, b_trends, a_score=70, b_score=85):
    a = json.dumps({"trends": a_trends, "score": a_score})
    b = json.dumps({"trends": b_trends, "score": b_score})
    return _run(f"cmpHeadline({a},{b},'bike',k=>k==='knee'?[138,145]:null)")


def test_a_knee_that_moves_into_its_range_is_called_so():
    h = _headline({"knee": 151.4}, {"knee": 144.7})
    assert "151.4° → 144.7°" in h
    assert "now inside the 138–145° range" in h
    assert "Score</b> 70 → 85" in h


def test_a_change_under_the_measurement_floor_is_not_reported():
    """3 deg is what the pose model disagrees with itself by on one clip."""
    h = _headline({"knee": 147.0, "trunk": 20.0}, {"knee": 145.0, "trunk": 21.5})
    assert "No measurement moved" in h
    assert "Knee" not in h and "Trunk" not in h


def test_moving_out_of_the_range_is_said_too():
    assert "now outside" in _headline({"knee": 143.0}, {"knee": 149.0})


def test_a_metric_with_no_range_still_reports_a_change_past_the_floor():
    h = _headline({"knee": 143.0, "trunk": 25.0}, {"knee": 143.0, "trunk": 19.0})
    assert "Trunk angle" in h and "25° → 19°" in h


def test_at_most_three_changes_the_biggest_first():
    h = _headline({"knee": 156.0, "hip": 60.0, "trunk": 25.0, "elbow": 80.0},
                  {"knee": 144.0, "hip": 52.0, "trunk": 19.0, "elbow": 90.0})
    assert h.count("<li>") == 4          # the score line plus three
    assert h.index("Knee angle") < h.index("Elbow") < h.index("Hip")
    assert "Trunk" not in h              # the smallest of four drops out


def test_run_metrics_with_no_floor_and_no_range_say_nothing_rather_than_guess():
    a = json.dumps({"trends": {"cadence": 160}, "score": 60})
    b = json.dumps({"trends": {"cadence": 178}, "score": 70})
    h = _run(f"cmpHeadline({a},{b},'run',k=>null)")
    assert "Cadence" not in h


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def test_the_report_and_the_history_page_offer_the_comparison():
    html = SPA.read_text(encoding="utf-8")
    for host in ("cmpNextV", "cmpNextP", "cmpNextH"):
        assert f'id="{host}"' in html
        assert f"renderCompareNext('{host}'" in html
    assert "openCompare(prev.id,e.id,from)" in html


def test_the_table_uses_the_floor():
    html = SPA.read_text(encoding="utf-8")
    assert "m.hi,m.noise" in html
    assert "within measurement error" in _fn(html, "cmpRow")


def test_the_landing_hero_column_cannot_outgrow_a_phone():
    """The hero badge is `width:max-content`: with the longer wording it was
    417 px wide on a 390 px phone, and the `1fr` column grew to hold it, so the
    headline's paragraph ran 51 px off the right edge of the screen."""
    land = (SPA.parent / "landing.html").read_text(encoding="utf-8")
    assert ".hero-news + .hero-badge{display:flex;width:max-content;max-width:100%}" in land
    assert "grid-template-columns:minmax(0,1fr);gap:70px" in land


def test_the_how_it_works_steps_each_carry_a_picture_and_end_on_the_recheck():
    land = (SPA.parent / "landing.html").read_text(encoding="utf-8")
    how = land[land.index('<section id="how"'):land.index("</section>", land.index('<section id="how"'))]
    assert how.count('class="step rv"') == 3
    assert how.count('<img class="shot') == 3
    assert "Change it. Film again. See what moved." in how
    for src in ("media/run-raw.webp", "media/run-overlay.webp", "media/compare.webp"):
        assert (SPA.parent / src).exists()
