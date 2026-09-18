"""What a two-sided session's report looks like, after Artur's second real pair.

He uploaded IMG_4525 + IMG_4527 and listed what was wrong with the page. Each
class below is one of those items, pinned:

* the single player above the panel showed the base clip -- the same video the
  panel showed again below, beside the other side;
* the stills sat in 16:9 boxes a card wide, where nothing on them could be read,
  and there was no way to open one;
* each card printed that clip's own knee angle, so "147.8 vs 139.5" read as a
  left-right verdict -- and the 9 deg was the chainring behind the right
  ankle, which the right clip's own warning said, two lines away;
* the joint table showed one clip's six rows and asked him to "analyze the
  right side", which he had just uploaded;
* the score tile was the width of the page with a third of it used;
* the share card drew "91" leftwards off the canvas (the header's right
  alignment was never reset) and "A" over "grade A".
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services.video_analysis import bilateral_session
from app.services.video_analysis.biomechanics import technique_scorer

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
CSS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.css"


@pytest.fixture(scope="module")
def html() -> str:
    return SPA.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return CSS.read_text(encoding="utf-8")


def _fn(html: str, name: str) -> str:
    i = html.index(f"function {name}(")
    j = html.index("\nfunction ", i + 10)
    return html[i:j]


class TestThePanelOrder:
    def test_videos_come_before_the_stills_in_both_branches(self, html):
        fn = _fn(html, "renderBilateral")
        refusal = fn[fn.index("if(!b.combined)"):fn.index("const unc=")]
        merged = fn[fn.index("const unc="):]
        for block in (refusal, merged):
            assert block.index("${vids}") < block.index('class="bil-sides"')

    def test_the_single_player_leaves_the_screen_for_a_pair(self, html, css):
        assert "classList.toggle('pair', pairVids)" in html
        assert ".result-top.pair > #videoCol{display:none}" in css
        # ...but not the printed page, which reads the still that column holds.
        assert "@media print{.result-top.pair > #videoCol{display:block}}" in css

    def test_the_pair_branch_does_not_load_the_base_overlay_twice(self, html):
        i = html.index("const pairVids=")
        branch = html[i:html.index("} else if(job.overlay_url){", i)]
        assert "removeAttribute('src')" in branch


class TestTheStills:
    def test_a_still_is_a_button_that_opens_the_lightbox(self, html):
        fn = _fn(html, "renderBilateral")
        assert 'class="bil-shot-btn"' in fn
        assert 'id="shotModal"' in html
        assert "function openShot(" in html
        assert "closest('.bil-shot-btn')" in html

    def test_the_lightbox_joins_the_shared_dialog_containment(self, html):
        fn = _fn(html, "openShot")
        assert "modalOpened(" in fn
        assert "modalClosed(" in _fn(html, "closeShot")

    def test_the_still_is_no_longer_letterboxed_into_16_9(self, css):
        i = css.index(".bil-shot{")
        rule = css[i:css.index("}", i)]
        assert "aspect-ratio:16/9" not in rule


class TestTheCards:
    def test_a_card_carries_no_knee_angle(self, html):
        """The only knee number in the panel is the session's own, in the
        merged header. Two per-clip numbers side by side are a left-right
        verdict whatever the caption says."""
        fn = _fn(html, "renderBilateral")
        assert fn.count("knee_at_bdc") == 1
        assert "b.knee_at_bdc" in fn

    def test_a_clip_with_no_pedal_circle_says_so_and_why(self, html):
        fn = _fn(html, "renderBilateral")
        assert "no usable pedal circle" in fn
        assert "quality_warnings" in fn

    def test_a_refusal_names_the_clip_that_failed(self, html):
        fn = _fn(html, "renderBilateral")
        assert "failedSide" in fn
        assert "-side clip couldn" in fn


class TestTheJointTable:
    def test_a_side_card_carries_its_clips_joint_table(self):
        result = {
            "camera_side": "right",
            "sport_specific_metrics": {"knee_at_bdc": 139.2},
            "bilateral_geometry": None,
            "angle_statistics": {
                "right_knee": {"min": 60.0, "mean": 110.0, "max": 150.0, "p05": 65.0,
                               "p95": 148.0, "valid_frames": 150, "nan_frames": 3,
                               "artefact_frames": 1, "artefact_pct": 0.6,
                               "values": list(range(150))},
                "left_knee": {"mean": None, "valid_frames": 0, "nan_frames": 153},
            },
        }
        card = bilateral_session._side_card(result)
        stats = card["angle_statistics"]
        assert stats["right_knee"]["p05"] == 65.0
        assert stats["right_knee"]["valid_frames"] == 150
        assert "values" not in stats["right_knee"], "per-frame detail must not ride along"
        assert stats["left_knee"]["mean"] is None

    def test_the_report_fills_the_far_rows_from_the_session(self, html):
        assert "function sessionOtherSide(" in html
        i = html.index("const sessionOther=sessionOtherSide(r);")
        block = html[i:i + 900]
        assert "mergeAngleStats(liveStats" in block
        assert "clip · this session" in block
        assert 'id="metricsNote"' in html


class TestTheScoreTile:
    def test_the_breakdown_is_rendered_from_the_result(self, html):
        assert 'id="scoreBd"' in html
        assert "renderScoreBreakdown(r, sport);" in html
        fn = _fn(html, "renderScoreBreakdown")
        assert "r.score_breakdown" in fn
        assert "r.score_coverage" in fn

    @pytest.mark.parametrize("sport,weights", [
        ("bike", technique_scorer.CYCLING_WEIGHTS),
        ("run", technique_scorer.RUNNING_WEIGHTS),
    ])
    def test_the_weights_on_the_tile_are_the_scorers(self, html, sport, weights):
        """The page prints a weight beside every component. It mirrors the
        scorer's table, and a mirror drifts -- so this reads both."""
        i = html.index("const SCORE_PARTS={")
        block = html[i:html.index("};", i)]
        j = block.index(f"{sport}:{{")
        sport_block = block[j:block.index("}", j)]
        found = {k: float(w) for k, w in re.findall(r"(\w+):\['[^']+',(\.\d+)\]", sport_block)}
        assert found == {k: float(v) for k, v in weights.items()}


class TestTheShareCard:
    def test_the_footer_resets_the_alignment_the_header_left_behind(self, html):
        fn = html[html.index("async function buildShareCard("):]
        fn = fn[:fn.index("\nfunction toast(")]
        header_right = fn.index("x.textAlign='right'")
        footer = fn.index("// ---- footer:")
        reset = fn.index("x.textAlign='left'", footer)
        assert header_right < footer < reset
        # and the reset comes before the score is drawn
        assert reset < fn.index("x.fillText(sTxt", footer)


class TestTheOverlayAfterARestart:
    """The job store is memory; a deploy empties it. The file is on the volume."""

    async def _stored(self, db, user, job_id):
        from app.models.analysis import Analysis

        db.add(Analysis(user_id=user.id, client_id="h1", created_at_ms=1,
                        job_id=job_id, sport="bike", kind="video", score=90, data={}))
        await db.commit()

    async def test_the_owner_still_gets_the_file(self, db, make_user, tmp_path, monkeypatch):
        from app import main
        from app.core import jobs

        # Settings is frozen; the lookup that matters is the one job_file makes.
        monkeypatch.setattr(jobs, "job_dir_for", lambda jid: tmp_path / jid)
        job_id = uuid.uuid4().hex[:12]
        (tmp_path / job_id).mkdir()
        (tmp_path / job_id / "overlay_left.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
        user = await make_user()
        await self._stored(db, user, job_id)
        assert job_id not in main.JOBS
        resp = await main.job_overlay(job_id, t=None, side="left", user=user, db=db)
        assert Path(resp.path).name == "overlay_left.mp4"

    async def test_a_stranger_and_an_anonymous_caller_get_nothing(
        self, db, make_user, tmp_path, monkeypatch,
    ):
        from app import main
        from app.core import jobs

        # Settings is frozen; the lookup that matters is the one job_file makes.
        monkeypatch.setattr(jobs, "job_dir_for", lambda jid: tmp_path / jid)
        job_id = uuid.uuid4().hex[:12]
        (tmp_path / job_id).mkdir()
        (tmp_path / job_id / "overlay.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
        owner = await make_user()
        other = await make_user()
        await self._stored(db, owner, job_id)
        with pytest.raises(HTTPException) as e:
            await main.job_overlay(job_id, t="tok", side=None, user=other, db=db)
        assert e.value.status_code == 404
        with pytest.raises(HTTPException):
            await main.job_overlay(job_id, t="tok", side=None, user=None, db=db)

    async def test_a_live_job_is_still_served_from_the_store(self, db, tmp_path):
        from app import main

        job_id = uuid.uuid4().hex[:12]
        overlay = tmp_path / "overlay.mp4"
        overlay.write_bytes(b"\x00\x00\x00\x1cftypisom")
        main.JOBS[job_id] = {
            "status": "completed", "overlay_path": str(overlay), "token": "tok",
            "owner_user_id": None, "created_at": time.time(), "job_dir": str(tmp_path),
        }
        try:
            resp = await main.job_overlay(job_id, t="tok", side=None, user=None, db=db)
            assert resp.path == str(overlay)
        finally:
            main.JOBS.pop(job_id, None)


class TestGettingBackToTheReport:
    """Artur followed the aero panel's link out of a two-sided report and could
    not get back: Back opened the analyze form, and History showed a card
    that did not say it was the session."""

    def test_the_report_keeps_its_job_in_the_url(self, html):
        fn = _fn(html, "recordRoute")
        assert "name==='results' && state.jobId ? '#job='+state.jobId" in fn

    def test_a_restored_report_reuses_its_history_entry(self, html):
        """init() restores /app#job=… by polling; the render that follows
        must overwrite the entry saved the first time, not add a twin."""
        i = html.index("async function pollOnce(")
        fn = html[i:html.index("\nfunction ", i + 10)]
        assert "e.jobId===state.jobId" in fn
        assert "reuseHistId: prior.id" in fn

    def test_the_history_card_says_both_sides(self, html):
        fn = _fn(html, "renderHistList")
        assert "e.bilateral||e.cameraSide==='both'?'both sides'" in fn

    def test_the_saved_entry_carries_the_sessions_verdict(self, html):
        fn = _fn(html, "entryFromResult")
        for k in ("partial:", "knee_single:", "agreement:", "sides:", "slots_swapped:"):
            assert k in fn, k
        assert "keyframe_base64" not in fn[fn.index("bilateral:"):fn.index("score:isV")], \
            "the side stills must not double the entry"

    def test_the_history_detail_renders_the_session(self, html):
        assert 'id="hdBilateral"' in html
        fn = _fn(html, "renderHistoryBilateral")
        assert "merged, except the knee" in fn and "not merged" in fn
        assert "renderHistoryBilateral(e);" in _fn(html, "renderHistoryDetail")
