"""The two-sided session in the shipped SPA.

String checks against index.html, same reasoning as test_camera_spa.py: what
breaks here is a guard, an endpoint name or a field name, and nothing else in
the suite would ever look at them.

The quiet failure modes worth pinning:

* the pair upload posting to /analyze (one clip analysed, the other silently
  dropped, and the rider charged for a session they did not get);
* the field names drifting from the endpoint's ``video_left``/``video_right``,
  which fails as a 422 the SPA would report as "upload failed";
* the panel printing a merged number when the merge was REFUSED -- the exact
  false confidence this whole feature was built to remove;
* the per-side cards growing a score each, which is how the rider ended up
  believing his legs were 9 deg apart in the first place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return SPA.read_text(encoding="utf-8")


def _fn(html: str, name: str) -> str:
    """Source of one top-level function, up to the next top-level one."""
    i = html.index(f"function {name}(")
    j = html.index("\nfunction ", i + 10)
    return html[i:j]


class TestTheUploadForm:
    def test_both_sides_is_offered(self, html):
        assert 'data-side="both"' in html

    def test_there_are_two_matching_clip_cards(self, html):
        """Two clips only read as one session if BOTH are visible as clips.

        The first version showed the second clip as a bare filename row above
        the first clip's big preview, and it looked like one clip had been
        added twice -- which is exactly what the rider reported.
        """
        assert 'id="pairSlot"' in html
        assert 'id="fileB"' in html
        assert 'id="pairCardA"' in html
        assert 'id="pairCardB"' in html

    def test_each_card_says_which_side_it_is(self, html):
        i = html.index("const PAIR_SLOTS=[")
        block = html[i:html.index("];", i)]
        assert "'Left side'" in block
        assert "'Right side'" in block

    def test_a_loaded_card_shows_a_frame_not_just_a_filename(self, html):
        fn = _fn(html, "renderPairCards")
        assert "paircard-thumb" in fn
        assert "<video" in fn

    def test_the_pair_cards_replace_the_single_clip_uploader(self, html):
        """Otherwise a third, unlabelled slot sits on screen beside them."""
        fn = _fn(html, "syncPairSlot")
        for el in ("#drop", "#filepreview", "#camRow"):
            assert el in fn
        assert "pairmode-hide" in fn

    def test_hiding_uses_a_class_only_this_code_owns(self, app_css):
        """`hidden` is managed by the drop zone, the preview and the recorder
        for their own reasons; borrowing it would leave one of them wrongly
        hidden after switching back out of pair mode.

        The rule lives in app.css since the stylesheet came out of the
        document."""
        assert ".pairmode-hide{display:none!important}" in app_css

    def test_the_second_slot_is_hidden_until_asked_for(self, html):
        i = html.index('id="pairSlot"')
        assert "hidden" in html[i - 120:i]

    def test_thumbnail_urls_are_revoked(self, html):
        """One object URL per clip, not one per repaint."""
        assert "revokeObjectURL" in _fn(html, "pairRevoke")
        assert "pairRevoke(slot)" in _fn(html, "pairThumb")

    def test_the_pair_mode_is_video_only(self, html):
        """A photo has no stride and no pedal circle to merge against.

        It was bike-only too, on the reasoning that a run side view already
        sees both legs. It does -- but it can only MEASURE the near one, and
        once there was a way to tell a trustworthy run clip from one whose
        legs swapped, running earned a pair of its own. See TestTheRunSession.
        """
        fn = _fn(html, "isPair")
        assert "state.mode==='video'" in fn


class TestTheUpload:
    def test_it_posts_to_the_pair_endpoint(self, html):
        fn = _fn(html, "analyzePair")
        assert "'/analyze-pair'" in fn
        assert "'/analyze'" not in fn

    def test_the_field_names_match_the_endpoint(self, html):
        fn = _fn(html, "analyzePair")
        assert "fd.append('video_left'" in fn
        assert "fd.append('video_right'" in fn

    def test_the_left_slot_sends_the_left_clip(self, html):
        """The slots ARE the side declaration -- swapping them here would
        merge two correctly-measured clips into a mirrored verdict."""
        fn = _fn(html, "analyzePair")
        assert "fd.append('video_left', state.file)" in fn
        assert "fd.append('video_right', state.fileB)" in fn

    def test_it_refuses_to_start_with_one_clip(self, html):
        assert "if(!state.fileB){" in _fn(html, "analyzePair")

    def test_the_paid_gate_is_reported_as_such(self, html):
        """402 is a plan boundary, not a broken upload."""
        assert "xhr.status===402" in _fn(html, "analyzePair")

    def test_analyze_routes_pairs_away_from_the_single_clip_path(self, html):
        assert "if(isPair()) return analyzePair();" in html


class TestProgress:
    def test_the_server_stage_is_shown_while_processing(self, html):
        """Two clips take about twice as long; a spinner with no words reads
        as a hang, and a reload loses the run."""
        assert "j.stage ? esc(j.stage)" in html


class TestThePanel:
    def test_a_refusal_prints_no_merged_number(self, html):
        fn = _fn(html, "renderBilateral")
        refusal = fn[fn.index("if(!b.combined)"):fn.index("const unc=")]
        assert "knee_at_bdc" not in refusal
        assert "asymmetry" not in refusal

    def test_a_refusal_says_which_clip_every_number_came_from(self, html):
        """Not just "the score": when the merge is refused, the whole metric
        table, the ranges and the saddle verdict are one clip's. A rider who
        filmed two sides otherwise reads them as both."""
        fn = _fn(html, "renderBilateral")
        assert "-side clip alone" in fn
        assert "b.metrics_side" in fn

    def test_a_loaded_side_card_shows_its_own_frame(self, html):
        """One photo for a two-sided session looks half-done -- reported."""
        fn = _fn(html, "renderBilateral")
        assert "bil-shot" in fn
        assert "keyframe_base64" in fn

    def test_both_overlays_are_offered(self, html):
        fn = _fn(html, "renderBilateral")
        assert "b.overlays" in fn
        assert "withJobToken" in fn
        assert "download" in fn

    def test_a_card_is_labelled_as_that_clip_alone(self, html):
        """The card is a fact about a clip, not a verdict about a leg."""
        fn = _fn(html, "renderBilateral")
        assert "side alone" in fn

    def test_every_refusal_reason_the_server_can_emit_has_words(self, html):
        """A reason code with no copy renders as a blank explanation, which
        reads as a bug rather than a decision.

        The list is READ OUT OF THE SERVER SOURCE rather than typed here, so
        adding a new refusal reason without writing its copy fails this test
        instead of shipping an empty panel.
        """
        import re
        from pathlib import Path
        backend = Path(__file__).resolve().parents[1]
        emitted = set()
        for mod in ("app/services/video_analysis/biomechanics/bilateral.py",
                    "app/services/video_analysis/bilateral_session.py"):
            src = (backend / mod).read_text(encoding="utf-8")
            # Only what is RETURNED reaches the panel. `logger.info` also
            # carries a `reason=`, and scraping those made this test demand
            # copy for a diagnostic tag no reader ever sees.
            emitted |= set(re.findall(
                r'BilateralFit\(\s*False,\s*reason=["\']([a-z_]+)["\']', src))
            emitted |= set(re.findall(
                r'_refusal\([^)]*?["\']([a-z_]+)["\']', src, re.S))
        assert emitted, "found no refusal reasons to check"
        i = html.index("const BIL_REASONS={")
        block = html[i:html.index("};", i)]
        missing = [r for r in sorted(emitted) if f"{r}:" not in block]
        assert not missing, f"refusal reasons with no copy: {missing}"

    def test_the_uncertainty_is_rendered_with_the_value(self, html):
        fn = _fn(html, "renderBilateral")
        assert "uncertainty_deg" in fn
        assert "\\u00b1" in fn

    def test_per_side_cards_carry_no_score(self, html):
        fn = _fn(html, "renderBilateral")
        cards = fn[fn.index("const cards="):fn.index("if(!b.combined)")]
        assert "score" not in cards

    def test_the_panel_publishes_no_asymmetry_number(self, html):
        """Left-right difference and scale error are degenerate in this
        method (see test_bilateral.TestWhyThereIsNoAsymmetryNumber), so the
        panel must not print one -- and must say why, or the two per-clip
        readings sitting side by side read as one."""
        fn = _fn(html, "renderBilateral")
        assert "asymmetry" not in fn.lower().replace("no asymmetry number", "")
        assert "no asymmetry number" in fn
        assert "the instrument, not your legs" in fn

    def test_the_panel_is_skipped_for_ordinary_single_clip_results(self, html):
        fn = _fn(html, "renderBilateral")
        assert "if(!b){ host.innerHTML=''; return; }" in fn


class TestPollingHasACeiling:
    """A job stuck in the QUEUE used to spin forever.

    `pollStart` is deliberately reset on every "queued" reply, so a clip
    waiting its turn is not failed for the queue's sake. The side effect was
    that a job which never left the queue reset the clock on every tick and
    the spinner ran until the tab was closed -- Artur watched one for two
    hours and reopened his editor thinking the tool had hung.
    """

    def test_there_is_an_outer_limit_that_nothing_resets(self, html):
        assert "const POLL_TOTAL_MS" in html
        fn = _fn(html, "pollOnce")
        assert "state.pollFirst" in fn
        assert "POLL_TOTAL_MS" in fn

    def test_the_outer_limit_is_longer_than_the_analysis_one(self, html):
        """Otherwise it fires on ordinary slow analyses instead of on the
        failure it exists for."""
        import math
        import re
        limits = {}
        for name in ("POLL_ANALYZE_MS", "POLL_TOTAL_MS"):
            m = re.search(rf"const {name} = ([0-9]+)\*([0-9]+)\*([0-9]+);", html)
            assert m, f"{name} not found or not written as a minutes product"
            limits[name] = math.prod(int(g) for g in m.groups())
        assert limits["POLL_TOTAL_MS"] > limits["POLL_ANALYZE_MS"]
        assert limits["POLL_TOTAL_MS"] >= 10 * 60 * 1000

    def test_the_clock_is_started_when_polling_starts(self, html):
        assert "state.pollFirst=Date.now()" in html


class TestHistoryKeepsOnlyWhatExists:
    def test_the_stored_session_carries_no_asymmetry_fields(self, html):
        """The server stopped publishing these; storing them would keep a
        withdrawn claim alive in every saved entry, waiting to be rendered."""
        i = html.index("bilateral:res.bilateral?")
        entry = html[i:html.index(":null,", i)]
        assert "asymmetry" not in entry
        assert "per_side" not in entry
        assert "knee_at_bdc" in entry and "uncertainty_deg" in entry


class TestTheRunSession:
    """A run pair is not a bike pair with different words.

    Two bike clips share a rigid object and get pooled against a common ruler.
    Two run clips share only the athlete -- but both of them measure cadence,
    contact time and trunk lean independently, so the gap between them is the
    session's own error, measured on the day. Every left/right difference is
    judged against that gap rather than against a number someone picked.
    """

    def test_pair_mode_is_offered_for_running(self, html):
        fn = _fn(html, "isPair")
        assert "state.sport==='bike'" not in fn
        assert "state.mode==='video'" in fn

    def test_running_is_not_offered_a_camera_side_override(self, html):
        """A run side view sees both legs, so a user-set unilateral lock would
        claim a certainty the sport does not have. Only Auto and Both sides."""
        fn = _fn(html, "syncPairSlot")
        assert "state.sport==='run'" in fn

    def test_the_upload_declares_its_sport(self, html):
        fn = _fn(html, "analyzePair")
        assert "fd.append('sport', state.sport)" in fn
        assert "state.sport==='bike') fd.append('position'" in fn

    def test_a_refusal_names_the_clip_and_the_reason(self, html):
        fn = _fn(html, "renderRunSession")
        assert "unstable_sides" in fn
        assert "RUN_SESSION_REASONS" in fn

    def test_differences_are_judged_against_the_measured_error(self, html):
        """The whole point: not a fixed threshold, but what these two clips
        showed on quantities that cannot differ."""
        fn = _fn(html, "renderRunSession")
        assert "r.readable" in fn
        assert "within this session" in fn

    def test_it_leaves_a_bike_session_alone(self, html):
        """Both panels render into the same slot."""
        fn = _fn(html, "renderRunSession")
        assert "if(!s) return;" in fn

    def test_every_run_refusal_reason_has_words(self, html):
        import re
        from pathlib import Path
        backend = Path(__file__).resolve().parents[1]
        src = (backend / "app/services/video_analysis/run_session.py").read_text(
            encoding="utf-8")
        emitted = set(re.findall(r'_refusal\([^)]*?"([a-z_]+)"', src, re.S))
        i = html.index("const RUN_SESSION_REASONS={")
        block = html[i:html.index("};", i)]
        missing = [r for r in sorted(emitted) if f"{r}:" not in block]
        assert not missing, f"run refusal reasons with no copy: {missing}"
