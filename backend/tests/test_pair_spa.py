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

    def test_hiding_uses_a_class_only_this_code_owns(self, html):
        """`hidden` is managed by the drop zone, the preview and the recorder
        for their own reasons; borrowing it would leave one of them wrongly
        hidden after switching back out of pair mode."""
        assert ".pairmode-hide{display:none!important}" in html

    def test_the_second_slot_is_hidden_until_asked_for(self, html):
        i = html.index('id="pairSlot"')
        assert "hidden" in html[i - 120:i]

    def test_thumbnail_urls_are_revoked(self, html):
        """One object URL per clip, not one per repaint."""
        assert "revokeObjectURL" in _fn(html, "pairRevoke")
        assert "pairRevoke(slot)" in _fn(html, "pairThumb")

    def test_the_pair_mode_is_bike_video_only(self, html):
        """A running side view already sees both legs, and a photo has no
        pedal circle to merge against."""
        fn = _fn(html, "isPair")
        assert "state.sport==='bike'" in fn
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

    def test_a_refusal_says_the_score_belongs_to_one_side(self, html):
        fn = _fn(html, "renderBilateral")
        assert "one side" in fn

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
            emitted |= set(re.findall(r'reason=["\']([a-z_]+)["\']', src))
            emitted |= set(re.findall(r'_refusal\([^)]*?["\']([a-z_]+)["\']', src,
                                      re.S))
        # `combine_sides` logs a short tag and returns a longer one; only the
        # returned values reach the panel.
        emitted -= {"scale", "leg"}
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

    def test_an_insignificant_asymmetry_is_named_as_the_method(self, html):
        fn = _fn(html, "renderBilateral")
        assert "asymmetry_significant" in fn
        assert "pose model, not you" in fn

    def test_the_panel_is_skipped_for_ordinary_single_clip_results(self, html):
        fn = _fn(html, "renderBilateral")
        assert "if(!b){ host.innerHTML=''; return; }" in fn
