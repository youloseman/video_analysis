"""The uploader's controls are reachable, and reachable by everyone.

Three things here shipped broken and rendered fine.

The camera-side control lived INSIDE the bike-position block, which is hidden
for running. So the two-sided run session -- backend, merge logic, run-specific
copy in syncPairSlot -- was unreachable from the interface. No runner could
click it, and nothing complained, because a hidden control is not an error.

The history card was a button with a button inside it (the delete X). A screen
reader announces the card and cannot reach the delete; Enter on the card had to
be told not to fire when it came from the X. Two controls, one inside the
other, is the axe rule `nested-interactive`, and it fired on every card.

The drop zone told every phone "Drag & drop" -- and most clips are filmed on a
phone. Measured against the pointer, not the screen width: a tablet with a
trackpad drags fine.
"""
from __future__ import annotations

import re
from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(
    encoding="utf-8",
)
UPLOADER = SPA[SPA.index('<section id="uploader"'):SPA.index('<section id="progress"')]


def _block(html, block_id):
    """The element with this id, to its matching close tag (div or details)."""
    m = re.search(r'<(div|details)\b[^>]*\bid="%s"' % block_id, html)
    assert m, block_id
    tag = m.group(1)
    i = m.end()
    depth = 1
    for t in re.finditer(r"<%s\b|</%s>" % (tag, tag), html[i:]):
        depth += 1 if t.group().startswith("<" + tag) else -1
        if depth == 0:
            return html[m.start(): i + t.end()]
    raise AssertionError("unbalanced " + block_id)


def test_the_camera_side_control_is_not_inside_the_bike_only_block():
    """Hidden with the bike position, the run session could never be started."""
    position = _block(UPLOADER, "positionField")
    assert 'id="sideField"' not in position, (
        "#sideField is nested in #positionField again -- hidden for running"
    )
    assert 'id="sideField"' in UPLOADER


def test_the_folded_controls_state_what_they_are_set_to():
    """A fold that says nothing while closed is a hidden control."""
    side = _block(UPLOADER, "sideField")
    assert side.startswith("<details"), "camera side is meant to fold"
    assert 'id="sideNow"' in side, "the folded camera-side control shows no setting"
    assert "function syncSideSummary(" in SPA
    opt = _block(UPLOADER, "optBox")
    assert 'id="profileField"' in opt and 'id="heightField"' in opt


def test_history_cards_do_not_nest_one_control_in_another():
    cards = re.findall(r'<div class="hcard"[^>]*>', SPA)
    assert cards, "history card markup moved"
    for c in cards:
        assert 'role="button"' not in c and "tabindex" not in c, (
            "the card is a control again, with the delete button nested inside it"
        )
    assert '<button type="button" class="htitle"' in SPA, "no keyboard way to open a card"


def test_the_drop_zone_copy_follows_the_pointer_not_the_screen():
    assert "function dropCopyFor(" in SPA
    assert "matchMedia('(pointer: coarse)')" in SPA
    assert "Tap to choose a clip" in SPA
    # Both the boot path and the mode switch go through it.
    assert SPA.count("dropCopyFor(") >= 3


def test_a_clip_this_browser_cannot_decode_gets_a_stand_in():
    """An iPhone .MOV is HEVC; Chrome on Windows shows a black rectangle."""
    assert 'id="previewPlace"' in UPLOADER
    assert "vid.onerror=fallback" in SPA
    assert "if(!vid.videoWidth) fallback()" in SPA


def test_choosing_a_clip_brings_the_button_into_view():
    """On a bike the button sits ~460px under the fold; scrolling beats shrinking."""
    assert "function bringAnalyzeIntoView(" in SPA
    body = SPA[SPA.index("function setFile("):SPA.index("function bringAnalyzeIntoView(")]
    assert "bringAnalyzeIntoView();" in body


def test_dates_are_formatted_in_the_interface_language():
    """`toLocaleDateString([])` is the browser's language; the UI is English."""
    assert "const UI_LOCALE='en-CA';" in SPA
    assert "toLocaleDateString([]" not in SPA
    assert "toLocaleTimeString([]" not in SPA
