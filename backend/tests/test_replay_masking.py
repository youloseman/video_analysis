"""Session replay must never record the athlete.

PostHog replay records the DOM, and on a results page the DOM holds a video of
the person who uploaded it: the camera preview while they film, the clip they
picked, the annotated frames, the kinogram, the overlay video, the joint
editors. Recording those would mean keeping footage of people's bodies in a
third-party analytics tool — for users in the EU among others — and the privacy
policy promises in writing that it does not happen.

``ph-no-capture`` is what keeps that promise: PostHog replaces the element with
a blank box of the same size and skips every DOM change inside it.

This is exactly the kind of thing that breaks silently. Someone adds a new
<img> showing the athlete, nothing errors, no test fails, and it quietly starts
appearing in replays. So the rule is pinned here by element id rather than left
to memory: a new media element has to be named in this list, which is the
moment to decide whether it shows a person.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import read_spa

# Elements that can render the athlete's OWN footage or frames.
MUST_BE_MASKED = [
    "camView",        # live camera preview -- the person, filming, right now
    "preview",        # the clip they picked, before analysis
    "previewImg",     # same, as a still
    "printKeyframe",  # the annotated frame
    "overlayVideo",   # the annotated overlay video
    "kinogramImg",    # five stride positions of them
    "pImage",         # photo analysis, annotated
    "hdImg",          # saved keyframe in history
    "hdKinogramImg",  # saved kinogram in history
    "shotImg",        # the full-size frame viewer
    "cardPreview",    # the share card, which contains the frame
    "vEdVideo",       # joint editor: video
    "vEdCanvas",
    "vEdLoupe",
    "pEdCanvas",      # joint editor: photo
    "pEdLoupe",
]

# Published sample reports. Artur's own footage, already public on /examples --
# masking them would blank out the one page where seeing the product matters.
DELIBERATELY_NOT_MASKED = ["exClip", "exKeyframe"]


def _opening_tag(html: str, el_id: str) -> str | None:
    m = re.search(r'<\w+[^>]*\bid="' + re.escape(el_id) + r'"[^>]*>', html)
    return m.group(0) if m else None


@pytest.mark.parametrize("el_id", MUST_BE_MASKED)
def test_athlete_media_is_excluded_from_replay(el_id):
    tag = _opening_tag(read_spa(), el_id)
    assert tag is not None, (
        f"#{el_id} is gone from the SPA. If it was renamed, rename it here too; "
        f"if it was deleted, delete it from the list -- do not leave it absent."
    )
    assert "ph-no-capture" in tag, (
        f"#{el_id} can show the athlete and is NOT masked from session replay:\n"
        f"  {tag[:160]}"
    )


@pytest.mark.parametrize("el_id", DELIBERATELY_NOT_MASKED)
def test_the_published_samples_stay_visible(el_id):
    """Not an oversight -- a decision, recorded so it is not 'fixed' later."""
    tag = _opening_tag(read_spa(), el_id)
    if tag is None:
        pytest.skip(f"#{el_id} is not in the static document")
    assert "ph-no-capture" not in tag


def test_no_unlisted_video_or_canvas_escapes_review():
    """A new <video> or <canvas> with an id is either athlete media or it is
    not, and somebody has to decide which. Failing here is the prompt to do so.

    Images are deliberately not swept the same way: the document is full of
    icons and illustrations, and a list of those would be noise. Video and
    canvas are the elements that only ever exist here to show somebody moving.
    """
    known = set(MUST_BE_MASKED) | set(DELIBERATELY_NOT_MASKED)
    found = set(re.findall(r'<(?:video|canvas)[^>]*\bid="([^"]+)"', read_spa()))
    unlisted = found - known
    assert not unlisted, (
        f"new video/canvas elements nobody has classified: {sorted(unlisted)}. "
        f"Add each to MUST_BE_MASKED (it can show the athlete) or to "
        f"DELIBERATELY_NOT_MASKED (it cannot)."
    )


def test_the_privacy_policy_describes_replay_as_it_now_runs():
    """The policy used to say 'Session replay is turned off' in two places.
    Turning replay on without rewriting those made the published document
    false -- which is a worse failure than any bug in this repo."""
    from pathlib import Path

    policy = (Path(__file__).resolve().parents[1] / "app" / "static"
              / "privacy.html").read_text(encoding="utf-8")
    assert "Session replay is turned off" not in policy
    assert "Session replay is off" not in policy
    # And it has to say what IS true instead.
    assert "blocked out of the recording" in policy
    assert "Session replay" in policy
