"""Analyze as drawn (Analyze / AnalyzePhone boards, step 5 of the redesign).

What the port must keep honest, not how it looks:

* the GO card ticks only what capture_check measured -- the athlete found
  and, on a clip, tall enough in the frame. The board's "Side view" and
  "Camera is steady" are not checked by anything, so they are not claimed;
* "Details" closes the setup card, outside step 03, because step 03 hides for
  a photo and a photo has details too;
* the trust lines sit under the guide and keep "in the app" (the landing runs
  the ad pixel);
* the bar beside the button shows what is LEFT, and only once the server has
  said so.
"""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
UPLOADER = SPA[SPA.index('<section id="uploader"'):SPA.index('<section id="progress"')]


def _fn(name):
    i = SPA.index(f"function {name}(")
    return SPA[i:SPA.index("\n}\n", i)]


def test_go_ticks_only_what_the_check_measures():
    passed = _fn("capPassed")
    assert "In shot" in passed and "Big enough in the frame" in passed
    for claim in ("Side view", "steady", "Steady"):
        assert claim not in passed, f"GO claims {claim!r}, which capture_check never checks"
    card = _fn("renderCapCheck")
    assert "c.verdict==='go'" in card and "cc-ok" in card
    # The corners turn green from the same verdict.
    assert "fp.dataset.cc=c.verdict" in card
    assert '.filepreview[data-cc="go"]' in CSS


def test_details_close_the_setup_card_not_step_three():
    setup = UPLOADER[UPLOADER.index('<div class="upsetup">'):UPLOADER.index("<!-- /upsetup -->")]
    assert 'id="stepDetails"' in setup
    film = UPLOADER[UPLOADER.index('id="stepFilm"'):UPLOADER.index('id="stepWho"')]
    assert 'id="stepDetails"' not in film, "step 03 hides for a photo; Details would go with it"


def test_trust_lines_sit_under_the_guide_and_stay_scoped():
    assert UPLOADER.index('id="capguide"') < UPLOADER.index('class="trustrow"')
    assert "No ad trackers in the app" in UPLOADER
    assert "—" not in UPLOADER[UPLOADER.index('class="trustrow"'):]


def test_the_bar_shows_what_is_left_once_the_server_said_so():
    quota = _fn("renderQuota")
    assert "q.remaining/q.limit" in quota
    assert "auth&&q&&q.limit" in quota, "the bar must not show a guessed allowance"


def test_the_choices_are_named_where_they_are():
    note = _fn("scanNote")
    assert "footageFirst()?'below':'on the left'" in note
    assert "—" not in note
