"""Profiles as drawn (Profiles / ProfilesPhone boards, step 6 of the redesign).

What the port must keep honest:

* the score trail is the server's (``scores`` from /me/profiles, the column
  History reads), never recomputed in the browser;
* a CdA step is shown as a number only between two measurements taken the
  same way; otherwise the row says "measured differently";
* every open CdA panel still says Flapp does not calculate CdA;
* no em dash in the screen's copy.
"""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")


def _fn(name):
    i = SPA.index(f"function {name}(")
    return SPA[i:SPA.index("\n}\n", i)]


def test_the_trail_is_the_servers():
    assert "p.scores" in _fn("profScoreHtml")


def test_a_step_is_a_number_only_like_for_like():
    like = _fn("cdaLike")
    assert "a.source===b.source" in like and "a.protocol===b.protocol" in like
    assert "single_ride" in like
    cell = _fn("cdaStepCell")
    assert "cdaLike(cur,prev)" in cell and "measured differently" in cell
    # The card's headline delta follows the same rule.
    assert "cdaLike(c[0],c[1])" in _fn("profCdaHead")


def test_the_panel_never_claims_flapp_measured_it():
    assert "CDA_DISCLAIMER" in _fn("cdaPanel")
    assert "Flapp does not calculate" in SPA[SPA.index("const CDA_DISCLAIMER="):][:200]


def test_no_em_dash_on_the_screen():
    sec = SPA[SPA.index('<section id="profiles"'):SPA.index('<section id="athletes"')]
    assert "—" not in sec
    for name in ("profSubLine", "profCardHtml", "renderProfiles", "cdaPanel", "cdaRowsHtml"):
        assert "—" not in _fn(name), name
    pos = SPA[SPA.index("const PROFILE_POS_LABEL="):][:200]
    assert "—" not in pos.split("\n")[0]
