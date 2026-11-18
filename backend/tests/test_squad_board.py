"""The Squad as drawn on the Squad board (2026-11-17): the counts under the title,
To review as cards, Also as tags, and a table per sport, one column per reading."""
from __future__ import annotations

import re
from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def _squad() -> str:
    return SPA[SPA.index("function renderSquadHome("):SPA.index("function athleteLink(")]


def test_the_header_carries_the_counts_and_the_coach_actions():
    body = _squad()
    assert "$('#dashMeta')" in body and "$('#sqHeadActs')" in body
    assert 'class="sq-kpis"' not in body                       # the five tiles are gone
    assert "Latest clips" not in body                          # To review shows the new ones


def test_a_card_only_says_in_range_for_readings_that_have_one():
    """A ride keeps a range for the knee alone: "everything in range" over an
    ungraded trunk and hip was a claim the data did not make."""
    body = _squad()
    assert "banded.length===shown.length?'everything in range'" in body
    assert "data-note=" in body and "goCoachNote()" in body
    assert "id=\"sqSeenAll\"" in body


def test_one_table_per_sport_and_no_empty_columns():
    body = _squad()
    assert "const sportTable=sp=>{" in body
    assert "list.some(r=>r.last&&r.last.trends&&r.last.trends[m.key]!=null)" in body
    assert "Flapp does not calculate it" in body               # never claim we compute CdA
    assert re.search(r"state\.squadSort==='name'", body)


def test_the_rig_seeds_a_squad_worth_looking_at():
    rig = (Path(__file__).resolve().parents[1] / "scripts" / "ui_rig" / "rig.py").read_text(encoding="utf-8")
    assert "/races" in rig and "/me/athlete-groups" in rig and "state.athleteId = String(id)" in rig
