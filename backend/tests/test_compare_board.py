"""Compare as drawn on the Compare board (2026-11-17)."""
from __future__ import annotations

from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_two_sessions_read_as_dumbbells_on_the_range():
    assert "function cmpDbRow(" in SPA and 'class="cmp-grid cmp-grid-db"' in SPA
    body = SPA[SPA.index("function renderCompare(){"):SPA.index("function renderSidePickers(")]
    # setups, sides and two people keep their own rows: no before/after verdict there
    assert "setups ? (m=>cmpRow(" in body and "cmpGapRow(m,A,B)" in body and "cmpSideRow(" in body
    assert "formBand(sport,k,B)" in body


def test_the_score_change_sits_between_the_pickers():
    assert 'class="cmp-mid"' in SPA
