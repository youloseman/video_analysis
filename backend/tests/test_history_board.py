"""History as drawn on the History board (2026-11-17)."""
from __future__ import annotations

from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_history_reads_by_date_and_by_month():
    assert "function histStripHtml(list)" in SPA and 'id="histStrip"' in SPA
    assert "h3 class=\"hmonth\"" in SPA and "function histCardHtml(e,list)" in SPA


def test_the_change_is_against_the_last_one_like_it():
    """cmpPrevious: same sport, side and setup -- a photo against a clip is the camera."""
    body = SPA[SPA.index("function histDelta(e,list){"):SPA.index("function histCardHtml(")]
    assert "cmpPrevious(e,list)" in body


def test_a_phone_deletes_by_swiping():
    assert "wireHistSwipe" in SPA and 'class="hswipe-del"' in SPA
    css = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.css").read_text(encoding="utf-8")
    assert ".hgroup .hcard.swiped .hswipe-del{display:flex}" in css
