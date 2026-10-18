"""The report's measurements table (redesign brief, section 2).

One row per reading: name and source | value | range bar | status word and
signed distance, grouped "needs attention" (furthest out first), "in range",
"not graded". Built from the same tile() calls the metric cards used, so the
band, its citation, the tolerance and the Kona field are the ones the score
was built on -- the table must never call "too high" what the rest of the
report counts as borderline.
"""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
TOKENS = (STATIC / "tokens.css").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    i = SPA.index(f"function {name}(")
    return SPA[i:SPA.index("\nfunction ", i + 10)]


def test_the_status_is_graded_by_the_reports_own_tolerance():
    tile = _fn("tile")
    assert "grZone(range.value,range.lo,range.hi,range.metric)" in tile
    status = _fn("measStatus")
    for word in ("'In range'", "'Borderline'", "'Too high'", "'Too low'", "'Not graded'"):
        assert word in status


def test_rows_are_grouped_with_the_furthest_out_first():
    render = _fn("renderTiles")
    assert "'Needs attention'" in render and "'In range'" in render and "'Not graded'" in render
    assert "Math.abs(b.delta||0)-Math.abs(a.delta||0)" in render


def test_the_scale_takes_the_reading_in():
    """From the brief: min(lo-0.8 span, v-0.2 span) .. max(hi+0.8 span, v+0.2 span)."""
    bar = _fn("measBar")
    assert "Math.min(lo-span*0.8, v-span*0.2)" in bar
    assert "Math.max(hi+span*0.8, v+span*0.2)" in bar


def test_status_is_never_colour_alone():
    assert 'class="meas-legend"' in SPA
    row = _fn("measRow")
    assert '<span class="m-pill">${st.word}</span>' in row


def test_amber_text_clears_aa():
    """#B26A00 is 4.2:1 on white: a marker colour, not a text colour."""
    assert "--m-near-text:#8A5600" in TOKENS
    assert ".mrow.m-near .m-pill{color:var(--m-near-text)}" in CSS


def test_a_phone_gets_a_two_line_row_per_reading():
    """A card per reading made the table ~180 px a row on a phone; a row of
    the list is name + value, then bar + verdict."""
    i = CSS.index("@media(max-width:640px){\n  .meastable,")
    phone = CSS[i:CSS.index("\n}", i)]
    assert ".meastable thead{display:none}" in phone
    assert ".meastable .mrow{display:grid" in phone
    assert ".meastable .m-val{grid-column:2;grid-row:1" in phone
    assert ".meastable td.m-st{grid-column:2;grid-row:2" in phone
