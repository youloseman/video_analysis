"""Where the speed gradient goes (Artur 2026-11-21): identity numerals,
progress, one accent edge, the page-title stripe and the score line over time.
Never the score inside the ring, never a reading with a range."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.css").read_text(encoding="utf-8")


def test_only_the_score_line_runs_in_the_gradient():
    body = re.search(r"function trendChart2\(.*?\n}\n", SPA, re.S).group(0)
    assert "const gid=o.grades?" in body          # grades:true is the score chart only
    assert 'stroke="url(#${gid})"' in body


def test_the_ring_numeral_and_readings_never_take_the_gradient():
    block = APP[APP.index("/* Numerals in the gradient"):]
    sel = block[block.index("{") + 1:block.index("{", block.index("{") + 1)]
    for never in (".ds-score", ".score-num", ".ring", ".pc-big", ".dsc-v", ".tdelta"):
        assert never not in sel, never
    # a count in a warning colour keeps it
    assert ".sql-kpi:not(.sql-kpi-race):not(.sql-kpi-bad) b" in sel
    # last in the file, so it beats each base rule's colour
    assert APP.rstrip().endswith("}")
    assert APP.index("/* Numerals in the gradient") > APP.index(".sql-kpi b{")


def test_drills_done_is_a_progress_bar_without_inline_style():
    assert '<progress class="gprog"' in SPA
    assert ".gprog::-webkit-progress-value{background:var(--g-speed)" in APP
