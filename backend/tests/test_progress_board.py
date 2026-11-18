"""Progress as drawn on the Progress board (2026-11-17)."""
from __future__ import annotations

from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def _progress() -> str:
    return SPA[SPA.index("function renderProgress(){"):SPA.index("function openProgress(){")]


def test_readings_are_grouped_by_what_they_did():
    body = _progress()
    assert "'Moving away'" in body and "'Improving'" in body and "Held steady" in body
    # a reading with no range is shown as a line, never counted as steady
    assert "loose=rest.filter(x=>!x.opt)" in body and "No range to read against" in body


def test_the_band_is_the_one_the_dashboard_and_report_use():
    assert "formBand(sport,m.key,lastE)" in _progress()


def test_one_side_per_line_still():
    body = _progress()
    assert "e.cameraSide===state.progressSide" in body
