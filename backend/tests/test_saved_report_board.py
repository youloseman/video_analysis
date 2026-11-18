"""The saved report reads as the fresh one (2026-11-17)."""
from __future__ import annotations

from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_the_saved_report_says_what_and_when_and_has_findings():
    assert 'id="hdHeadMeta"' in SPA and 'id="hdActsTop"' in SPA
    assert "function renderHistoryFindings(e)" in SPA and "renderHistoryFindings(e);" in SPA
    assert "formTiles(e.sport,[e])" in SPA[SPA.index("function renderHistoryFindings(e)"):]


def test_joint_angles_fold_and_open_for_print():
    assert 'id="hdAnglesFold"' in SPA
    assert "#hdAnglesFold:not([open])" in SPA
