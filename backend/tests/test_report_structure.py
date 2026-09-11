"""The report reads in fix-order, and the rail points at sections that exist.

The report ran to roughly 7,000px with no way to move around it, and it opened
with the answer before the question: kinogram, then the coach's prose, then the
drills, and only then the findings all of that was written about. It now goes
score and footage, what is wrong, what it means, what to do, then the evidence
and the numbers -- with a sticky rail listing exactly the sections this
particular report has.

Order is markup order here, which is the thing a later edit silently breaks: a
block moved for a layout reason reads fine in isolation and rearranges the
argument.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(
    encoding="utf-8",
)
RESULTS = SPA[SPA.index('<section id="results"'):SPA.index('<section id="photoResults"')]


def _positions(*ids):
    out = []
    for block_id in ids:
        m = re.search(r'id="%s"' % block_id, RESULTS)
        assert m, f"{block_id} is not in the results section any more"
        out.append(m.start())
    return out


def test_the_problem_comes_before_the_prose_about_it():
    findings, coach, plan = _positions("issuesBlock", "coachBlock", "planBlock")
    assert findings < coach < plan, (
        "reading order broke: it must be Findings -> Coach -> Plan"
    )


def test_the_bike_read_outs_sit_with_the_sections_they_belong_to():
    findings, aero, coach = _positions("issuesBlock", "aeroBlock", "coachBlock")
    assert findings < aero < coach, "Aero is a finding, not a footnote to the coach"
    plan, fit, kino = _positions("planBlock", "fitBlock", "kinogramBlock")
    assert plan < fit, "fit adjustments ARE the plan on a bike -- keep them adjacent"
    assert fit < kino, "the kinogram is evidence; it follows what to do about it"


def test_the_numbers_come_after_the_argument_and_the_lab_comes_last():
    kino, metrics, angles, lab, export = _positions(
        "kinogramBlock", "metricsBlock", "anglesBlock", "biomechBlock", "exportBlock",
    )
    assert kino < metrics < angles < lab < export


def test_the_rail_only_points_at_sections_that_exist():
    """A rail entry for a missing id is a button that silently does nothing."""
    listed = re.search(r"const RAIL_SECTIONS=\[(.*?)\];", SPA, re.S)
    assert listed, "the rail's section table is gone"
    ids = re.findall(r"\['(\w+)',", listed.group(1))
    assert ids, "the rail lists no sections"
    for block_id in ids:
        assert f'id="{block_id}"' in RESULTS, f"rail points at missing #{block_id}"
    # ...and in the same order the page is in, or the contents contradict it.
    assert _positions(*ids) == sorted(_positions(*ids)), (
        "the rail lists sections in a different order than the report shows them"
    )


def test_the_lab_section_is_folded_and_says_something_while_folded():
    """Eight charts, the deepest thing in the report and the least actionable.

    Opened flat it put a red "23% match" directly under an A grade. Folded, the
    header still has to carry the two overall figures -- a fold that says
    nothing is just a hidden section.
    """
    assert '<details class="block foldblock hidden" id="biomechBlock">' in RESULTS
    assert 'id="biomechPeek"' in RESULTS, "the folded header shows no numbers"
    assert 'id="biomechVerdict"' in RESULTS, "nothing reconciles the cards with the score"
    assert "function renderBiomechSummary(" in SPA
    # Sport-aware on purpose: RUNNING_WEIGHTS scores these, CYCLING_WEIGHTS does not.
    assert "A ride is scored on its fit angles" in SPA


@pytest.mark.parametrize("needle", [
    'class="t-far hidden"',          # the rows start folded
    "data-showfar",                  # ...and there is a control to unfold them
    "the camera could not see",      # ...that says why they are empty
])
def test_the_joints_the_camera_could_not_see_are_folded_away(needle):
    """Six rows of em-dashes in the middle of a bike table read as failed data."""
    assert needle in SPA
