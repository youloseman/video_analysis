"""Where the bike shows up on the landing page, and where its scores sit.

Two problems, both found by looking at the page rather than the code.

**The bike was eight blocks down.** Most visitors arrive for a fit check, not
for running form -- and the first photograph of anyone on a bicycle was the
before/after slider, well past the fold on any screen. Everything above it was
a runner, a phone mock, or screenshots of the report UI. The fix is an order,
which is exactly the kind of thing that gets undone by the next edit that
"just moves a section", so the order is pinned here.

**The score pills covered the evidence.** They were absolutely positioned
inside the frame, which is fine at 980px and wrong at 390: two ~150px pills
across a ~350px image sit on top of the angle labels the section exists to
show. They now live outside the frame and become a row above it on a phone.

String assertions on markup, for the same reason `test_mobility_spa.py` uses
them: what breaks here is a class name, an id, or an order, and no other test
in this suite ever looks at this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

LANDING = Path(__file__).resolve().parents[1] / "app" / "static" / "landing.html"


@pytest.fixture(scope="module")
def html() -> str:
    return LANDING.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def order(html: str) -> list[str]:
    """Section banner comments, in document order."""
    return [m.strip() for m in
            re.findall(r"<!-- ={5,} ([A-Z][^=]*?) ={5,} -->", html)]


def at(order: list[str], needle: str) -> int:
    for i, name in enumerate(order):
        if name.startswith(needle):
            return i
    raise AssertionError(f"no section starting {needle!r} in {order}")


# --------------------------------------------------------------------------
# the bike, early
# --------------------------------------------------------------------------

def test_the_first_photograph_of_a_bike_comes_right_after_the_report(order):
    """PROOF carries the only landscape shot of a rider on the page. It sits
    directly under the report showcase so the fit story lands in the first
    couple of screens."""
    assert at(order, "PROOF") == at(order, "REPORT SHOWCASE") + 1


@pytest.mark.parametrize("later", [
    "TRACKSIDE", "HOW IT WORKS", "SPLIT 1", "SPLIT 2", "DUAL PATH", "PRICING",
])
def test_the_bike_lands_before_the_rest_of_the_pitch(order, later):
    assert at(order, "PROOF") < at(order, later)


def test_the_explainer_is_still_above_the_deep_sections(order):
    """Moving PROOF up must not push "what do I actually have to do" past the
    two long feature splits."""
    assert at(order, "HOW IT WORKS") < at(order, "SPLIT 1")


def test_the_background_rhythm_still_alternates(html, order):
    """The sections stripe --c-bg / --c-panel, so a reorder that lands two
    panels together reads as one long grey slab. This is the constraint that
    decided which pair could be swapped."""
    bg = [                       # prefix -> painted background, in one list so
        ("REPORT SHOWCASE", "bg"),   # adjacency here is adjacency on the page
        ("PROOF", "panel"),
        ("TRACKSIDE", "bg"),
        ("HOW IT WORKS", "panel"),
        ("SPLIT 1", "dark"),
        ("SPLIT 2", "bg"),           # .split paints nothing -> body, --c-bg
        # WHY FLAPP (the dark divider) came out 2026-09-18: it restated the
        # hero, and the page was 23 phone screens long. bg -> panel still
        # alternates without it.
        ("DUAL PATH", "panel"),
        ("PRICING", "white"),
    ]
    # `at` raises if a section is missing, so this also pins that the list is
    # the whole striped run and not a subset that hides a collision.
    painted = sorted(bg, key=lambda kv: at(order, kv[0]))
    seq = [paint for _, paint in painted]
    assert [p for p, _ in painted] == [p for p, _ in bg], painted
    assert all(a != b for a, b in zip(seq, seq[1:])), seq


def test_the_page_names_bike_fit_before_you_scroll(html):
    """A nav link is the only bike-fit wording that is visible at the top."""
    nav = html[html.index('<div class="nav-links">'):]
    nav = nav[:nav.index("</div>")]
    assert 'href="#proof"' in nav
    assert "Bike fit" in nav


def test_that_nav_link_points_at_a_section_that_exists(html):
    """A dead anchor scrolls nowhere and is invisible to whoever wrote it."""
    for href in re.findall(r'<a href="#([a-z-]+)"', html):
        assert f'id="{href}"' in html, href


# --------------------------------------------------------------------------
# the score pills, off the evidence
# --------------------------------------------------------------------------

def test_the_pills_are_not_inside_the_frame(html):
    """Inside `.cmp` they cannot leave it: it is `overflow:hidden`, so no
    media query could lift them off the image."""
    frame = html[html.index('<div class="cmp" id="cmp">'):]
    frame = frame[:frame.index("</div>\n      <div class=\"plancard")]
    assert "cmp-tag" not in frame


def test_the_pills_sit_with_the_frame_and_not_somewhere_else(html):
    wrap = html[html.index('<div class="cmp-wrap'):]
    assert wrap.index('class="cmp-tags"') < wrap.index('<div class="cmp" id="cmp">')


def test_they_are_still_overlaid_on_a_wide_screen(html):
    block = html[html.index(".cmp-tags{"):]
    block = block[:block.index("}")]
    assert "position:absolute" in block


def test_and_become_a_row_above_the_image_on_a_phone(html):
    """The whole point: static flow, so the frame starts below them."""
    mq = html[html.index("@media(max-width:640px){\n  /* The frame is"):]
    mq = mq[:mq.index("\n}")]
    assert "position:static" in mq
    assert "margin-bottom" in mq


def test_the_slider_still_finds_its_control(html):
    """The pills moved out of the element the slider script queries."""
    assert "cmp.querySelector('input[type=range]')" in html
    frame = html[html.index('<div class="cmp" id="cmp">'):]
    assert "input type=\"range\"" in frame[:frame.index("</div>")]
