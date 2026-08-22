"""The printed report — the artefact a Full subscriber pays for and hands over.

These are string checks against the shipped SPA rather than browser tests,
because the bug they guard was a CSS selector, and a selector is exactly the
kind of thing no Python test would otherwise ever look at.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return SPA.read_text(encoding="utf-8")


def _strip_comments(css: str) -> str:
    """CSS comments out. The comment above the fixed rule names the banned
    selector in order to explain it, and a scanner that cannot tell the
    warning from the offence would forbid documenting it."""
    return re.sub(r"/\*.*?\*/", " ", css, flags=re.S)


@pytest.fixture(scope="module")
def print_css(html: str) -> str:
    """Everything inside the @media print block."""
    start = html.index("@media print{")
    depth, i = 0, start + len("@media print")
    while i < len(html):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return html[start:i + 1]
        i += 1
    raise AssertionError("unterminated @media print block")


def test_no_wildcard_hides_the_whole_report(print_css: str):
    """`[class*="lock"]` also matches "block".

    It shipped, and it hid all fifteen sections: the paid Save-as-PDF printed a
    score with an empty page under it. Any substring selector that a plain
    `.block` would match is banned here.
    """
    for sel in re.findall(r'\[class\*=("[^"]*"|\'[^\']*\')\]', _strip_comments(print_css)):
        needle = sel.strip("\"'")
        assert needle not in "block", (
            f'[class*="{needle}"] in the print stylesheet also matches "block", '
            "which is every section of the report"
        )


def test_locked_sections_are_still_hidden_by_name(print_css: str):
    """Removing the wildcard must not un-hide the blurred placeholders."""
    for cls in (".tlock", ".tcard-lock", ".lockbar"):
        assert cls in print_css, f"{cls} should still be hidden in print"


def test_the_masthead_prints_before_the_report(html: str):
    """A header printed halfway down page one is not a header."""
    results = html.index('<section id="results"')
    head = html.index('id="printHead"')
    first_block = html.index('class="result-top"')
    assert results < head < first_block


def test_the_closing_caveats_print_last(html: str):
    """'How to read this' belongs after the numbers, not before them."""
    foot = html.index('id="printFoot"')
    last_block = html.index('id="exportBlock"')
    assert last_block < foot


def test_screen_only_furniture_is_hidden(print_css: str):
    """Things with no job on paper: the on-screen masthead, the ephemeral
    note, the feedback strip, and anything that exists to be clicked."""
    for cls in (".screenhead", ".ephem-note", ".fb", ".again-row", ".upsell"):
        assert cls in print_css


def test_the_video_is_replaced_by_a_still(print_css: str):
    """Paper cannot play an overlay; it can show the annotated frame."""
    assert "#videoCol video" in print_css
    assert "#printKeyframe" in print_css


def test_the_kinogram_is_allowed_to_fit(print_css: str):
    """On screen it scrolls sideways; on paper that clips it."""
    assert ".kinowrap" in print_css
