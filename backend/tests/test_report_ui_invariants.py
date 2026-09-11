"""Four one-line rules whose absence nobody notices until it is on production.

Each of these shipped broken, and none of them announced it. They are pinned
here because they are exactly the kind of line a later edit drops silently: the
page still renders, the tests still pass, and the defect is visual.

The real verification is a browser: a headless pass over the report at 1366x768
and 390x844 measuring ``scrollWidth``, the clipped-cell check and axe-core.
These are the cheap half that runs on every push.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
CSS = (STATIC / "app.css").read_text(encoding="utf-8")
SPA = (STATIC / "index.html").read_text(encoding="utf-8")


def test_the_print_keyframe_never_shows_on_screen():
    """It is a 720x1280 still, and it was sitting above the player.

    ``renderResults`` un-hides it whenever a keyframe exists, because paper
    cannot play a video and the still stands in for it. Nothing hid it again:
    every video report opened with the frame above its own player, and the page
    scrolled sideways (1401px on a 1366px viewport) because the image is wider
    than the column it lands in.
    """
    assert re.search(r"#printKeyframe\{display:none\}", CSS), (
        "the screen-scope rule is gone; the print-only still is back on screen"
    )
    # ...and print must still override it, or the paid PDF loses its only image.
    assert "#printKeyframe{display:block !important" in CSS


def test_every_data_table_can_scroll_sideways():
    """Seven columns do not fit a phone. Unwrapped, the last one is just cut.

    Two tables in the history detail were injected straight into ``innerHTML``
    with no ``.tablewrap``, so on a 390px screen their status column read
    "OPTIMA" and "PHASE DEPEN" with nothing to suggest there was more.
    """
    tables = [m.start() for m in re.finditer(r"<table class=\"data\"", SPA)]
    assert tables, "no data tables found -- has the markup moved?"
    for pos in tables:
        preceding = SPA[max(0, pos - 400):pos]
        assert "tablewrap" in preceding, (
            f"a data table at offset {pos} is not inside .tablewrap: "
            f"...{SPA[max(0, pos - 90):pos + 40]!r}"
        )
    assert ".tablewrap{" in CSS and "overflow-x:auto" in CSS


def test_the_angle_table_uses_the_asymmetric_band():
    """One hip, two verdicts, four rows apart.

    Closing the hip past its band is the risk; opening it past the band is a
    comfort trade-off and never a fault. The key-metrics tile encodes that by
    stretching the upper bound to the measurement, and the joint-angle table did
    not -- so a bike report printed "71° · in range (40–71)" on the card and
    "71 · out of range (40–55)" in the table.
    """
    assert "function angleBandFor(" in SPA
    body = SPA[SPA.index("function angleTableRows("):][:1400]
    assert "angleBandFor(" in body, "the table went back to the raw band"
    assert "angleOptimal(k" not in body, (
        "the table is reading the raw band again, bypassing the hip rule"
    )


def test_navigation_icons_are_not_all_the_same_one():
    """A sidebar you navigate by shape stops working when the shapes repeat.

    Pricing, Examples, What's new and Reviews all drew the same spark, and two
    report sections wore the compare arrows that belong to "Compare two
    analyses".
    """
    nav = SPA[SPA.index('<nav class="sidenav"'):]
    nav = nav[:nav.index("</nav>")]
    icons = re.findall(r'href="#(i-[a-z-]+)"', nav)
    assert len(icons) == len(set(icons)), f"duplicate nav icons: {icons}"
    # And the two sections that had borrowed one:
    assert '<use href="#i-film"/></svg> Kinogram' in SPA
    assert '<use href="#i-list"/></svg> Training plan' in SPA
    for symbol in ("i-tag", "i-grid", "i-bell", "i-film", "i-list"):
        assert f'id="{symbol}"' in SPA, f"{symbol} is referenced but not defined"
