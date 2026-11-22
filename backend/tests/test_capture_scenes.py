"""The capture guide on Analyze (Artur 2026-11-22): six drawn pairs, each the
way to film it beside the usual mistake, in the house figures."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
GUIDE = SPA[SPA.index('<section class="capguide" id="capguide"'):SPA.index("</section>", SPA.index('id="capguide"'))]


def test_six_pairs_each_with_a_yes_and_a_no():
    steps = re.findall(r'<li class="cg-step">(.*?)</li>', GUIDE, re.S)
    assert len(steps) == 6
    for s in steps:
        assert s.count('class="cgs-ok"') == 1 and s.count('class="cgs-no"') == 1
        assert re.search(r"<p><b>[^<.]+</b></p>", s)        # a heading, no full stop


def test_the_swipe_hint_counts_the_steps():
    assert 'swipe · 6' in GUIDE


def test_the_athlete_is_the_sprite_figure_for_each_sport():
    assert 'href="#pic-run" class="fig-run"' in GUIDE and 'href="#pic-bike" class="fig-bike"' in GUIDE
    assert '<img class="cg-fig' not in GUIDE
