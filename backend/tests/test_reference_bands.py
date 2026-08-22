"""Bands and their citations, served rather than duplicated in the client.

The client used to carry its own copy of every range - literals for running, a
second position table for cycling. That is the arrangement that let the landing
page and the price catalogue drift apart twice, so these pin the served copy to
the config the analyser actually grades against.
"""
from __future__ import annotations

import pytest

from app.services.video_analysis.biomechanics.cycling_positions import (
    get_cycling_reference,
)
from app.services.video_analysis.biomechanics.sport_configs import (
    RUNNING_REFERENCE,
    RUNNING_REFERENCE_SOURCES,
    reference_bands,
)


def test_running_bands_match_the_config_the_analyser_grades_against():
    bands = reference_bands("run")
    assert bands["cadence_spm"]["lo"], "cadence band must be served"
    assert (bands["cadence_spm"]["lo"], bands["cadence_spm"]["hi"]) == \
        RUNNING_REFERENCE["cadence_spm"]
    # knee_max is the CONTACT angle, not the swing one -- an easy thing to get
    # backwards, and getting it backwards attaches the wrong citation.
    assert (bands["knee_max"]["lo"], bands["knee_max"]["hi"]) == \
        RUNNING_REFERENCE["knee_at_initial_contact"]
    assert (bands["knee_min"]["lo"], bands["knee_min"]["hi"]) == \
        RUNNING_REFERENCE["knee_at_swing"]


def test_every_running_band_carries_a_citation():
    for field, band in reference_bands("run").items():
        assert band.get("source"), f"{field} is served without a source"


def test_citations_only_name_papers_the_config_actually_cites():
    """A citation invented at the render layer would be worse than none."""
    cited = " ".join(RUNNING_REFERENCE_SOURCES.values())
    for name in ("Heiderscheit", "Folland", "Weyand", "Napier", "Souza"):
        assert name in cited
    # ...and nothing claims a study the module docstring never mentions.
    assert "Retul" not in cited          # that is a cycling source
    assert "Bini" not in cited


@pytest.mark.parametrize("position", ["road_hoods", "triathlon", "tt_aero", "casual"])
def test_cycling_bands_follow_the_position(position: str):
    bands = reference_bands("bike", position)
    ref = get_cycling_reference(position)
    assert (bands["knee_at_bdc"]["lo"], bands["knee_at_bdc"]["hi"]) == \
        tuple(ref["knee_at_bdc"])
    # A rider in the drops is not graded against a triathlete's trunk band.
    assert (bands["trunk_angle_avg"]["lo"], bands["trunk_angle_avg"]["hi"]) == \
        tuple(ref["trunk_angle"])


def test_two_positions_really_do_differ():
    road = reference_bands("bike", "road_hoods")
    tri = reference_bands("bike", "triathlon")
    assert road["trunk_angle_avg"] != tri["trunk_angle_avg"]


def test_an_unsourced_band_says_so_by_omission():
    """Silence, not a borrowed name: several cycling ranges have no citation."""
    bands = reference_bands("bike", "road_hoods")
    unsourced = [f for f, b in bands.items() if "source" not in b]
    assert unsourced, "if every band gained a source, this test should be rewritten"
    for field in unsourced:
        assert "source" not in bands[field]


def test_an_unknown_sport_serves_nothing_rather_than_running_bands():
    assert reference_bands("swim") == {}
