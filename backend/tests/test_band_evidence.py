"""Why each range is what it is (Artur, 2026-11-22, after John Bye asked why
our ranges are "optimal"): every served band carries a reason, the kind of
evidence it is and links to the studies; an explanation can never drift from
the band it explains."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.services.video_analysis.biomechanics import band_evidence as BE
from app.services.video_analysis.biomechanics.sport_configs import reference_bands

POSITIONS = ("road_hoods", "road_drops", "tt_aero", "triathlon")
SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def test_every_entry_explains_the_live_band():
    for (sport, position, field), e in BE.EVIDENCE.items():
        band = reference_bands(sport, position).get(field)
        assert band, (sport, position, field)
        assert (e["lo"], e["hi"]) == (band["lo"], band["hi"]), (sport, position, field)


def test_every_served_band_has_an_answer():
    for pos in POSITIONS:
        for field in reference_bands("bike", pos):
            if field == "forearm_tilt_avg" and pos.startswith("road"):
                continue  # an aero-bar measure: shown, not graded off the bars
            assert ("bike", pos, field) in BE.EVIDENCE, (pos, field)
    for field in reference_bands("run"):
        assert ("run", None, field) in BE.EVIDENCE, field


def test_entries_are_honest_and_well_formed():
    for key, e in BE.EVIDENCE.items():
        assert e["basis"] in BE.BASIS_LABEL, key
        for r in e.get("refs", []):
            assert r in BE.REFS, (key, r)
        out = BE._entry_out(e)
        assert out["why"].endswith("."), key
        assert "—" not in out["why"], key  # no em dash in Flapp copy
        # a band said to be measured links the measurement
        if e["basis"] == "measured":
            assert out["refs"], key
    for rid, r in BE.REFS.items():
        assert r["url"].startswith("https://"), rid
        assert r["cite"] and r["title"], rid


def test_the_endpoint_serves_the_catalog():
    from app.main import app

    j = TestClient(app).get("/bands/evidence.json").json()
    hoods = j["bike"]["road_hoods"]["trunk_angle_avg"]
    assert (hoods["lo"], hoods["hi"]) == (32, 50)
    assert hoods["basis_label"] == "Measured in studies"
    assert any("Skovereng" in r["cite"] for r in hoods["refs"])
    assert j["run"]["_"]["cadence_spm"]["article"] == "/academy/running-cadence"


def test_the_spa_puts_the_question_mark_on_the_range():
    s = SPA.read_text(encoding="utf-8")
    assert "/bands/evidence.json" in s
    assert 'aria-label="Why this range?"' in s
    # an entry shows only for the band it was written for
    assert re.search(r"if\(!e\|\|e\.lo!==lo\) return null;", s)
    assert "noopener" in s.split("function evRow", 1)[1][:900]
