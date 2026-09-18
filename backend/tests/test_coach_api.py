"""The coach profile (``/me/coach``) and the note in the export.

Pinned: the profile is Full/admin only and says so; it round-trips and is
cleaned (whitespace, duplicate cues, the cue cap); a logo that is not an
image or too large is refused; and a note saved on a history entry reaches
the Markdown export under the coach's name.
"""
from __future__ import annotations

import base64

import pytest
from fastapi import HTTPException

from app.api import coach as api
from app.models.user import TIER_ENTHUSIAST, TIER_FULL
from app.services.export import ai_export


async def test_only_the_full_plan_has_a_coach_profile(make_user, db):
    starter = await make_user()
    got = await api.get_coach(starter, db)
    assert got["allowed"] is False and got["name"] == ""
    with pytest.raises(HTTPException) as e:
        await api.put_coach(api.CoachIn(name="X"), starter, db)
    assert e.value.status_code == 402
    enth = await make_user(tier=TIER_ENTHUSIAST)
    with pytest.raises(HTTPException):
        await api.put_coach(api.CoachIn(name="X"), enth, db)


async def test_the_profile_round_trips_and_is_cleaned(make_user, db):
    user = await make_user(tier=TIER_FULL)
    body = api.CoachIn(
        name="  Coach   Artur ", tagline="UESCA  · Nanaimo",
        cues=["Land under the hips", "land under the hips", "  ", "Quiet arms"],
    )
    saved = await api.put_coach(body, user, db)
    assert saved["name"] == "Coach Artur" and saved["tagline"] == "UESCA · Nanaimo"
    assert saved["cues"] == ["Land under the hips", "Quiet arms"]
    again = await api.get_coach(user, db)
    assert again["allowed"] is True and again["cues"] == saved["cues"]
    # Update replaces; nothing is merged from the old row.
    saved2 = await api.put_coach(api.CoachIn(name="A", cues=[]), user, db)
    assert saved2["cues"] == [] and saved2["tagline"] == ""


async def test_a_mark_must_be_a_small_image(make_user, db):
    user = await make_user(tier=TIER_FULL)
    with pytest.raises(HTTPException) as e:
        await api.put_coach(api.CoachIn(logo="data:text/html;base64,PGI+"), user, db)
    assert e.value.status_code == 422
    big = "data:image/png;base64," + base64.b64encode(b"\x89PNG" + b"0" * (250 * 1024)).decode()
    with pytest.raises(HTTPException) as e:
        await api.put_coach(api.CoachIn(logo=big), user, db)
    assert e.value.status_code == 413
    ok = "data:image/png;base64," + base64.b64encode(b"\x89PNG tiny").decode()
    saved = await api.put_coach(api.CoachIn(logo=ok), user, db)
    assert saved["logo"] == ok


def test_the_export_carries_the_coachs_note_under_their_name():
    result = {"sport_type": "run", "technique_score": 80, "letter_grade": "B",
              "sport_specific_metrics": {}, "angle_statistics": {}}
    md = ai_export.build_markdown(result, coach_notes="Land softer.\nSee you Tuesday.", coach_name="Coach Artur")
    assert "## The athlete's own coach's notes (Coach Artur)" in md
    assert "Land softer." in md and "See you Tuesday." in md
    assert "coach's notes" not in ai_export.build_markdown(result)
