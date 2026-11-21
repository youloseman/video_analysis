"""A session renews itself on every visit, and signing in loads what a page
load loads (2026-11-21: a coach was signed out a month after the last sign-in
and, signed in again, saw an empty Squad until a reload)."""
from __future__ import annotations

import re
from pathlib import Path

from app.api import auth as auth_api
from app.core.security import _decode_uid
from app.models.user import TIER_FULL

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


async def test_me_hands_back_a_fresh_token_for_the_same_person(db, make_user):
    user = await make_user(tier=TIER_FULL)
    out = await auth_api.me(user, db)
    assert out.token and _decode_uid(out.token) == user.id


def _fn(name: str) -> str:
    m = re.search(r"(?:async )?function " + name + r"\(.*?\n}\n", SPA, re.S)
    assert m, f"{name} is gone"
    return m.group(0)


def test_the_spa_keeps_the_renewed_token_and_never_holds_it_in_auth():
    body = _fn("checkAuth")
    assert "if(auth.token){ try{ localStorage.setItem(TKEY,auth.token); }catch(_e){} delete auth.token; }" in body


def test_signing_in_loads_the_roster_like_a_page_load():
    loader = _fn("loadSignedIn")
    assert "await loadAthletes();" in loader and "loadCoach();" in loader and "loadAllLinked()" in loader
    assert "await loadSignedIn();" in _fn("checkAuth")
    assert "renderAuth(); closeAuth(); loadSignedIn(); renderPromoBanner();" in SPA
