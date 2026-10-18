"""Promo codes: free full reports, once per account, until the end date."""
from __future__ import annotations

from datetime import date

import pytest

from app.services import promo

RAW = "RIDE10:10:2026-11-18, OLD5:5:2026-01-01, bad entry, x:1:2026-12-01"


def test_codes_parse_from_the_variable_and_skip_junk():
    c = promo.codes(RAW)
    assert c["RIDE10"] == {"credits": 10, "until": date(2026, 11, 18)}
    assert "OLD5" in c and "X" not in c and len(c) == 2


def test_a_code_works_until_its_day_once_per_account_any_case():
    assert promo.check(" ride10 ", already=None, today=date(2026, 11, 18), raw=RAW) == (10, "RIDE10")
    with pytest.raises(ValueError, match="expired"):
        promo.check("RIDE10", already=None, today=date(2026, 11, 19), raw=RAW)
    with pytest.raises(ValueError, match="already used"):
        promo.check("RIDE10", already="OLD5", today=date(2026, 10, 20), raw=RAW)
    with pytest.raises(ValueError, match="doesn't exist"):
        promo.check("NOPE", already=None, today=date(2026, 10, 20), raw=RAW)


async def test_redeeming_adds_the_reports_and_a_second_try_is_refused(db, make_user, monkeypatch):
    from fastapi import HTTPException

    from app.api.me import PromoIn, redeem_promo

    monkeypatch.setenv("PROMO_CODES", "RIDE10:10:2099-01-01")
    user = await make_user()
    out = await redeem_promo(PromoIn(code="ride10"), user, db)
    assert out == {"code": "RIDE10", "added": 10, "unlock_credits": 10}
    await db.refresh(user)
    assert user.unlock_credits == 10 and user.promo_code == "RIDE10"
    with pytest.raises(HTTPException) as e:
        await redeem_promo(PromoIn(code="RIDE10"), user, db)
    assert e.value.status_code == 400 and user.unlock_credits == 10
