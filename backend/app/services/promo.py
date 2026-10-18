"""Promo codes: free full reports for a group we invite.

One shape only, on purpose: a code adds unlock credits (each opens one full
report, see billing._redeem_unlock_credit) to an account, once per account,
and can be redeemed until its end date. Codes live in the PROMO_CODES
environment variable, ``CODE:credits:YYYY-MM-DD`` comma-separated, so a new
one is a variable change, not a deploy.

First use (Artur, 2026-10-18): his road-cycling WhatsApp group -- ten free
full reports, the code good for a month.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from typing import Any

_ITEM = re.compile(r"^\s*([A-Za-z0-9_-]{3,32})\s*:\s*(\d{1,3})\s*:\s*(\d{4}-\d{2}-\d{2})\s*$")


def codes(raw: str | None = None) -> dict[str, dict[str, Any]]:
    """The configured codes: ``{"RIDE10": {"credits": 10, "until": date}}``."""
    raw = os.environ.get("PROMO_CODES", "") if raw is None else raw
    out: dict[str, dict[str, Any]] = {}
    for item in raw.split(","):
        m = _ITEM.match(item)
        if m:
            out[m.group(1).upper()] = {"credits": int(m.group(2)),
                                       "until": date.fromisoformat(m.group(3))}
    return out


def check(code: str, *, already: str | None, today: date | None = None,
          raw: str | None = None) -> tuple[int, str]:
    """``(credits, CODE)`` this code gives this account, or ``ValueError``
    with a sentence for the athlete."""
    key = (code or "").strip().upper()
    found = codes(raw).get(key)
    if not found:
        raise ValueError("That code doesn't exist. Check the spelling.")
    today = today or datetime.now(timezone.utc).date()
    if today > found["until"]:
        raise ValueError("That code has expired.")
    if already:
        raise ValueError("This account has already used a promo code.")
    return found["credits"], key


__all__ = ["check", "codes"]
