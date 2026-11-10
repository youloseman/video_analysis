"""The one gate between an athlete's account and aero research.

Hard rule (Artur, 2026-11-08): a user's clips and measured CdA are not used
for R&D without the athlete's own, separate, revocable yes
(``users.research_opt_in``). Anything that takes data out for research -- a
script pairing clips with measured CdA to calibrate the trunk-angle estimate,
a benchmark set -- selects through ``research_eligible`` and nothing else.

Three things follow from whose consent it is:

* the switch is the OWNER's: a coach's measurement on a linked athlete's setup
  goes with the athlete's answer, not the coach's;
* an athlete on a coach's roster with no account of their own (model B) has
  nobody who can say yes, so their data is never eligible -- the coach cannot
  consent for them;
* test, staff and admin accounts are not athletes and are left out.

The swim clip's per-clip box (services/swim.py ``consent_training``) is a
different consent for a different purpose and is not read here.
"""
from __future__ import annotations

from typing import Any

from app.models.user import TIER_ADMIN
from app.services.newsletter import is_test_address


def research_eligible(owner: Any) -> bool:
    """Whether this account's clips and CdA rows may be used for research."""
    return (
        owner is not None
        and bool(getattr(owner, "research_opt_in", False))
        and getattr(owner, "tier", None) != TIER_ADMIN
        and not is_test_address(getattr(owner, "email", None))
    )


def profile_research_eligible(profile: Any, owner: Any) -> bool:
    """A setup's data: its owner's yes, and never a roster athlete's setup kept
    on the coach's account (``athlete_id`` set) -- that athlete never had the
    switch to turn on."""
    if profile is None or getattr(profile, "athlete_id", None) is not None:
        return False
    return research_eligible(owner)
