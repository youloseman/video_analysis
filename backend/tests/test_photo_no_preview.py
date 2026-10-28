"""A photo never spends the free preview, and the photo report sells the clip.

Paywall read of 2026-10-28 (PostHog, the first 183 accounts to meet a
paywall): 95 had read a photo first and 2 of them paid; 88 had read a clip
first and 14 paid; nobody who only ever read photos paid. The photo used to
claim the account's one full report -- a still our own banner calls
approximate -- and its card offered a plan and nothing else. Now:

* a photo is always the score-and-skeleton teaser on a free account
  (``main.photo_access``), the preview stays for the first clip;
* a ``photo:`` claim left behind by the old rule does not count as spent;
* the photo result says whether that first clip would still open in full
  (``locked.preview_waiting``), which is the photo card's main offer;
* the SPA renders the photo report's own card and a loud two-line banner.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.user import TIER_ENTHUSIAST, TIER_STARTER

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
CSS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.css"


def _user(tier: str = TIER_STARTER, claimed: str | None = None):
    return SimpleNamespace(id=1, tier=tier, free_preview_job_id=claimed)


def test_a_photo_is_the_teaser_on_a_free_account_and_full_on_a_plan():
    assert main.photo_access(_user(TIER_STARTER)) == main.ACCESS_TEASER
    assert main.photo_access(_user(TIER_ENTHUSIAST)) == main.ACCESS_FULL


def test_a_photo_claim_from_the_old_rule_hands_the_preview_back():
    """95 accounts spent their one full report on a still. Their first clip
    is the report the preview exists to show, so the claim does not count."""
    assert main.preview_available(_user(claimed="photo:1700000000000")) is True
    # A real job claim is still read the old way: gone from the store = spent.
    assert main.preview_available(_user(claimed="job-long-gone")) is False
    assert main.preview_available(_user(claimed=None)) is True


def test_the_photo_route_never_claims_the_preview():
    """The route gates through photo_access and writes nothing to the claim."""
    import inspect

    src = inspect.getsource(main.analyze_photo_endpoint)
    assert "photo_access(user)" in src
    assert "free_preview_job_id" not in src
    assert "PREVIEW_CLAIMED" not in src
    assert 'result["locked"]["preview_waiting"]' in src


def test_the_spa_renders_the_photo_card_and_the_loud_banner():
    spa = SPA.read_text(encoding="utf-8")
    assert "function renderPhotoUpsell(" in spa
    assert "renderPhotoUpsell('pUpsellHost', res.locked" in spa
    assert "renderUpsell('pUpsellHost'" not in spa
    # A saved photo in the cabinet gets the same card, never the $4 one.
    assert "if(e.kind==='photo'){ renderPhotoUpsell('hdUpsellHost'" in spa
    # The banner, on every photo report whatever the tier, in two lines.
    assert "<b>A photo is a rough read.</b> One frame, approximate angles." in spa
    assert "switchToVideo('banner')" in spa
    assert "card: 'c_photo'" in spa
    assert ".banner.ph-limit{" in CSS.read_text(encoding="utf-8")
