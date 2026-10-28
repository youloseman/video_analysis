"""The teaser card (a locked video report, in the app and in History).

Paywall read of 2026-10-28: on the old card people bought the $4 unlock and
nothing else; the preview card, which names its counts and starts a plan
checkout from the card, took 23% to checkout against 6% for a card whose
only button opened the price list; yearly checkouts were abandoned (3 paid
of 25) while monthly closed 4 of 6. The card now names counts and one
finding (never a value), keeps the $4 as a first-class offer, and puts the
plan beside it with monthly pressed first. These pin the wiring; the
server's half is in test_result_gating (lock_summary).
"""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


def test_the_locked_video_report_renders_the_teaser_card_not_the_bullet_list():
    assert "function teaserCardHtml(" in SPA
    assert "host.innerHTML=teaserCardHtml(locked, sellable?unlockId:null, items, offer, delta, ctx);" in SPA
    # The preview keeps its own card (step 5 of the spec is separate).
    assert "title:'That was one report. The plan is what changes the next one.'" in SPA


def test_the_card_names_the_finding_and_blurs_the_number():
    assert "Biggest thing off in this clip:" in SPA
    assert 'class="tc-blur"' in SPA
    assert "const WORST_LABELS={" in SPA
    assert ".tc-blur{" in CSS and "color:transparent" in CSS


def test_both_offers_sit_on_the_card_with_monthly_pressed_first():
    assert "startCheckout('unlock'," in SPA
    assert "function planOffer(){" in SPA
    assert "function tcSetPeriod(per){" in SPA
    # The toggle starts from billPeriod, which the SPA opens on monthly.
    assert "let billPeriod='monthly';" in SPA
    assert "const yearly = billPeriod==='yearly' && offer.yearly;" in SPA


def test_the_fresh_report_and_the_cabinet_both_pass_their_context():
    assert "entryId, {sport:r.sport_type||state.sport, score:r.technique_score, entryId:entryId});" in SPA
    assert "lock_summary:e.lockSummary||null}" in SPA
    assert "if(res.locked.lock_summary) e.lockSummary=res.locked.lock_summary;" in SPA


def test_the_impression_says_which_card_it_was():
    assert "card: locked.reason==='preview' ? 'a_preview' : (containerId==='hdUpsellHost' ? 'e_history' : 'b_teaser')," in SPA
    assert "if(ctx.rerender) return true;" in SPA
