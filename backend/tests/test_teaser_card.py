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
    assert ": teaserCardHtml(locked, sellable?unlockId:null, items, offer, delta, ctx);" in SPA
    # The preview has its own card (step 5), with the same title as before.
    assert "? previewCardHtml(locked, offer, ctx)" in SPA
    assert "<h3>That was one report. The plan is what changes the next one.</h3>" in SPA


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


# Step 5 (2026-10-28): the preview card counts what sits behind the blur and
# carries the same plan box, toggle included, as the teaser card.

def test_the_preview_card_counts_off_lock_summary_and_never_lists_what_it_showed():
    assert "function previewCardHtml(locked, offer, ctx){" in SPA
    assert "const PREVIEW_UNLOCKS=['plan','fit','angles','ranges','kinogram','video','export'];" in SPA
    body = SPA.split("function previewCardHtml(locked, offer, ctx){", 1)[1].split("\n}\n", 1)[0]
    assert "ls.adjustments" in body and "ls.drills" in body and "ls.angles" in body
    assert "has('coaching')" not in body and "has('issues')" not in body
    assert "startCheckout('unlock'" not in body  # the preview is not for sale a second time
    assert "function previewBody(" not in SPA


def test_both_cards_draw_the_one_plan_box():
    assert "function planOfferHtml(offer, primary, src, lines){" in SPA
    assert SPA.count("planOfferHtml(offer, true, 'preview'") == 1
    assert SPA.count("planOfferHtml(offer, !unlockId, 'teaser'") == 1
    assert "data-src=\"${esc(src)}\"" in SPA
    # The toggle redraws whichever card is on screen; both branches remember it.
    assert "state.teaserCard={containerId:containerId, locked:locked, unlockId:unlockId, ctx:ctx};" in SPA
    assert "const offer = auth ? planOffer() : null;" in SPA
    assert ".tc-offers.tc-one{grid-template-columns:1fr}" in CSS


def test_the_two_offers_are_priced_in_the_same_unit():
    # Artur, 2026-10-28, off the restyled card: "$4" and "$9 /mo" read as the
    # same thing for the price of a coffee. The $4 says it opens one report;
    # the plan's first line says how many a month, off the tier table the
    # server sends with the card (never a number typed into the SPA).
    assert "<small>one report</small>" in SPA and ".tc-price.tc-per{" in CSS
    assert "function planLines(offer){" in SPA
    # Artur, 2026-10-29: "30 reports" is the value against the $4 for one, so
    # it is under the price and first in the list, with the price of one report.
    assert "${q.limit} full reports a ${q.window}${each>0?`, ${each}¢ each`:''}" in SPA
    assert "const each = perMonth>0 ? Math.round(perMonth/q.limit) : 0;" in SPA
    assert "' · '+esc(String(offer.quota.limit))+' reports'" in SPA
    assert "quota:(card&&card.quota)||null," in SPA
    assert SPA.count("planLines(offer)") == 3  # the definition and both cards
