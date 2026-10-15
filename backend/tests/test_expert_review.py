"""The Expert Review deliverable: template, draft, and delivery.

Three things here are worth more than the rest:

* an order has to know WHICH analysis was bought, or fulfilling it means
  emailing the customer to ask — which is what this replaces;
* a draft must never be readable by the customer, because "delivered" is a
  promise the account view makes on the strength of it;
* the sample on the pricing page has to stay a valid report, or the one thing a
  buyer sees before paying drifts away from the thing they get.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.api.billing import _apply_event, _sport_of
from app.models.order import (
    ORDER_DELIVERED,
    ORDER_IN_REVIEW,
    ORDER_PAID,
    Order,
)
from app.services import expert_review as er


def expert_purchase(user_id, *, session_id="cs_expert_1", analysis="h1"):
    """A one-time Expert Review checkout, as Stripe reports it."""
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id),
            "customer": "cus_1",
            "mode": "payment",
            "amount_total": 2900,
            "currency": "usd",
            "metadata": {
                "user_id": str(user_id), "plan": "expert", "tier": "",
                "analysis_client_id": analysis,
            },
        }},
    }


# --------------------------------------------------------------------------
# The template
# --------------------------------------------------------------------------
def test_the_fit_table_is_cycling_only():
    """A runner has no saddle height; showing the section would be noise."""
    assert "fit" in [s["key"] for s in er.sections_for("bike")]
    assert "fit" not in [s["key"] for s in er.sections_for("run")]


def test_normalize_is_an_allowlist_over_the_section_list():
    dirty = er.blank_report("run") | {
        "verdict": "ok", "internal_cost": 12, "__proto__": "x",
    }
    clean = er.normalize_report(dirty, "run")
    assert clean["verdict"] == "ok"
    assert "internal_cost" not in clean
    assert "__proto__" not in clean


def test_a_runners_report_cannot_smuggle_in_fit_rows():
    clean = er.normalize_report({"fit": [{"part": "Saddle"}]}, "run")
    assert "fit" not in clean


def test_blank_rows_are_dropped_rather_than_stored():
    report = er.normalize_report({"priorities": [
        {"title": "Land closer", "why": "", "change": "", "check": ""},
        {"title": "", "why": "", "change": "", "check": ""},
    ]}, "run")
    assert len(report["priorities"]) == 1


def test_long_prose_is_capped():
    clean = er.normalize_report({"verdict": "x" * (er.MAX_PROSE_CHARS + 500)}, "run")
    assert len(clean["verdict"]) == er.MAX_PROSE_CHARS


def test_priorities_are_capped_because_everything_cannot_be_first():
    rows = [{"title": f"p{i}", "why": "w", "change": "c", "check": "k"} for i in range(9)]
    assert len(er.normalize_report({"priorities": rows}, "run")["priorities"]) == er.MAX_PRIORITIES


# --------------------------------------------------------------------------
# Prefill — the draft the reviewer opens on
# --------------------------------------------------------------------------
def test_capture_notes_repeat_the_analyzers_own_caveat():
    notes = er.capture_notes({"sport": "run", "gated": True, "confidence": "low"})
    assert "partial" in notes
    assert "low" in notes


def test_capture_notes_say_so_when_the_capture_was_clean():
    """Silence would read as 'not assessed'."""
    assert "clean capture" in er.capture_notes({"sport": "run", "gated": False, "confidence": "high"})


def test_an_entry_with_no_flags_is_not_reported_as_clean():
    """Analyses saved before the flags were persisted look identical to clean
    ones. Calling them clean would invent an all-clear nothing ever gave."""
    notes = er.capture_notes({"sport": "run"})
    assert "clean capture" not in notes
    assert "before capture-quality flags were recorded" in notes


def test_capture_notes_survive_the_clients_metrics_being_a_display_list():
    """``metrics`` on a stored entry is the angle-row list the history card
    renders, not a metrics dict — reading it as one used to 500 the editor."""
    entry = {"sport": "bike", "gated": True,
             "metrics": [{"label": "Right knee", "value": "104°"}]}
    assert "partial" in er.capture_notes(entry)


def test_the_coach_is_not_asked_to_judge_the_numbers():
    """Artur, 2026-10-14: the coach's eye leads; how far to trust a measurement
    is the product's statement (capture_notes), not a section to fill."""
    keys = [s["key"] for s in er.sections_for("bike", coach=True)]
    assert "trust" not in keys and "caveats" not in keys
    assert keys[:2] == ["verdict", "seen"]


def test_prefill_seeds_only_the_filming_setup():
    """Seeding judgement sections would invite shipping a report nobody wrote."""
    draft = er.prefill({"sport": "bike"})
    assert draft["retest"] == er.RETEST_DEFAULT["bike"]
    assert not draft["verdict"] and not draft["seen"] and not draft["priorities"]


# --------------------------------------------------------------------------
# The coach's internal notes to the product
# --------------------------------------------------------------------------
def test_the_disagreement_box_is_stored_but_never_reaches_the_athlete():
    clean = er.normalize_report({"verdict": "v", "disagree": "knee drawn on the chainring"}, "bike")
    assert clean["disagree"] == "knee drawn on the chainring"
    assert "disagree" not in er.public_report(clean)
    assert "disagree" not in [s["key"] for s in er.sections_for("bike")]
    assert "disagree" in [s["key"] for s in er.sections_for("bike", coach=True)]


def test_a_review_delivered_on_the_old_template_still_reads_whole():
    old = {"verdict": "v", "trust": "solid", "working": "w", "caveats": "one plane"}
    keys = [s["key"] for s in er.sections_for("run", report=old)]
    assert keys.index("trust") == keys.index("verdict") + 1
    assert keys[-1] == "caveats"
    # and re-saving it in the editor does not drop them
    assert er.normalize_report(old, "run")["trust"] == "solid"
    # a new review never grows them
    assert "trust" not in [s["key"] for s in er.sections_for("run", report={"verdict": "v"})]


def test_prefill_survives_an_order_with_no_analysis_attached():
    draft = er.prefill(None)
    assert er.missing_required(draft, "run")      # nothing written yet
    assert draft["version"] == er.REPORT_VERSION


# --------------------------------------------------------------------------
# The sample shown on the pricing page
# --------------------------------------------------------------------------
def test_the_sample_is_a_complete_report():
    """It is the only thing a buyer sees before paying — it cannot rot."""
    assert er.missing_required(er.SAMPLE_REPORT, er.SAMPLE_SPORT) == []


def test_the_sample_survives_normalisation_unchanged_in_substance():
    clean = er.normalize_report(er.SAMPLE_REPORT, er.SAMPLE_SPORT)
    assert clean["verdict"] == er.SAMPLE_REPORT["verdict"]
    assert len(clean["priorities"]) == len(er.SAMPLE_REPORT["priorities"])


def test_every_sample_priority_answers_all_four_questions():
    """A priority without a check is an opinion, not an instruction."""
    for p in er.SAMPLE_REPORT["priorities"]:
        for field in er.PRIORITY_FIELDS:
            assert p.get(field), f"sample priority missing {field}"


# --------------------------------------------------------------------------
# Which analysis was bought
# --------------------------------------------------------------------------
async def test_the_order_records_which_analysis_to_review(db, make_user):
    user = await make_user()
    await _apply_event(db, expert_purchase(user.id, analysis="h1785607210387"))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.analysis_client_id == "h1785607210387"
    assert order.status == ORDER_PAID


async def test_an_order_with_no_analysis_is_still_recorded(db, make_user):
    """Losing the link is bad; losing the paid order is worse."""
    user = await make_user()
    await _apply_event(db, expert_purchase(user.id, analysis=""))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.analysis_client_id is None


async def test_a_new_order_starts_with_no_report(db, make_user):
    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    assert er.is_empty(order.report)
    assert order.delivered_at_ms is None


# --------------------------------------------------------------------------
# Publishing
# --------------------------------------------------------------------------
def test_an_incomplete_report_names_what_is_missing():
    half = er.normalize_report({"verdict": "written", "working": "written"}, "run")
    missing = er.missing_required(half, "run")
    assert "What I see" in missing
    assert "The verdict" not in missing


def test_a_complete_report_has_nothing_missing():
    assert er.missing_required(er.SAMPLE_REPORT, "run") == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [({"fit": [{"part": "Saddle height"}]}, "bike"), ({"verdict": "x"}, "run")],
)
def test_the_sport_is_recoverable_from_a_stored_report(report, expected):
    """Stored reports have to stay renderable without a second source of truth."""
    assert _sport_of(report) == expected


# --------------------------------------------------------------------------
# The customer's side of the wall
# --------------------------------------------------------------------------
async def test_a_draft_is_not_visible_as_a_delivered_report(db, make_user):
    """`has_report` drives the "Read your review" link in the account view."""
    from app.api.billing import _order_out

    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    order.report = er.normalize_report({"verdict": "half a thought"}, "run")
    order.status = ORDER_IN_REVIEW
    await db.commit()
    assert _order_out(order)["has_report"] is False

    order.status = ORDER_DELIVERED
    await db.commit()
    assert _order_out(order)["has_report"] is True


# --------------------------------------------------------------------------
# Telling the athlete
#
# Delivering a review nobody is told about is a paid product found by accident.
# Two channels: an email, and a card on the dashboard that keeps shouting until
# it is opened (an email can be missed, and Pricing is where you go to BUY).
# --------------------------------------------------------------------------
def test_the_email_leads_with_the_verdict_not_a_notification():
    """The one paragraph the reviewer wrote as the takeaway is what earns the
    click. A bare "you have an update" does not."""
    subject, text, html = er.ready_email(er.SAMPLE_REPORT, "https://x.test/app")
    assert "Your cadence is not the problem" in text
    assert "Your cadence is not the problem" in html
    assert "review" in subject.lower()


def test_the_email_links_to_the_app_rather_than_pasting_the_report():
    """The plan and the re-test only make sense together, in the app — and a
    review pasted into an inbox is one nobody re-reads in week three."""
    _s, text, html = er.ready_email(er.SAMPLE_REPORT, "https://x.test/app")
    assert "https://x.test/app" in text and "https://x.test/app" in html
    # The four-week plan is NOT in the mail.
    assert "calf work twice a week" not in text


def test_the_email_strips_markdown_the_mail_client_would_show_raw():
    report = {"verdict": "Your **saddle** is the *whole* problem.", "reviewer": "A"}
    _s, text, _h = er.ready_email(report, "https://x.test/app")
    assert "**" not in text and "*" not in text


def test_the_email_escapes_the_reviewers_prose_into_html():
    report = {"verdict": "Knee <5 deg & drifting", "reviewer": "A"}
    _s, _t, html = er.ready_email(report, "https://x.test/app")
    assert "&lt;5 deg &amp; drifting" in html


def test_a_long_verdict_is_trimmed_for_the_inbox():
    report = {"verdict": "word " * 400, "reviewer": "A"}
    _s, text, _h = er.ready_email(report, "https://x.test/app")
    assert "…" in text


def test_email_is_off_until_a_provider_is_configured():
    """Same posture as Stripe and Gemini: unset means degrade, never crash."""
    from app.services import notify

    assert notify.email_enabled() is False
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False


async def test_a_delivered_review_starts_unread(db, make_user):
    from app.api.billing import _order_out

    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    order.report = er.SAMPLE_REPORT
    order.status = ORDER_DELIVERED
    await db.commit()
    assert _order_out(order)["unread"] is True

    order.read_at_ms = 123
    await db.commit()
    assert _order_out(order)["unread"] is False


async def test_the_athletes_copy_leaves_the_coachs_disagreement_out(db, make_user):
    from app.api.billing import my_report

    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    order.report = er.SAMPLE_REPORT | {"disagree": "the app drew the far knee"}
    order.status = ORDER_DELIVERED
    await db.commit()
    out = await my_report(order.id, user=user, db=db)
    assert "disagree" not in out["report"]
    assert "disagree" not in [s["key"] for s in out["sections"]]
    assert out["report"]["seen"] == er.SAMPLE_REPORT["seen"]


async def test_an_undelivered_order_is_never_unread(db, make_user):
    """`unread` drives a loud card. It must not fire on a draft."""
    from app.api.billing import _order_out

    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    order.report = er.SAMPLE_REPORT
    order.status = ORDER_IN_REVIEW
    await db.commit()
    assert _order_out(order)["unread"] is False


async def test_a_delivered_order_with_no_report_still_claims_none(db, make_user):
    """Orders delivered by email before this existed must not offer a dead link."""
    from app.api.billing import _order_out

    user = await make_user()
    await _apply_event(db, expert_purchase(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    order.status = ORDER_DELIVERED
    await db.commit()
    assert _order_out(order)["has_report"] is False


# --------------------------------------------------------------------------
# The athlete's own pictures in the delivered review (2026-10-15): a $39
# review arrived as text only, while a kinogram with the leg straight at the
# bottom says "saddle too high" before a word is read.
# --------------------------------------------------------------------------
KF = "data:image/jpeg;base64,AAAA"
KINO = "data:image/jpeg;base64,BBBB"


def test_the_review_carries_the_frame_and_the_kinogram():
    figs = er.review_figures({"keyframe": KF, "kinogram": KINO}, "bike")
    assert [f["kind"] for f in figs] == ["kinogram", "keyframe"]
    assert "pedal revolution" in figs[0]["caption"]
    assert "stride" in er.review_figures({"kinogram": KINO}, "run")[0]["caption"]
    assert er.review_figures({"keyframe": None, "kinogram": "http://x/y.jpg"}, "bike") == []


async def test_the_delivered_report_endpoint_serves_them(db, make_user):
    from app.api.billing import my_report
    from app.models.analysis import Analysis

    user = await make_user()
    db.add(Analysis(user_id=user.id, client_id="h1", created_at_ms=1, sport="bike", kind="video",
                    data={"sport": "bike", "kind": "video"},
                    result={"keyframe_base64": KF, "kinogram_base64": KINO}))
    await db.commit()
    await _apply_event(db, expert_purchase(user.id, analysis="h1"))
    order = (await db.execute(select(Order))).scalar_one()
    order.report = er.normalize_report({"verdict": "Lower the saddle."}, "bike")
    order.status = ORDER_DELIVERED
    await db.commit()
    out = await my_report(order.id, user, db)
    assert [f["src"] for f in out["figures"]] == [KINO, KF]


def test_the_page_draws_them_after_the_verdict_everywhere():
    from pathlib import Path
    spa = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "function renderReportHtml(report, sections, meta, figures, capture)" in spa
    assert "s.key==='verdict'" in spa
    # the athlete's copy, the sample, and the reviewer's preview
    assert spa.count("j.report), j.figures") == 2
    # internal sections never render on the athlete's page or in the preview
    assert "filter(s=>!s.internal)" in spa
    assert "{src:d.media.kinogram, kind:'kinogram'" in spa
