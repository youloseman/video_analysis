"""Every mail a customer gets is drawn by app/services/mail_layout.py: the
app's ground, a white card, the title in the brand face, one coral button,
the same foot. No em dash in our copy, and none of the old blue."""

import inspect
from types import SimpleNamespace

import pytest

from app.api import auth, coach_link, coach_reports
from app.services import expert_review as er
from app.services import guest, mail_layout, notify, plan_lapse


def _sent(monkeypatch):
    box = []
    monkeypatch.setattr(notify, "send_email", lambda to, subject, text, html: box.append((subject, text, html)) or True)
    return box


def _mails(monkeypatch):
    report = {"verdict": "Lower the saddle.", "first": "Raise the cockpit.", "reviewer": "B. Coach",
              "note_1": "A", "note_2": "B", "note_3": "C"}
    out = [
        notify.analysis_ready_email("bike", 83, "B", "https://x/app#job=a", "https://x/unsubscribe?u=1&t=z"),
        auth._reset_mail("https://x/app#reset=t"),
        er.notes_ready_email(report, "https://x/app#history=h1"),
        er.ready_email(report, "https://x/app#reviews", recheck="Book a re-check by 1 December."),
        guest.welcome_email("notes", "bike", "https://x/app#reset=t"),
        guest.welcome_email("expert", "swim", "https://x/app#reset=t"),
        plan_lapse.lapse_email("notice", "full", 3, 1795000000000),
        plan_lapse.lapse_email("reminder", "full", 1, 1795000000000),
    ]
    box = _sent(monkeypatch)
    coach_link._mail_note(SimpleNamespace(email="a@x"), "Coach K", SimpleNamespace(sport="bike", client_id="h1"))
    coach_reports._mail("a@x", "Sam", "Coach K", "run", "https://x/r/abc")
    return out + box


def test_every_customer_mail_is_drawn_by_the_layout(monkeypatch):
    mails = _mails(monkeypatch)
    assert len(mails) == 10
    for subject, text, html in mails:
        assert 'data-flapp-mail="1"' in html, subject
        assert mail_layout.CORAL in html, subject              # one filled button
        assert html.count(f"background:{mail_layout.CORAL}") == 1, subject
        assert "#2F6DE0" not in html, subject                  # the retired blue
        for part in (subject, text, html):
            assert "—" not in part, (subject, part[:200])


@pytest.mark.parametrize("fn", [notify.analysis_ready_email, auth._reset_mail, er.notes_ready_email,
                                er.ready_email, guest.welcome_email, plan_lapse.lapse_email,
                                coach_link._mail_note, coach_reports._mail])
def test_no_mail_lays_itself_out_by_hand(fn):
    """The owner's own sale mail is not a customer's and keeps its table."""
    src = inspect.getsource(fn)
    assert "#2F6DE0" not in src and "font-family:-apple-system" not in src
    assert "mail_layout" in src


def test_the_lane_mail_too():
    from app.services import coach_lane
    src = inspect.getsource(coach_lane._tell_the_coach)
    assert "m.page(" in src and "#2F6DE0" not in src


def test_text_keeps_its_quotes_and_attributes_are_escaped():
    html = mail_layout.page(title="It's ready", body=mail_layout.para("Don't <b>"),
                            foot=mail_layout.foot_link('https://x/?a=1&b="2"', "Stop"))
    assert "It's ready" in html and "Don't &lt;b&gt;" in html
    assert 'href="https://x/?a=1&amp;b=&quot;2&quot;"' in html
