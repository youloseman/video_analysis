"""Outbound email.

Two things must hold no matter what: the message a provider receives is the one
we meant to send, and a mail outage never fails the thing that triggered it. A
review that is written, paid for and complete must reach the customer's account
even when the notification does not.
"""

from __future__ import annotations

import dataclasses
import json
import smtplib
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import notify


@pytest.fixture
def resend(monkeypatch):
    """Configure the Resend provider and capture the request it would make."""
    cfg = dataclasses.replace(
        settings, email_provider="resend",
        email_api_key="re_test_key", email_from="Flapp <reviews@flapp.test>",
    )
    monkeypatch.setattr(notify, "settings", cfg)
    sent = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["method"] = req.get_method()
        sent["headers"] = {k.lower(): v for k, v in req.headers.items()}
        sent["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    return sent


@pytest.fixture
def smtp(monkeypatch):
    """Configure the SMTP provider and capture the message handed to it."""
    cfg = dataclasses.replace(
        settings, email_provider="smtp", smtp_host="mail.flapp.test",
        smtp_port=587, smtp_user="u", smtp_password="p",
        email_from="reviews@flapp.test",
    )
    monkeypatch.setattr(notify, "settings", cfg)
    calls = SimpleNamespace(started_tls=False, logged_in=False, message=None)

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            calls.host, calls.port = host, port
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context=None): calls.started_tls = True
        def login(self, u, p): calls.logged_in = True
        def send_message(self, msg): calls.message = msg

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    return calls


# --------------------------------------------------------------------------
# Configuration posture
# --------------------------------------------------------------------------
def test_no_provider_means_no_send_and_no_exception():
    """The current production deploy runs without mail. That must be a quiet
    no-op, not a 500 on the reviewer's Send button."""
    assert notify.email_enabled() is False
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False


def test_a_provider_without_a_from_address_counts_as_unconfigured(monkeypatch):
    """Half-configured is the dangerous state: it looks live and silently is
    not. Treat it as off so the startup log says so."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        settings, email_provider="resend", email_api_key="re_x", email_from=None,
    ))
    assert notify.email_enabled() is False


def test_an_empty_recipient_is_refused(resend):
    assert notify.send_email("", "s", "t", "<p>h</p>") is False
    assert resend == {}


# --------------------------------------------------------------------------
# Resend
# --------------------------------------------------------------------------
def test_resend_posts_the_message_we_meant_to_send(resend):
    ok = notify.send_email(
        "athlete@example.com", "Your review is ready", "plain", "<p>rich</p>",
    )
    assert ok is True
    assert resend["method"] == "POST"
    assert resend["url"] == notify.RESEND_ENDPOINT
    assert resend["headers"]["authorization"] == "Bearer re_test_key"
    assert resend["body"] == {
        "from": "Flapp <reviews@flapp.test>",
        "to": ["athlete@example.com"],
        "subject": "Your review is ready",
        "text": "plain",
        "html": "<p>rich</p>",
    }


def test_resend_identifies_itself_so_the_edge_does_not_block_it(resend):
    """Resend is behind Cloudflare, which answers the stdlib's default
    "Python-urllib/3.x" with a 1010 bot challenge — never reaching the API, and
    reading in the log like a bad token. Caught by testing against the real
    endpoint; this keeps it caught."""
    notify.send_email("a@b.c", "s", "t", "<p>h</p>")
    ua = resend["headers"]["user-agent"]
    assert ua == notify.USER_AGENT
    assert "urllib" not in ua.lower()


def test_resend_carries_reply_to_when_one_is_configured(resend, monkeypatch):
    """A review notification invites a reply; it has to reach a human."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        settings, email_provider="resend", email_api_key="re_test_key",
        email_from="Flapp <reviews@flapp.test>", email_reply_to="artur@flapp.test",
    ))
    notify.send_email("a@b.c", "s", "t", "<p>h</p>")
    assert resend["body"]["reply_to"] == "artur@flapp.test"


def test_resend_omits_reply_to_when_unset(resend):
    notify.send_email("a@b.c", "s", "t", "<p>h</p>")
    assert "reply_to" not in resend["body"]


def test_a_rejected_send_surfaces_the_providers_own_reason(resend, monkeypatch):
    """urlopen RAISES on 4xx, and the useful part is the body. "HTTP Error 403"
    alone costs an hour; "The domain is not verified" does not."""
    import io

    def refuse(req, timeout=None):
        raise notify.urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {},
            io.BytesIO(b'{"message":"The flapp.test domain is not verified."}'),
        )
    monkeypatch.setattr(notify.urllib.request, "urlopen", refuse)

    captured = {}
    monkeypatch.setattr(notify.logger, "warning",
                        lambda evt, **kw: captured.update(kw))
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False
    assert "not verified" in captured["err"]
    assert "403" in captured["err"]


def test_a_resend_outage_is_reported_not_raised(resend, monkeypatch):
    def boom(req, timeout=None):
        raise notify.urllib.error.URLError("connection refused")
    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False


def test_a_rejected_resend_request_is_not_reported_as_sent(resend, monkeypatch):
    class _Bad:
        status = 422
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda req, timeout=None: _Bad())
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False


# --------------------------------------------------------------------------
# SMTP
# --------------------------------------------------------------------------
def test_smtp_sends_over_starttls_with_both_bodies(smtp):
    ok = notify.send_email(
        "athlete@example.com", "Your review is ready", "plain", "<p>rich</p>",
    )
    assert ok is True
    assert (smtp.host, smtp.port) == ("mail.flapp.test", 587)
    assert smtp.started_tls is True, "credentials must not cross the wire in clear"
    assert smtp.logged_in is True
    msg = smtp.message
    assert msg["To"] == "athlete@example.com"
    assert msg["Subject"] == "Your review is ready"
    # Multipart/alternative: plain text for clients that refuse HTML.
    types = {p.get_content_type() for p in msg.walk()}
    assert {"text/plain", "text/html"} <= types


def test_smtp_carries_reply_to_when_one_is_configured(smtp, monkeypatch):
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        settings, email_provider="smtp", smtp_host="mail.flapp.test",
        email_from="reviews@flapp.test", email_reply_to="artur@flapp.test",
    ))
    notify.send_email("a@b.c", "s", "t", "<p>h</p>")
    assert smtp.message["Reply-To"] == "artur@flapp.test"


def test_smtp_without_credentials_still_sends(smtp, monkeypatch):
    """Some relays authenticate by IP; a missing user must not mean a login
    attempt with an empty password."""
    monkeypatch.setattr(notify, "settings", dataclasses.replace(
        settings, email_provider="smtp", smtp_host="mail.flapp.test",
        email_from="reviews@flapp.test", smtp_user=None,
    ))
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is True
    assert smtp.logged_in is False


def test_an_smtp_outage_is_reported_not_raised(smtp, monkeypatch):
    class Dead:
        def __init__(self, *a, **k): raise smtplib.SMTPConnectError(421, "down")
    monkeypatch.setattr(notify.smtplib, "SMTP", Dead)
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False


def test_a_dns_failure_is_reported_not_raised(smtp, monkeypatch):
    class Dead:
        def __init__(self, *a, **k): raise OSError("name resolution failed")
    monkeypatch.setattr(notify.smtplib, "SMTP", Dead)
    assert notify.send_email("a@b.c", "s", "t", "<p>h</p>") is False
