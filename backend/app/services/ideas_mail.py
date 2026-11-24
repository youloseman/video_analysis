"""Mail for feature ideas: one to the owner when an idea arrives, one to its
author when it ships.

The second mail is the point of the whole channel. A form that swallows ideas
is worse than no form; "you asked for this, it is in Flapp now" is the reply
that makes a paying rider feel heard, and it costs one click in the inbox.
"""

from __future__ import annotations

from app.core.config import settings
from app.services import mail_layout as M
from app.services.notify import base_url, send_email

_SNIPPET = 90


def _snippet(text: str) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= _SNIPPET else t[: _SNIPPET - 1].rstrip() + "…"


def idea_admin_email(idea: dict, admin_url: str) -> tuple[str, str, str]:
    """(subject, text, html) for the owner: one new idea."""
    who = idea.get("email") or "anonymous"
    tier = idea.get("tier") or "guest"
    subject = f"Flapp idea: {_snippet(idea.get('message') or '')}"
    text = (f"{idea.get('message') or ''}\n\nFrom: {who} ({tier})\n\n"
            f"Inbox: {admin_url}\n")
    body = (M.para(idea.get("message") or "")
            + M.para(f"From {who} · {tier}", muted=True, size=14)
            + M.button(admin_url, "Open the inbox"))
    return subject, text, M.page(title="New feature idea", body=body, base=base_url())


def notify_idea(idea: dict, admin_url: str) -> bool:
    """Tell the owner. Blocking: run it off the event loop."""
    to = settings.admin_email
    if not to:
        return False
    subject, text, html = idea_admin_email(idea, admin_url)
    return send_email(to, subject, text, html)


def idea_shipped_email(message: str, note: str | None) -> tuple[str, str, str]:
    """(subject, text, html) for the rider whose idea shipped."""
    url = f"{base_url()}/app#whatsnew"
    subject = "You asked for it: it is in Flapp now"
    lead = "A while ago you sent us an idea for Flapp:"
    done = "It is built and live. Thank you for taking the time to write it down."
    note = (note or "").strip()
    text = (f"{lead}\n\n\"{message}\"\n\n{done}\n\n"
            + (f"{note}\n\n" if note else "")
            + f"See what changed: {url}\n\nArtur, Flapp\n")
    body = (M.para(lead)
            + M.callout("Your idea", message)
            + M.para(done)
            + (M.para(note) if note else "")
            + M.button(url, "See what changed")
            + M.para("Artur, Flapp", muted=True, size=14))
    return subject, text, M.page(title="Your idea is in Flapp", body=body,
                                 preheader=_snippet(message), base=base_url())


def send_idea_shipped(to: str, message: str, note: str | None) -> bool:
    """Blocking: run it off the event loop."""
    subject, text, html = idea_shipped_email(message, note)
    return send_email(to, subject, text, html)
