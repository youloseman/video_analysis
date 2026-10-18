"""Outbound email. Optional at runtime, like every other integration here.

A delivered Expert Review that nobody is told about is a paid product the
customer finds by accident. In-app placement helps, but it only reaches someone
who happens to open the app -- so this exists.

Two providers, because a solo product should not have to pick a vendor to get
started and should not be stuck with one either:

* ``resend``  -- one HTTPS POST, no SDK. The default recommendation.
* ``smtp``    -- the universal escape hatch (Fastmail, Gmail app password, SES
  SMTP, anything). Port 587 + STARTTLS; port 25 is blocked on most hosts.

With neither configured the message is written to the log at INFO and the caller
is told it did not send. That is deliberate: local development and the current
production deploy both run without mail, and a missing key must never turn
"review delivered" into a 500 for the reviewer.

Sending is BLOCKING (a socket, or an HTTPS round trip). Every caller must hand
it to a threadpool -- this service runs a single worker, so an inline send would
stall every other request for its duration.
"""

from __future__ import annotations

import json
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage

import structlog

from app.core.config import settings

logger = structlog.get_logger()

RESEND_ENDPOINT = "https://api.resend.com/emails"
TIMEOUT_S = 15
# Sent on every API call. Not cosmetic: Resend is behind Cloudflare, and the
# stdlib default ("Python-urllib/3.11") is met with a 1010 bot challenge instead
# of the API -- which reads in the log as a permission problem and is not one.
USER_AGENT = "Flapp/1.0 (+https://getflapp.com)"


def email_enabled() -> bool:
    """Whether a real message can actually leave the building."""
    if settings.email_provider == "resend":
        return bool(settings.email_api_key and settings.email_from)
    if settings.email_provider == "smtp":
        return bool(settings.smtp_host and settings.email_from)
    return False


def log_email_configuration() -> None:
    """Say at startup whether anyone will ever be told their review is ready.

    Silent by construction otherwise: the reviewer clicks Send, the order flips
    to delivered, and the absence of an email looks exactly like a working one.
    """
    if email_enabled():
        logger.info(
            "EMAIL_READY", provider=settings.email_provider, sender=settings.email_from,
        )
    else:
        logger.warning(
            "EMAIL_DISABLED",
            provider=settings.email_provider or "none",
            detail="delivered reviews will not be announced; the athlete only "
                   "finds out by opening the app",
        )


def _send_resend(to: str, subject: str, text: str, html: str) -> None:
    body: dict[str, object] = {
        "from": settings.email_from,     # Resend takes "Name <a@b.com>" as a string
        "to": [to],
        "subject": subject,
        "text": text,
        "html": html,
    }
    if settings.email_reply_to:
        body["reply_to"] = settings.email_reply_to
    req = urllib.request.Request(
        RESEND_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.email_api_key}",
            "Content-Type": "application/json",
            # Resend sits behind Cloudflare, which rejects the default
            # "Python-urllib/3.x" agent with a 1010 challenge before the request
            # ever reaches the API. Identify the client properly.
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    # stdlib on purpose: this is one POST, and the prod image has no HTTP client
    # of its own (httpx only arrives transitively via other SDKs).
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"resend returned {resp.status}")
    except urllib.error.HTTPError as e:
        # urlopen RAISES on 4xx/5xx rather than returning them, and the useful
        # part is in the body: "domain is not verified", "from is invalid".
        # Without this the log says "HTTP Error 403" and costs an hour.
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 -- diagnostics must not mask the error
            pass
        raise RuntimeError(f"resend returned {e.code}: {detail}") from e


def _send_smtp(to: str, subject: str, text: str, html: str) -> None:
    msg = EmailMessage()
    msg["From"] = settings.email_from
    msg["To"] = to
    msg["Subject"] = subject
    if settings.email_reply_to:
        msg["Reply-To"] = settings.email_reply_to
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=TIMEOUT_S) as s:
        s.starttls(context=ssl.create_default_context())
        if settings.smtp_user:
            s.login(settings.smtp_user, settings.smtp_password or "")
        s.send_message(msg)


def send_email(to: str, subject: str, text: str, html: str) -> bool:
    """Send one message. Returns whether it actually went.

    Never raises: this is called from the publish path, and a mail outage must
    not fail the delivery of a review that is otherwise complete and paid for.
    The customer can still read it in the app; the log carries the failure.
    """
    if not to:
        return False
    if not email_enabled():
        logger.info(
            "EMAIL_SKIPPED", to=to, subject=subject,
            reason="no provider configured",
        )
        return False
    try:
        if settings.email_provider == "resend":
            _send_resend(to, subject, text, html)
        else:
            _send_smtp(to, subject, text, html)
    except (urllib.error.URLError, OSError, smtplib.SMTPException, RuntimeError) as e:
        logger.warning(
            "EMAIL_FAILED", to=to, subject=subject,
            provider=settings.email_provider, err=str(e),
        )
        return False
    logger.info("EMAIL_SENT", to=to, subject=subject, provider=settings.email_provider)
    return True


# --------------------------------------------------------------------------
# "Your analysis is ready"
#
# Only ever sent to somebody who is NOT looking at the result -- the caller
# checks that first (main._maybe_notify_ready). An analysis takes 30-60
# seconds, which is long enough to put the phone down and short enough that a
# mail to somebody still watching would be pure noise.
#
# Every copy carries a working one-click unsubscribe. Not a preference buried
# in an account screen: the person receiving this may have signed up ten
# minutes ago, and the polite version of "we will email you" is "and here is
# how to stop".
# --------------------------------------------------------------------------
def _esc_html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def analysis_ready_email(
    sport: str, score: int | None, grade: str | None,
    report_url: str, unsubscribe_url: str,
) -> tuple[str, str, str]:
    """(subject, text, html) for a finished analysis nobody was watching."""
    noun = "ride" if sport == "bike" else "run"
    verdict = (
        f"{score}/100" + (f" ({grade})" if grade else "")
        if score is not None else "ready to read"
    )
    subject = f"Your {noun} analysis is ready — {verdict}"
    text = (
        f"Your {noun} analysis finished while you were away.\n\n"
        f"Technique score: {verdict}\n\n"
        f"Open the report — joint angles, what to fix first and the plan:\n"
        f"{report_url}\n\n"
        f"Don't want these? Turn them off in one click:\n{unsubscribe_url}\n"
    )
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        'font-size:16px;line-height:1.6;color:#14294B;max-width:520px">'
        '<p style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;'
        'color:#2F6DE0;margin:0 0 6px">Flapp</p>'
        f'<h1 style="font-size:22px;margin:0 0 8px">Your {_esc_html(noun)} '
        'analysis is ready</h1>'
        f'<p style="margin:0 0 22px;font-size:15px;color:#5A6478">Technique score '
        f'<b style="color:#14294B">{_esc_html(verdict)}</b></p>'
        f'<p style="margin:0 0 24px"><a href="{_esc_html(report_url)}" '
        'style="background:#2F6DE0;color:#fff;text-decoration:none;'
        'padding:12px 20px;border-radius:8px;display:inline-block;'
        'font-weight:600">Open the report</a></p>'
        '<p style="font-size:13px;color:#5A6478;margin:0 0 18px">Joint angles '
        'against reference ranges, what to fix first, and the plan.</p>'
        f'<p style="font-size:12px;color:#8A94A3;margin:0">'
        f'<a href="{_esc_html(unsubscribe_url)}" style="color:#8A94A3">'
        'Stop sending me these</a></p></div>'
    )
    return subject, text, html


# --------------------------------------------------------------------------
# "Somebody paid" -- to the owner, not the customer.
#
# The first Coach Notes order (2026-10-11) was found two days in, by
# accident, in the Stripe dashboard: nothing told anybody a review was owed,
# and it comes with a 72-hour promise. Every sale is announced; the ones that
# owe work say so in the subject line.
# --------------------------------------------------------------------------
REVIEW_PLANS = {"expert": "Expert Review", "notes": "Coach Notes",
                "expert_recheck": "Expert Review (re-check)"}
NOTES_DUE_HOURS = 72


def _local_time(ms: int) -> str:
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    t = datetime.fromtimestamp(ms / 1000, UTC)
    utc = t.strftime("%a %d %b %H:%M UTC")
    try:
        pt = t.astimezone(ZoneInfo("America/Vancouver"))
    except ZoneInfoNotFoundError:   # no tz database in the image: UTC alone is still right
        return utc
    return f"{pt.strftime('%a %d %b %H:%M')} Pacific ({utc})"


def sale_email(sale: dict, admin_url: str) -> tuple[str, str, str]:
    """(subject, text, html) for the owner when a payment lands."""
    plan = str(sale.get("plan") or "")
    amount = sale.get("amount_total")
    money = (f"{amount / 100:.2f} {str(sale.get('currency') or '').upper()}".strip()
             if isinstance(amount, (int, float)) else "amount unknown")
    review = REVIEW_PLANS.get(plan)
    if sale.get("mode") == "subscription":
        what = f"Subscription: {sale.get('tier') or 'unknown tier'}"
    elif review:
        what = review
    elif plan == "unlock":
        what = "Report unlock"
    else:
        what = plan or "One-time purchase"

    rows = [("What", what), ("Amount", money), ("Buyer", sale.get("email") or "?"),
            ("Paid", _local_time(sale["paid_ms"]))]
    if review:
        rows.append(("Analysis", sale.get("analysis_client_id")
                     or "NOT LINKED -- pick one in the queue"))
        if plan == "notes":
            rows.append(("Due", _local_time(sale["paid_ms"] + NOTES_DUE_HOURS * 3600_000)
                         + f" -- promised within {NOTES_DUE_HOURS} h"))
    if sale.get("credit_granted"):
        rows.append(("Note", "Includes an Expert Review credit -- the athlete can now order one."))

    subject = (f"ACTION: new {review} order — {money}" if review
               else f"New sale: {what} — {money}")
    text = "\n".join(f"{k}: {v}" for k, v in rows)
    if review:
        text += f"\n\nOpen the review queue:\n{admin_url}\n"
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        'font-size:15px;line-height:1.6;color:#14294B;max-width:520px">'
        f'<h1 style="font-size:20px;margin:0 0 12px">{_esc_html(subject)}</h1>'
        '<table style="border-collapse:collapse">'
        + "".join(
            f'<tr><td style="padding:3px 14px 3px 0;color:#5A6478">{_esc_html(k)}</td>'
            f'<td style="padding:3px 0"><b>{_esc_html(str(v))}</b></td></tr>'
            for k, v in rows)
        + "</table>"
        + (f'<p style="margin:20px 0 0"><a href="{_esc_html(admin_url)}" '
           'style="background:#2F6DE0;color:#fff;text-decoration:none;'
           'padding:10px 18px;border-radius:8px;display:inline-block;'
           'font-weight:600">Open the review queue</a></p>' if review else "")
        + "</div>"
    )
    return subject, text, html


def notify_sale(sale: dict, admin_url: str) -> bool:
    """Email the owner about one sale. Blocking: run it off the event loop."""
    to = settings.admin_email
    if not to:
        logger.warning("SALE_NOT_ANNOUNCED", reason="ADMIN_EMAIL not set",
                       plan=sale.get("plan"), user_id=sale.get("user_id"))
        return False
    subject, text, html = sale_email(sale, admin_url)
    return send_email(to, subject, text, html)
