"""Product news to registered athletes: who gets it, and the one email each.

Rare and deliberate -- an issue goes out by hand (scripts/newsletter.py),
never on a schedule. Three rules hold it together:

* **Every copy carries its own one-click unsubscribe** for this list only
  (``/unsubscribe?list=news``); leaving the news never silences the mail
  about the athlete's own analyses and purchases.
* **It identifies the sender with a postal address.** Canada's anti-spam
  law (CASL) requires it in every commercial message; an issue whose
  template still has the placeholder is refused, not sent.
* **Test and staff accounts are not athletes.** The addresses the deploy
  probes and funnel checks sign up with, and admin accounts, are left out.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.models.user import TIER_ADMIN

CONTENT_DIR = Path(__file__).resolve().parents[2] / "content" / "newsletters"
ADDRESS_PLACEHOLDER = "[MAILING ADDRESS]"
UNSUB_PLACEHOLDER = "{{UNSUBSCRIBE_URL}}"

# Accounts the service itself created: deploy probes, funnel checks, selftests,
# the App Store review login, and anything on a reserved test domain.
_TEST_EMAIL = re.compile(
    r"(@example\.(com|org|net)$|\.test$|^deploytest|^funnel\d|^t\d{6,}@|^audit-probe|"
    r"^flapp-selftest|^appreview@getflapp\.com$)",
    re.I,
)


def is_test_address(email: str | None) -> bool:
    return not email or bool(_TEST_EMAIL.search(email.strip()))


def is_recipient(user: Any) -> bool:
    """Whether this account gets product news."""
    return (
        not getattr(user, "news_opt_out", False)
        and getattr(user, "tier", None) != TIER_ADMIN
        and not is_test_address(getattr(user, "email", None))
    )


def load_issue(slug: str) -> dict[str, str]:
    """``content/newsletters/<slug>.html`` (+ ``.txt`` and ``.subject``)."""
    base = CONTENT_DIR / slug
    html = (base.with_suffix(".html")).read_text(encoding="utf-8")
    text = (base.with_suffix(".txt")).read_text(encoding="utf-8")
    subject = (base.with_suffix(".subject")).read_text(encoding="utf-8").strip()
    return {"slug": slug, "subject": subject, "html": html, "text": text}


def render(issue: dict[str, str], unsubscribe_url: str, address: str) -> tuple[str, str, str]:
    """(subject, text, html) for one recipient. Raises ``ValueError`` when the
    issue cannot legally go out (no postal address, or no unsubscribe slot)."""
    address = (address or "").strip()
    if not address or ADDRESS_PLACEHOLDER in address:
        raise ValueError("No postal address for the sender: set NEWSLETTER_ADDRESS.")
    for part in ("html", "text"):
        if UNSUB_PLACEHOLDER not in issue[part]:
            raise ValueError(f"The issue's {part} has no {UNSUB_PLACEHOLDER} slot.")
    def fill(s: str, url: str, addr: str) -> str:
        return s.replace(UNSUB_PLACEHOLDER, url).replace(ADDRESS_PLACEHOLDER, addr)
    import html as _html

    return (
        issue["subject"],
        fill(issue["text"], unsubscribe_url, address),
        fill(issue["html"], _html.escape(unsubscribe_url), _html.escape(address)),
    )


__all__ = ["ADDRESS_PLACEHOLDER", "CONTENT_DIR", "UNSUB_PLACEHOLDER",
           "is_recipient", "is_test_address", "load_issue", "render"]
