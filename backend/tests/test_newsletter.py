"""Product news: who gets it, and that every copy can be left in one click.

What would hurt if it broke: news going to an athlete who opted out, a news
unsubscribe also switching off the analysis-ready mail (or the reverse), a
deploy probe receiving the issue, or an issue going out without the postal
address the law asks for.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import newsletter as NL


def _u(email, tier="starter", out=False):
    return SimpleNamespace(email=email, tier=tier, news_opt_out=out)


@pytest.mark.parametrize("email", [
    "funnel1785385959@getflapp.com", "deploytest+1785364754@getflapp.com",
    "t1785385024@getflapp.com", "audit-probe-1785404925@example.com",
    "flapp-selftest-ui2@example.com", "appreview@getflapp.com", "someone@example.com",
])
def test_the_services_own_accounts_get_nothing(email):
    assert not NL.is_recipient(_u(email))


def test_athletes_get_it_unless_they_left_or_run_the_place():
    assert NL.is_recipient(_u("normap.ca@gmail.com"))
    assert NL.is_recipient(_u("jwrdds@comcast.net", tier="full"))
    assert not NL.is_recipient(_u("normap.ca@gmail.com", out=True))
    assert not NL.is_recipient(_u("top.raider90@gmail.com", tier="admin"))


def test_the_issue_in_the_repo_renders_and_refuses_without_an_address():
    issue = NL.load_issue("2026-10-kona-crank")
    assert "88°" in issue["html"] and "88°" in issue["text"] and issue["subject"]
    with pytest.raises(ValueError, match="postal address"):
        NL.render(issue, "https://x/unsub", "")
    with pytest.raises(ValueError, match="postal address"):
        NL.render(issue, "https://x/unsub", NL.ADDRESS_PLACEHOLDER)
    subject, text, html = NL.render(issue, "https://x/unsubscribe?u=1&t=a&list=news", "1 Main St, Nanaimo BC")
    assert "https://x/unsubscribe?u=1&amp;t=a&amp;list=news" in html
    assert "https://x/unsubscribe?u=1&t=a&list=news" in text
    assert "1 Main St, Nanaimo BC" in html and NL.ADDRESS_PLACEHOLDER not in html + text
    assert NL.UNSUB_PLACEHOLDER not in html + text


async def test_a_news_unsubscribe_leaves_the_analysis_mail_alone(db, make_user):
    main = pytest.importorskip("app.main")
    user = await make_user()
    assert user.news_opt_out is False and user.notify_on_ready is True
    # the analysis-ready token cannot be used on the news list, or the reverse
    await main.unsubscribe(user.id, main.unsubscribe_token(user.id), "news", db)
    await db.refresh(user)
    assert user.news_opt_out is False
    await main.unsubscribe(user.id, main.unsubscribe_token(user.id, "news"), "news", db)
    await db.refresh(user)
    assert user.news_opt_out is True and user.notify_on_ready is True
    await main.unsubscribe(user.id, main.unsubscribe_token(user.id, "news"), None, db)
    await db.refresh(user)
    assert user.notify_on_ready is True
