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


def _u(email, tier="starter", out=False, marketing=True):
    return SimpleNamespace(
        email=email, tier=tier, news_opt_out=out, marketing_opt_in=marketing
    )


@pytest.mark.parametrize(
    "email",
    [
        "funnel1785385959@getflapp.com",
        "deploytest+1785364754@getflapp.com",
        "t1785385024@getflapp.com",
        "audit-probe-1785404925@example.com",
        "flapp-selftest-ui2@example.com",
        "appreview@getflapp.com",
        "someone@example.com",
    ],
)
def test_the_services_own_accounts_get_nothing(email):
    assert not NL.is_recipient(_u(email))


def test_athletes_get_it_unless_they_left_or_run_the_place():
    assert NL.is_recipient(_u("normap.ca@gmail.com"))
    assert NL.is_recipient(_u("jwrdds@comcast.net", tier="full"))
    assert not NL.is_recipient(_u("normap.ca@gmail.com", out=True))
    assert not NL.is_recipient(_u("top.raider90@gmail.com", tier="admin"))


def test_no_consent_no_news():
    """Since 2026-10-24 news needs the athlete's own yes (CASL): an account
    that never ticked the box is not on the list, however old it is."""
    assert not NL.is_recipient(_u("normap.ca@gmail.com", marketing=False))


@pytest.mark.parametrize("slug", ["2026-10-kona-crank", "2026-10-android-test"])
def test_every_issue_in_the_repo_carries_its_unsubscribe_and_address(slug):
    """Both parts of every issue, or the law is broken by the copy, not the code."""
    issue = NL.load_issue(slug)
    for part in ("html", "text"):
        assert "{{UNSUBSCRIBE_URL}}" in issue[part], f"{slug}.{part}"
        assert "[MAILING ADDRESS]" in issue[part], f"{slug}.{part}"
    assert issue["subject"]


def test_the_issue_in_the_repo_renders_and_refuses_without_an_address():
    issue = NL.load_issue("2026-10-kona-crank")
    assert "88°" in issue["html"] and "88°" in issue["text"] and issue["subject"]
    with pytest.raises(ValueError, match="postal address"):
        NL.render(issue, "https://x/unsub", "")
    with pytest.raises(ValueError, match="postal address"):
        NL.render(issue, "https://x/unsub", NL.ADDRESS_PLACEHOLDER)
    subject, text, html = NL.render(
        issue, "https://x/unsubscribe?u=1&t=a&list=news", "1 Main St, Nanaimo BC"
    )
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


# --- --only: narrowing an issue to some of the list -------------------------
# An issue can be for a subset ("help test the Android app" is of no use to an
# iPhone athlete). The flag that does it must only ever remove recipients: a
# file of addresses is an easy way to accidentally mail someone who never gave
# consent, and that is exactly what the consent rule exists to prevent.


def _only_list(tmp_path, body):
    p = tmp_path / "only.txt"
    p.write_text(body, encoding="utf-8")
    return str(p)


def _recipients_after_only(users, only):
    """What scripts/newsletter.py keeps: is_recipient first, then the file."""
    people = [u for u in users if NL.is_recipient(u)]
    if only is not None:
        people = [u for u in people if (u.email or "").lower() in only]
    return people


def test_only_file_is_parsed_leniently(tmp_path):
    from scripts.newsletter import read_only_list

    got = read_only_list(
        _only_list(
            tmp_path,
            "\n".join(
                [
                    "# the android testers",
                    "One@Gmail.com",  # mixed case, as a person types it
                    "two@gmail.com, three@gmail.com",
                    "",
                    "four@gmail.com   # already said yes",
                ]
            ),
        )
    )
    assert got == {
        "one@gmail.com",
        "two@gmail.com",
        "three@gmail.com",
        "four@gmail.com",
    }


def test_only_cannot_reach_someone_without_consent(tmp_path):
    from scripts.newsletter import read_only_list

    users = [_u("yes@gmail.com", marketing=True), _u("no@gmail.com", marketing=False)]
    only = read_only_list(_only_list(tmp_path, "yes@gmail.com\nno@gmail.com\n"))
    assert [u.email for u in _recipients_after_only(users, only)] == ["yes@gmail.com"]


def test_only_removes_the_consenting_people_it_does_not_name(tmp_path):
    from scripts.newsletter import read_only_list

    users = [_u("android@gmail.com"), _u("iphone@gmail.com")]
    only = read_only_list(_only_list(tmp_path, "android@gmail.com\n"))
    assert [u.email for u in _recipients_after_only(users, only)] == [
        "android@gmail.com"
    ]


def test_without_only_the_list_is_unchanged():
    users = [_u("a@gmail.com"), _u("b@gmail.com"), _u("c@gmail.com", out=True)]
    assert [u.email for u in _recipients_after_only(users, None)] == [
        "a@gmail.com",
        "b@gmail.com",
    ]
