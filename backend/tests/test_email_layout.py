"""The letter layout: one marked-up text, two parts that say the same thing.

What would hurt if it broke: an HTML part and a text part that drift apart (the
text part is what spam filters and some clients read first), a Markdown issue
that goes out without its unsubscribe link or postal address, or a mark that
leaks into the letter as a literal ``**`` or ``## ``.
"""

from __future__ import annotations

import pytest

from app.services import email_layout as L
from app.services import newsletter as NL

SAMPLE = """# Kona pros sit further forward

Hi,

We measured **25 pros** and 58 age-groupers.

## What we found
- **It isn't saddle height.** Same knee angle.
- It's where you sit.

> Three degrees at the axle is 4-5 cm at the hip.

1. Tap **Mark the crank axle**.
2. Drag it if it's off.

[Measure your position](https://getflapp.com/app?a=1&b=2)

Read [the study](https://getflapp.com/academy/x) first.

Ride fast,"""


def test_every_mark_becomes_its_element_and_none_leaks():
    text, html = L.render(SAMPLE, kicker="Product news")
    assert "<h1" in html and "<h2" in html
    assert "&#9632;" in html                                   # bullets
    assert L.PANEL in html                                    # the panel
    assert f'bgcolor="{L.ACTION}"' in html                     # the button
    assert 'href="https://getflapp.com/app?a=1&amp;b=2"' in html
    assert '<a href="https://getflapp.com/academy/x"' in html
    for leak in ("**", "## ", "](", "\n> "):
        assert leak not in html, leak
        assert leak not in text, leak
    assert "Product news".upper() not in text                  # the kicker is HTML only


def test_the_text_part_says_what_the_html_says():
    text, html = L.render(SAMPLE)
    assert text.startswith("Kona pros sit further forward\n\nHi,")
    assert "WHAT WE FOUND" in text
    assert "- It isn't saddle height. Same knee angle." in text
    assert "1. Tap Mark the crank axle." in text
    assert "Measure your position: https://getflapp.com/app?a=1&b=2" in text
    assert "the study (https://getflapp.com/academy/x)" in text
    # signature in both
    assert "Artur" in text and "support@getflapp.com" in text
    assert "Founder, Flapp" in html and "mailto:support@getflapp.com" in html


def test_user_text_is_escaped():
    text, html = L.render("Hi <script>alert(1)</script> & co")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<script>" in text  # plain text is plain


def test_crlf_templates_still_split_into_blocks():
    _, html = L.render("## One\r\n\r\nPara one\r\n\r\nPara two")
    assert html.count('<p style="margin:0 0 16px') == 2


def test_no_signature_and_a_footer():
    text, html = L.render("Hi", signature=None, footer="Sent by [us](https://x.test)")
    assert "Founder" not in html and "Founder" not in text
    assert text.endswith("--\nSent by us (https://x.test)\n")


def test_layout_copy_keeps_the_house_rules():
    """No em dash in anything the layout itself writes (brandbook section 10)."""
    text, html = L.render("Hi", footer=NL.FOOTER, preheader="p")
    assert "—" not in text + html
    # Flapp and Artur are in Vancouver (Artur, 2026-11-25); the postal address
    # CASL asks for comes from NEWSLETTER_ADDRESS, never from the layout.
    assert "Vancouver, BC" in text and "Vancouver, BC" in html
    assert "Nanaimo" not in text + html


def test_a_markdown_issue_renders_and_carries_its_legal_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(NL, "CONTENT_DIR", tmp_path)
    (tmp_path / "x.md").write_text(
        "---\nsubject: A subject\nkicker: Product news\npreheader: Preview line\n---\n"
        "# Title\n\nHi,\n\n[Open the app](https://getflapp.com/app)\n\nRide fast,\n",
        encoding="utf-8",
    )
    issue = NL.load_issue("x")
    assert issue["subject"] == "A subject"
    assert "Preview line" in issue["html"] and "PRODUCT NEWS" not in issue["text"]
    for part in ("html", "text"):
        assert NL.UNSUB_PLACEHOLDER in issue[part]
        assert NL.ADDRESS_PLACEHOLDER in issue[part]
    subject, text, html = NL.render(issue, "https://x/unsubscribe?u=1&t=a&list=news", "1 Main St")
    assert "https://x/unsubscribe?u=1&amp;t=a&amp;list=news" in html
    assert "https://x/unsubscribe?u=1&t=a&list=news" in text
    assert "1 Main St" in html + text


def test_a_markdown_issue_without_a_subject_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(NL, "CONTENT_DIR", tmp_path)
    (tmp_path / "x.md").write_text("# Title\n\nHi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subject"):
        NL.load_issue("x")
