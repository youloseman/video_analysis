"""One look for every mail Flapp sends a customer.

The app's own tokens (tokens.css), written out as inline styles because mail
clients drop stylesheets: the warm ground, a white card on a hairline, the
title in the brand face, one coral button, the same foot. Tables, not divs,
for the layout, because Outlook still lays mail out with Word.

Every customer mail is built from these pieces, so a change of colour or
spacing is made here once. Copy stays in the module that sends the mail:
this file knows how a mail looks, not what it says.
"""

from __future__ import annotations

from html import escape

# tokens.css, light theme. Mail is light only: dark-mode clients repaint
# colours on their own and do it worse when the mail fights them.
GROUND = "#F6F5F2"
CARD = "#FFFFFF"
LINE = "#DEDBD4"
INK = "#14181F"
HEAD = "#14294B"
SOFT = "#3C424C"
FAINT = "#5B6470"
CORAL = "#CE3F2B"       # the action: the one filled button
LINK = "#1C46A3"
NAVY = "#14294B"

BODY_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
BRAND_FONT = "'Archivo','Helvetica Neue',Arial,sans-serif"


def esc(s: object) -> str:
    """For attribute values."""
    return escape(str(s), quote=True)


def txt(s: object) -> str:
    """For text between tags: quotes stay as typed."""
    return escape(str(s), quote=False)


def para(text: str, *, muted: bool = False, size: int = 16) -> str:
    """A paragraph of plain text (escaped here)."""
    color = FAINT if muted else INK
    return (f'<p style="margin:0 0 16px;font-size:{size}px;line-height:1.55;'
            f'color:{color}">{txt(text)}</p>')


def para_html(html: str, *, muted: bool = False, size: int = 16) -> str:
    """A paragraph whose HTML the caller has already escaped."""
    color = FAINT if muted else INK
    return (f'<p style="margin:0 0 16px;font-size:{size}px;line-height:1.55;'
            f'color:{color}">{html}</p>')


def link(href: str, label: str) -> str:
    return f'<a href="{esc(href)}" style="color:{LINK};font-weight:600;text-decoration:underline">{txt(label)}</a>'


def button(href: str, label: str) -> str:
    """The one filled button a mail has. A table cell, so it keeps its shape
    in Outlook; the link fills it so the whole pill is the target."""
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:8px 0 20px">'
        f'<tr><td style="border-radius:8px;background:{CORAL}">'
        f'<a href="{esc(href)}" style="display:inline-block;padding:14px 24px;font-family:{BODY_FONT};'
        f'font-size:16px;font-weight:600;line-height:1;color:#FFFFFF;text-decoration:none;'
        f'border-radius:8px">{txt(label)}</a></td></tr></table>'
    )


def callout(label: str, text: str, by: str | None = None) -> str:
    """The coach's words in the dark card the app shows them in: a line in
    the brand face, a paragraph in the body face."""
    sign = (f'<p style="margin:12px 0 0;font-size:14px;line-height:1.4;color:#C7D2EE">{txt(by)}</p>'
            if by else "")
    face = (f"font-family:{BRAND_FONT};font-style:italic;font-weight:900;font-size:20px;line-height:1.25"
            if len(text) <= 140 else "font-size:16px;line-height:1.55")
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:0 0 20px">'
        f'<tr><td style="background:{NAVY};border-radius:12px;padding:18px 20px">'
        f'<p style="margin:0 0 6px;font-size:14px;font-weight:600;color:#C7D2EE">{txt(label)}</p>'
        f'<p style="margin:0;{face};color:#FFFFFF">{txt(text)}</p>{sign}</td></tr></table>'
    )


def foot_link(href: str, label: str) -> str:
    """A link in the grey foot (unsubscribe and the like)."""
    return f'<a href="{esc(href)}" style="color:{FAINT};text-decoration:underline">{txt(label)}</a>'


def fallback_link(href: str) -> str:
    """For a button some clients will not render: the address itself."""
    return para_html(f'If the button does nothing, copy this address into your browser:<br>'
                     f'<a href="{esc(href)}" style="color:{LINK};word-break:break-all">{txt(href)}</a>',
                     muted=True, size=14)


def page(*, title: str, body: str, kicker: str = "", preheader: str = "",
         foot: str = "", base: str = "https://getflapp.com") -> str:
    """The whole mail: ground, wordmark, the card with the title and body, foot.

    ``body`` and ``foot`` are HTML the caller built from the pieces above;
    ``title``, ``kicker`` and ``preheader`` are plain text.
    """
    base = base.rstrip("/")
    pre = (f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:{GROUND}">'
           f'{txt(preheader)}</div>') if preheader else ""
    kick = (f'<td align="right" style="font-family:{BODY_FONT};font-size:14px;font-weight:600;'
            f'color:{FAINT}">{txt(kicker)}</td>') if kicker else ""
    extra = (f'<p style="margin:0 0 8px">{foot}</p>') if foot else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light"><meta name="supported-color-schemes" content="light">'
        f'<title>{txt(title)}</title></head>'
        f'<body style="margin:0;padding:0;background:{GROUND}" data-flapp-mail="1">{pre}'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{GROUND}">'
        '<tr><td align="center" style="padding:24px 12px 32px">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:560px">'
        # wordmark
        '<tr><td style="padding:0 4px 14px">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
        f'<td><a href="{esc(base)}" style="text-decoration:none">'
        f'<img src="{esc(base)}/icons/icon-192.png" width="28" height="28" alt="" '
        'style="vertical-align:middle;border:0;border-radius:6px">'
        f'<span style="vertical-align:middle;margin-left:8px;font-family:{BRAND_FONT};font-style:italic;'
        f'font-weight:900;font-size:22px;color:{HEAD}">Flapp</span></a></td>{kick}</tr></table></td></tr>'
        # the card
        f'<tr><td style="background:{CARD};border:1px solid {LINE};border-radius:16px;padding:28px 28px 12px;'
        f'font-family:{BODY_FONT};color:{INK}">'
        f'<h1 style="margin:0 0 16px;font-family:{BRAND_FONT};font-style:italic;font-weight:900;'
        f'font-size:26px;line-height:1.15;color:{HEAD}">{txt(title)}</h1>{body}</td></tr>'
        # foot
        f'<tr><td style="padding:16px 4px 0;font-family:{BODY_FONT};font-size:12px;line-height:1.5;color:{FAINT}">'
        f'{extra}<p style="margin:0">Flapp · running and cycling technique from one side-view clip · '
        f'<a href="{esc(base)}" style="color:{FAINT}">getflapp.com</a></p></td></tr>'
        '</table></td></tr></table></body></html>'
    )
