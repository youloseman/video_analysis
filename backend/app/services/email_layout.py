"""One letter, two parts: a marked-up text becomes the plain-text and the HTML part.

Ported from Clariva's prospect letters (E:/Clariva/prospect/letter-html.mjs, layout
of 12 Oct 2026), set in Flapp's colours: a white 600 px column on the warm ground,
the speed stripe across the top, the mark and a kicker line centred, the letter in
sections under headings with a hairline above each, lists set as lists, the one
filled button in action coral, a signature set the way a bank sets one, and the
legal lines small and grey at the foot.

Why a renderer and not a hand-made HTML file per issue. The plain-text part is not
a fallback: some clients show it first and a spam filter reads it first, so it has
to say the same thing as the HTML. Two files written by hand drift; one text
rendered twice cannot.

Why it stays this plain. Tables, inline styles and the reader's own fonts, because
that is what every mail client draws the same. No web font, no CSS gradient (Gmail
and Outlook drop them: the stripe is an image with the gradient's middle colour
behind it), no tracking pixel. The two images (stripe and mark) are served from
getflapp.com; when a client blocks them the stripe falls back to a flat colour and
the mark to its alt text, and the letter still reads.

The marks, one block per blank-line-separated paragraph:

    # Title                    the letter's headline (once, at the top)
    ## Heading                 a section: hairline above, navy heading
    **words**                  bold, in navy
    [label](https://...)       a link inside a paragraph
    [label](https://...)       ...alone in its own block: the coral button
    - item                     a bulleted list (every line of the block)
    1. item                    a numbered list (every line of the block)
    > words                    a panel: tinted, with an accent rule on the left

Copy rules from the brandbook apply to what is written in them: no full stop at
the end of a heading, and no em dash anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---- Flapp tokens (static/tokens.css), as mail clients need them: literal ----
GROUND = "#F6F5F2"        # --c-ground
CARD = "#FFFFFF"          # --c-bg
LINE = "#DEDBD4"          # --c-line
INK = "#14181F"           # --c-ink
INK_SOFT = "#3C424C"      # --c-ink-soft
HEAD = "#14294B"          # --c-head, brand navy
FAINT = "#5B6470"         # --c-ink-faint
FOOT = "#8A929E"          # --c-ink-disabled: the legal lines only
ACCENT = "#2457C5"        # --c-blue: links, list marks, panel rule
PANEL = "#EDF2FD"         # --c-panel-blue
ACTION = "#CE3F2B"        # --c-coral-btn: the one filled button
STRIPE_MID = "#6A5AC8"    # the speed gradient's middle, behind the stripe image

FONT = "'IBM Plex Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'IBM Plex Mono',Menlo,Consolas,'Courier New',monospace"
# --f-brand: the wordmark and the step numerals, nothing else
BRAND = "'Archivo','Arial Black',Arial,sans-serif"

SITE = "https://getflapp.com"
STRIPE_IMG = f"{SITE}/media/email/rule.png"   # 1200x8, blue into coral
MARK_IMG = f"{SITE}/icons/icon-192.png"


@dataclass(frozen=True)
class Signature:
    name: str
    title: str
    email: str
    city: str = "Vancouver, BC"
    site: str = "getflapp.com"
    href: str = SITE


ARTUR = Signature(name="Artur", title="Founder, Flapp", email="support@getflapp.com")


# --------------------------------------------------------------------------
# inline marks
# --------------------------------------------------------------------------
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BUTTON = re.compile(r"^\[([^\]]+)\]\(([^)\s]+)\)$")


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _inline(s: str, *, link_colour: str = ACCENT, bold_colour: str = HEAD) -> str:
    """Escape, then bold and links. Escaping first is safe: the marks use no
    character the escape touches."""
    out = _esc(s)
    out = _BOLD.sub(lambda m: f'<b style="color:{bold_colour}">{m.group(1)}</b>', out)
    return _LINK.sub(
        lambda m: f'<a href="{m.group(2)}" style="color:{link_colour};text-decoration:underline">'
                  f"{m.group(1)}</a>", out)


def _plain_inline(s: str) -> str:
    s = _BOLD.sub(r"\1", s)
    return _LINK.sub(lambda m: m.group(1) if m.group(1) == m.group(2) else f"{m.group(1)} ({m.group(2)})", s)


def _lines(b: str) -> list[str]:
    return [ln.strip() for ln in b.split("\n") if ln.strip()]


def _blocks(text: str) -> list[str]:
    # CRLF first: a template saved on Windows otherwise reads as one paragraph.
    blocks = [b.strip() for b in re.split(r"\n{2,}", text.replace("\r\n", "\n")) if b.strip()]
    out: list[str] = []
    for b in blocks:
        # a heading written on the line above its first paragraph is its own block
        first, _, rest = b.partition("\n")
        if re.match(r"^#{1,2} ", first) and rest.strip():
            out += [first, rest.strip()]
        else:
            out.append(b)
    return out


def _all(b: str, pat: str) -> bool:
    return all(re.match(pat, ln) for ln in _lines(b))


# --------------------------------------------------------------------------
# blocks -> HTML
# --------------------------------------------------------------------------
def _h1(b: str) -> str:
    return (f'<h1 style="margin:0 0 18px;font-family:{FONT};font-size:24px;line-height:1.25;'
            f'font-weight:700;color:{HEAD}">{_inline(b[2:])}</h1>')


def _h2(b: str) -> str:
    return (f'<h2 style="margin:28px 0 12px;padding-top:16px;border-top:1px solid {LINE};'
            f'font-family:{FONT};font-size:19px;line-height:1.3;font-weight:700;color:{HEAD}">'
            f"{_inline(b[3:])}</h2>")


def _para(b: str) -> str:
    return (f'<p style="margin:0 0 16px;font-size:16px;line-height:1.6;color:{INK}">'
            + "<br>".join(_inline(ln) for ln in _lines(b)) + "</p>")


def _panel(b: str) -> str:
    body = "<br>".join(_inline(re.sub(r"^>\s?", "", ln)) for ln in _lines(b))
    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            'style="margin:4px 0 18px;border-collapse:collapse"><tr>'
            f'<td style="background:{PANEL};border-left:4px solid {ACCENT};padding:14px 18px;'
            f'font-size:16px;line-height:1.5;font-weight:600;color:{HEAD}">{body}</td></tr></table>')


def _bullets(b: str) -> str:
    rows = "".join(
        f'<tr><td valign="top" style="padding:0 10px 8px 2px;color:{ACCENT};font-size:14px;'
        f'line-height:1.75">&#9632;</td><td valign="top" style="padding:0 0 8px;font-size:16px;'
        f'line-height:1.5;color:{INK}">{_inline(ln[2:])}</td></tr>'
        for ln in _lines(b))
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin:0 0 16px;border-collapse:collapse">{rows}</table>')


def _numbered(b: str) -> str:
    rows = []
    for ln in _lines(b):
        n, _, rest = ln.partition(". ")
        # The app's step numeral (.ustep-n): heavy italic brand face, two digits,
        # in blue -- the app's own fallback where the gradient cannot clip to text.
        rows.append(
            f'<tr><td valign="top" style="padding:0 12px 10px 0;font-family:{BRAND};font-size:20px;'
            f'font-weight:900;font-style:italic;letter-spacing:-.02em;line-height:1.2;color:{ACCENT}">'
            f'{int(n):02d}</td><td valign="top" style="padding:1px 0 10px;font-size:16px;'
            f'line-height:1.5;color:{INK}">{_inline(rest)}</td></tr>')
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin:0 0 16px;border-collapse:collapse">{"".join(rows)}</table>')


def _button(label: str, url: str) -> str:
    # A table cell with the colour, not a padded link alone: Outlook ignores
    # padding on <a>, and the cell keeps the button a button there too.
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
            'style="margin:6px 0 20px;border-collapse:separate"><tr>'
            f'<td bgcolor="{ACTION}" style="background:{ACTION};border-radius:4px">'
            f'<a href="{_esc(url)}" style="display:inline-block;padding:13px 22px;font-family:{FONT};'
            f'font-size:16px;font-weight:500;line-height:1;color:#FFFFFF;text-decoration:none">'
            f"{_esc(label)}</a></td></tr></table>")


def _block_html(b: str) -> str:
    if b.startswith("# "):
        return _h1(b)
    if b.startswith("## "):
        return _h2(b)
    if _all(b, r"^>"):
        return _panel(b)
    if _all(b, r"^- "):
        return _bullets(b)
    if _all(b, r"^\d+\. "):
        return _numbered(b)
    m = _BUTTON.match(b)
    if m:
        return _button(m.group(1), m.group(2))
    return _para(b)


def _signature_html(s: Signature) -> str:
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'style="margin:24px 0 22px;border-collapse:collapse"><tr><td style="padding:0 0 10px">'
        f'<img src="{STRIPE_IMG}" width="240" height="2" alt="" style="display:block;width:240px;'
        f'height:2px;border:0;background:{STRIPE_MID}"></td></tr>'
        f'<tr><td style="font-size:14px;line-height:1.55;color:{INK}">'
        f'<span style="font-size:15px;font-weight:700;color:{HEAD}">{_esc(s.name)}</span><br>'
        f'<span style="color:{FAINT}">{_esc(s.title)} &middot; {_esc(s.city)}</span><br>'
        f'<span style="font-size:13px;color:{FAINT}"><b style="font-family:{MONO};color:{ACCENT};'
        f'font-weight:600">E</b>&nbsp; <a href="mailto:{_esc(s.email)}" style="color:{FAINT};'
        f'text-decoration:none">{_esc(s.email)}</a></span><br>'
        f'<a href="{_esc(s.href)}" style="font-size:13px;font-weight:600;color:{ACCENT};'
        f'text-decoration:none">{_esc(s.site)}</a></td></tr></table>'
    )


def _footer_html(footer: str) -> str:
    body = "<br>".join(_inline(ln, link_colour=FOOT, bold_colour=FAINT) for ln in _lines(footer))
    return (f'<p style="margin:8px 0 0;padding-top:12px;border-top:1px solid {LINE};'
            f'font-size:12px;line-height:1.5;color:{FOOT}">{body}</p>')


def _masthead(kicker: str) -> str:
    return (
        f'<tr><td align="center" style="padding:24px 28px 16px;border-bottom:1px solid {LINE}">'
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'style="margin:0 auto;border-collapse:collapse"><tr>'
        f'<td valign="middle" style="padding:0 9px 0 0"><a href="{SITE}" style="text-decoration:none">'
        f'<img src="{MARK_IMG}" width="30" height="30" alt="" style="display:block;width:30px;'
        f'height:30px;border:0;border-radius:6px;background:{HEAD}"></a></td>'
        # The app's .wordmark: Archivo 900 italic, tight, in brand navy.
        f'<td valign="middle" style="font-family:{BRAND};font-size:22px;font-weight:900;'
        f'font-style:italic;letter-spacing:-.02em;line-height:1;color:{HEAD}"><a href="{SITE}" '
        f'style="color:{HEAD};'
        'text-decoration:none">Flapp</a></td></tr></table>'
        f'<div style="margin-top:9px;font-family:{MONO};font-size:11px;letter-spacing:.14em;'
        f'text-transform:uppercase;color:{FAINT}">{_esc(kicker)}</div></td></tr>'
    )


def render(
    markup: str,
    *,
    kicker: str = "Run & bike technique",
    preheader: str | None = None,
    signature: Signature | None = ARTUR,
    footer: str | None = None,
) -> tuple[str, str]:
    """(text, html) for one letter. ``footer`` takes the same inline marks
    (bold, links) and is set small under a hairline."""
    blocks = _blocks(markup)

    # ---- HTML ----
    pre = ""
    if preheader:
        # The line an inbox shows after the subject. Hidden in the open letter;
        # the padding after it stops the client pulling in the masthead text.
        pre = (f'<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;font-size:1px;'
               f'line-height:1px;color:{GROUND}">{_esc(preheader)}'
               + "&nbsp;&zwnj;" * 40 + "</div>")
    body = "".join(_block_html(b) for b in blocks)
    if signature:
        body += _signature_html(signature)
    if footer:
        body += _footer_html(footer)
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light"><meta name="supported-color-schemes" content="light">'
        f'</head><body style="margin:0;padding:0;background:{GROUND}">{pre}'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'bgcolor="{GROUND}" style="background:{GROUND};border-collapse:collapse"><tr>'
        '<td align="center" style="padding:24px 10px">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;max-width:600px;background:{CARD};border-collapse:collapse">'
        f'<tr><td bgcolor="{STRIPE_MID}" style="background:{STRIPE_MID};font-size:0;line-height:0">'
        f'<img src="{STRIPE_IMG}" width="600" height="4" alt="" style="display:block;width:100%;'
        'max-width:600px;height:4px;border:0"></td></tr>'
        + _masthead(kicker)
        + f'<tr><td style="padding:26px 28px 14px;font-family:{FONT};font-size:16px;color:{INK}">'
        + body
        + "</td></tr></table></td></tr></table></body></html>"
    )

    # ---- plain text ----
    parts: list[str] = []
    for b in blocks:
        if b.startswith("# "):
            parts.append(_plain_inline(b[2:]))
        elif b.startswith("## "):
            parts.append(_plain_inline(b[3:]).upper())
        elif _all(b, r"^>"):
            parts.append("\n".join(_plain_inline(re.sub(r"^>\s?", "", ln)) for ln in _lines(b)))
        elif (m := _BUTTON.match(b)):
            parts.append(f"{m.group(1)}: {m.group(2)}")
        else:
            parts.append("\n".join(_plain_inline(ln) for ln in _lines(b)))
    if signature:
        parts.append(f"{signature.name}\n{signature.title}, {signature.city}\n"
                     f"{signature.email} · {signature.site}")
    if footer:
        parts.append("--\n" + "\n".join(_plain_inline(ln) for ln in _lines(footer)))
    return "\n\n".join(parts) + "\n", html


__all__ = ["ARTUR", "Signature", "render"]
