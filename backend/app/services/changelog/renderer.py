"""Server-rendered ``/changelog`` page.

The page shell (sidebar, footer, base CSS, account box, SEO head) is reused
verbatim from the Academy renderer. Importing its private ``_page``/``_esc`` is
deliberate: the alternative is a second copy of the shell, and two copies drift
until the two server-rendered surfaces stop looking like the same product.
"""

from __future__ import annotations

from app.services.academy.renderer import SITE_NAME, _esc, _page
from app.services.changelog.parser import Entry

_CSS = """
/* Hero mirrors the Academy hub's so the two server-rendered pages read as one
   product (see _HUB_CSS::.acad-hero). */
.cl-hero{background:linear-gradient(135deg,#14294B 0%,#1D3E77 60%,#2F6DE0 100%);color:#fff;
  position:relative;overflow:hidden}
.cl-hero .wrap{max-width:760px;padding:56px 24px 56px;position:relative;z-index:1}
.cl-hero .eyebrow{color:rgba(255,255,255,.75)}
.cl-hero h1{font-family:var(--f-display);font-weight:900;font-style:italic;text-transform:uppercase;
  font-size:clamp(30px,5vw,48px);line-height:1;letter-spacing:-.01em;margin:14px 0 14px}
.cl-hero p{font-size:17px;line-height:1.6;color:rgba(255,255,255,.9);max-width:60ch;margin:0}
.cl-list{max-width:760px;margin:0 auto;padding:48px 24px 72px}
.cl-entry{position:relative;padding:0 0 40px 28px;border-left:2px solid var(--c-line)}
.cl-entry:last-child{border-left-color:transparent;padding-bottom:16px}
.cl-entry::before{content:"";position:absolute;left:-7px;top:6px;width:12px;height:12px;
  border-radius:50%;background:var(--c-blue);border:3px solid #fff}
.cl-entry .when{font-family:var(--f-mono);font-size:12px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--c-ink-faint);display:block;margin-bottom:6px}
.cl-entry h2{font-family:var(--f-display);text-transform:uppercase;color:var(--c-navy);
  font-size:clamp(20px,3vw,26px);line-height:1.12;margin:0 0 12px}
.cl-entry p{font-size:15px;line-height:1.65;color:var(--c-ink);margin:12px 0;max-width:66ch}
.cl-entry ul,.cl-entry ol{margin:12px 0;padding-left:22px}
.cl-entry li{font-size:15px;line-height:1.65;color:var(--c-ink);margin:6px 0;max-width:64ch}
.cl-entry strong{color:var(--c-navy);font-weight:800}
.cl-empty{color:var(--c-ink-soft);font-size:15px}
.cl-foot{max-width:760px;margin:0 auto 72px;padding:20px 24px;border-top:1px solid var(--c-line)}
.cl-foot p{font-size:14px;color:var(--c-ink-soft);line-height:1.6;margin:0}
"""

_LEDE = (
    "Everything we've shipped, newest first. Most of it starts as someone "
    "telling us a result didn't match what they saw."
)


def render_changelog(entries: list[Entry], base_url: str) -> str:
    canonical = f"{base_url}/changelog"
    items = "".join(
        '<article class="cl-entry">'
        f'<span class="when">{_esc(e.date_label)}</span>'
        f"<h2>{_esc(e.title)}</h2>"
        f"{e.body_html}"
        "</article>"
        for e in entries
    )
    inner = items or '<p class="cl-empty">Nothing here yet.</p>'
    body = (
        '<div class="cl-hero"><div class="wrap">'
        '<div class="eyebrow">Changelog</div>'
        "<h1>What's new in Flapp</h1>"
        f"<p>{_esc(_LEDE)}</p>"
        "</div></div>"
        f'<main class="cl-list">{inner}</main>'
        '<div class="cl-foot"><p>Found something the analyzer gets wrong? '
        "Rate your next result — the prompt sits at the bottom of every report, "
        "and it is where this list comes from.</p></div>"
    )
    jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": f"{SITE_NAME} changelog",
        "url": canonical,
        "description": _LEDE,
    }
    return _page(
        title=f"What's new — {SITE_NAME}",
        description="Release notes for Flapp: what changed in the running and "
        "cycling form analyzer, newest first.",
        canonical=canonical,
        body=body,
        extra_css=_CSS,
        active="changelog",
        jsonld=jsonld,
    )
