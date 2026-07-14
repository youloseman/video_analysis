"""Render :class:`Article` objects into full, SEO-friendly HTML pages.

Each page is a standalone document (own ``<title>``/meta/OpenGraph + JSON-LD)
styled with the same brand tokens as the single-page app (``static/index.html``)
so Academy reads as part of Flapp, not a bolted-on blog. Server-rendered on
purpose: the article text is in the initial HTML, which is what search engines
index — that is the whole point of the section.
"""

from __future__ import annotations

import html
import json
from typing import Any

from app.services.academy.parser import Article, ArticleMeta, get_categories

# Public base URL for canonical/OG tags + sitemap. Override via env in main.py.
SITE_NAME = "Flapp"
SITE_TAGLINE = "Running & Cycling Form Analysis"

# Shared design tokens + base chrome, lifted from static/index.html so the
# Academy pages match the app. Kept intentionally compact (a subset of the
# app's full stylesheet — just what article/hub pages use).
_BASE_CSS = """
:root{
  --c-blue:#2F6DE0; --c-blue-dk:#2459C2; --c-navy:#14294B;
  --c-coral:#F1553F; --c-coral-btn:#CE3F2B; --c-coral-dk:#C13A26;
  --c-ink:#1E2530; --c-ink-soft:#5C6675; --c-ink-faint:#8A94A3;
  --c-panel:#F3F5F8; --c-panel-blue:#EAF1FC; --c-line:#DCE2EA; --c-bg:#FFFFFF;
  --c-bike:#F2A33C; --c-run:#EF5B5B;
  --f-display:'Archivo','Arial Black',sans-serif;
  --f-body:'Manrope','Segoe UI',sans-serif;
  --f-mono:'IBM Plex Mono',monospace;
  --radius:10px; --radius-btn:8px; --skew:-8deg;
  --shadow:0 2px 10px rgba(20,41,75,.07);
  --shadow-lg:0 12px 34px rgba(20,41,75,.12);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--f-body);font-size:16px;line-height:1.6;color:var(--c-ink);background:var(--c-bg)}
.wrap{max-width:820px;margin:0 auto;padding:0 24px}
a{color:var(--c-blue)}
.speedline{height:6px;width:120px;border-radius:3px;transform:skewX(var(--skew));
  background:linear-gradient(90deg,var(--c-blue),var(--c-coral))}
.eyebrow{font-size:11px;letter-spacing:.2em;text-transform:uppercase;font-weight:800;color:var(--c-blue)}
/* header */
header{border-bottom:1px solid var(--c-line);background:#fff;position:sticky;top:0;z-index:20}
header .wrap{max-width:1000px;display:flex;align-items:center;justify-content:space-between;height:64px}
.wordmark{font-family:var(--f-display);font-weight:900;font-style:italic;font-size:24px;
  text-transform:uppercase;letter-spacing:-.01em;color:var(--c-navy);display:flex;align-items:center;gap:8px;text-decoration:none}
.wordmark .dot{width:9px;height:9px;border-radius:50%;background:var(--c-coral);transform:translateY(-9px)}
header nav a{color:var(--c-ink-soft);text-decoration:none;font-size:13px;font-weight:700;padding:8px 0;margin-left:22px}
header nav a:hover{color:var(--c-blue)}
header nav a[aria-current="page"]{color:var(--c-navy)}
/* footer */
footer{border-top:1px solid var(--c-line);padding:24px 0 48px;font-size:12px;color:var(--c-ink-soft);margin-top:64px}
footer .wrap{max-width:1000px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
footer a{color:var(--c-blue);text-decoration:none}
footer a:hover{text-decoration:underline;text-underline-offset:3px}
/* pills */
.pill{display:inline-flex;align-items:center;gap:8px;font-size:11px;font-weight:800;
  letter-spacing:.06em;text-transform:uppercase;color:var(--c-navy);padding:4px 12px;border-radius:999px;background:var(--c-panel)}
.pill .dot{width:8px;height:8px;border-radius:50%;flex:none}
.pill-run{background:#FDE7E7}.pill-run .dot{background:var(--c-run)}
.pill-bike{background:#FCEED8}.pill-bike .dot{background:var(--c-bike)}
"""

_HUB_CSS = """
.acad-hero{background:linear-gradient(135deg,#14294B 0%,#1D3E77 60%,#2F6DE0 100%);color:#fff;
  position:relative;overflow:hidden}
.acad-hero .wrap{max-width:1000px;padding:56px 24px 60px;position:relative;z-index:1}
.acad-hero .eyebrow{color:rgba(255,255,255,.75)}
.acad-hero h1{font-family:var(--f-display);font-weight:900;font-style:italic;text-transform:uppercase;
  font-size:clamp(32px,5vw,52px);line-height:1;letter-spacing:-.01em;margin:14px 0 14px}
.acad-hero p{max-width:560px;font-size:17px;color:rgba(255,255,255,.9)}
.hub{max-width:1000px;margin:0 auto;padding:44px 24px 8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:20px}
.acard{display:flex;flex-direction:column;border:1px solid var(--c-line);border-radius:var(--radius);
  background:#fff;box-shadow:var(--shadow);overflow:hidden;text-decoration:none;color:inherit;transition:.15s}
.acard:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px)}
.acard .stripe{height:5px}
.acard.run .stripe{background:var(--c-run)}
.acard.bike .stripe{background:var(--c-bike)}
.acard .body{padding:20px 22px 22px;display:flex;flex-direction:column;gap:10px;flex:1}
.acard .cat{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--c-ink-soft)}
.acard h2{font-family:var(--f-display);font-weight:800;font-size:20px;line-height:1.15;color:var(--c-navy)}
.acard p{font-size:14px;color:var(--c-ink-soft);line-height:1.5;flex:1}
.acard .meta{font-family:var(--f-mono);font-size:12px;color:var(--c-ink-faint);margin-top:4px}
.acard.feat{grid-column:1/-1;flex-direction:row;align-items:stretch}
.acard.feat .stripe{height:auto;width:6px;flex:none}
.acard.feat h2{font-size:26px}
.empty{color:var(--c-ink-soft);padding:24px 0}
@media(max-width:640px){.acard.feat{flex-direction:column}.acard.feat .stripe{width:auto;height:5px}}
"""

_ARTICLE_CSS = """
.article-wrap{max-width:760px;margin:0 auto;padding:36px 24px 16px}
.crumbs{font-size:13px;color:var(--c-ink-soft);margin-bottom:20px}
.crumbs a{color:var(--c-ink-soft);text-decoration:none}
.crumbs a:hover{color:var(--c-blue)}
.crumbs .sep{margin:0 8px;color:var(--c-line)}
.article-head{margin-bottom:8px}
.article-head .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.article-head .rt{font-family:var(--f-mono);font-size:12px;color:var(--c-ink-faint)}
.article-head h1{font-family:var(--f-display);font-weight:900;font-style:italic;text-transform:uppercase;
  font-size:clamp(28px,4.5vw,42px);line-height:1.02;letter-spacing:-.01em;color:var(--c-navy)}
.article-head .lede{font-size:19px;line-height:1.5;color:var(--c-ink-soft);margin-top:16px}
.article-head + .speedline{margin:22px 0 30px}
/* article body typography */
.prose{font-size:17px;line-height:1.7}
.prose > *:first-child{margin-top:0}
.prose h2{font-family:var(--f-display);font-weight:800;text-transform:uppercase;font-size:24px;
  color:var(--c-navy);letter-spacing:-.005em;margin:40px 0 14px;line-height:1.15}
.prose h3{font-family:var(--f-body);font-weight:800;font-size:19px;color:var(--c-navy);margin:28px 0 10px}
.prose h4{font-weight:800;font-size:16px;color:var(--c-navy);margin:22px 0 8px}
.prose p{margin:0 0 18px}
.prose ul,.prose ol{margin:0 0 18px;padding-left:24px}
.prose li{margin:0 0 8px}
.prose li::marker{color:var(--c-blue)}
.prose strong{color:var(--c-ink);font-weight:800}
.prose a{color:var(--c-blue);text-underline-offset:3px}
.prose blockquote{border-left:3px solid var(--c-blue);background:var(--c-panel-blue);
  padding:14px 20px;border-radius:0 var(--radius) var(--radius) 0;margin:0 0 20px;color:var(--c-navy);font-weight:600}
.prose code{font-family:var(--f-mono);font-size:.9em;background:var(--c-panel);padding:2px 6px;border-radius:5px;color:var(--c-navy)}
.prose pre{background:var(--c-navy);color:#EAF1FC;padding:18px 20px;border-radius:var(--radius);overflow-x:auto;margin:0 0 20px}
.prose pre code{background:none;color:inherit;padding:0}
.prose table{width:100%;border-collapse:collapse;margin:0 0 22px;font-size:15px;display:block;overflow-x:auto}
.prose th,.prose td{border:1px solid var(--c-line);padding:10px 14px;text-align:left;vertical-align:top}
.prose thead th{background:var(--c-panel);font-weight:800;color:var(--c-navy);font-size:13px;
  text-transform:uppercase;letter-spacing:.04em}
.prose tbody tr:nth-child(even){background:var(--c-panel)}
/* sources */
.sources{margin-top:44px;padding-top:28px;border-top:1px solid var(--c-line)}
.sources h2{font-family:var(--f-display);font-weight:800;text-transform:uppercase;font-size:16px;
  color:var(--c-navy);letter-spacing:.04em;margin-bottom:14px}
.sources ol{padding-left:22px;font-size:14px;color:var(--c-ink-soft);line-height:1.5}
.sources li{margin-bottom:10px}
.sources .finding{display:block;color:var(--c-ink-faint);font-style:italic;margin-top:2px}
/* back-to-hub / cta */
.article-cta{margin-top:40px;padding:26px 28px;border:1px solid var(--c-line);border-radius:var(--radius);
  background:var(--c-panel-blue);display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
.article-cta p{font-weight:700;color:var(--c-navy);font-size:16px}
.btn{font-family:var(--f-body);font-weight:800;border-radius:var(--radius-btn);text-decoration:none;
  padding:12px 22px;font-size:15px;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:var(--c-blue);color:#fff}.btn-primary:hover{background:var(--c-blue-dk)}
.backlink{display:inline-block;margin-bottom:24px;font-size:13px;font-weight:700;color:var(--c-blue);text-decoration:none}
.backlink:hover{text-decoration:underline;text-underline-offset:3px}
"""

_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,700;0,800;0,900;'
    '1,800;1,900&family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" '
    'rel="stylesheet">'
)


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _header(active: str) -> str:
    """Site header with the Academy nav link marked active on Academy pages."""
    acad_attr = ' aria-current="page"' if active == "academy" else ""
    return (
        "<header><div class=\"wrap\">"
        '<a class="wordmark" href="/">Flapp<span class="dot"></span></a>'
        '<nav aria-label="Site">'
        '<a href="/">Analyze</a>'
        f'<a href="/academy"{acad_attr}>Academy</a>'
        '<a href="/docs" target="_blank" rel="noopener">API</a>'
        "</nav></div></header>"
    )


def _footer() -> str:
    return (
        "<footer><div class=\"wrap\">"
        f"<span>{SITE_NAME} · side-view running &amp; cycling form analysis</span>"
        '<span><a href="/academy">Academy</a> · '
        '<a href="/docs" target="_blank" rel="noopener">API</a> · '
        '<a href="/health" target="_blank" rel="noopener">status</a></span>'
        "</div></footer>"
    )


def _page(
    *,
    title: str,
    description: str,
    canonical: str,
    body: str,
    extra_css: str,
    active: str,
    jsonld: dict[str, Any] | list[dict[str, Any]] | None = None,
    og_type: str = "website",
) -> str:
    """Assemble a full HTML document with SEO head, header, body, footer."""
    ld = ""
    if jsonld is not None:
        ld = (
            '<script type="application/ld+json">'
            + json.dumps(jsonld, ensure_ascii=False)
            + "</script>"
        )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>{_esc(title)}</title>\n"
        f'<meta name="description" content="{_esc(description)}">\n'
        f'<link rel="canonical" href="{_esc(canonical)}">\n'
        f'<meta property="og:type" content="{og_type}">\n'
        f'<meta property="og:title" content="{_esc(title)}">\n'
        f'<meta property="og:description" content="{_esc(description)}">\n'
        f'<meta property="og:url" content="{_esc(canonical)}">\n'
        f'<meta property="og:site_name" content="{SITE_NAME}">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f"{_FONTS_LINK}\n"
        f"<style>{_BASE_CSS}{extra_css}</style>\n"
        f"{ld}\n"
        "</head>\n<body>\n"
        f"{_header(active)}\n{body}\n{_footer()}\n"
        "</body>\n</html>"
    )


def _lede_and_rest(body_html: str) -> tuple[str, str]:
    """Pull the first ``<p>`` out as a large lede; return (lede, remaining)."""
    if body_html.startswith("<p>"):
        end = body_html.find("</p>")
        if end != -1:
            lede = body_html[3:end]
            rest = body_html[end + 4 :].lstrip("\n")
            return lede, rest
    return "", body_html


def _sport_pill(sport: str) -> str:
    label = "Running" if sport == "run" else "Cycling"
    cls = "pill-run" if sport == "run" else "pill-bike"
    return f'<span class="pill {cls}"><span class="dot"></span>{label}</span>'


# ---------------------------------------------------------------------------
# Public: render the hub and a single article
# ---------------------------------------------------------------------------
def render_hub(articles: list[ArticleMeta], base_url: str) -> str:
    canonical = f"{base_url}/academy"
    cards: list[str] = []
    for m in articles:
        feat = " feat" if m.featured else ""
        cards.append(
            f'<a class="acard {m.sport}{feat}" href="/academy/{_esc(m.slug)}">'
            '<span class="stripe"></span>'
            '<span class="body">'
            f'<span class="cat">{_esc(m.category_label)}</span>'
            f"<h2>{_esc(m.title)}</h2>"
            f"<p>{_esc(m.description)}</p>"
            f'<span class="meta">{m.read_time} min read</span>'
            "</span></a>"
        )
    grid = (
        f'<div class="grid">{"".join(cards)}</div>'
        if cards
        else '<p class="empty">No articles yet — check back soon.</p>'
    )
    body = (
        '<div class="acad-hero"><div class="wrap">'
        '<div class="eyebrow">Flapp Academy</div>'
        "<h1>Train the technique,<br>not just the miles</h1>"
        "<p>Evidence-based guides on running form, running economy, cycling "
        "position and aerodynamics — the science behind what our analyzer "
        "measures.</p>"
        "</div></div>"
        f'<main class="hub">{grid}</main>'
    )
    jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": f"{SITE_NAME} Academy",
        "url": canonical,
        "description": "Evidence-based guides on running form, running "
        "economy, cycling position and aerodynamics.",
    }
    return _page(
        title=f"Academy — {SITE_NAME} · Running & Cycling Technique Guides",
        description="Evidence-based guides on running form, running economy, "
        "cycling position and aerodynamics — the science behind good technique.",
        canonical=canonical,
        body=body,
        extra_css=_HUB_CSS,
        active="academy",
        jsonld=jsonld,
        og_type="website",
    )


def render_article(article: Article, base_url: str) -> str:
    m = article.meta
    canonical = f"{base_url}/academy/{m.slug}"
    lede, rest = _lede_and_rest(article.body_html)

    # sources
    sources_html = ""
    if article.sources:
        items = []
        for s in article.sources:
            author = _esc(str(s.get("author", "")))
            year = s.get("year")
            year_s = f" ({year})" if year else ""
            stitle = _esc(str(s.get("title", "")))
            finding = s.get("finding")
            find_s = f'<span class="finding">{_esc(str(finding))}</span>' if finding else ""
            items.append(f"<li><strong>{author}</strong>{year_s}. {stitle}{find_s}</li>")
        sources_html = (
            '<div class="sources"><h2>Sources</h2><ol>'
            + "".join(items)
            + "</ol></div>"
        )

    body = (
        '<div class="article-wrap">'
        '<a class="backlink" href="/academy">← Academy</a>'
        '<nav class="crumbs" aria-label="Breadcrumb">'
        '<a href="/">Flapp</a><span class="sep">/</span>'
        '<a href="/academy">Academy</a><span class="sep">/</span>'
        f"<span>{_esc(m.category_label)}</span>"
        "</nav>"
        '<article>'
        '<div class="article-head">'
        f'<div class="row">{_sport_pill(m.sport)}'
        f'<span class="rt">{m.read_time} min read</span></div>'
        f"<h1>{_esc(m.title)}</h1>"
        + (f'<p class="lede">{lede}</p>' if lede else "")
        + "</div>"
        '<div class="speedline"></div>'
        f'<div class="prose">{rest}</div>'
        f"{sources_html}"
        '<div class="article-cta">'
        "<p>Want this checked on your own form? Upload a side-view clip.</p>"
        '<a class="btn btn-primary" href="/">Analyze my video →</a>'
        "</div>"
        "</article>"
        "</div>"
    )

    jsonld = [
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": m.title,
            "description": m.description,
            "url": canonical,
            "mainEntityOfPage": canonical,
            "author": {"@type": "Organization", "name": SITE_NAME},
            "publisher": {"@type": "Organization", "name": SITE_NAME},
            **({"datePublished": m.published} if m.published else {}),
        },
        {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": "Academy", "item": f"{base_url}/academy"},
                {"@type": "ListItem", "position": 2, "name": m.title, "item": canonical},
            ],
        },
    ]
    return _page(
        title=f"{m.title} · {SITE_NAME} Academy",
        description=m.description,
        canonical=canonical,
        body=body,
        extra_css=_ARTICLE_CSS,
        active="academy",
        jsonld=jsonld,
        og_type="article",
    )


def render_sitemap(articles: list[ArticleMeta], base_url: str) -> str:
    urls = [
        (base_url + "/", None),
        (base_url + "/academy", None),
    ]
    for m in articles:
        urls.append((f"{base_url}/academy/{m.slug}", m.published))
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for loc, lastmod in urls:
        entry = f"<url><loc>{_esc(loc)}</loc>"
        if lastmod:
            entry += f"<lastmod>{_esc(lastmod)}</lastmod>"
        entry += "</url>"
        parts.append(entry)
    parts.append("</urlset>")
    return "".join(parts)


def render_robots(base_url: str) -> str:
    return (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {base_url}/sitemap.xml\n"
    )
