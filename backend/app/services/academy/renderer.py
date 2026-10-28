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
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.services import ad_pixel, analytics
from app.services.academy.parser import Article, ArticleMeta
from app.services.academy.widgets import render_widgets

# Public base URL for canonical/OG tags + sitemap. Override via env in main.py.
SITE_NAME = "Flapp"
SITE_TAGLINE = "Running & Cycling Form Analysis"

# Shared design tokens + base chrome, lifted from static/index.html so the
# Academy pages match the app. Kept intentionally compact (a subset of the
# app's full stylesheet — just what article/hub pages use).
_BASE_CSS = """
/* Tokens come from /tokens.css (linked in the head) — this file used to hold
   its own copy of the palette, which is how the Academy ended up shipping
   text at 3.07:1 after the app had already fixed it. */
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--f-body);font-size:15px;line-height:1.6;color:var(--c-ink);background:var(--c-ground)}
.wrap{max-width:820px;margin:0 auto;padding:0 24px}
a{color:var(--c-blue-dk)}
/* Brand focus ring on every interactive element (mirrors the app). */
:is(a,button,input,select,summary,[tabindex]):focus-visible{outline:2px solid var(--c-blue);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
/* Speedlines are gone with the old brand; the element stays in the markup. */
.speedline{display:none}
.eyebrow{font-family:var(--f-mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:500;color:var(--c-ink-faint)}
/* ---- shared left-sidebar shell (mirrors the app in static/index.html);
       --sidebar-w comes from tokens.css, which both surfaces link ---- */
.wordmark{font-family:var(--f-brand,var(--f-display));font-weight:900;font-style:italic;font-size:22px;
  text-transform:none;letter-spacing:-.02em;color:var(--c-ink);display:flex;align-items:center;gap:9px;text-decoration:none;cursor:pointer}
.wordmark .dot{width:6px;height:6px;border-radius:50%;background:var(--c-coral);transform:translateY(-10px)}
.sidebar{position:fixed;top:0;left:0;bottom:0;width:var(--sidebar-w);z-index:40;background:var(--c-bg);
  border-right:1px solid var(--c-line);display:flex;flex-direction:column;padding:0 8px 12px;gap:0}
.sidebar > .wordmark{height:56px;padding:0 10px 0 18px;font-size:22px;border-bottom:1px solid var(--c-line);margin:0 -8px 8px;align-items:center}
.sidebar > .wordmark .dot{transform:translateY(-6px)}
.sidenav{display:flex;flex-direction:column;gap:1px;margin-top:4px;flex:1}
.navlink{position:relative;display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:var(--radius);min-height:36px;
  color:var(--c-ink-soft);text-decoration:none;font-size:13.5px;font-weight:500;transition:background-color .15s ease-out,color .15s ease-out}
.navlink svg{width:16px;height:16px;flex:none;color:var(--c-ink-faint)}
.navlink:hover{background:var(--c-panel);color:var(--c-ink)}
.navlink:focus-visible{outline:2px solid var(--c-blue);outline-offset:-2px}
.navlink[aria-current="page"]{background:var(--c-panel-blue);color:var(--c-ink)}
.navlink[aria-current="page"] svg{color:var(--c-blue)}
.navlink[aria-current="page"]::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--c-blue)}
/* [hidden] has to beat .navlink{display:flex}, or the signed-in and admin
   entries show to every visitor. */
.navlink[hidden]{display:none}
.navlink-cta{order:5;margin-top:10px;font-weight:600}
.navlink-cta svg{width:30px;height:30px;padding:7px;box-sizing:border-box;border-radius:50%;background:var(--c-action);color:var(--c-action-ink)}
.navlink-cta:hover svg{color:var(--c-action-ink)}
.sidenav .navgroup-bottom{order:10}
.wordmark .wm-tile{flex:none;width:30px;height:30px;border-radius:var(--radius-lg);background:var(--c-brand-navy);display:flex;align-items:center;justify-content:center;box-sizing:border-box}
.wordmark .wm-tile .mark{width:21px;height:17px;color:#fff}
.planline{display:block;width:100%;text-decoration:none;color:inherit;padding:2px 0}
.planline .pl-name{font-family:var(--f-mono);font-size:11px;color:var(--c-ink-faint);display:flex;justify-content:space-between;gap:8px}
.planline .pl-name span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:110px}
.planline:hover .pl-name{color:var(--c-ink)}
.navgroup-bottom{margin-top:auto;padding-top:8px;border-top:1px solid var(--c-line);display:flex;flex-direction:column;gap:1px}
/* account box (bottom of sidebar + mobile top bar), mirrors the app */
.btn{font-family:var(--f-body);font-weight:500;letter-spacing:0;border-radius:var(--radius);cursor:pointer;
  border:1px solid transparent;transition:background-color .15s ease-out,color .15s ease-out,border-color .15s ease-out;display:inline-flex;align-items:center;justify-content:center;
  gap:8px;text-decoration:none;min-height:40px;padding:9px 20px;font-size:14px}
.btn svg{width:16px;height:16px;flex:none}
.btn-primary{background:var(--c-action);color:var(--c-action-ink)}.btn-primary:hover{background:var(--c-action-dk)}
.btn-ghost{background:var(--c-bg);color:var(--c-ink);border-color:var(--c-line-strong)}.btn-ghost:hover{background:var(--c-panel);border-color:var(--c-ink)}
.btn-sm{padding:5px 12px;font-size:13px;min-height:32px}
.btn:focus-visible{outline:2px solid var(--c-blue);outline-offset:2px}
.authbox{display:flex;align-items:center;gap:10px}
.usermail{font-size:12px;font-weight:500;color:var(--c-ink-soft);max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sidebar > .authbox{margin-top:0;padding:12px 10px 4px;border-top:1px solid var(--c-line)}
.sidebar > .authbox .btn{width:100%;justify-content:center}
.topbar .authbox-m{margin-left:auto}
.appshell{margin-left:var(--sidebar-w);min-height:100vh;display:flex;flex-direction:column}
.appshell > main,.appshell > .article-wrap{flex:1}
/* mobile top bar + off-canvas */
.topbar{display:none;position:sticky;top:0;z-index:30;background:var(--c-bg);border-bottom:1px solid var(--c-line);
  align-items:center;gap:12px;height:52px;padding:0 12px 0 16px}
.topbar-toggle{width:40px;height:40px;border:0;border-radius:var(--radius);background:transparent;
  color:var(--c-ink);display:inline-flex;align-items:center;justify-content:center;cursor:pointer;flex:none;order:3}
.topbar-toggle:hover{background:var(--c-panel)}.topbar-toggle svg{width:20px;height:20px}
.topbar .wordmark{font-size:20px;order:1}
.topbar .authbox-m{order:2}
.scrim{position:fixed;inset:0;background:rgba(14,17,22,.44);z-index:35;border:0}
@media(max-width:900px){
  .topbar{display:flex}
  .appshell{margin-left:0}
  .sidebar{transform:translateX(-100%);transition:transform .15s ease-out;box-shadow:var(--shadow-lg)}
  body.nav-open .sidebar{transform:translateX(0)}
  body.nav-open .scrim{display:block}.scrim[hidden]{display:none}
}
@media(min-width:901px){.scrim{display:none!important}}
/* footer */
footer{border-top:1px solid var(--c-line);padding:20px 0 40px;font-size:11.5px;font-family:var(--f-mono);color:var(--c-ink-faint);margin-top:64px}
footer .wrap{max-width:1000px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
footer a{color:var(--c-ink-soft);text-decoration:none}
footer a:hover{text-decoration:underline;text-underline-offset:3px}
/* pills */
.pill{display:inline-flex;align-items:center;gap:7px;font-family:var(--f-mono);font-size:11px;font-weight:500;
  letter-spacing:.02em;color:var(--c-ink-soft);padding:2px 8px;border-radius:2px;border:1px solid var(--c-line-strong);background:var(--c-bg)}
.pill .dot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--c-ink-faint)}
"""

# Article cards with their covers: the hub's rows and an article's "Keep
# reading" share them. Artur, 2026-10-26: one drawn template on every card read
# as dry and the articles as interchangeable, so every guide has its own
# picture -- our footage, or Marey's plates from the 1880s (public domain).
_CARD_CSS = """
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;grid-auto-flow:dense}
.acard{display:flex;flex-direction:column;border:1px solid var(--c-line);border-radius:var(--radius-lg);min-width:0;
  background:var(--c-bg);overflow:hidden;text-decoration:none;color:inherit;transition:border-color .15s ease-out}
.acard:hover{border-color:var(--c-ink)}
.acard.wide{grid-column:span 2}
.acard.full{grid-column:1/-1;flex-direction:row}
.acard.full .media{flex:3 1 0;border-bottom:0;border-right:1px solid var(--c-line)}
.acard.full .body{flex:2 1 0;justify-content:center;padding:26px 28px}
.acard .stripe{height:3px;background:var(--c-blue)}
.media{position:relative;display:block;aspect-ratio:16/9;background:var(--c-media-bg);border-bottom:1px solid var(--c-line);overflow:hidden}
.media img{display:block;width:100%;height:100%;object-fit:cover}
.media.contain{background:#fff}
.media.contain img{object-fit:contain;padding:6px;box-sizing:border-box}
.media .src{position:absolute;left:10px;bottom:10px;font-family:var(--f-mono);font-size:10.5px;letter-spacing:.03em;
  color:#fff;background:rgba(11,17,29,.74);padding:3px 7px;border-radius:var(--radius-xs);
  max-width:calc(100% - 20px);box-sizing:border-box;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.media.contain .src{background:rgba(255,255,255,.92);color:var(--c-ink-soft);border:1px solid var(--c-line)}
.media .bdg{position:absolute;top:12px;right:14px;font-family:var(--f-mono);font-weight:500;color:#fff;font-size:32px;
  line-height:1;letter-spacing:-.02em;text-shadow:0 1px 10px rgba(0,0,0,.5)}
.media .bdg.sm{left:14px;right:auto;font-size:12px;letter-spacing:.05em;color:#DCE2EC}
.media .bdg.l,.media .bdg.r{right:auto;font-size:20px}
.media .bdg.l{left:14px}.media .bdg.r{left:calc(50% + 14px)}
.acard .body{padding:18px 20px 20px;display:flex;flex-direction:column;gap:8px;flex:1}
.acard .cat{font-family:var(--f-mono);font-size:10.5px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:var(--c-ink-faint)}
.acard h3{font-family:var(--f-display);font-weight:600;font-size:17px;line-height:1.3;color:var(--c-ink)}
.acard.wide h3,.acard.full h3{font-size:21px;line-height:1.25}
.acard p{font-size:14px;color:var(--c-ink-soft);line-height:1.5;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.acard.full p{-webkit-line-clamp:4}
.acard .meta{font-family:var(--f-mono);font-size:11.5px;color:var(--c-ink-faint);margin-top:auto;padding-top:4px}
.acard.cta{background:var(--c-brand-navy);border-color:var(--c-brand-navy);color:#fff}
.acard.cta:hover{border-color:var(--c-ink)}
.acard.cta .media{border-bottom-color:var(--c-brand-navy)}
.acard.cta .cat{color:#C9D1DE}.acard.cta h3{color:#fff}
.acard.cta .go{align-self:flex-start;margin-top:4px;background:var(--c-action);color:var(--c-action-ink);padding:9px 16px;
  border-radius:var(--radius-btn);font-weight:600;font-size:14px}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:640px){
  .grid{grid-template-columns:minmax(0,1fr)}
  .acard.wide{grid-column:auto}
  .acard.full{flex-direction:column}
  .acard.full .media{flex:none;border-right:0;border-bottom:1px solid var(--c-line)}
  .acard.full .body{padding:18px 20px 20px}
}
"""

_HUB_CSS = _CARD_CSS + """
.acad-hero{position:relative;background:#0B111D;color:#fff;overflow:hidden}
.acad-hero .bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:60% 50%;opacity:.8}
.acad-hero .veil{position:absolute;inset:0;background:linear-gradient(90deg,rgba(11,17,29,.95) 0%,rgba(11,17,29,.8) 40%,rgba(11,17,29,.15) 80%)}
.acad-hero .wrap{max-width:1000px;padding:64px 24px 26px;position:relative;z-index:1}
.acad-hero .eyebrow{color:#C9D1DE}
.acad-hero h1{font-family:var(--f-display);font-weight:600;text-transform:none;color:#fff;
  font-size:clamp(30px,4.4vw,46px);line-height:1.08;letter-spacing:-.015em;margin:12px 0 14px;max-width:620px}
.acad-hero p{max-width:540px;font-size:16px;line-height:1.55;color:#DCE2EC}
.hchips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}
.hchip{display:inline-flex;align-items:center;gap:8px;min-height:40px;padding:0 16px;border:1px solid rgba(255,255,255,.35);
  border-radius:999px;font-size:14px;font-weight:500;color:#fff;text-decoration:none;background:rgba(11,24,47,.55);
  transition:border-color .15s ease-out}
.hchip:hover{border-color:#fff}
.hchip .n{font-family:var(--f-mono);font-size:12px;color:#C9D1DE}
.acad-hero .credit{margin-top:34px;font-family:var(--f-mono);font-size:11px;color:#AEB8C8}
.hub{max-width:1000px;margin:0 auto;padding:40px 24px 8px}
.sec{margin-bottom:44px;scroll-margin-top:84px}
.sec-h{display:flex;align-items:baseline;justify-content:space-between;gap:12px;border-bottom:1px solid var(--c-line);
  padding-bottom:10px;margin-bottom:16px}
.sec-h h2{font-family:var(--f-display);font-weight:600;font-size:20px;color:var(--c-ink)}
.empty{color:var(--c-ink-soft);padding:24px 0}
@media(max-width:640px){
  .acad-hero .veil{background:rgba(11,17,29,.8)}
  .acad-hero .wrap{padding:44px 16px 22px}
  .hub{padding:32px 16px 8px}
}
"""

# The reading-progress bar: how far through the <article> the reader is.
_READ_JS = """
(function(){
  var a=document.querySelector('article'); if(!a) return;
  var bar=document.createElement('div'); bar.id='readbar'; bar.setAttribute('aria-hidden','true'); document.body.appendChild(bar);
  var t=null;
  function upd(){ t=null; var top=a.offsetTop, h=a.offsetHeight-window.innerHeight;
    var p=h>0?Math.max(0,Math.min(1,(window.scrollY-top)/h)):1; bar.style.width=(p*100).toFixed(1)+'%'; }
  window.addEventListener('scroll',function(){ if(!t) t=requestAnimationFrame(upd); },{passive:true});
  window.addEventListener('resize',upd); upd();
})();
"""

# Figures open full screen (Artur, 2026-10-25: most readers are on a phone, and a
# chart or a joint table at column width is too small to read). Tap a figure:
# it fills the screen; tap again for its own pixels, pan with a finger, the
# spot you tapped centred. The cross, the backdrop or Escape closes it and
# focus goes back to the figure.
_ZOOM_JS = """
(function(){
  var imgs=[].slice.call(document.querySelectorAll('.cover img, .prose figure img')); if(!imgs.length) return;
  var z=document.createElement('div'); z.id='zoom'; z.hidden=true;
  z.setAttribute('role','dialog'); z.setAttribute('aria-modal','true'); z.setAttribute('aria-label','Enlarged figure');
  z.innerHTML='<button type="button" class="zoom-x" aria-label="Close">\u00d7</button>'
    +'<span class="zoom-hint">Tap to zoom</span><div class="zoom-box"><img alt=""></div><p class="zoom-cap"></p>';
  document.body.appendChild(z);
  var box=z.querySelector('.zoom-box'), im=box.querySelector('img'), cap=z.querySelector('.zoom-cap'),
      hint=z.querySelector('.zoom-hint'), from=null;
  function fit(){ z.classList.remove('full'); hint.textContent='Tap to zoom'; }
  function open(img){
    from=img; im.src=img.currentSrc||img.src; im.alt=img.alt||'';
    var fc=img.closest('figure').querySelector('figcaption'); cap.textContent=fc?fc.textContent:''; cap.hidden=!fc;
    fit(); z.hidden=false; document.body.style.overflow='hidden'; z.querySelector('.zoom-x').focus();
  }
  function close(){ z.hidden=true; im.removeAttribute('src'); document.body.style.overflow=''; if(from) from.focus(); from=null; }
  imgs.forEach(function(img){
    img.tabIndex=0; img.setAttribute('role','button');
    img.setAttribute('aria-label',(img.alt||'Figure')+' \u2014 open full screen');
    img.addEventListener('click',function(){ open(img); });
    img.addEventListener('keydown',function(e){ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); open(img); } });
  });
  im.addEventListener('click',function(e){
    e.stopPropagation();
    if(z.classList.contains('full')){ fit(); return; }
    var rx=e.offsetX/im.clientWidth, ry=e.offsetY/im.clientHeight;
    z.classList.add('full'); hint.textContent='Tap to fit';
    requestAnimationFrame(function(){
      box.scrollLeft=rx*im.clientWidth-box.clientWidth/2; box.scrollTop=ry*im.clientHeight-box.clientHeight/2; });
  });
  z.addEventListener('click',function(e){ if(e.target===z||e.target===box||e.target.closest('.zoom-x')) close(); });
  document.addEventListener('keydown',function(e){ if(e.key==='Escape'&&!z.hidden) close(); });
})();
"""

_ARTICLE_CSS = """
.article-wrap{max-width:760px;margin:0 auto;padding:36px 24px 16px;
  /* .appshell is a column flex and this centres itself with auto margins, so
     without a width it shrink-wraps its content: a five-column table pushed the
     whole article 12px past a 390px phone. Same trap as the examples page. */
  width:100%;min-width:0;box-sizing:border-box}
.crumbs{font-size:13px;color:var(--c-ink-soft);margin-bottom:20px}
.crumbs a{color:var(--c-ink-soft);text-decoration:none}
.crumbs a:hover{color:var(--c-ink)}
.crumbs .sep{margin:0 8px;color:var(--c-line)}
.article-head{margin-bottom:8px}
.article-head .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.article-head .rt{font-family:var(--f-mono);font-size:12px;color:var(--c-ink-soft)}
.article-head h1{font-family:var(--f-display);font-weight:600;font-style:normal;text-transform:none;
  font-size:clamp(24px,3.4vw,32px);line-height:1.2;letter-spacing:0;color:var(--c-ink)}
/* The speedline under the title, as under every page title in the app. */
.article-head h1::after{content:"";display:block;width:72px;height:4px;border-radius:999px;background:var(--g-speed);margin-top:14px}
/* Reading progress: the speed gradient along the top of the viewport, as far
   as the reader has scrolled through the article. */
#readbar{position:fixed;top:0;left:0;height:3px;width:0;background:var(--g-speed);z-index:120;pointer-events:none}
.article-head .lede{font-size:17px;line-height:1.55;color:var(--c-ink-soft);margin-top:14px}
.article-head{padding-bottom:22px;border-bottom:1px solid var(--c-line);margin-bottom:28px}
/* article body typography */
.prose{font-size:16px;line-height:1.7}
.prose > *:first-child{margin-top:0}
.prose h2{font-family:var(--f-display);font-weight:600;text-transform:none;font-size:22px;
  color:var(--c-ink);letter-spacing:0;margin:40px 0 14px;line-height:1.25}
.prose h3{font-family:var(--f-body);font-weight:600;font-size:18px;color:var(--c-ink);margin:28px 0 10px}
.prose h4{font-weight:600;font-size:16px;color:var(--c-ink);margin:22px 0 8px}
.prose p{margin:0 0 18px}
.prose figure{margin:26px 0 28px}
.prose figure img{display:block;width:100%;height:auto;border:1px solid var(--c-line);background:#fff}
.prose figure img{cursor:zoom-in}
.prose figure img:focus-visible{outline:2px solid var(--c-blue);outline-offset:3px}
#zoom{position:fixed;inset:0;z-index:300;background:#0B111D;display:flex;flex-direction:column}
#zoom[hidden]{display:none}
#zoom .zoom-box{flex:1;min-height:0;overflow:auto;display:flex;align-items:center;justify-content:center;
  padding:60px 12px 8px;-webkit-overflow-scrolling:touch}
#zoom .zoom-box img{max-width:100%;max-height:100%;object-fit:contain;background:#fff;border-radius:4px;cursor:zoom-in}
#zoom.full .zoom-box{display:block}
#zoom.full .zoom-box img{max-width:none;max-height:none;cursor:zoom-out}
#zoom .zoom-x{position:absolute;top:10px;right:12px;z-index:1;width:44px;height:44px;border:0;border-radius:50%;
  background:rgba(255,255,255,.16);color:#fff;font-size:28px;line-height:44px;cursor:pointer}
#zoom .zoom-hint{position:absolute;top:22px;left:16px;font:12px var(--f-mono);color:#C9D1DE;letter-spacing:.04em}
#zoom .zoom-cap{margin:0 auto;max-width:760px;padding:8px 16px 16px;font-size:13px;line-height:1.45;color:#C9D1DE;text-align:center}
.prose figcaption{margin-top:8px;font-size:13.5px;line-height:1.5;color:var(--c-ink-faint)}
.prose ul,.prose ol{margin:0 0 18px;padding-left:24px}
.prose li{margin:0 0 8px}
.prose li::marker{color:var(--c-ink-faint)}
.prose strong{color:var(--c-ink);font-weight:600}
.prose a{color:var(--c-blue-dk);text-underline-offset:3px}
.prose blockquote{border-left:3px solid var(--c-blue);background:var(--c-panel-blue);
  padding:14px 20px;border-radius:var(--radius);margin:0 0 20px;color:var(--c-ink);font-weight:500}
.prose code{font-family:var(--f-mono);font-size:.9em;background:var(--c-panel);padding:2px 6px;border-radius:2px;color:var(--c-ink)}
.prose pre{background:var(--c-media-bg);color:#C6CCD6;padding:18px 20px;border-radius:var(--radius);overflow-x:auto;margin:0 0 20px}
.prose pre code{background:none;color:inherit;padding:0}
.prose table{width:100%;border-collapse:collapse;margin:0 0 22px;font-size:15px;display:block;overflow-x:auto}
.prose th,.prose td{border:1px solid var(--c-line);padding:10px 14px;text-align:left;vertical-align:top}
.prose thead th{background:var(--c-bg);font-family:var(--f-mono);font-weight:500;color:var(--c-ink-faint);font-size:11px;
  text-transform:uppercase;letter-spacing:.06em}
.prose tbody tr:nth-child(even){background:var(--c-panel)}
/* sources */
.sources{margin-top:44px;padding-top:28px;border-top:1px solid var(--c-line)}
.sources h2{font-family:var(--f-mono);font-weight:500;text-transform:uppercase;font-size:11px;
  color:var(--c-ink-faint);letter-spacing:.08em;margin-bottom:14px}
.sources ol{padding-left:22px;font-size:14px;color:var(--c-ink-soft);line-height:1.5}
.sources li{margin-bottom:10px}
.sources .finding{display:block;color:var(--c-ink-soft);font-style:italic;margin-top:2px}
/* back-to-hub / cta */
.article-cta{margin-top:40px;padding:22px 24px;border:1px solid var(--c-line);border-radius:var(--radius);
  background:var(--c-panel-blue);display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
.article-cta p{font-weight:500;color:var(--c-ink);font-size:15px}
.backlink{display:inline-flex;align-items:center;gap:6px;margin-bottom:24px;font-size:13px;font-weight:500;color:var(--c-blue-dk);text-decoration:none}
.backlink svg{width:15px;height:15px;flex:none}
.backlink:hover{text-decoration:underline;text-underline-offset:3px}
/* byline: a person, not "Flapp" */
.byline{display:flex;align-items:center;gap:12px;margin-top:20px}
.byline .av{width:40px;height:40px;border-radius:50%;background:var(--c-brand-navy);color:#fff;font-size:13px;font-weight:600;
  display:flex;align-items:center;justify-content:center;flex:none}
.byline .who{font-size:14px;line-height:1.45;color:var(--c-ink-soft)}
.byline .who strong{color:var(--c-ink);font-weight:600}
.byline .when{display:block;font-family:var(--f-mono);font-size:12px;color:var(--c-ink-faint)}
/* the cover: the same picture as the article's hub card, credited on the page */
.cover{margin:0 0 32px}
.cover .frame{border:1px solid var(--c-line);border-radius:var(--radius-lg);overflow:hidden;background:var(--c-media-bg)}
.cover.contain .frame{background:#fff;padding:8px}
.cover img{display:block;width:100%;height:auto;cursor:zoom-in}
.cover img:focus-visible{outline:2px solid var(--c-blue);outline-offset:3px}
.cover figcaption{margin-top:8px;font-size:13.5px;line-height:1.5;color:var(--c-ink-faint)}
.cover .credit{display:block;margin-top:3px;font-family:var(--f-mono);font-size:11.5px;color:var(--c-ink-faint)}
/* keep reading */
.more{margin-top:52px}
.more .sec-h{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--c-line);padding-bottom:10px;margin-bottom:16px}
.more .sec-h h2{font-family:var(--f-display);font-weight:600;font-size:20px;color:var(--c-ink)}
.more .grid{grid-template-columns:repeat(3,minmax(0,1fr))}
.more .acard h3{font-size:15.5px}
.more .acard p{display:none}
@media(max-width:640px){.more .grid{grid-template-columns:minmax(0,1fr)}}
"""

_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    # The app's faces (tokens.css): Archivo for the wordmark only, Plex Sans
    # for the UI and prose, Plex Mono for every numeral.
    '<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@1,900;0,800;0,900'
    '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" '
    'rel="stylesheet">'
    # Shared design tokens, ahead of _BASE_CSS so the Academy can still
    # override one locally if it ever needs to. This link is why the Academy
    # cannot drift from the app again: its copy of the palette used to ship
    # the quietest text tier at 3.07:1 long after the app had fixed it.
    '<link rel="stylesheet" href="/tokens.css">'
)


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


# Inline nav icons (Academy pages have no shared SVG sprite of their own).
_ICONS = {
    "analyze": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M11 8v3l2 2"/><path d="M20 20l-3.5-3.5"/></svg>',
    "book": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H20"/></svg>',
    "ext": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 5h5v5"/><path d="M19 5l-8 8"/><path d="M18 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4"/></svg>',
    "menu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
    "arr-l": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12H5M11 6l-6 6 6 6"/></svg>',
    "arr-r": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h15M13 6l6 6-6 6"/></svg>',
    "progress": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4v16h16"/><path d="M7 15l3.5-4 3 2.5L20 7"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5V12l3 2"/></svg>',
    # Same path as the SPA's #i-bulb, so the Feedback entry looks identical on
    # both documents rather than merely similar.
    "bulb": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.3 1 2.5h6c0-1.2.3-1.8 1-2.5A6 6 0 0 0 12 3z"/></svg>',
    "home": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 11l8-7 8 7"/><path d="M6 10v10h12V10"/></svg>',
    "spark": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M18 16.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/></svg>',
    # The same paths as the SPA's #i-video / #i-tag / #i-grid / #i-bell / #i-list.
    "video": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
    "tag": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 3.5h8l9 9-8 8-9-9z"/><circle cx="7.5" cy="7.5" r="1.4"/></svg>',
    "grid": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5"/></svg>',
    "bell": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8.5a6 6 0 1 0-12 0c0 6.5-2.5 8.5-2.5 8.5h17S18 15 18 8.5"/><path d="M13.8 20.5a2 2 0 0 1-3.6 0"/></svg>',
    "list": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 6h11M10 12h11M10 18h11"/><path d="M3 6.4l1.4 1.4L7 5.2"/><path d="M3 12.4l1.4 1.4L7 11.2"/><path d="M3 18.4l1.4 1.4L7 17.2"/></svg>',
    "run": '<svg viewBox="63 47 433 433" fill="currentColor"><path d="M69 63.5L63.5 69L64.5 76L69 79.5L186 79.5L191.5 74L190.5 67L186 63.5ZM69 255.5L63.5 261L64.5 268L69 271.5L202 271.5L207.5 266L206.5 259L202 255.5Z"/><path d="M230 81.5L146 117.5L139.5 124L136.5 130L135.5 142L140.5 153L147 158.5L154 161.5L165 161.5L227 134.5L230 134.5L263.5 167L194.5 260L188.5 271L185.5 282L185.5 295L187.5 304L190.5 311L198.5 322L209 330.5L258.5 362L219.5 402L213.5 412L211.5 419L210.5 425L211.5 436L216.5 448L229 462.5L235 463.5L239.5 460L240.5 453L228.5 437L226.5 429L228.5 418L230.5 414L279.5 364L279.5 358L273 351.5L214 314.5L207.5 308L203.5 301L201.5 294L201.5 283L208.5 268L280.5 172L282.5 166L281.5 163L247 128.5L233 116.5L228 116.5L162 145.5L155 144.5L151.5 140L151.5 136L156 130.5L232 97.5L236 97.5L240 99.5L294 138.5L313.5 158L395.5 252L399 254.5L406 254.5L471 218.5L477 219.5L480.5 224L480.5 230L477 233.5L402 274.5L395 273.5L363 247.5L361 246.5L355 247.5L351.5 251L350.5 254L351.5 258L383 284.5L394 290.5L408 289.5L488 245.5L495.5 235L496.5 221L491.5 211L485 205.5L477 202.5L469 202.5L455 208.5L404 236.5L322.5 144L307 128.5L245 83.5L239 81.5ZM393 95.5L382 98.5L374 102.5L360.5 115L352.5 132L351.5 150L354.5 161L358.5 169L371 182.5L388 190.5L406 191.5L417 188.5L425 184.5L438.5 172L446.5 155L447.5 137L444.5 126L440.5 118L428 104.5L411 96.5ZM394.5 112L409 112.5L420 118.5L426.5 126L431.5 139L430.5 153L424.5 164L417 170.5L404 175.5L390 174.5L379 168.5L372.5 161L367.5 148L368.5 134L374.5 123L385 114.5ZM336 225.5L328.5 232L267.5 298L267.5 304L271 308.5L321 344.5L326.5 350L328.5 354L327.5 366L323 371.5L250 422.5L247.5 425L246.5 430L251 436.5L256 437.5L260 435.5L331 385.5L339.5 377L343.5 369L344.5 352L337.5 338L287.5 300L345.5 238L346.5 236L345.5 229L341 225.5Z"/></svg>',
}


# The same paths as the SPA's #i-compare / #i-person / #i-bike / #i-mark.
_ICONS["compare"] = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 4H4v16h4"/><path d="M16 4h4v16h-4"/><path d="M12 3v18"/></svg>'
_ICONS["runner"] = (
    '<svg viewBox="50 45 383 383" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><defs><mask id="nav-run-cut" maskUnits="userSpaceOnUse" x="50" y="45" width="383" height="383"><rect x="50" y="45" width="383" height="383" fill="#fff"/><path d="M322 160 L232 232 L298 292 L200 332" fill="none" stroke="#000" stroke-width="66" stroke-linecap="round" stroke-linejoin="round"/></mask></defs><g mask="url(#nav-run-cut)"><path d="M215 250 L78 378" stroke-width="44"/><path d="M318 150 L236 97 L188 158" stroke-width="38"/><path d="M330 190 L350 222 L412 232" stroke-width="30"/></g><path d="M322 160 L232 232" stroke-width="54"/><path d="M232 232 L298 292 L200 332" stroke-width="42"/><circle cx="379" cy="113" r="40" fill="currentColor" stroke="none"/></svg>'
)
_ICONS["person"] = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="3.4"/><path d="M5.5 20a6.5 6.5 0 0 1 13 0"/></svg>'
_ICONS["bike"] = (
    '<svg viewBox="20 -10.5 258 258" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><defs><mask id="nav-bike-cut-a" maskUnits="userSpaceOnUse" x="20" y="-10.5" width="258" height="258"><rect x="20" y="-10.5" width="258" height="258" fill="#fff"/><path d="M100 82 L178 70 L186 108 L226 110" fill="none" stroke="#000" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/><path d="M112 92 L164 118 L156 176" fill="none" stroke="#000" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/></mask><mask id="nav-bike-cut-b" maskUnits="userSpaceOnUse" x="20" y="-10.5" width="258" height="258"><rect x="20" y="-10.5" width="258" height="258" fill="#fff"/><path d="M112 92 L164 118 L156 176" fill="none" stroke="#000" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/></mask></defs><g mask="url(#nav-bike-cut-a)"><circle cx="70" cy="170" r="36" stroke-width="16"/><circle cx="228" cy="170" r="36" stroke-width="16"/><path d="M70 170 L140 172" stroke-width="14"/><path d="M140 172 L112 112" stroke-width="14"/><path d="M112 112 L70 170" stroke-width="14"/><path d="M140 172 L202 118" stroke-width="14"/><path d="M202 118 L112 112" stroke-width="14"/><path d="M202 118 L228 170" stroke-width="14"/><path d="M202 118 L206 108 L232 112" stroke-width="14"/><path d="M96 106 L126 108" stroke-width="17"/><path d="M150 180 L162 180" stroke-width="14"/><circle cx="140" cy="172" r="10" fill="currentColor" stroke="none"/></g><g mask="url(#nav-bike-cut-b)"><path d="M100 82 L178 70" stroke-width="34"/><path d="M178 70 L186 108 L226 110" stroke-width="16"/></g><path d="M112 92 L164 118 L156 176" stroke-width="20"/><circle cx="222" cy="46" r="23" fill="currentColor" stroke="none"/></svg>'
)
# The lockup: the navy app-icon tile, then the name -- as in the app.
_WORDMARK = (
    '<span class="wm-tile">'
    + '<svg class="mark" viewBox="0 0 780 632" aria-hidden="true"><g transform="translate(148 -168)"><path d="M0 224 L430 224 L430 364 L150 364 L150 470 L390 470 L390 596 L150 596 L150 800 L0 800 Z" fill="currentColor" opacity=".22" transform="translate(155.5 0) skewX(-11) translate(-148 0)"/><path d="M0 224 L430 224 L430 364 L150 364 L150 470 L390 470 L390 596 L150 596 L150 800 L0 800 Z" fill="currentColor" opacity=".45" transform="translate(155.5 0) skewX(-11) translate(-74 0)"/><path d="M0 224 L430 224 L430 364 L150 364 L150 470 L390 470 L390 596 L150 596 L150 800 L0 800 Z" fill="currentColor" transform="translate(155.5 0) skewX(-11)"/><circle cx="446" cy="246" r="78" fill="#F1553F" transform="translate(155.5 0) skewX(-11)"/></g></svg>'
    + "</span>Flapp"
)


def _sidebar(active: str) -> str:
    """Left sidebar, kept in lock-step with the app's sidebar in
    ``static/index.html`` so Academy reads as the same product. The app-view
    links point back to the SPA: Progress/History deep-link via hash so the SPA
    opens that view on load. ``#authbox`` is filled in by ``_AUTH_JS``."""
    acad = ' aria-current="page"' if active == "academy" else ""
    exmp = ' aria-current="page"' if active == "examples" else ""
    chlog = ' aria-current="page"' if active == "changelog" else ""
    return (
        # mobile top bar + scrim
        '<div class="topbar">'
        '<button class="topbar-toggle" id="navToggle" type="button" aria-label="Open menu" '
        'aria-expanded="false" aria-controls="sidebar">' + _ICONS["menu"] + "</button>"
        '<a class="wordmark" href="/">' + _WORDMARK + "</a>"
        '<div class="authbox authbox-m" id="authboxM"></div>'
        "</div>"
        '<div class="scrim" id="navScrim" hidden></div>'
        # sidebar
        '<aside class="sidebar" id="sidebar" aria-label="Main">'
        '<a class="wordmark" href="/">' + _WORDMARK + "</a>"
        '<nav class="sidenav" aria-label="Site">'
        # Home = the signed-in dashboard; hidden until _AUTH_JS confirms a session
        # (mirrors the SPA, where #navHome is auth-gated). Fixes Home vanishing
        # from the menu on Academy pages.
        f'<a href="/app" class="navlink navlink-cta">{_ICONS["video"]}<span>Analyze</span></a>'
        f'<a href="/app#home" class="navlink" id="acadHome" hidden>{_ICONS["runner"]}<span>My technique</span></a>'
        f'<a href="/app#progress" class="navlink">{_ICONS["progress"]}<span>Progress</span></a>'
        f'<a href="/app#history" class="navlink">{_ICONS["clock"]}<span>History</span></a>'
        f'<a href="/app#compare" class="navlink" id="acadCompare" hidden>{_ICONS["compare"]}<span>Compare</span></a>'
        f'<a href="/app#profiles" class="navlink" id="acadProfiles" hidden>{_ICONS["bike"]}<span>Profiles</span></a>'
        # Admin-only, hidden until _AUTH_JS confirms the tier -- same treatment as
        # Home above. Without these the menu visibly LOSES entries the moment you
        # open an Academy page, which reads as the app breaking rather than as
        # two different documents sharing a look.
        f'<a href="/app#feedback" class="navlink" id="acadFeedback" hidden>{_ICONS["bulb"]}<span>Feedback</span></a>'
        f'<a href="/app#reviews" class="navlink" id="acadReviews" hidden>{_ICONS["list"]}<span>Reviews</span></a>'
        '<div class="navgroup-bottom">'
        f'<a href="/app#pricing" class="navlink">{_ICONS["tag"]}<span>Pricing</span></a>'
        f'<a href="/examples" class="navlink"{exmp}>{_ICONS["grid"]}<span>Examples</span></a>'
        f'<a href="/academy" class="navlink"{acad}>{_ICONS["book"]}<span>Academy</span></a>'
        f'<a href="/changelog" class="navlink"{chlog}>{_ICONS["bell"]}<span>What\'s new</span></a>'
        f'<a href="/app#account" class="navlink" id="acadAccount" hidden>{_ICONS["person"]}<span>Account</span></a>'
        "</div>"
        "</nav>"
        '<div class="authbox" id="authbox"></div>'
        "</aside>"
    )


def _guest_nav(active: str) -> str:
    """The landing page's top menu, for visitors who are not signed in.

    Artur, 2026-10-24: a visitor who clicks "See 4 real reports" on the
    landing page arrives here, and the menu they were reading (The report,
    Bike fit, ... Log in, Analyze free) used to vanish into the app's
    sidebar -- Progress and History, empty for them. Visitors keep the
    site's menu; signed-in athletes keep the app's sidebar. Which one shows
    is decided by the ``signed-in`` class on <html> (see ``_SHELL_PICK`` and
    ``_AUTH_JS``). Same items and order as landing.html's nav.
    """
    acad = ' aria-current="page"' if active == "academy" else ""
    return (
        '<nav class="gnav" aria-label="Site">'
        '<div class="gnav-inner">'
        '<a class="wordmark" href="/">' + _WORDMARK + "</a>"
        '<div class="gnav-links">'
        '<a href="/#report">The report</a>'
        '<a href="/#proof">Bike fit</a>'
        '<a href="/#how">How it works</a>'
        '<a href="/app#pricing">Pricing</a>'
        f'<a href="/academy"{acad}>Academy</a>'
        "</div>"
        '<div class="gnav-cta">'
        '<a class="gnav-login" href="/app#login">Log in</a>'
        '<a class="gnav-go" href="/app">Analyze free</a>'
        "</div></div></nav>"
    )


# Runs in <head>, before anything paints: a stored session means the app's
# sidebar, no session means the site's top menu. _AUTH_JS corrects it once
# /auth/me answers (an expired token drops back to the guest menu).
_SHELL_PICK = (
    "<script>try{if(localStorage.getItem('flapp_token'))"
    "document.documentElement.classList.add('signed-in');}catch(e){}</script>"
)

_GUEST_CSS = (
    "/* ---- guest vs signed-in shell (see _guest_nav) ---- */"
    ".gnav{position:sticky;top:0;z-index:50;background:#0B182F;padding:10px 16px}"
    ".signed-in .gnav{display:none}"
    "html:not(.signed-in) .sidebar,html:not(.signed-in) .topbar,"
    "html:not(.signed-in) .scrim{display:none!important}"
    "html:not(.signed-in) .appshell{margin-left:0}"
    ".gnav-inner{max-width:920px;margin:0 auto;height:56px;display:flex;align-items:center;gap:26px;"
    "padding:0 8px 0 18px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:999px}"
    ".gnav .wordmark{color:#fff;font-size:22px}"
    ".gnav .wm-tile{border:1px solid rgba(255,255,255,.18)}"
    ".gnav-links{display:flex;gap:22px}"
    ".gnav-links a,.gnav-login{color:rgba(255,255,255,.8);text-decoration:none;font-weight:700;font-size:14px;"
    "transition:color .15s ease-out}"
    ".gnav-links a:hover,.gnav-links a[aria-current],.gnav-login:hover{color:#fff}"
    ".gnav-cta{margin-left:auto;display:flex;align-items:center;gap:12px}"
    ".gnav-go{background:var(--c-action);color:var(--c-action-ink);padding:10px 20px;border-radius:999px;"
    "font-weight:600;font-size:14px;text-decoration:none;transition:background-color .15s ease-out}"
    ".gnav-go:hover{background:var(--c-action-dk)}"
    "@media(max-width:760px){.gnav-links,.gnav-login{display:none}.gnav-inner{height:52px}}"
)


def _footer() -> str:
    return (
        "<footer><div class=\"wrap\">"
        f"<span>{SITE_NAME} · side-view running &amp; cycling form analysis</span>"
        '<span><a href="/academy">Academy</a> · '
        '<a href="/privacy">Privacy</a> · '
        '<a href="/terms">Terms</a> · '
        # Reopens the ad-pixel choice; only while the pixel is on.
        + ('<a href="/privacy#ads" data-ad-settings>Ad settings</a> · '
           if ad_pixel.enabled() else "")
        + '<a href="/docs" target="_blank" rel="noopener">API</a> · '
        '<a href="/health" target="_blank" rel="noopener">status</a></span>'
        "</div></footer>"
    )


# Tiny script: mobile drawer toggle (open/close, scrim, Escape).
_NAV_JS = (
    "var t=document.getElementById('navToggle'),s=document.getElementById('navScrim');"
    "function o(){document.body.classList.add('nav-open');t.setAttribute('aria-expanded','true');s.hidden=false;}"
    "function c(){document.body.classList.remove('nav-open');t.setAttribute('aria-expanded','false');s.hidden=true;}"
    "if(t){t.addEventListener('click',function(){document.body.classList.contains('nav-open')?c():o();});"
    "s.addEventListener('click',c);"
    "document.addEventListener('keydown',function(e){if(e.key==='Escape'&&document.body.classList.contains('nav-open'))c();});}"
)

# Account box: mirror the app's renderAuth on these static pages. Anonymous
# visitors (no token -> no fetch) get a "Log in" button that deep-links to the
# SPA's login modal (/app#login); signed-in visitors see their email + "Log out".
_AUTH_JS = (
    "(function(){"
    "var boxes=[document.getElementById('authbox'),document.getElementById('authboxM')]"
    ".filter(Boolean);if(!boxes.length)return;"
    "function esc(x){return String(x).replace(/[&<>\"]/g,function(c){"
    "return{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c];});}"
    "function render(a){"
    "['acadHome','acadCompare','acadProfiles','acadAccount'].forEach(function(id){"
    "var e=document.getElementById(id);if(e)e.hidden=!a;});"
    "var adm=!!(a&&a.tier==='admin');"
    "['acadFeedback','acadReviews'].forEach(function(id){"
    "var e=document.getElementById(id);if(e)e.hidden=!adm;});"
    "var plan={starter:'Starter',enthusiast:'Enthusiast',full:'Full',admin:'Admin'};"
    # Signed in: the app's plan line (Log out lives on Account). `ph-no-capture`
    # keeps the address out of analytics autocapture.
    "var side=a?'<a href=\"/app#account\" class=\"planline\"><span class=\"pl-name\"><span>'"
    "+esc(plan[a.tier]||'Starter')+'</span><span class=\"ph-no-capture\" title=\"'+esc(a.email)+'\">'"
    "+esc(a.email.split('@')[0])+'</span></span></a>':"
    "'<a class=\"btn btn-ghost btn-sm\" href=\"/app#login\">Log in</a>';"
    "var m=a?'':side;"
    "boxes.forEach(function(b){b.innerHTML=b.id==='authbox'?side:m;});"
    "}"
    "var tk=null;try{tk=localStorage.getItem('flapp_token');}catch(e){}"
    "render(null);"
    "if(tk){fetch('/auth/me',{headers:{Authorization:'Bearer '+tk}})"
    ".then(function(r){return r.ok?r.json():null;})"
    ".then(function(u){if(u)render(u);"
    "else document.documentElement.classList.remove('signed-in');})"
    ".catch(function(){});}"
    "})();"
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
    extra_js: str = "",
    image: str | None = None,
    image_alt: str = "",
) -> str:
    """Assemble a full HTML document with SEO head, header, body, footer.

    ``image`` is the absolute URL of the page's picture: a shared link, Google
    Discover and an AI answer all show it, and without one they show nothing
    or somebody else's."""
    pic = ""
    if image:
        size = _image_size(image.split("/media/", 1)[-1] if "/media/" in image else "")
        pic = (
            f'<meta property="og:image" content="{_esc(image)}">\n'
            + (f'<meta property="og:image:width" content="{size[0]}">\n'
               f'<meta property="og:image:height" content="{size[1]}">\n' if size else "")
            + (f'<meta property="og:image:alt" content="{_esc(image_alt)}">\n' if image_alt else "")
            + f'<meta name="twitter:image" content="{_esc(image)}">\n'
        )
    ld = ""
    if jsonld is not None:
        ld = (
            '<script type="application/ld+json">'
            + json.dumps(jsonld, ensure_ascii=False)
            + "</script>"
        )
    # Always include the mobile-nav + account-box scripts; append widget JS after.
    script = f"<script>{_NAV_JS}{_AUTH_JS}{extra_js}</script>\n"
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"{_SHELL_PICK}\n"
        f"<title>{_esc(title)}</title>\n"
        f'<meta name="description" content="{_esc(description)}">\n'
        f'<link rel="canonical" href="{_esc(canonical)}">\n'
        f'<meta property="og:type" content="{og_type}">\n'
        f'<meta property="og:title" content="{_esc(title)}">\n'
        f'<meta property="og:description" content="{_esc(description)}">\n'
        f'<meta property="og:url" content="{_esc(canonical)}">\n'
        f'<meta property="og:site_name" content="{SITE_NAME}">\n'
        f"{pic}"
        '<meta name="twitter:card" content="summary_large_image">\n'
        # Discover shows a large picture only when the page allows it.
        '<meta name="robots" content="max-image-preview:large">\n'
        f"{_FONTS_LINK}\n"
        f"<style>{_BASE_CSS}{_GUEST_CSS}{extra_css}</style>\n"
        f"{ld}\n"
        # Academy is the top of the funnel -- these pages are what search
        # brings people in on, so leaving them unmeasured would hide the one
        # traffic source that is not us telling people about ourselves.
        f"{analytics.snippet_html()}\n"
        # Public pages, so the consent-gated ad pixel too (sample reports and
        # What's new share this shell). See services/ad_pixel.py.
        f"{ad_pixel.snippet_html()}\n"
        "</head>\n<body>\n"
        f'{_guest_nav(active)}\n'
        f'{_sidebar(active)}\n<div class="appshell">\n{body}\n{_footer()}\n</div>\n'
        f"{script}"
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


# The byline. Search and answer engines weigh who wrote a page; "Flapp" as the
# author of a study a person measured by hand undersold it.
AUTHOR = "Artur Podgornyi"
AUTHOR_ROLE = "triathlete, founder of Flapp"

MEDIA_DIR = Path(__file__).resolve().parents[2] / "static" / "media"
HERO = "/media/academy/covers/marey-runner-1883.webp"
HERO_ALT = ("Chronophotograph by Étienne-Jules Marey, about 1883: one runner photographed "
            "many times on a single plate, every phase of the stride at once")
HERO_CREDIT = "Photo: É.-J. Marey, c. 1883 · Wellcome Collection · CC BY 4.0"
CTA_PICTURE = "/media/report-frame.webp"


@lru_cache(maxsize=64)
def _image_size(rel: str) -> tuple[int, int] | None:
    """(width, height) of a file under static/media, for width/height
    attributes (no layout shift) and og:image:width/height."""
    if not rel:
        return None
    try:
        from PIL import Image

        with Image.open(MEDIA_DIR / rel.removeprefix("/media/")) as im:
            return im.size
    except Exception:  # noqa: BLE001 - a missing size must never break a page
        return None


def _wh(src: str) -> str:
    size = _image_size(src)
    return f' width="{size[0]}" height="{size[1]}"' if size else ""


def _date(iso: str | None) -> str:
    """2026-07-15 -> 15 Jul 2026."""
    try:
        d = date.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return ""
    return f"{d.day} {d.strftime('%b')} {d.year}"


def _badge(text: str) -> str:
    if not text:
        return ""
    if "|" in text:  # a split picture: one label over each half
        left, right = (p.strip() for p in text.split("|", 1))
        return f'<span class="bdg l">{_esc(left)}</span><span class="bdg r">{_esc(right)}</span>'
    return f'<span class="bdg{" sm" if len(text) > 8 else ""}">{_esc(text)}</span>'


def _card(m: ArticleMeta, *, size: str | None = None) -> str:
    """One article card: its cover, who the picture belongs to, the title."""
    cls = size if size is not None else m.card
    media = '<span class="stripe"></span>'
    if m.cover:
        fit = " contain" if m.cover_fit == "contain" else ""
        src_chip = f'<span class="src">{_esc(m.cover_credit)}</span>' if m.cover_credit else ""
        media = (
            f'<span class="media{fit}"><img src="{_esc(m.cover)}" alt="{_esc(m.cover_alt)}"{_wh(m.cover)} '
            f'loading="lazy" decoding="async">{_badge(m.cover_badge)}{src_chip}</span>'
        )
    when = _date(m.published)
    return (
        f'<a class="acard {m.sport}{" " + cls if cls else ""}" href="/academy/{_esc(m.slug)}">'
        f"{media}"
        '<span class="body">'
        f'<span class="cat">{_esc(m.category_label)} · {m.read_time} min</span>'
        f"<h3>{_esc(m.title)}</h3>"
        f"<p>{_esc(m.description)}</p>"
        + (f'<span class="meta">{when}</span>' if when else "")
        + "</span></a>"
    )


def _cta_card() -> str:
    return (
        '<a class="acard cta wide" href="/app">'
        f'<span class="media"><img src="{CTA_PICTURE}" alt="A Flapp report frame: a triathlete on a trainer '
        f'with knee, hip, trunk and elbow angles drawn over the video"{_wh(CTA_PICTURE)} loading="lazy" '
        'decoding="async" style="object-position:center 58%"></span>'
        '<span class="body"><span class="cat">Your clip</span>'
        "<h3>Every angle in these guides, measured on your own video.</h3>"
        '<span class="go">Analyze free</span></span></a>'
    )


# ---------------------------------------------------------------------------
# Public: render the hub and a single article
# ---------------------------------------------------------------------------
def render_hub(articles: list[ArticleMeta], base_url: str) -> str:
    canonical = f"{base_url}/academy"
    # Studies built on our own data first, then the guides by sport.
    rows = [
        ("research", "Research", "Original research", "Our data",
         [m for m in articles if m.kind == "research"]),
        ("running", "Running", "Running form", "Guides",
         [m for m in articles if m.kind != "research" and m.sport == "run"]),
        ("cycling", "Cycling", "Bike position and aero", "Guides",
         [m for m in articles if m.kind != "research" and m.sport == "bike"]),
    ]
    sections: list[str] = []
    chips: list[str] = []
    for key, chip, title, label, metas in rows:
        if not metas:
            continue
        cards = "".join(_card(m) for m in metas)
        if key == "cycling":
            cards += _cta_card()
        n = len(metas)
        sections.append(
            f'<section class="sec" id="{key}" aria-labelledby="h-{key}">'
            f'<div class="sec-h"><h2 id="h-{key}">{title}</h2>'
            f'<span class="eyebrow">{label} · {n}</span></div>'
            f'<div class="grid">{cards}</div></section>'
        )
        chips.append(f'<a class="hchip" href="#{key}">{chip} <span class="n">{n}</span></a>')
    grid = "".join(sections) or '<p class="empty">No articles yet — check back soon.</p>'
    body = (
        '<header class="acad-hero">'
        f'<img class="bg" src="{HERO}" alt="{_esc(HERO_ALT)}"{_wh(HERO)} fetchpriority="high">'
        '<div class="veil"></div><div class="wrap">'
        '<div class="eyebrow">Flapp Academy</div>'
        "<h1>Running form and bike position, measured</h1>"
        "<p>In 1883 Étienne-Jules Marey photographed runners many times a second "
        "to see what the eye misses. We measure the same angles from your phone "
        "video. These guides explain what the numbers mean, and the studies show "
        "what we found.</p>"
        f'<nav class="hchips" aria-label="Sections">{"".join(chips)}</nav>'
        f'<div class="credit">{_esc(HERO_CREDIT)}</div>'
        "</div></header>"
        f'<main class="hub">{grid}</main>'
    )
    jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": f"{SITE_NAME} Academy",
        "url": canonical,
        "description": "Evidence-based guides on running form, running "
        "economy, cycling position and aerodynamics.",
        "image": base_url + HERO,
        "mainEntity": {
            "@type": "ItemList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": i + 1,
                    "url": f"{base_url}/academy/{m.slug}",
                    "name": m.title,
                    **({"image": base_url + m.cover} if m.cover else {}),
                }
                for i, m in enumerate(articles)
            ],
        },
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
        image=base_url + HERO,
        image_alt=HERO_ALT,
    )


def render_article(article: Article, base_url: str) -> str:
    m = article.meta
    canonical = f"{base_url}/academy/{m.slug}"
    # Swap any [[widget:...]] placeholders for real markup + collect their
    # CSS/JS. Runs before the lede split so a widget can never be the lede.
    body_html, widget_css, widget_js = render_widgets(article.body_html)
    lede, rest = _lede_and_rest(body_html)

    # The cover. A study whose cover is one of its own figures shows it once,
    # at the top, with that figure's caption.
    cover_html = ""
    if m.cover:
        caption = ""
        fig = re.search(r'<figure><img src="' + re.escape(m.cover) + r'"[^>]*>(?:<figcaption>(.*?)</figcaption>)?</figure>\n?',
                        rest, flags=re.S)
        if fig:
            caption = fig.group(1) or ""
            rest = rest[: fig.start()] + rest[fig.end():]
        credit = f'<span class="credit">{_esc(m.cover_credit)}</span>' if m.cover_credit else ""
        cap = f"<figcaption>{caption}{credit}</figcaption>" if caption or credit else ""
        cover_html = (
            f'<figure class="cover{" contain" if m.cover_fit == "contain" else ""}"><div class="frame">'
            f'<img src="{_esc(m.cover)}" alt="{_esc(m.cover_alt)}"{_wh(m.cover)} fetchpriority="high">'
            f"</div>{cap}</figure>"
        )

    when = _date(m.published)
    upd = _date(m.updated) if m.updated and m.updated != m.published else ""
    byline = (
        '<div class="byline"><span class="av" aria-hidden="true">AP</span>'
        f'<span class="who"><strong>{AUTHOR}</strong> · {AUTHOR_ROLE}'
        + (f'<span class="when">Published {when}{" · Updated " + upd if upd else ""}</span>' if when else "")
        + "</span></div>"
    )

    # Keep reading: the same sport first, then the rest; never this article.
    from app.services.academy.parser import get_all_articles

    others = [a for a in get_all_articles() if a.slug != m.slug]
    others.sort(key=lambda a: a.sport != m.sport)
    more = others[:3]
    more_html = (
        '<section class="more" aria-labelledby="h-more"><div class="sec-h"><h2 id="h-more">Keep reading</h2>'
        '<a class="eyebrow" href="/academy">All articles</a></div>'
        f'<div class="grid">{"".join(_card(a, size="") for a in more)}</div></section>'
        if more else ""
    )

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
        f'<a class="backlink" href="/academy">{_ICONS["arr-l"]}Academy</a>'
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
        + byline
        + "</div>"
        + cover_html
        + '<div class="speedline"></div>'
        f'<div class="prose">{rest}</div>'
        f"{sources_html}"
        '<div class="article-cta">'
        "<p>Want this checked on your own form? Upload a side-view clip.</p>"
        f'<a class="btn btn-primary" href="/">Analyze my video{_ICONS["arr-r"]}</a>'
        "</div>"
        "</article>"
        f"{more_html}"
        "</div>"
    )

    image = base_url + m.cover if m.cover else None
    jsonld = [
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": m.title,
            "description": m.description,
            "url": canonical,
            "mainEntityOfPage": canonical,
            "author": {"@type": "Person", "name": AUTHOR, "jobTitle": "Founder, Flapp"},
            "publisher": {"@type": "Organization", "name": SITE_NAME, "url": base_url},
            **({"image": [image]} if image else {}),
            **({"datePublished": m.published} if m.published else {}),
            **({"dateModified": m.updated or m.published} if (m.updated or m.published) else {}),
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
        extra_css=_ARTICLE_CSS + _CARD_CSS + widget_css,
        active="academy",
        jsonld=jsonld,
        og_type="article",
        extra_js=widget_js + _READ_JS + _ZOOM_JS,
        image=image,
        image_alt=m.cover_alt,
    )


def render_sitemap(articles: list[ArticleMeta], base_url: str) -> str:
    urls = [
        (base_url + "/", None),
        (base_url + "/academy", None),
        (base_url + "/changelog", None),
    ]
    # Sample reports. Imported here rather than duplicated: a page that is not
    # in the sitemap is a page search does not know about, which for the one
    # surface built to be found before signup would be the whole point missed.
    try:
        from app.services.examples import render_sitemap_entries

        urls += [(u, None) for u in render_sitemap_entries(base_url)]
    except Exception:  # noqa: BLE001 - a missing sample must not break robots
        pass
    images = {f"{base_url}/academy": [base_url + HERO]}
    for m in articles:
        loc = f"{base_url}/academy/{m.slug}"
        urls.append((loc, m.updated or m.published))
        if m.cover:
            images[loc] = [base_url + m.cover]
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
             'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">']
    for loc, lastmod in urls:
        entry = f"<url><loc>{_esc(loc)}</loc>"
        if lastmod:
            entry += f"<lastmod>{_esc(lastmod)}</lastmod>"
        for img in images.get(loc, []):
            entry += f"<image:image><image:loc>{_esc(img)}</image:loc></image:image>"
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
