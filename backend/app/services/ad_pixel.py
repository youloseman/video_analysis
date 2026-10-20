"""The Meta ad pixel on Flapp's public pages: asked first in Europe, notice
and opt-out elsewhere.

Added 2026-10-19 to measure paid social ads. Decisions, all Artur's:

* **Public pages only** -- the landing page, the Academy, the sample reports,
  What's new and the privacy policy (where the choice can be changed). Never
  inside /app: that is where the athlete's footage, results and account email
  live, and the app's upload screen promises "no ad trackers in the app".
* **Consent where the law asks for it, opt-out elsewhere** (changed the same
  day: the banner everywhere cost too many visits). Visitors whose browser is
  in a European time zone (EU/EEA/UK and the Atlantic islands that belong to
  them) or set to French -- the nearest browser-side proxy for Quebec, whose
  Law 25 also wants tracking off by default -- get the banner, and Meta's
  script is not requested until they press Allow. A browser that cannot say
  its time zone is treated as European: the error goes the safe way.
  Everyone else gets the pixel with the page, a notice in the privacy policy
  and an "Ad settings" link in every public footer that turns it off at once
  and for good (Canada's PIPEDA/BC PIPA accept that for non-sensitive data).
  "Do Not Track" is a no everywhere, without asking.
* **No automatic collection.** ``autoConfig`` is off, so Meta does not record
  button clicks or page metadata on its own; the pixel sends page views and
  nothing else. (Automatic advanced matching -- Meta reading email fields --
  is a switch in Events Manager, not in code; it must stay OFF there too.)

The snippet Meta hands out also carries a ``<noscript><img>`` fallback. It is
left out on purpose: it fires for every visitor without JavaScript, which is
to say without anyone having been asked.

Like analytics.py, everything here is inert when no pixel id is configured:
no banner, no script, and the privacy policy keeps its "no ad pixel" branch.
"""

from __future__ import annotations

import json

from app.core.config import settings

# localStorage key holding "yes" / "no". Same origin for every public page, so
# one answer covers the site.
CONSENT_KEY = "flapp_ad_consent"


def enabled() -> bool:
    return bool(settings.meta_pixel_id)


# Templated with plain replacement, like analytics._LOADER_JS: a `%` or a
# brace in a future edit to the JS must not break every page.
_SNIPPET = """<style>
.adc{position:fixed;left:16px;right:16px;bottom:16px;z-index:300;max-width:560px;margin:0 auto;
  background:#fff;color:#14181F;border:1px solid #D9DDE3;border-radius:12px;
  box-shadow:0 12px 32px rgba(20,24,31,.18);padding:16px 18px;
  font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.adc[hidden]{display:none}
.adc p{margin:0;flex:1 1 260px;color:#3A4250}
.adc a{color:#1F4FB0}
.adc .adc-b{display:flex;gap:8px}
.adc button{font:600 14px/1 "IBM Plex Sans",system-ui,sans-serif;padding:10px 16px;border-radius:6px;
  cursor:pointer;border:1px solid #14294B;background:#fff;color:#14294B}
.adc button:hover{background:#F3F5F8}
.adc button:focus-visible{outline:2px solid #2F6DE0;outline-offset:2px}
</style>
<script>
(function(w,d){
  var ID=__PIXEL_ID__, KEY=__KEY__;
  function get(){ try{ return w.localStorage.getItem(KEY); }catch(e){ return null; } }
  function put(v){ try{ w.localStorage.setItem(KEY,v); }catch(e){} }
  function load(){
    if(w.fbq) { w.fbq('consent','grant'); return; }
    /* Meta's base code, unchanged except that it runs only from here. */
    !function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?
    n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;
    n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;
    t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}(w,d,'script',
    'https://connect.facebook.net/en_US/fbevents.js');
    w.fbq('set','autoConfig',false,ID);
    w.fbq('init',ID);
    w.fbq('track','PageView');
  }
  var bar=null;
  function hide(){ if(bar) bar.hidden=true; }
  function show(){
    if(!bar){
      bar=d.createElement('div');
      bar.className='adc'; bar.setAttribute('role','region'); bar.setAttribute('aria-label','Ad measurement');
      bar.innerHTML='<p>May we use Meta\\'s pixel on these public pages to measure our ads? '+
        'It never runs inside the app. '+
        '<a href="/privacy#ads">Details</a></p>'+
        '<div class="adc-b"><button type="button" data-a="no">No thanks</button>'+
        '<button type="button" data-a="yes">Allow</button></div>';
      bar.addEventListener('click',function(e){
        var a=e.target.getAttribute && e.target.getAttribute('data-a');
        if(a==='yes') w.flappAds.allow(); else if(a==='no') w.flappAds.deny();
      });
      d.body.appendChild(bar);
    }
    bar.hidden=false;
  }
  w.flappAds={
    choice:function(){ return get(); },
    allow:function(){ put('yes'); hide(); load(); },
    deny:function(){ put('no'); hide(); if(w.fbq) w.fbq('consent','revoke'); },
    ask:function(){ show(); }
  };
  /* Every link marked data-ad-settings (the footers, the policy) reopens the
     choice; without this script it is a plain link to the policy section. */
  d.addEventListener('click',function(e){
    var a=e.target.closest && e.target.closest('[data-ad-settings]');
    if(!a) return;
    e.preventDefault(); show();
  });
  function asksFirst(){
    var tz='';
    try{ tz=Intl.DateTimeFormat().resolvedOptions().timeZone||''; }catch(e){}
    if(!tz) return true;
    if(tz.indexOf('Europe/')===0) return true;
    if(['Atlantic/Reykjavik','Atlantic/Canary','Atlantic/Madeira','Atlantic/Azores','Atlantic/Faroe','Arctic/Longyearbyen'].indexOf(tz)>=0) return true;
    var langs=(navigator.languages&&navigator.languages.length)?navigator.languages:[navigator.language||''];
    for(var i=0;i<langs.length;i++){ var l=String(langs[i]||'').toLowerCase(); if(l==='fr'||l.indexOf('fr-')===0) return true; }
    return false;
  }
  var dnt=(navigator.doNotTrack==='1'||w.doNotTrack==='1');
  var c=get();
  if(dnt || c==='no') return;
  if(c==='yes' || !asksFirst()){ load(); return; }
  if(d.body) show(); else d.addEventListener('DOMContentLoaded',show);
})(window,document);
</script>"""


def snippet_html() -> str:
    """The banner + consent-gated loader, or ``""`` when no pixel is set."""
    if not enabled():
        return ""
    return (
        _SNIPPET
        .replace("__PIXEL_ID__", json.dumps(settings.meta_pixel_id))
        .replace("__KEY__", json.dumps(CONSENT_KEY))
    )


def inject(html: str) -> str:
    """Put the snippet just before ``</head>``; untouched when off or headless."""
    snippet = snippet_html()
    if not snippet or "</head>" not in html:
        return html
    return html.replace("</head>", snippet + "\n</head>", 1)


_ON_OPEN, _ON_CLOSE = "<!--ADPIXEL:ON-->", "<!--/ADPIXEL:ON-->"
_OFF_OPEN, _OFF_CLOSE = "<!--ADPIXEL:OFF-->", "<!--/ADPIXEL:OFF-->"


def _strip_blocks(html: str, opener: str, closer: str) -> str:
    out, i = [], 0
    while True:
        j = html.find(opener, i)
        if j < 0:
            out.append(html[i:])
            return "".join(out)
        k = html.find(closer, j)
        if k < 0:  # unbalanced: keep the rest rather than eat the page
            out.append(html[i:])
            return "".join(out)
        out.append(html[i:j])
        i = k + len(closer)


def apply_disclosure(html: str) -> str:
    """Keep whichever privacy-policy branch is true: pixel on, or none."""
    if enabled():
        html = _strip_blocks(html, _OFF_OPEN, _OFF_CLOSE)
        return html.replace(_ON_OPEN, "").replace(_ON_CLOSE, "")
    html = _strip_blocks(html, _ON_OPEN, _ON_CLOSE)
    return html.replace(_OFF_OPEN, "").replace(_OFF_CLOSE, "")
