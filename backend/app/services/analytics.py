"""Product analytics (PostHog). Optional at runtime, like every other
integration here.

Why this exists at all: every question that decides what to build next --
how many people who land on the marketing page ever upload a clip, how many
who see a locked report open the pricing screen, which of the two sports is
actually used -- is currently answered by guessing. The server log knows a
clip was analysed; it does not know whether the athlete then closed the tab.

Two halves, deliberately:

* **In the browser** (:func:`snippet_html`) -- pageviews, the analyse funnel,
  paywall and checkout intent. This is where the drop-offs are, and none of
  them reach the server.
* **On the server** (:func:`capture` / :func:`capture_bg`) -- the events that
  must not be lost to a closed tab, a blocked script or a failed redirect.
  Money is the whole list: Stripe tells *us* a payment succeeded, and that is
  the only trustworthy account of it.

Both are inert without ``POSTHOG_KEY``: no snippet is injected, no request
leaves the process. Local development and a self-hosted copy stay silent
unless somebody opts in, and the test suite never phones home.

Sending is BLOCKING (an HTTPS round trip), same as ``notify.py`` -- so no
request handler may call :func:`capture` directly. Use :func:`capture_bg`,
which hands it to a thread and forgets about it. Analytics may never slow a
response down, and may *never* fail one: every error here is swallowed and
logged at debug level. A metric nobody sees is worth strictly less than the
purchase it would otherwise have 500'd.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Short on purpose. This runs in a worker thread on a single-instance service;
# a hung analytics socket must not tie one up for a minute.
TIMEOUT_S = 5
# Same reasoning as notify.py: a stdlib default User-Agent gets bot-challenged
# by more edges than you would expect.
USER_AGENT = "Flapp/1.0 (+https://getflapp.com)"


def enabled() -> bool:
    """Whether anything is actually being recorded."""
    return settings.analytics_enabled


def person_id(user: Any) -> str:
    """The PostHog person id for a signed-in athlete.

    The database id, not the email: it is stable across an address change and
    it keeps the identifier itself out of the analytics vendor's key space
    (the email travels as a person *property*, which we can delete on request
    without orphaning the history).

    The browser computes this from the same account payload -- see
    ``UserOut.id`` in api/auth.py. The two must agree, or a purchase recorded
    server-side lands on a different person than the funnel that led to it.
    """
    return f"u{user.id}"


# --------------------------------------------------------------------------
# Browser side
# --------------------------------------------------------------------------
def _assets_host(host: str) -> str:
    """Where ``array.js`` is served from for a given ingestion host.

    PostHog Cloud serves the library from a sibling host
    (``us.i.posthog.com`` -> ``us-assets.i.posthog.com``); a reverse proxy or a
    self-hosted instance serves it from itself.
    """
    if ".i.posthog.com" in host:
        return host.replace(".i.posthog.com", "-assets.i.posthog.com")
    return host


# The loader. Deliberately hand-written rather than the vendor's minified
# stub, because it has to survive two things the stub does not cover:
#
#  * being blocked -- ad-blockers eat requests to posthog.com, and roughly a
#    third of a cycling audience runs one. `onerror` then makes every call a
#    no-op instead of leaving `posthog.capture` undefined for the page to trip
#    over.
#  * being called before it loads -- the app fires events during boot, so
#    calls are queued (bounded) and flushed on init.
#
# Everything the app touches goes through `window.flappAnalytics`, which is
# absent entirely when analytics is off -- so every call site is already
# written to no-op, with or without a key.
#
# Templated with plain string replacement, not %-formatting: a stray `%` in a
# future edit to the JS would otherwise turn every page into a 500.
_LOADER_JS = """
(function(w,d){
  var KEY=__PH_KEY__, HOST=__PH_HOST__, ASSETS=__PH_ASSETS__, REC=__PH_REC__;
  var queue=[], ready=false, dead=false;
  function run(fn){
    if(dead) return;
    if(ready){ try{ fn(w.posthog); }catch(e){} return; }
    if(queue.length<50) queue.push(fn);
  }
  w.flappAnalytics={
    capture:function(name,props){ run(function(p){ p.capture(name, props||{}); }); },
    identify:function(id,props){ run(function(p){ p.identify(String(id), props||{}); }); },
    register:function(props){ run(function(p){ p.register(props||{}); }); },
    reset:function(){ run(function(p){ p.reset(); }); },
    optOut:function(){ run(function(p){ p.opt_out_capturing(); }); },
    optIn:function(){ run(function(p){ p.opt_in_capturing(); }); },
    optedOut:function(){
      try{ return !!(w.posthog && w.posthog.has_opted_out_capturing
        && w.posthog.has_opted_out_capturing()); }catch(e){ return false; }
    }
  };
  var s=d.createElement('script');
  s.src=ASSETS+'/static/array.js'; s.async=true; s.crossOrigin='anonymous';
  s.onerror=function(){ dead=true; queue.length=0; };
  s.onload=function(){
    if(!w.posthog||!w.posthog.init){ dead=true; queue.length=0; return; }
    w.posthog.init(KEY,{
      api_host:HOST,
      /* Anonymous visitors are counted but get no person profile: cheaper, and
         nothing about a passer-by is worth storing under a name. */
      person_profiles:'identified_only',
      /* One pageview per document load. NOT 'history_change': the app pushes
         history entries on the SAME url as it moves between views, which would
         bill each of them as another view of /app. In-app navigation is
         reported as `view_opened` instead. */
      capture_pageview:true,
      capture_pageleave:true,
      autocapture:true,
      respect_dnt:true,
      /* Replay records the DOM, and on a results page that includes the
         athlete's own footage. Off unless POSTHOG_SESSION_RECORDING says
         otherwise, and masked when on. */
      disable_session_recording:!REC,
      session_recording:{maskAllInputs:true},
      persistence:'localStorage+cookie',
      loaded:function(){
        ready=true;
        var q=queue; queue=[];
        for(var i=0;i<q.length;i++){ try{ q[i](w.posthog); }catch(e){} }
      }
    });
  };
  (d.head||d.documentElement).appendChild(s);
})(window,document);
""".strip()


@lru_cache(maxsize=1)
def snippet_html() -> str:
    """The ``<script>`` block to drop into every page, or ``""`` when off.

    Cached: ``settings`` is frozen and read once at import, so this is the same
    string for the life of the process -- and it is concatenated onto a 400 KB
    document on every uncached request.
    """
    if not enabled():
        return ""
    body = _LOADER_JS
    for token, value in (
        ("__PH_KEY__", json.dumps(settings.posthog_key)),
        ("__PH_HOST__", json.dumps(settings.posthog_host)),
        ("__PH_ASSETS__", json.dumps(_assets_host(settings.posthog_host))),
        ("__PH_REC__", "true" if settings.posthog_session_recording else "false"),
    ):
        body = body.replace(token, value)
    return f"<script>{body}</script>"


# The privacy policy has to describe the service as it is actually running,
# and analytics is a deploy-time switch -- so the page carries both branches
# and the server keeps the true one. A policy that claims we run PostHog when
# POSTHOG_KEY was never set is wrong in the same way as one that denies it.
_ON_OPEN, _ON_CLOSE = "<!--ANALYTICS:ON-->", "<!--/ANALYTICS:ON-->"
_OFF_OPEN, _OFF_CLOSE = "<!--ANALYTICS:OFF-->", "<!--/ANALYTICS:OFF-->"


def _strip_blocks(html: str, opener: str, closer: str) -> str:
    """Remove every ``opener … closer`` block, markers included."""
    out = html
    while True:
        start = out.find(opener)
        if start == -1:
            return out
        end = out.find(closer, start)
        if end == -1:                     # unbalanced marker: drop just the tag
            return out.replace(opener, "")
        out = out[:start] + out[end + len(closer):]


def apply_disclosure(html: str) -> str:
    """Keep whichever of the two privacy-policy branches is true right now."""
    dead, live = (_OFF_OPEN, _OFF_CLOSE) if enabled() else (_ON_OPEN, _ON_CLOSE)
    out = _strip_blocks(html, dead, live)
    for marker in (_ON_OPEN, _ON_CLOSE, _OFF_OPEN, _OFF_CLOSE):
        out = out.replace(marker, "")
    return out


def inject(html: str) -> str:
    """Put the snippet in ``html`` just before ``</head>``.

    A page with no ``</head>`` (or with analytics off) is returned untouched --
    this is a helper for whole documents, and a partial one is not worth an
    exception.
    """
    snippet = snippet_html()
    if not snippet or "</head>" not in html:
        return html
    return html.replace("</head>", snippet + "\n</head>", 1)


# --------------------------------------------------------------------------
# Server side
# --------------------------------------------------------------------------
def _endpoint() -> str:
    return f"{settings.posthog_host}/i/v0/e/"


def capture(
    distinct_id: str,
    event: str,
    properties: dict[str, Any] | None = None,
) -> bool:
    """Record one event. BLOCKING -- call from a thread, not a handler.

    Returns whether it was accepted, for tests and for the caller that wants to
    log a miss. Never raises: an analytics outage is not an outage.
    """
    if not enabled():
        return False
    props = dict(properties or {})
    # Everything sent from here is about a known account, so the person profile
    # is wanted -- unlike the browser, which stays anonymous until sign-in.
    props.setdefault("$process_person_profile", True)
    props.setdefault("source", "server")
    payload = json.dumps({
        "api_key": settings.posthog_key,
        "event": event,
        "distinct_id": distinct_id,
        "properties": props,
    }).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(),
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            ok = 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError) as e:  # noqa: BLE001
        logger.debug("ANALYTICS_CAPTURE_FAILED", analytics_event=event, err=str(e))
        return False
    if not ok:
        logger.debug("ANALYTICS_CAPTURE_REJECTED", analytics_event=event)
    return ok


def capture_bg(
    distinct_id: str,
    event: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget :func:`capture` from async code.

    The default executor keeps the future alive, so nothing has to be awaited
    or held on to. Outside a running loop (a script, a test) it falls back to
    sending inline, which is the behaviour that makes it testable.
    """
    if not enabled():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        capture(distinct_id, event, properties)
        return
    loop.run_in_executor(None, capture, distinct_id, event, properties)


def log_analytics_configuration() -> None:
    """Say in the deploy log whether anything is being measured.

    Same class of problem as billing and email: an unset key looks exactly like
    a working integration until somebody opens an empty dashboard a week later
    and cannot tell whether the product has no users or no analytics.
    """
    if not enabled():
        logger.info(
            "ANALYTICS_DISABLED",
            detail="POSTHOG_KEY unset -- no product analytics is being collected",
        )
        return
    logger.info(
        "ANALYTICS_READY",
        host=settings.posthog_host,
        session_recording=settings.posthog_session_recording,
    )
