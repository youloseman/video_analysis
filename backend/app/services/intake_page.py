"""The page behind an athlete's intake link (/a/{token}).

A person who has never heard of Flapp opens this from a coach's message, on a
phone, probably next to the bike. So it says whose it is, how to film in four
short lines, and asks for one file. The verdict on the framing comes back while
they are still by the camera, on the coach's own bars.

Server-rendered, one self-contained document: it has no business loading the app.
"""

from __future__ import annotations

from html import escape
from typing import Any

_POSITIONS = (
    ("triathlon", "Triathlon / aero"), ("tt_aero", "Time trial"), ("road_hoods", "Road, hoods"),
    ("road_drops", "Road, drops"), ("casual", "Upright"),
)

_STYLE = """
*{box-sizing:border-box}
body{margin:0;background:var(--c-ground,#F4F2EE);color:var(--c-ink);font-family:var(--f-body);font-size:16px;line-height:1.5}
main{max-width:520px;margin:0 auto;padding:28px 16px 48px}
.who{display:flex;align-items:center;gap:12px;margin:0 0 20px}
.who img{height:44px;width:auto;max-width:150px}
.who b{display:block;font-family:var(--f-display);font-weight:600;color:var(--c-head)}
.who small{display:block;color:var(--c-ink-faint);font-size:13px}
h1{font-family:var(--f-display);font-weight:600;font-size:22px;line-height:1.25;color:var(--c-head);margin:0 0 6px}
.lede{margin:0 0 18px;color:var(--c-ink-soft)}
.card{background:var(--c-bg);border:1px solid var(--c-line);border-radius:var(--radius);padding:16px;margin:0 0 14px}
.tips{margin:0;padding-left:20px}.tips li{margin:3px 0}
label.f{display:block;font-family:var(--f-mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--c-ink-faint);margin:12px 0 4px}
label.f:first-child{margin-top:0}
select,input[type=file]{width:100%;font:inherit;padding:10px;border:1px solid var(--c-line-strong);border-radius:var(--radius);background:var(--c-bg);color:var(--c-ink)}
.ok{display:flex;gap:10px;align-items:flex-start;margin:14px 0;font-size:14px}
.ok input{margin-top:4px;width:18px;height:18px;flex:none}
button{width:100%;font:inherit;font-weight:600;padding:13px;border:0;border-radius:var(--radius-btn);background:var(--c-action);color:var(--c-action-ink);cursor:pointer}
button[disabled]{opacity:.55;cursor:default}
.bar{height:6px;background:var(--c-line);border-radius:3px;overflow:hidden;margin:12px 0 0}.bar i{display:block;height:100%;width:0;background:var(--c-blue)}
.msg{margin:12px 0 0;font-size:14.5px}.msg.bad{color:var(--c-coral-btn)}
.foot{font-size:12.5px;color:var(--c-ink-faint);margin:18px 0 0}.foot a{color:inherit}
.hidden{display:none}
"""

_SCRIPT = """
(function(){
  var f=document.getElementById('f'); if(!f) return;
  var token=f.dataset.token, sport=document.getElementById('sport'), posBox=document.getElementById('posBox');
  var go=document.getElementById('go'), msg=document.getElementById('msg'), bar=document.getElementById('bar'), fill=bar.firstElementChild;
  function say(t,bad){ msg.textContent=t; msg.className='msg'+(bad?' bad':''); }
  function sync(){ posBox.classList.toggle('hidden', sport.value!=='bike'); }
  sport.addEventListener('change',sync); sync();
  function watch(id){
    var tries=0;
    (function tick(){
      fetch('/a/'+encodeURIComponent(token)+'/clips/'+encodeURIComponent(id)).then(function(r){return r.ok?r.json():null;}).then(function(j){
        tries++;
        if(j&&j.status==='no_go'){ say((j.reasons&&j.reasons[0]||'This clip needs filming again.')+' Film again and send the new one.',true); go.disabled=false; go.textContent='Send another clip'; return; }
        if(j&&j.status==='failed'){ say('We could not read this clip. Film it again, side-on, with the whole body in shot.',true); go.disabled=false; go.textContent='Send another clip'; return; }
        if(j&&(j.status==='running'||j.status==='done')){ say('Looks good. Your coach has it.'); go.disabled=false; go.textContent='Send another clip'; return; }
        if(tries>200){ say('It is in line. Your coach will have it soon; you can close this page.'); go.disabled=false; go.textContent='Send another clip'; return; }
        setTimeout(tick,3000);
      }).catch(function(){ setTimeout(tick,5000); });
    })();
  }
  f.addEventListener('submit',function(e){
    e.preventDefault();
    var file=f.video.files[0];
    if(!file){ say('Pick a clip first.',true); return; }
    if(!f.accept.checked){ say('Tick the box so your coach can see this clip.',true); return; }
    var d=new FormData(); d.append('video',file); d.append('sport',sport.value); d.append('accept','true');
    if(sport.value==='bike') d.append('position',f.position.value);
    var x=new XMLHttpRequest(); x.open('POST','/a/'+encodeURIComponent(token)+'/clip');
    go.disabled=true; bar.classList.remove('hidden'); fill.style.width='0'; say('Sending…');
    x.upload.onprogress=function(ev){ if(ev.lengthComputable) fill.style.width=Math.round(100*ev.loaded/ev.total)+'%'; };
    x.onload=function(){
      bar.classList.add('hidden');
      var j={}; try{ j=JSON.parse(x.responseText); }catch(_e){}
      if(x.status>=200&&x.status<300){ say('Sent. Checking the framing…'); f.video.value=''; watch(j.id); }
      else { say(typeof j.detail==='string'?j.detail:'That did not go through. Please try again.',true); go.disabled=false; }
    };
    x.onerror=function(){ bar.classList.add('hidden'); say('The connection dropped. Please try again.',true); go.disabled=false; };
    x.send(d);
  });
})();
"""


def _doc(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{escape(title)}</title><meta name="robots" content="noindex">'
        '<meta name="color-scheme" content="light">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">'
        f'<link rel="stylesheet" href="/tokens.css"><style>{_STYLE}</style></head><body data-theme="light"><main>{body}</main></body></html>'
    )


def render_intake_page(ctx: dict[str, Any] | None) -> str:
    """``None`` is a link that is not live. It does not say why: a rotated link
    and a guessed one look the same from here, and should."""
    if ctx is None:
        return _doc("This link isn't active", (
            "<h1>This link isn't active</h1>"
            '<p class="lede">Ask your coach for a new one.</p>'
        ))
    coach, athlete = escape(ctx["coach"]), escape(ctx["athlete"])
    logo = ctx.get("logo") or ""
    head = (
        '<div class="who">'
        + (f'<img src="{escape(logo, quote=True)}" alt="">' if logo.startswith("data:image/") else "")
        + f'<div><b>{coach}</b>'
        + (f'<small>{escape(ctx.get("tagline") or "")}</small>' if ctx.get("tagline") else "")
        + "</div></div>"
    )
    if ctx.get("linked"):
        return _doc(f"Your clip for {ctx['coach']}", head + (
            f"<h1>Hi {athlete}</h1>"
            f'<p class="lede">Your Flapp account is already linked to {coach}. '
            "Film and upload in the app, and they will see it.</p>"
            '<div class="card"><a href="/app">Open Flapp</a></div>'
        ))
    sport = ctx.get("sport")
    options = "".join(
        f'<option value="{v}"{" selected" if v == sport else ""}>{label}</option>'
        for v, label in (("bike", "Bike"), ("run", "Run"))
    )
    positions = "".join(f'<option value="{v}">{label}</option>' for v, label in _POSITIONS)
    token = escape(ctx["token"], quote=True)
    return _doc(f"Send your clip to {ctx['coach']}", head + (
        f"<h1>{athlete}, send your clip to {coach}</h1>"
        '<p class="lede">No account needed. It takes a minute to send, and you will know right away if it needs filming again.</p>'
        '<div class="card"><ul class="tips">'
        "<li>Film from the side, square on, with the camera at hip height.</li>"
        "<li>Fill most of the frame, with your whole body in shot.</li>"
        "<li>Keep the camera still for 10 to 15 seconds.</li>"
        "<li>On a bike, film the side without the chainring.</li>"
        "</ul></div>"
        f'<form class="card" id="f" data-token="{token}">'
        f'<label class="f" for="sport">What is it</label><select id="sport">{options}</select>'
        f'<div id="posBox"><label class="f" for="position">Position</label><select id="position" name="position">{positions}</select></div>'
        '<label class="f" for="video">Your clip</label><input id="video" name="video" type="file" accept="video/*">'
        '<label class="ok"><input type="checkbox" name="accept">'
        f"<span>I want {coach} to see this clip and the report made from it. They keep it in their account.</span></label>"
        '<button id="go" type="submit">Send to my coach</button>'
        '<div class="bar hidden" id="bar"><i></i></div><p class="msg" id="msg" role="status" aria-live="polite"></p>'
        "</form>"
        '<p class="foot">Measured by Flapp. Your clip is kept for 60 days, then deleted. '
        '<a href="/privacy#coach-upload">How this works</a>.</p>'
        f"<script>{_SCRIPT}</script>"
    ))
