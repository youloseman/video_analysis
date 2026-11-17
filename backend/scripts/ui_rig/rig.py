"""Pixel rig for the SPA: the same screens, before and after a change, compared pixel by pixel.

Scratch SQLite only -- never prod. Nothing here talks to anything but 127.0.0.1.

    python scripts/ui_rig/rig.py seed          # once: out/seed.db with an athlete (4 example
                                                 analyses) and a coach (3 athletes)
    python scripts/ui_rig/rig.py shoot before  # server from THIS checkout, screenshots to out/before
    ...change the code...
    python scripts/ui_rig/rig.py shoot after
    python scripts/ui_rig/rig.py diff before after   # side-by-side crops of what moved -> out/diff-before-after

What it shoots (desktop 1366 and phone 390, full page, clock frozen at 2026-11-17 18:00 UTC,
animations off, fonts settled):
  athlete: home, history, progress, compare, profiles, account
  coach:   home (Squad), history, progress, compare, profiles, account, pricing, whatsnew, swim,
           reviews, athletes, batch, feedback
  results: every content/examples/*.json (video through render(), photo through renderPhoto())
  share cards for the bike examples; logged out: login, signup.
Run-to-run noise is text anti-aliasing (~1.4k px on some desktop shots): `diff` calls a shot
changed only above NOISE pixels or when its height moves.

Needs the backend venv (uvicorn, playwright with msedge or chromium, Pillow, numpy).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = OUT / "seed.db"
RUN = OUT / "run.db"
EXAMPLES = BACKEND / "content" / "examples"
NOISE = 6000
T0 = "Date.UTC(2026,10,17,18,0,0)"
CLOCK = """(()=>{ window.__T=%s; const R=Date;
  class D extends R{ constructor(...a){ super(...(a.length?a:[window.__T])); } static now(){ return window.__T; } }
  window.Date=D; })()""" % T0
FREEZE = "*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important}"
ATHLETE = ["home", "history", "progress", "compare", "profiles", "account"]
COACH = ATHLETE + ["pricing", "whatsnew", "swim", "reviews", "athletes", "batch", "feedback"]
PW = "Rig-pass-2026!"


class Server:
    def __init__(self, db: Path, port: int, python: str):
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, DATABASE_URL="sqlite+aiosqlite:///" + str(db).replace("\\", "/"))
        log = open(OUT / "server.log", "w")
        self.p = subprocess.Popen([python, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                                  cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(90):
            try:
                urllib.request.urlopen(self.base + "/app", timeout=2)
                return
            except Exception:
                time.sleep(1)
        self.p.kill()
        sys.exit("server did not start -- see out/server.log")

    def call(self, method, path, body=None, token=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()[:300]

    def stop(self):
        self.p.kill()


def examples() -> list[tuple[str, dict]]:
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(EXAMPLES.glob("*.json"))]


def browser(pw):
    try:
        return pw.chromium.launch(channel="msedge")
    except Exception:
        return pw.chromium.launch()


def seed(a):
    OUT.mkdir(exist_ok=True)
    SEED.unlink(missing_ok=True)
    srv = Server(SEED, a.port, a.python)
    try:
        toks = {}
        for who in ("athlete", "coach"):
            st, j = srv.call("POST", "/auth/register", {"email": f"rig-{who}@example.com", "password": PW, "accept_privacy": True})
            if st >= 300:
                sys.exit(f"register {who}: {st} {j}")
            toks[who] = j["token"]
        con = sqlite3.connect(SEED)
        con.execute("update users set tier='full' where email='rig-athlete@example.com'")
        con.execute("update users set tier='coach' where email='rig-coach@example.com'")
        con.commit()
        con.close()
        for name, sport in (("Anna Reid", "both"), ("Marc Olsen", "bike"), ("Ines Duarte", "run")):
            srv.call("POST", "/me/athletes", {"name": name, "sport": sport}, toks["coach"])
        # The athlete's history is written the way a user writes it: the app's own
        # saveToHistory, one example a day apart, so Progress has a line to draw.
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = browser(pw)
            ctx = b.new_context()
            ctx.add_init_script(CLOCK)
            ctx.add_init_script("try{localStorage.setItem('flapp_token',%s)}catch(e){}" % json.dumps(toks["athlete"]))
            pg = ctx.new_page()
            pg.goto(srv.base + "/app#home", wait_until="networkidle")
            pg.wait_for_timeout(1000)
            for i, (name, r) in enumerate(examples()):
                kind = "photo" if r.get("mode") == "photo" else "video"
                pg.evaluate("""([R, kind, back]) => { window.__T = %s - back*86400000; state.sport = R.sport_type;
                                 return saveToHistory(kind, R); }""" % T0, [r, kind, len(examples()) - i])
                pg.wait_for_timeout(600)
            b.close()
        (OUT / "tokens.json").write_text(json.dumps(toks), encoding="utf-8")
        con = sqlite3.connect(SEED)
        n = con.execute("select count(*) from analyses").fetchone()[0]
        con.close()
        print(f"seeded {SEED}: {n} analyses, coach with 3 athletes")
    finally:
        srv.stop()


def shoot(a):
    from playwright.sync_api import sync_playwright
    if not SEED.exists():
        sys.exit("run `rig.py seed` first")
    out = OUT / a.tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(SEED, RUN)
    toks = json.loads((OUT / "tokens.json").read_text(encoding="utf-8"))
    srv = Server(RUN, a.port, a.python)
    errs: list[str] = []
    try:
        with sync_playwright() as pw:
            b = browser(pw)
            for vt, vp in (("d", {"width": 1366, "height": 900}), ("p", {"width": 390, "height": 844})):
                for who in ("athlete", "coach", None):
                    ctx = b.new_context(viewport=vp, device_scale_factor=1)
                    ctx.add_init_script(CLOCK)
                    if who:
                        ctx.add_init_script("try{localStorage.setItem('flapp_token',%s)}catch(e){}" % json.dumps(toks[who]))
                    pg = ctx.new_page()
                    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))

                    def snap(name):
                        pg.add_style_tag(content=FREEZE)
                        pg.evaluate("document.fonts.ready.then(()=>1)")
                        pg.wait_for_timeout(700)
                        h = min(pg.evaluate("document.documentElement.scrollHeight"), 9000)
                        pg.set_viewport_size({"width": vp["width"], "height": h})
                        pg.wait_for_timeout(250)
                        pg.screenshot(path=str(out / f"{name}-{vt}.png"))
                        pg.set_viewport_size(vp)

                    if who is None:
                        for r in ("login", "signup"):
                            pg.goto(srv.base + "/app#" + r, wait_until="networkidle")
                            pg.reload(wait_until="networkidle")
                            pg.wait_for_timeout(1200)
                            snap("out-" + r)
                        ctx.close()
                        continue
                    for r in (ATHLETE if who == "athlete" else COACH):
                        # A hash-only goto is a same-document jump and the app reads the hash
                        # on load, so every route needs a real reload.
                        pg.goto(srv.base + "/app#" + r, wait_until="networkidle")
                        pg.reload(wait_until="networkidle")
                        pg.wait_for_timeout(1500)
                        snap(("a-" if who == "athlete" else "c-") + r)
                    if who == "coach":
                        ctx.close()
                        continue
                    for name, r in examples():
                        pg.goto(srv.base + "/app#home", wait_until="networkidle")
                        pg.wait_for_timeout(1000)
                        if r.get("mode") == "photo":
                            pg.evaluate("R => { state.sport = R.sport_type; renderPhoto(R); }", r)
                        else:
                            pg.evaluate("R => { state.sport = R.sport_type; render({result: R, job_id: 'rig'}, {}); }", r)
                        pg.wait_for_timeout(1500)
                        snap("result-" + name)
                        if vt == "d" and name.startswith("bike") and r.get("mode") != "photo":
                            b64 = pg.evaluate("""async (a) => { const blob = await buildShareCard({sport:'bike', score:a.s, grade:a.g, keyframe:null});
                                const buf = new Uint8Array(await blob.arrayBuffer()); let s=''; for (const x of buf) s += String.fromCharCode(x); return btoa(s); }""",
                                              {"s": r.get("technique_score") or 80, "g": r.get("letter_grade") or "B"})
                            (out / f"card-{name}-d.png").write_bytes(base64.b64decode(b64))
                    ctx.close()
            b.close()
    finally:
        srv.stop()
    from PIL import Image, ImageChops
    for who, home in (("a", "a-home"), ("c", "c-home")):
        for v in "dp":
            h = Image.open(out / f"{home}-{v}.png").convert("RGB")
            for f in out.glob(f"{who}-*-{v}.png"):
                if f.stem == f"{home}-{v}":
                    continue
                x = Image.open(f).convert("RGB")
                if x.size == h.size and ImageChops.difference(x, h).getbbox() is None:
                    print(f"WARNING {f.name} is identical to {home}: the route did not open")
    print(f"{a.tag}: {len(list(out.iterdir()))} shots; page errors: {errs[:5]}")


def diff(a):
    import numpy as np
    from PIL import Image, ImageChops
    A, B = OUT / a.a, OUT / a.b
    D = OUT / f"diff-{a.a}-{a.b}"
    shutil.rmtree(D, ignore_errors=True)
    D.mkdir(parents=True)
    same = changed = 0
    for f in sorted(p.name for p in A.iterdir()):
        if not (B / f).exists():
            print("MISSING", f)
            continue
        x, y = Image.open(A / f).convert("RGB"), Image.open(B / f).convert("RGB")
        if x.size != y.size:
            print(f"SIZE   {f:30} {x.size} -> {y.size}")
            changed += 1
            continue
        d = ImageChops.difference(x, y)
        n = int((np.asarray(d.convert("L")) > 0).sum())
        if n == 0 or n < NOISE:
            same += 1
            continue
        changed += 1
        x0, y0, x1, y1 = d.getbbox()
        pad = 60
        box = (max(0, x0 - pad), max(0, y0 - pad), min(x.width, x1 + pad), min(x.height, y1 + pad))
        w, h = box[2] - box[0], box[3] - box[1]
        side = Image.new("RGB", (w * 2 + 10, h), "red")
        side.paste(x.crop(box), (0, 0))
        side.paste(y.crop(box), (w + 10, 0))
        side.save(D / f)
        print(f"DIFF   {f:30} {n} px  box={box}")
    print(f"unchanged {same}, changed {changed}; crops in {D}")


def main():
    venv = BACKEND / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--python", default=str(venv) if venv.exists() else sys.executable, help="python that runs the server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    s = sub.add_parser("shoot")
    s.add_argument("tag")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"seed": seed, "shoot": shoot, "diff": diff}[a.cmd](a)


if __name__ == "__main__":
    main()
