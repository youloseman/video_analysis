import json
from playwright.sync_api import sync_playwright
B="http://127.0.0.1:8765"
with sync_playwright() as p:
    br=p.chromium.launch(); out={}
    for vp,opts in {"desk":dict(viewport={"width":1366,"height":768}),
                    "mob":dict(viewport={"width":390,"height":844},device_scale_factor=2,is_mobile=True,has_touch=True)}.items():
        ctx=br.new_context(**opts); page=ctx.new_page()
        page.goto(B+"/app", wait_until="networkidle"); page.wait_for_timeout(400)
        page.locator("button[data-login]:visible").first.click(); page.wait_for_timeout(300)
        page.fill("#authEmail","audit-ui@example.com"); page.fill("#authPass","auditpass123")
        page.click("#authSubmit"); page.wait_for_timeout(2200)
        page.evaluate("window.scrollTo(0,0)"); page.wait_for_timeout(200)
        page.locator("#navAnalyze:visible, #botAnalyze:visible").first.click(); page.wait_for_timeout(700)
        for sport in ("bike","run"):
            page.click(f"#sportSeg [data-sport='{sport}']"); page.wait_for_timeout(400)
            out[f"{vp}_{sport}"]=page.evaluate("""() => {
              const r=s=>{const e=document.querySelector(s);
                if(!e||e.offsetParent===null) return null;
                const b=e.getBoundingClientRect();
                return {top:Math.round(b.top+scrollY), h:Math.round(b.height)};};
              return {vh:innerHeight,
                modeSeg:r('#modeSeg'), sport:r('#sportSeg'), position:r('#positionField'),
                side:r('#sideField'), drop:r('.drop'), camrow:r('#camRow'),
                profile:r('#profileField'), height:r('#heightField'),
                overlay:r('#overlayField'), focus:r('#focusField'), analyze:r('#analyze'),
                analyzeBottom: (()=>{const e=document.querySelector('#analyze');
                  return e? Math.round(e.getBoundingClientRect().bottom+scrollY):null;})()};
            }""")
        ctx.close()
    br.close(); print(json.dumps(out,indent=1))
