"""Saving what the app builds (Artur 2026-11-21): on an iPhone the result card
goes straight into Photos, and the PDF report leaves the app through the share
sheet. A WebView ignores <a download> on a blob, so in the app both did nothing."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPA = (ROOT / "backend" / "app" / "static" / "index.html").read_text(encoding="utf-8")
BRIDGE = (ROOT / "mobile" / "bridge" / "bridge.js").read_text(encoding="utf-8")
PKG = json.loads((ROOT / "mobile" / "package.json").read_text(encoding="utf-8"))


def test_the_app_ships_the_plugins_it_calls():
    deps = PKG["dependencies"]
    assert "@capacitor/filesystem" in deps and "@capacitor-community/media" in deps and "@capacitor/share" in deps


def test_the_bridge_puts_an_image_in_photos_on_ios_and_shares_the_rest():
    save = BRIDGE[BRIDGE.index("function saveFile("):BRIDGE.index("function shareFile(")]
    assert "kind === 'image' && ios" in save and "Media.savePhoto({ path: dataUrl })" in save
    # a failed save falls back to the share sheet rather than doing nothing
    assert "shareViaFile(dataUrl, name, title)" in save
    via = BRIDGE[BRIDGE.index("function shareViaFile("):BRIDGE.index("function saveFile(")]
    assert "directory: 'CACHE'" in via and "files: [w.uri]" in via
    assert re.search(r"saveFile: saveFile,\s*shareFile: shareFile", BRIDGE)


def test_the_card_modal_saves_through_one_saver_not_a_download_link():
    modal = SPA[SPA.index('id="cardModal"'):SPA.index("</div>\n</div>", SPA.index('id="cardModal"'))]
    assert 'id="cardSaveBtn"' in modal and " download" not in modal
    assert "saveBlob(file, file.name, 'image')" in SPA
    saver = SPA[SPA.index("async function saveBlob("):SPA.index("function openCardModal(")]
    # the app first, then the phone's share sheet, then a download
    assert saver.index("nativeSave()") < saver.index("webSheetFor(file)") < saver.index("a.download=name")


def test_the_pdf_goes_through_the_same_saver():
    pdf = SPA[SPA.index("async function downloadReportPdf("):SPA.index("function printWaitHtml(")]
    assert "saveBlob(new File([blob], name, {type:'application/pdf'}), name, 'pdf')" in pdf
    assert "a.download" not in pdf
