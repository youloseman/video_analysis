"""Metric pictograms on the readings (Artur 2026-11-22): Findings, the Home
sport card and What to fix lead each reading with the house rider or runner,
the logo's dot on the joint it measures, in the row's status colour. A fix
that is one change of one part reads as a chip; the sentence stays for a
screen reader."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.css").read_text(encoding="utf-8")


def test_every_figure_the_code_names_is_in_the_sprite():
    keys = set(re.findall(r"'([a-z_]+)'\]", SPA[SPA.index("const PM_RX="):SPA.index("function metricPic(")]))
    keys -= {"KNEE", "TRUNK"}
    keys |= {"knee_run", "knee_bdc", "lean", "trunk"}
    keys |= set(re.findall(r"'([a-z_]+)'", SPA[SPA.index("const PM_RUN="):SPA.index("const PM_RX=")]))
    for k in keys:
        assert f'<symbol id="pm-{k}" ' in SPA, k
    for base in set(re.findall(r'<use href="#(pmb-\d+)"/>', SPA)):
        assert f'<symbol id="{base}" ' in SPA, base


def test_a_run_reading_never_gets_a_bike_figure():
    fn = SPA[SPA.index("function pmKey("):SPA.index("function metricPic(")]
    assert "return k&&run===PM_RUN.has(k)?k:null;" in fn


def test_findings_lead_every_reading_with_its_pictogram():
    assert SPA.count('<span class="fd-name">${metricPic(m.lbl,') == 3      # photo, fresh clip, saved report
    assert '<span class="fd-name">${esc(m.lbl)}</span>' not in SPA


def test_the_home_card_keeps_the_words_for_a_screen_reader():
    li = SPA[SPA.index("const li=({m,st})=>"):SPA.index("const list=[...shown")]
    assert "metricPic(m.lbl,sport)" in li and "pmBar(m.range)" in li
    assert '<span class="visually-hidden">· ${st.say||DSC_WORD[st.cls]}</span>' in li


def test_every_chip_belongs_to_a_sentence_the_table_has():
    table = SPA[SPA.index("function fixHowTable(e){"):SPA.index("function fixHow(e, key, dir)")]
    chips = SPA[SPA.index("const FIX_CHIP={"):SPA.index("function fixChip(")]
    for sport_block in re.findall(r"(?:bike|aero|run):\{(.*?)\}\},?\n", chips, re.S):
        for key in re.findall(r"(\w+):\{(?:low|high):\[", sport_block):
            assert re.search(rf"\b{key}:\{{", table), key


def test_a_chip_keeps_the_full_sentence_for_a_screen_reader():
    row = SPA[SPA.index("const pic=i.fx==='fix'?metricPic("):SPA.index("return `<label class=\"dplan-item fx-item")]
    assert 'class="fx-chip" aria-hidden="true"' in row
    assert '<span class="visually-hidden">${esc(i.how)}</span>' in row


def test_the_dot_takes_the_status_colour_from_the_row():
    for rule in (".fd-row.m-out .pm{--pm-dot:var(--m-out)}", ".dsc-rows .m-out .pm{--pm-dot:var(--m-out)}",
                 ".dsc-rows .m-near .pm{--pm-dot:var(--c-ink-faint)}", ".fx-out .pm{--pm-dot:var(--m-out)}"):
        assert rule in APP, rule


def test_progress_cards_lead_with_the_reading_in_its_current_status():
    i = SPA.index("const stNow=Object.fromEntries(formTiles(sport,[lastE])")
    block = SPA[i:SPA.index("const group=(cls,h,sub,list)=>", i)]
    assert "measStatus(t).cls" in block
    assert '<div class="tc-head">${pic(x)}' in block and "${pic(x)}<span class=\"pcf-l\">" in block
    assert ".pg-st.m-out .pm{--pm-dot:var(--m-out)}" in APP
