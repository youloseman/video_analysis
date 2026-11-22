"""The report chat and the bike fields, as the app ships them.

String checks, like the other SPA tests: what matters here is wiring that a
Python test can see -- the block exists on both screens, the free plan is
sent to the plans instead of a box that answers 402, and the profile form
sends the fields the server stores.
"""

from __future__ import annotations

import re

from conftest import read_app_css, read_spa

SPA = read_spa()
CSS = read_app_css()


def _fn(name: str) -> str:
    m = re.search(r"(?:async )?function " + name + r"\(.*?\n}\n", SPA, re.S)
    assert m, f"{name} is gone"
    return m.group(0)


def test_both_screens_have_a_chat_block():
    assert 'id="chatBlock"' in SPA and 'id="hdChatBlock"' in SPA
    assert "renderChat('chatBlock'" in SPA
    assert "renderChat('hdChatBlock', e)" in SPA


def test_the_chat_sits_after_the_report_not_inside_it():
    assert SPA.index('id="exportBlock"') < SPA.index('id="chatBlock"') < SPA.index('id="rawBlock"')
    assert SPA.index('id="hdExportBlock"') < SPA.index('id="hdChatBlock"') < SPA.index('id="hdNotesOffer"')


def test_only_a_report_read_in_full_offers_the_chat():
    body = _fn("renderChat")
    assert "e.access==='full'" in body
    # The server decides who has it (services/report_chat.chat_state). A $4
    # unlock is full access on a free plan: told which plan has the chat, not
    # handed a box the server refuses.
    assert "auth.report_chat" in body
    assert "const paid=mode==='open';" in body
    assert "openPricing()" in body


def test_a_gated_fresh_report_clears_the_previous_chat():
    i = SPA.index("const entryId=saveToHistory('video', r);")
    assert "renderChat('chatBlock', null);" in SPA[i:i + 200]


def test_the_question_goes_to_the_saved_report():
    body = _fn("chatSend")
    assert "meUrl(e,'chat')" in body
    assert "method:'POST'" in body
    # A failed question goes back in the box rather than vanishing.
    assert "ta.value=q" in body


def test_the_profile_form_sends_what_the_bike_is():
    for field in ("bike_family", "bike_model", "frame_size", "bike_year"):
        assert field + ":" in SPA, field
    for el in ("profFamily", "profModel", "profSize", "profYear"):
        assert f'id="{el}"' in SPA
    assert "PROFILES.bike_families" in _fn("profFamilyOptions")


def test_chat_styles_use_tokens():
    block = CSS[CSS.index("/* Ask about this report"):]
    block = block[:block.index("@media")]
    assert "border-radius:var(--radius" in block
    assert "#" not in re.sub(r"/\*.*?\*/", "", block), "a literal colour in the chat styles"


def test_a_cda_measurement_says_what_it_tested():
    assert "variable:(q('Var')&&q('Var').value)||null" in SPA
    assert "PROFILES.cda_variables" in _fn("cdaVarLabel")


def test_the_setup_table_shows_the_servers_difference_not_its_own():
    body = _fn("cdaVsCell")
    assert "r.vs_baseline" in body and "d.watts" in body
    assert "<th scope=\"col\">Vs baseline</th>" in SPA


def test_the_ladder_is_drawn_on_both_setup_views():
    assert "${cdaLadderHtml(p.ladder)}" in SPA
    assert "${cdaLadderHtml(s.ladder)}" in SPA


def test_parts_cards_come_from_the_server_and_say_what_does_not_fit():
    panel = _fn("partsResultHtml")
    assert "r.incompatible" in panel and "Does not fit this frame" in panel
    assert "Check with the seller before you buy" in panel
    card = _fn("partsCardHtml")
    assert "p.evidence" in card and "p.status_label" in card
    assert "data-partrep" in card


def test_the_generation_is_picked_from_the_data():
    assert 'id="profGen"' in SPA
    assert "bike_generation:$('#profGen').value||null" in SPA
    assert "/me/parts/frames" in _fn("loadPartsFrames")


def test_home_asks_about_the_latest_report_without_a_second_coral_button():
    """Artur 2026-11-22: the chat on the dashboard too, under What to fix,
    on the latest full report; Home keeps Analyze a clip as its one coral."""
    s = SPA
    assert 'id="dashChatBlock"' in s
    body = s.split("function renderDashChat", 1)[1][:1200]
    assert "auth.report_chat==='open'" in body and "quiet:true" in body and "max:2" in body
    assert "opts&&opts.quiet?'btn-ghost':'btn-primary'" in s



def test_progress_asks_about_the_series_the_page_draws():
    s = SPA
    assert 'id="progChatBlock"' in s
    body = s.split("function renderProgChat", 1)[1][:1500]
    assert "/me/progress/chat" in body and "linkedAthleteId" in body and "max:3" in body
    assert "renderProgChat(all, sport);" in s
    # the chat stays plain: no coach-review link under it (2026-11-22)
    assert "data-chathuman" not in s
