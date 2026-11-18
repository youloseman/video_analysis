"""What's new lists each release as date, audience, title and one line
(board: WhatsNew). The line and the audience come from two optional
frontmatter lines in each changelog file."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.changelog import parser
from app.services.changelog.parser import AUDIENCES, _first_sentence, _parse_file


def test_every_entry_has_a_short_line_and_a_known_audience():
    for e in parser.get_entries():
        assert e.summary, e.id
        assert len(e.summary) <= 110, (e.id, len(e.summary))
        assert "—" not in e.summary, e.id
        assert e.audience and set(e.audience) <= set(AUDIENCES), (e.id, e.audience)


def test_frontmatter_is_not_part_of_the_note(tmp_path: Path):
    p = tmp_path / "2026-11-20-a-thing.md"
    p.write_text("---\naudience: bike, Run, nonsense\nsummary: One line.\n---\n# A thing\n\nBody text.\n", encoding="utf-8")
    e = _parse_file(p)
    assert e.title == "A thing"
    assert e.summary == "One line."
    assert e.audience == ("bike", "run")
    assert "audience" not in e.body_html and "Body text." in e.body_html


def test_an_entry_without_frontmatter_still_gets_a_line(tmp_path: Path):
    p = tmp_path / "2026-11-20-plain.md"
    p.write_text("# Plain\n\nA **bold** start [here](/x). Second sentence.\n", encoding="utf-8")
    e = _parse_file(p)
    assert e.audience == ("everyone",)
    assert e.summary == "A bold start here."
    assert _first_sentence("- a list only\n") == ""


def test_the_json_carries_the_line_and_the_audience():
    j = TestClient(app).get("/changelog.json").json()
    first = j["entries"][0]
    assert first["summary"] and isinstance(first["audience"], list)
    assert "html" in first and first["id"] == j["latest"]


def test_the_public_page_does_not_print_the_frontmatter():
    html = TestClient(app).get("/changelog").text
    assert "audience:" not in html and "summary:" not in html
