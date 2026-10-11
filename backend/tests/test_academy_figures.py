"""Academy figures, the Kona 2026 draft, and the picture agreeing with its table.

The Kona article is the first Academy piece with images, so the converter
learned one block: a line holding ``![caption](/media/...)``. Only our own
/media/ paths become an <img>; an article never hotlinks someone else's
picture. The article went out as a ``.DRAFT.md`` and was published on Artur's word
(2026-10-11); every figure it points at must exist.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.services.academy.parser import CONTENT_DIR, markdown_to_html

BACKEND = Path(__file__).resolve().parents[1]
MEDIA = BACKEND / "app" / "static" / "media"
DRAFT = CONTENT_DIR / "kona-2026-pro-bike-positions.md"


class TestTheFigureBlock:
    def test_our_media_becomes_a_figure_with_its_caption(self):
        html = markdown_to_html("Before.\n![The **field** against the bands](/media/academy/kona-2026/field.webp)\nAfter.")
        assert '<figure><img src="/media/academy/kona-2026/field.webp"' in html
        assert 'alt="The field against the bands"' in html, "markdown stripped from alt"
        assert "<figcaption>The <strong>field</strong> against the bands</figcaption>" in html
        assert 'loading="lazy"' in html
        assert html.count("<p>") == 2, "the figure ends the paragraph before it"

    def test_someone_elses_image_is_never_embedded(self):
        assert "<img" not in markdown_to_html("![x](https://example.com/photo.jpg)")

    def test_an_attribute_injection_stays_text(self):
        html = markdown_to_html('![x](/media/a.webp" onerror="alert(1))')
        assert "<img" not in html and "onerror=\"" not in html


class TestTheKonaDraft:
    def test_it_is_published_with_its_sources(self):
        from app.services.academy import get_article, invalidate_cache
        invalidate_cache()
        art = get_article("kona-2026-pro-bike-positions")
        assert art is not None and art.meta.category == "bike-aero"
        assert len(art.sources) >= 7
        assert art.body_html.count("<figure>") >= 5

    def test_every_figure_it_shows_exists(self):
        text = DRAFT.read_text(encoding="utf-8")
        srcs = re.findall(r"^!\[[^\]]*\]\((/media/[^)]+)\)$", text, flags=re.M)
        assert len(srcs) >= 5
        for src in srcs:
            assert (MEDIA / src.removeprefix("/media/")).is_file(), src

    def test_all_fourteen_positions_are_shown_on_their_photographs(self):
        """Artur's call (2026-10-11): the figures first went out drawn on a
        neutral ground; he asked for the race photographs, every one. The
        frame where the model found the motorbike passenger stays out."""
        text = DRAFT.read_text(encoding="utf-8")
        shown = re.findall(r"\(/media/academy/kona-2026/position-(\d\d)\.webp\)", text)
        assert shown == [f"{n:02d}" for n in range(1, 15)]
        tilts = [float(m) for m in re.findall(r"Position \d+ of 14: forearm tilt (\d+)", text)]
        assert tilts == sorted(tilts), "ordered from the flattest forearms to the steepest"

    def test_the_one_hand_corrected_point_is_disclosed(self):
        """Position 4: the model put the shoulder on the top edge of the back
        (torso read 39 deg); moved into the joint by hand it reads 30. The
        article says so, in the caption and in the method."""
        text = DRAFT.read_text(encoding="utf-8")
        assert "Position 4 of 14: forearm tilt 21°, torso 30°" in text
        assert "moved by hand" in text and "from 39° to 30°" in text

    def test_the_numbers_in_the_text_match_the_table(self):
        text = DRAFT.read_text(encoding="utf-8")
        for row in ("| Torso, from horizontal | 20° | 17–24° | 12–32° | 20–30° |",
                    "| Shoulder, torso to upper arm | 85° | 77–88° | 71–97° | 80–105° |",
                    "| Elbow | 80° | 76–89° | 62–107° | 75–110° |",
                    "| Forearm tilt, above horizontal | 25° | 21–26° | 10–31° | 5–30° |"):
            assert row in text, row

    def test_the_article_column_cannot_be_pushed_wider_than_a_phone(self):
        css = (BACKEND / "app" / "services" / "academy" / "renderer.py").read_text(encoding="utf-8")
        i = css.index(".article-wrap{max-width")
        rule = css[i:css.index("}", i)]
        assert "width:100%" in rule and "min-width:0" in rule


class TestThePictureAgreesWithItsTable:
    def test_chips_take_the_rows_status(self):
        from app.services.video_analysis import photo_analyzer as PA
        assert PA._ROW_TO_CHIP == {
            "optimal": "good", "acceptable": "warn", "needs_work": "bad",
            "phase_dependent": "muted", "insufficient_visibility": "muted",
        }
        src = (BACKEND / "app" / "services" / "video_analysis" / "photo_analyzer.py").read_text(encoding="utf-8")
        assert 'row_status={k: v.get("status") for k, v in angles_with_context.items()}' in src
