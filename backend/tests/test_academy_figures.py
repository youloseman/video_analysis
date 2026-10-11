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
        assert len(art.sources) >= 8
        assert art.body_html.count("<figure>") >= 5

    def test_every_figure_it_shows_exists(self):
        text = DRAFT.read_text(encoding="utf-8")
        srcs = re.findall(r"^!\[[^\]]*\]\((/media/[^)]+)\)$", text, flags=re.M)
        assert len(srcs) >= 5
        for src in srcs:
            assert (MEDIA / src.removeprefix("/media/")).is_file(), src

    def test_all_fifteen_positions_are_shown_on_their_photographs(self):
        """Artur's call (2026-10-11): the figures first went out drawn on a
        neutral ground; he asked for the race photographs, every one. The
        frame where the model found the motorbike passenger stays out. The
        fifteenth is Trevor Foley, a reader's photo from an earlier Kona,
        and its caption says so."""
        text = DRAFT.read_text(encoding="utf-8")
        shown = re.findall(r"\(/media/academy/kona-2026/position-(\d\d)\.webp\)", text)
        assert shown == [f"{n:02d}" for n in range(1, 16)]
        assert "Trevor Foley, from a photo a reader shared; an earlier Kona, not the 2026 race." in text
        tilts = [float(m) for m in re.findall(r"Position \d+ of 15: forearm tilt (\d+)", text)]
        assert tilts == sorted(tilts), "ordered from the flattest forearms to the steepest"

    def test_the_hand_corrections_are_disclosed(self):
        """Readers pointed out points sitting off the joints; Artur moved the
        ones that missed on all fifteen photos (2026-10-11). The article says
        it was updated, how many points moved and by how much, rings every
        moved point, and gives the before-and-after medians."""
        text = DRAFT.read_text(encoding="utf-8")
        assert text.split("---", 2)[2].lstrip().startswith("**Updated 11 October 2026.**")
        assert "36 of the 60 points moved" in text
        assert "A ringed point is one we moved into the joint by hand." in text
        assert "shoulder 84° to 87°, elbow 78° to 83°" in text

    def test_uci_is_not_presented_as_a_triathlon_rule(self):
        """UCI equipment rules govern road time trials, not Ironman, whose 2025
        rules limit extension length and not tilt (Artur, 2026-10-11)."""
        text = DRAFT.read_text(encoding="utf-8")
        assert "Those rules do not apply to Ironman." in text
        assert "the UCI limit." not in text
        assert "The rules moved first." not in text

    def test_the_numbers_in_the_text_match_the_table(self):
        text = DRAFT.read_text(encoding="utf-8")
        for row in ("| Torso, from horizontal | 20° | 19–22° | 13–25° | 20–30° |",
                    "| Shoulder, torso to upper arm | 87° | 82–89° | 74–96° | 80–105° |",
                    "| Elbow | 83° | 76–88° | 66–93° | 75–110° |",
                    "| Forearm tilt, above horizontal | 25° | 22–27° | 19–32° | 5–30° |"):
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
