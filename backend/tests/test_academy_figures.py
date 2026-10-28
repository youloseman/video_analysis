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
# Article positions whose frame is a still from video (kona2026.json ``video``).
# Shown until 2026-10-14, when Artur took every frame out of the article.
VIDEO_FRAMES: tuple[int, ...] = ()
# Article positions on Scott Flathouse's race photographs, shown again since
# 2026-10-28 (Artur) with his credit on every one: the frame carries it, the
# caption says it, and the gallery links his Instagram and his events page.
SCOTT_FRAMES: tuple[int, ...] = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19,
                                 23, 24, 26, 28, 30, 31, 32)
# Riders A, B and C, shown inline in sections 1 and 3.
INLINE_FRAMES: tuple[int, ...] = (1, 23, 14)


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
        # Our chart, the app's table, riders A-C and Scott's photographs.
        assert art.body_html.count("<figure>") == 2 + len(INLINE_FRAMES) + len(SCOTT_FRAMES) + len(VIDEO_FRAMES)

    def test_every_figure_it_shows_exists(self):
        text = DRAFT.read_text(encoding="utf-8")
        srcs = re.findall(r"^!\[[^\]]*\]\((/media/[^)]+)\)$", text, flags=re.M)
        pos = lambda n: f"/media/academy/kona-2026/position-{n:02d}.webp"  # noqa: E731
        assert srcs == (["/media/academy/kona-2026/field.webp", pos(1), pos(23), pos(14),
                         "/media/academy/kona-2026/rider-c-table.webp"]
                        + [pos(n) for n in sorted(SCOTT_FRAMES + VIDEO_FRAMES)])
        for src in srcs:
            assert (MEDIA / src.removeprefix("/media/")).is_file(), src

    def test_every_position_is_shown_on_its_photograph(self):
        """Artur's call (2026-10-11): the figures first went out drawn on a
        neutral ground; he asked for the race photographs, every one. The
        frame where the model found the motorbike passenger stays out, and so
        does the age-grouper (kept for a pro-vs-age-group comparison). One is
        Trevor Foley, a reader's photo from an earlier Kona, and its caption
        says so. Twenty-one since 2026-10-12; thirty-two since the broadcast stills came in and the
        duplicates went out,
        every one shown (Artur, 2026-10-12: the stills go in too; the article
        says how many are stills). Then Slowtwitch asked for its photographs to
        come down (2026-10-13): no race photograph is published any more, and
        the thirty-two positions are a table of the numbers. The frames taken
        from video -- broadcast and riders' own posts -- stay (Artur, same day).
        Since 2026-10-14 no frame at all; since 2026-10-28 Scott Flathouse's
        photographs again, with his permission and credit, and only those."""
        text = DRAFT.read_text(encoding="utf-8")
        shown = [int(n) for n in re.findall(r"\(/media/academy/kona-2026/position-(\d\d)\.webp\)", text)]
        assert shown == list(INLINE_FRAMES) + sorted(SCOTT_FRAMES + VIDEO_FRAMES)
        for n in SCOTT_FRAMES:
            caps = re.findall(rf"^!\[([^\]]*)\]\(/media/academy/kona-2026/position-{n:02d}\.webp\)$", text, re.M)
            assert caps[-1].startswith(f"Position {n} of 32"), n
            assert all(c.endswith("Photo: Scott Flathouse (@scottflathousephoto).") for c in caps), n
        assert "https://www.instagram.com/scottflathousephoto/" in text
        assert "https://www.scottflathousephoto.com/events/" in text
        assert "The race photographs are Scott Flathouse's, published with his permission." in text
        assert not re.search(r"/media/academy/kona-2026/rider-[abc]\.webp", text)
        rows = re.findall(r"^\| (\d+) \| [^|]+ \| (?:Men|Women) \| (\d+)° \|", text, re.M)
        assert [int(n) for n, _ in rows] == list(range(1, 33))
        tilts = [int(t) for _, t in rows]
        assert tilts == sorted(tilts), "ordered from the flattest forearms to the steepest"
        assert "Position 27 is Trevor Foley, from a photo a reader shared; an earlier Kona, not the 2026 race." in text
        assert "the stills from video and the reader's photo are measured but not reproduced" in text
        # Athletes named where Artur could tell (2026-10-14); the one rider in
        # two frames is kept and said so, not hidden.
        assert "| 1 | Kristian Blummenfelt | Men |" in text
        assert "Positions 5 and 18 are the same athlete, Matthew Hanson" in text
        assert "We do not name the 2026 athletes" not in text
        # Nor are the photographs served: a removed figure is a deleted one.
        gone = [f"position-{n:02d}" for n in range(1, 33) if n not in SCOTT_FRAMES + VIDEO_FRAMES]
        for name in gone + ["rider-a", "rider-b", "rider-c"]:
            assert not (MEDIA / "academy" / "kona-2026" / f"{name}.webp").exists(), name

    def test_the_hand_corrections_are_disclosed(self):
        """Readers pointed out points sitting off the joints; Artur moved the
        ones that missed on all fifteen photos (2026-10-11). The article says
        it was updated, how many points moved and by how much, rings every
        moved point, and gives the before-and-after medians."""
        text = DRAFT.read_text(encoding="utf-8")
        assert text.split("---", 2)[2].lstrip().startswith("**Updated 12 October 2026.**")
        assert "76 of the 128 joint points moved" in text
        assert "on the original fifteen photos, the medians moved by 2° at most" in text

    def test_uci_is_not_presented_as_a_triathlon_rule(self):
        """UCI equipment rules govern road time trials, not Ironman, whose 2025
        rules limit extension length and not tilt (Artur, 2026-10-11)."""
        text = DRAFT.read_text(encoding="utf-8")
        assert "Those rules do not apply to Ironman." in text
        assert "the UCI limit." not in text
        assert "The rules moved first." not in text
        # Not a source for an Ironman article either (Artur, 2026-10-12).
        assert 'author: "UCI"' not in text

    def test_the_numbers_in_the_text_match_the_table(self):
        text = DRAFT.read_text(encoding="utf-8")
        for row in ("| Torso, from horizontal | 20° | 19–22° | 12–27° | 20–30° |",
                    "| Shoulder, torso to upper arm | 88° | 83–92° | 74–100° | 80–105° |",
                    "| Elbow | 84° | 76–88° | 66–102° | 75–110° |",
                    "| Forearm tilt, above horizontal | 26° | 23–28° | 19–32° | 5–30° |",
                    "| Helmet above the back, % of torso length | 18% | 14–21% | 7–33% | up to 23% |"):
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


def test_the_head_section_measures_the_helmet():
    """The helmet reading, its level reference, the hand moves on flush helmet
    tails, and the one study behind the penalty. The old "head position"
    100/100 score is no longer in the app, so the article no longer talks
    about it (Artur, 2026-10-12)."""
    text = DRAFT.read_text(encoding="utf-8")
    assert "100/100" not in text and '"head position"' not in text
    assert "## 4. The head: how far the helmet rises above the back" in text
    assert "level because the wind is" in text
    assert "on sixteen of the thirty-two positions we moved a helmet point by hand" in text
    assert "up to 6.4%" in text and "Beaumont" in text
    from app.services.video_analysis.biomechanics.head_silhouette import KONA_PRO_RANGE
    from app.services.video_analysis.biomechanics.head_silhouette import KONA_PRO_FREE_UP_TO
    assert KONA_PRO_RANGE == (7.0, 33.0), "the app's reference is the article's"
    assert KONA_PRO_FREE_UP_TO == 23.0


def test_the_stills_count_and_the_article_says_so():
    """Nine positions are broadcast or social-media stills, shown with the
    rest; the reader is told how many. The men-vs-women gap is reported as
    what it is: within the noise (Mann-Whitney, every angle p > 0.2)."""
    text = DRAFT.read_text(encoding="utf-8")
    assert "Six of them are broadcast and social-media stills" in text
    assert "## 5. Men and women: a first look" in text
    assert "none of these differences stands out from noise" in text


class TestTheHipOverCrankArticle:
    """Kona 2026, part two (2026-10-25): where the rider sits over the crank."""

    SLUG = "kona-2026-hip-over-crank"

    def test_it_is_published_and_every_figure_exists(self):
        from app.services.academy import get_article, invalidate_cache
        invalidate_cache()
        art = get_article(self.SLUG)
        assert art is not None and art.meta.category == "bike-position" and len(art.sources) == 4
        text = (CONTENT_DIR / f"{self.SLUG}.md").read_text(encoding="utf-8")
        srcs = re.findall(r"^!\[[^\]]*\]\((/media/[^)]+)\)$", text, flags=re.M)
        assert len(srcs) == art.body_html.count("<figure>") == 6
        for src in srcs:
            assert (MEDIA / src.removeprefix("/media/")).is_file(), src

    def test_the_app_links_here_and_the_link_into_the_app_opens_the_crank_goal(self):
        from app.services.video_analysis.biomechanics import hip_over_bb
        assert hip_over_bb.ARTICLE == f"/academy/{self.SLUG}"
        assert "(/app?goal=crank)" in (CONTENT_DIR / f"{self.SLUG}.md").read_text(encoding="utf-8")

    def test_the_numbers_are_the_fields(self):
        # The article's middle halves are the app's FIELD, rounded: one study, one set of numbers.
        from app.services.video_analysis.biomechanics.hip_over_bb import FIELD
        text = (CONTENT_DIR / f"{self.SLUG}.md").read_text(encoding="utf-8")
        assert f"{FIELD['pro']['n']} pros" in text and f"{FIELD['age_group']['n']} age-groupers" in text
        assert f"{round(FIELD['pro']['q1'])}–{round(FIELD['pro']['q3'])}°" in text
        assert f"{round(FIELD['age_group']['q1'])}–{round(FIELD['age_group']['q3'])}°" in text


def test_every_article_with_figures_opens_them_full_screen():
    # Most readers are on a phone (Artur, 2026-10-25): a figure must open full size.
    from app.services.academy import get_article, invalidate_cache
    from app.services.academy.renderer import render_article
    invalidate_cache()
    page = render_article(get_article("kona-2026-hip-over-crank"), "https://getflapp.com")
    assert "querySelectorAll('.cover img, .prose figure img')" in page and "#zoom{" in page
    assert "cursor:zoom-in" in page
