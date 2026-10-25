"""The analytics email's renderer: one block list, three outputs that agree.

No PostHog here -- the queries are read against live data by hand; what can
silently go wrong offline is the rendering (an unescaped page path, a delta
painted the wrong colour, a bar scaled off a None).
"""
from scripts import analytics_report as r


def _blocks(*fns):
    r.BLOCKS.clear()
    for f in fns:
        f()


def test_html_escapes_and_colours_deltas():
    _blocks(lambda: r.head("Pages"),
            lambda: r.table(["page", "visitors", "change"],
                            [["/a?x=<b>", 10, "+25%"], ["/b", None, "-40%"]], bar=1))
    out = r.render_html("t", "s")
    assert "&lt;b&gt;" in out and "<b>" not in out
    assert r.UP in out and r.DOWN in out
    assert out.count("height:8px") == 2          # a None still gets a (minimal) bar


def test_html_stays_under_gmail_clip():
    # Gmail cuts a message at ~102 KB. 300 rows is more than the real report
    # carries (~42 KB on 2026-10-25), so this leaves headroom, not slack.
    _blocks(*[lambda: r.table(["page", "visitors", "before", "change"],
                              [["/academy/some-long-article-path", 1234, 999, "+24%"]] * 12)
              for _ in range(25)])
    assert len(r.render_html("t", "s")) < 90_000


def test_text_and_markdown_carry_the_same_cells():
    _blocks(lambda: r.table(["plan", "n"], [["unlock", 1234]]),
            lambda: r.bullets(["one"]))
    assert "1,234" in r.render_text() and "1,234" in r.render_md()
    assert "* one" in r.render_text() and "- one" in r.render_md()


def test_delta_and_money():
    assert r.delta(15, 10) == "+50%"
    assert r.delta(0, 0) == "-" and r.delta(3, 0) == "new"
    assert r.money([("unlock", 2), ("full_yearly", 1), ("mystery", 5)]) == (107, 8)
