"""Values interpolated into an inline `onclick` have to survive the HTML parser.

The $4 unlock button was built as::

    onclick="startCheckout('unlock',${JSON.stringify(opts.unlockId)})"

`JSON.stringify` returns its result **with double quotes around it**, and the
attribute is delimited by double quotes. So the browser ended the attribute at
the id's opening quote and read the rest as new attributes::

    onclick="startCheckout('unlock',"  h1788334942060")"=""

The handler was cut off mid-call, the click did nothing, and nothing errored:
no exception, no console warning, no failed request. Every "Unlock just this
report" click since the button shipped was silent -- the whole one-off purchase
path, on the screen the free tier is funnelled to.

Two tests, because one instance is not the lesson: the parser test says why the
old spelling failed, and the sweep says no `onclick` in the file may interpolate
an unescaped value again. `esc()` turns `"` into `&quot;`, which the parser
decodes back to `"` inside the attribute -- so the JS receives the quoted
string it wanted, and the attribute survives.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
HTML = SPA.read_text(encoding="utf-8")

UNLOCK_ID = "h1788334942060"


class Attrs(HTMLParser):
    """Collect the attributes of the first <button> as the browser reads them."""

    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, str | None] = {}

    def handle_starttag(self, tag, attrs):
        if tag == "button" and not self.attrs:
            self.attrs = dict(attrs)

    @classmethod
    def of(cls, markup: str) -> dict[str, str | None]:
        p = cls()
        p.feed(markup)
        return p.attrs


def button(onclick_body: str) -> str:
    return f'<button type="button" onclick="{onclick_body}">Unlock</button>'


# --------------------------------------------------------------------------
# why the old spelling could not work
# --------------------------------------------------------------------------

def test_a_raw_json_string_breaks_the_attribute():
    """The bug, reproduced through a real HTML parser rather than by eye."""
    attrs = Attrs.of(button(f'startCheckout(\'unlock\',"{UNLOCK_ID}")'))
    assert attrs["onclick"] == "startCheckout('unlock',"     # cut off mid-call
    assert UNLOCK_ID not in (attrs["onclick"] or "")
    # ...and the remainder became attributes of its own, which is the tell in
    # the DOM inspector.
    assert len(attrs) > 2


def test_the_escaped_form_delivers_the_id():
    """`esc()` maps `"` to `&quot;`; the parser maps it back inside the value."""
    attrs = Attrs.of(button(f"startCheckout('unlock',&quot;{UNLOCK_ID}&quot;)"))
    assert attrs["onclick"] == f"startCheckout('unlock',\"{UNLOCK_ID}\")"
    assert set(attrs) == {"type", "onclick"}


# --------------------------------------------------------------------------
# the rule, across the whole file
# --------------------------------------------------------------------------

ONCLICK = re.compile(r'onclick="([^"]*)"')
INTERPOLATION = re.compile(r"\$\{([^}]*)\}")


def onclick_interpolations() -> list[tuple[int, str, str]]:
    out = []
    for m in ONCLICK.finditer(HTML):
        line = HTML[: m.start()].count("\n") + 1
        for expr in INTERPOLATION.findall(m.group(1)):
            out.append((line, expr.strip(), m.group(1)))
    return out


def test_every_value_interpolated_into_an_onclick_is_escaped():
    unescaped = [
        f"index.html:{line}  ${{{expr}}}"
        for line, expr, _ in onclick_interpolations()
        if not expr.startswith("esc(")
    ]
    assert not unescaped, (
        "an unescaped value in an inline onclick truncates the handler at the "
        "first quote it contains, silently: " + "; ".join(unescaped)
    )


def test_the_sweep_actually_looks_at_something():
    """A regex that matches nothing passes the test above for free."""
    assert len(ONCLICK.findall(HTML)) >= 8
    assert onclick_interpolations(), "no interpolating onclick found at all"


# --------------------------------------------------------------------------
# the button this was found on
# --------------------------------------------------------------------------

def test_the_unlock_button_still_passes_the_report_id():
    """Escaping it away -- or dropping the argument -- would trade a dead
    button for a live one that unlocks nothing: startCheckout('unlock') with
    no id shows "Open a report first" and returns."""
    line = next(
        (m.group(1) for m in ONCLICK.finditer(HTML)
         if "startCheckout('unlock'" in m.group(1)),
        None,
    )
    assert line is not None, "the one-off unlock button is gone"
    assert "unlockId" in line
    assert "esc(JSON.stringify(" in line


@pytest.mark.parametrize("handler", ["startCheckout"])
def test_the_handler_is_reachable_from_an_inline_attribute(handler):
    """Inline onclick resolves against `window`, so a handler moved inside a
    module or an IIFE would leave the same dead button with the markup intact."""
    assert re.search(rf"^\s*(async\s+)?function {handler}\(", HTML, re.M), (
        f"{handler} is no longer a top-level function declaration"
    )
