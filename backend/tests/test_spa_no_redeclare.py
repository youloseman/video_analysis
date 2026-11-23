"""No two top-level functions in the SPA share a name.

The SPA is one classic script: a second ``function copyText`` silently replaced
the first for the whole app, and the paid AI export's Copy reported "Clipboard
blocked" on every click until a code audit found it. There is no JS linter in
CI, so this is the ``no-redeclare`` rule for the one shape that bit.
"""

from __future__ import annotations

import collections
import re

from conftest import read_spa


def test_no_top_level_function_is_declared_twice():
    names = re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", read_spa(), re.M)
    dupes = sorted(n for n, k in collections.Counter(names).items() if k > 1)
    assert len(names) > 500, "the pattern stopped matching the SPA's functions"
    assert dupes == [], dupes


def test_copy_with_a_message_goes_through_the_one_copier():
    spa = read_spa()
    body = spa[spa.index("async function copyWithToast("):][:300]
    assert "copyText(" in body
    assert "navigator.clipboard" not in body
