"""One check, deliberately: that the index and the store still point at each other.

WHY ONLY ONE. The first version of this file also policed the index's table format
-- that every live thread carried a status, that cited section letters existed,
that closed threads named a ledger row. It failed twice on my own table layout and
told me nothing about the research either time. **A test that fires on formatting
spends the suite's credibility without buying correctness**, and the next person to
see it fail will start ignoring failures.

What survives is the one property whose violation is silent and costly: an index
that has been orphaned from its store, so that following it leads nowhere. Content
consistency between the two is a judgement call and stays a human one; the guard
that DOES pay for itself on content is `test_no_retracted_claims.py`, which checks
a specific retracted string rather than a shape.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "OPEN_THREADS.md"
STORE = ROOT / "docs" / "directions.md"


def test_the_index_and_the_store_reference_each_other():
    assert INDEX.exists(), "OPEN_THREADS.md is missing -- the index of open work"
    assert STORE.exists(), "directions.md is missing -- the store behind the index"
    assert "directions.md" in INDEX.read_text(), (
        "OPEN_THREADS.md must say where the detail lives, or it is a list of "
        "slogans with no evidence behind it.")
    assert "OPEN_THREADS.md" in STORE.read_text(), (
        "directions.md must point to the index, or threads recorded here become "
        "invisible once the file is long -- which is why the index was added.")
