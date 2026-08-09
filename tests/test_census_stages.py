"""A failed screening draw must not shift a stage-2 draw into stage 1.

WHY THIS EXISTS (thread A7). `run_census_analysis.split_stages` separates the
screening draws from the deepening draws by APPEND ORDER within an item, which is
the only thing that makes the two estimates separable at all. It originally skipped
rows with `ok: false` BEFORE incrementing the per-item counter. These kernels append
a row for every attempt -- `{"ok": False, "why": ...}` on exception -- so position in
the list is the draw index, and skipping a failed row without consuming its slot
shifts every later draw of that item down one, tagging a stage-2 draw as stage 1.

The failure is silent, and it lands only on items that had a failure: exactly the
items most likely to be unusual in other ways. It was found by reading the code
before the census run landed, not by seeing a wrong number.
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "census_analysis", ROOT / "scripts" / "run_census_analysis.py")
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)

_OK = {"best_rank": 1, "best_depth": 3, "correct": True, "final_rank": 1,
       "n_tokens": 10, "multi_token_gold": False, "gold": "4"}


def _rows(n_ok_screen: int, n_failed_screen: int, n_deep: int) -> list[dict]:
    rows = [{**_OK, "family": "f", "item": 0, "ok": True} for _ in range(n_ok_screen)]
    rows += [{"family": "f", "item": 0, "ok": False, "why": "boom"}
             for _ in range(n_failed_screen)]
    rows += [{**_OK, "family": "f", "item": 0, "ok": True} for _ in range(n_deep)]
    return rows


def test_a_failed_screening_draw_consumes_its_slot():
    """3 good + 1 failed screening draw, then 2 deepening draws, at screen_draws=4."""
    df = census.split_stages(_rows(3, 1, 2), screen_draws=4)
    assert int((df["stage"] == 1).sum()) == 3
    assert int((df["stage"] == 2).sum()) == 2, (
        "a stage-2 draw was tagged stage 1: the failed screening row did not "
        "consume its slot, so every later draw of this item shifted down one"
    )


def test_the_clean_case_is_unchanged():
    df = census.split_stages(_rows(4, 0, 20), screen_draws=4)
    assert int((df["stage"] == 1).sum()) == 4
    assert int((df["stage"] == 2).sum()) == 20


def test_the_guard_would_fire_on_the_old_behaviour():
    """Reproduce the bug's arithmetic directly, so the test is not itself an assumption.

    The old code skipped non-ok rows before incrementing, so with 3 good + 1 failed
    screening rows the first deepening row saw index 3 < 4 and was tagged stage 1.
    """
    rows = _rows(3, 1, 2)
    seen, stages_buggy = 0, []
    for r in rows:
        if not r.get("ok"):
            continue          # the bug: skip WITHOUT consuming a slot
        stages_buggy.append(1 if seen < 4 else 2)
        seen += 1
    assert stages_buggy.count(1) == 4 and stages_buggy.count(2) == 1, (
        "the old behaviour should mis-tag one stage-2 draw as stage 1")
