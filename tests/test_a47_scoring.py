"""Regression test for the bug that killed A47's third run.

WHAT HAPPENED. `scratch/ds_h0inject/job.py` de-duplicates the gold token ids it tracks, because
a recipient and its donor can share a gold (3 of 18 items in that run -- both single digits) and
the hook would otherwise append twice per unroll. So `base_ranks` has ONE key for those items.
The P4 scoring block then did `list(r["base_ranks"].keys())[1]`, which raises IndexError on
exactly those items. It sat in the post-loop scoring section, outside the per-item try/except,
and before the results file was written -- so the job burned 77 minutes of GPU, died, and
DataSphere reported ERROR with no output. It was abandoned undiagnosed on submission day.

WHAT IS ASSERTED HERE.
  1. `_score` survives a record set containing a shared-gold item (the crash case).
  2. `_score` writes no files and returns normally -- it is printing only, because `main()` now
     banks results to disk *before* calling it.
  3. The kernel still de-duplicates gold ids, which is what makes single-key rows possible; if
     that changes, this test's premise is gone and it should be revisited rather than deleted.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics as st
import sys

import pytest

KERNEL = pathlib.Path(__file__).resolve().parents[1] / "scratch" / "ds_h0inject" / "job.py"


def _load_kernel():
    if not KERNEL.exists():
        pytest.skip(f"kernel not present at {KERNEL}")
    spec = importlib.util.spec_from_file_location("a47_job", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a47_job"] = mod
    spec.loader.exec_module(mod)          # module level is stdlib-only by construction
    return mod


def _row(gid_self, gid_don, gid_don_x, n=48):
    """A record shaped exactly as the per-item loop emits one."""
    ids = sorted({gid_self, gid_don, gid_don_x})
    ranks = {str(g): list(range(n, 0, -1)) for g in ids}
    arm = {"ranks": ranks, "tops": [[1, [[1, 0.5]]]], "dev": [0.0] * n,
           "rank_delta": [0] * n, "donor_norm": 1.0, "donor_gold_shared": gid_don == gid_self}
    return {
        "family": "add1", "item": 0, "gold": "3", "ok": True,
        "donor_gold": "4", "donor_gold_xfam": "5",
        "gid_self": gid_self, "gid_don": gid_don, "gid_don_xfam": gid_don_x,
        "n_tokens": 40, "norm_h0": 45.3, "norm_star": 76.4,
        "base_ranks": ranks, "base_tops": [[1, [[1, 0.5]]]],
        "seed_sweep": [{"seed": s, "best_rank": 1, "best_depth": 4, "final_rank": 1,
                        "oracle": 1, "final_correct": 1, "h0_norm": 45.3}
                       for s in (1, 2, 3, 4, 5, 6)],
        "arms": {k: dict(arm) for k in
                 ("own_h0", "redraw_h0", "mid4_same", "star_same", "star_xfam", "star_scaled")},
    }


def test_score_survives_a_shared_gold_item():
    """The exact crash: donor and recipient share a gold, so base_ranks has one key."""
    mod = _load_kernel()
    ok = [
        _row(7, 9, 11),      # all distinct
        _row(7, 9, 11),
        _row(7, 7, 7),       # <-- shared gold: base_ranks has ONE key. Killed the third run.
    ]
    assert any(len(r["base_ranks"]) == 1 for r in ok), "test premise lost: no single-key row"
    mod._score(ok, ok, [], st)      # must not raise


def test_score_writes_nothing(tmp_path, monkeypatch):
    """_score is printing only; main() banks before calling it, so it must not touch disk."""
    mod = _load_kernel()
    monkeypatch.chdir(tmp_path)
    mod._score([_row(7, 9, 11)], [_row(7, 9, 11)], [], st)
    assert list(tmp_path.iterdir()) == [], "scoring wrote files; it must not"


def test_kernel_still_deduplicates_gold_ids():
    """The premise of the bug: gold ids are de-duplicated, so a row can carry one key."""
    src = KERNEL.read_text() if KERNEL.exists() else pytest.skip("kernel absent")
    assert "gold_ids = sorted({g_self, g_don, g_don_x})" in src, (
        "gold-id construction changed; re-derive whether single-key rows are still possible"
    )


def test_results_are_banked_before_scoring():
    """The structural fix: 77 minutes of compute must not be lost to a summary bug."""
    src = KERNEL.read_text() if KERNEL.exists() else pytest.skip("kernel absent")
    bank = src.index('"banked_before_scoring": True')
    call = src.index("_score(ok, rows, dropped, st)")
    assert bank < call, "results must be written to disk before _score is called"
