"""Which banked runs measured an outcome that never varied?

WHY THIS EXISTS. D148 concluded that a causally-controlled change of dynamical regime
"produces no change in what the model answers", from *0 of 48 units changed correctness*. Its
banked run has `correct = False` in **432 of 432** orbits. The comparison ran on a variable
that is identically zero on both sides of the boundary, so there was nothing for the regime to
change, and the conclusion was an artefact of a floor nobody looked for. The task was "report
the largest/smallest {noun} of the sequence" and Huginn cannot do it.

That is CLAUDE.md §1 -- *before scoring geometry on a task, ask whether the model can do the
task* -- and prose review did not catch it in either direction: the row reads as a strong
bounded null and its own headline number, "0 of 48", is exactly what a floor produces.

The check is one line of arithmetic over data this repo already has, so it should not depend on
anyone remembering to do it. A degenerate outcome is not a small effect; it is **no experiment
at all**, and it looks identical to a clean null in every summary.

WHAT IT DOES. Walks every banked result JSON, finds outcome-like fields, and reports the base
rate. A rate of exactly 0.0 or 1.0 is flagged. Rank fields are converted to "reached rank 1"
first, because that is the project's standard capability axis.

WHAT A FLAG MEANS -- AND THAT IT IS NOT AUTOMATICALLY A BUG. D147 flags at 1.000 and is
**correct to**: its finding IS that the oracle capability axis reads 100% on a model that
answers `C` regardless of the question. A flag says *this run's outcome had no variance*; whether
that is the finding or the failure is a question for the row, not for the script.

    uv run python scripts/outcome_variance_scan.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

BOOL_FIELDS = ("correct", "is_correct", "correct_letter", "letter_correct",
               "text_raw_correct", "text_norm_correct", "acc", "hit", "took",
               "first_exact", "contains_gold")
RANK_FIELDS = ("best_rank", "min_rank", "rank", "final_rank")


def rows_of(path: pathlib.Path):
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    rows = d.get("rows") if isinstance(d, dict) else (d if isinstance(d, list) else None)
    if not rows or not isinstance(rows[0], dict):
        return None
    return [r for r in rows if r.get("ok", True)]


def main() -> int:
    flagged, scanned = [], 0
    print(f"{'run':26s} {'rows':>6s} {'field':18s} {'base rate':>10s}")
    print("-" * 64)
    for path in sorted(ROOT.glob("scratch/*/*.json")):
        if path.name in ("kernel-metadata.json", "manifest.json"):
            continue
        ok = rows_of(path)
        if not ok:
            continue
        scanned += 1
        run = path.parent.name
        for k in BOOL_FIELDS + RANK_FIELDS:
            if k not in ok[0]:
                continue
            v = [r[k] for r in ok if isinstance(r.get(k), (int, float, bool))]
            if not v:
                continue
            rate = (sum(1 for x in v if x == 1) / len(v)) if k in RANK_FIELDS \
                else (sum(bool(x) for x in v) / len(v))
            label = f"{k} (rank1)" if k in RANK_FIELDS else k
            flag = ""
            if rate in (0.0, 1.0):
                flag = "   <== NO VARIANCE"
                flagged.append((run, label, rate, len(v)))
            print(f"{run:26s} {len(ok):6d} {label:18s} {rate:10.3f}{flag}")

    print("-" * 64)
    print(f"scanned {scanned} banked result files; {len(flagged)} degenerate outcomes\n")
    for run, label, rate, n in flagged:
        print(f"  {run}.{label} = {rate:.3f} over {n} rows")
    print("\nA degenerate outcome is not a small effect -- it is no experiment at all, and it")
    print("reads as a clean null in every summary. For each flag, decide which it is:")
    print("  * the FINDING       -- e.g. D147, where 1.000 on a constant responder IS the point")
    print("  * the FAILURE       -- e.g. D148, whose behavioural null had no variance to explain")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
