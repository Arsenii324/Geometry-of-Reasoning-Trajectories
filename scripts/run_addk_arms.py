"""D110's three identification arms, re-derivable instead of ad hoc.

WHY THIS EXISTS. An independent provenance audit on 2026-08-10 flagged D110's
answer-token-fixed arm as a MISMATCH (it recomputed -0.6239 against the row's -0.481)
and observed, correctly, that **no script under `scripts/` referenced `ds_addk`, so the
analysis was not re-derivable**. Re-running it here reproduces all three arms exactly:

    fix GOLD (answer token), vary k   ->  -0.4810   (row: -0.481)
    fix v (first operand),  vary k    ->  -0.5265   (row: -0.527)
    fix k, vary v                     ->  +0.0719   (row: +0.072)

So the number was right and the audit's mismatch is a false positive — most likely a
different minimum-stratum rule; this module requires a stratum to span **>= 3 distinct
levels** of the varying axis, which is what makes a within-stratum Spearman meaningful.

**But the audit's underlying criticism was correct and is the reason for this file.**
The analysis existed only in shell heredocs. A claim whose derivation cannot be re-run
is one nobody can check, including its author six hours later — and this project has
retracted twelve claims, several of which were exactly that. The rule this file
enforces by existing: **a load-bearing number gets a script, not a heredoc.**

Run:  uv run python scripts/run_addk_arms.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics as st

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID = ROOT / "scratch/ds_addk/out/grid.json"
MIN_LEVELS = 3          # a stratum must span this many levels of the varying axis


def spearman(x, y) -> float:
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def problems() -> list[tuple[int, int, int, float]]:
    """One row per DISTINCT PROBLEM: (k, gold, v, median best_depth over draws).

    Keyed on (level, gold) rather than (level, item) because `_addk` sets
    v = i % max(1, 10 - k), so item slots COLLIDE at high k -- 6 slots but 3 distinct
    problems at k = 7. D110(b): `require_units` passed 42 keys that were 36 prompts.
    """
    rows = json.loads(GRID.read_text())["rows"]
    cells = collections.defaultdict(list)
    for r in rows:
        if r.get("ok"):
            cells[(r["level"], int(r["gold"]))].append(r["best_depth"])
    return [(k[0], k[1], k[1] - k[0], float(np.median(v)))
            for k, v in sorted(cells.items())]


def arm(units, fix_idx: int, vary_idx: int) -> tuple[float, int]:
    """Mean within-stratum Spearman, holding one axis fixed and varying another."""
    groups = collections.defaultdict(list)
    for u in units:
        groups[u[fix_idx]].append(u)
    rhos = [spearman([x[vary_idx] for x in v], [x[3] for x in v])
            for v in groups.values()
            if len({x[vary_idx] for x in v}) >= MIN_LEVELS]
    rhos = [r for r in rhos if r == r]
    return (st.mean(rhos) if rhos else float("nan")), len(rhos)


def main() -> int:
    u = problems()
    print(f"{len(u)} distinct problems from {GRID.relative_to(ROOT)}")
    print("(addk obeys gold = v + k, so the three axes carry only TWO degrees of "
          "freedom;\n no stratification separates all three, which is why D110 uses "
          "two arms that move v\n in OPPOSITE directions and a third that holds k.)\n")
    for fix, vary, name, claimed in ((1, 0, "GOLD (answer token) fixed, k varies", -0.481),
                                     (2, 0, "v (first operand) fixed, k varies", -0.527),
                                     (0, 2, "k (difficulty) fixed, v varies", +0.072)):
        r, n = arm(u, fix, vary)
        flag = "OK" if abs(r - claimed) < 5e-3 else "DIFFERS FROM D110"
        print(f"  {name:38s} rho = {r:+.4f} over {n} strata   "
              f"(D110: {claimed:+.3f})  {flag}")
    print("\n  If the answer token drove this, arm 1 would kill it; if the first "
          "operand did,\n  arm 2 would. Both survive at ~-0.5 while arm 3 is null, "
          "so the driver is k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
