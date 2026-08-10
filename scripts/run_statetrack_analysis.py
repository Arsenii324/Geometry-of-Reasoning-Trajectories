"""A27's analysis: can Huginn state-track, and if so does depth rise with difficulty?

Implements the gates pre-registered in `scratch/ds_statetrack/job.py`. The order is
enforced, not suggested: P1 (capability) gates P2 (dynamic range) gates P3 (H2). This
project has twice computed geometry on an axis the model could not do — D118 went VOID at
0.0% above one level, and D100's "working difficulty ladder" was withdrawn.

  P1  3-way accuracy at m = 1 must beat the MAJORITY-CLASS baseline of that level at
      binomial p < 0.05. Not the 1/3 chance floor: golds are balanced but not perfectly,
      and a constant responder must not pass.
  P2  at least two m-levels strictly inside (0.05, 0.95).
  P3  H2 at the (m, item) unit, m = 0 excluded a priori — D143's headline was entirely
      one trivial level and this is the same shape of trap.
  P5  shortcut probe: is accuracy predicted by whether the LAST swap touches the ball
      rather than by m? If so the model is pattern-matching, not tracking.

Run:  uv run python scripts/run_statetrack_analysis.py
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import random

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch/ds_statetrack/statetrack.json"


def binom_p(k, n, p0):
    """One-sided binomial tail P(X >= k) under p0 — exact, no normal approximation."""
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, n=20000, seed=0):
    o = spearman(x, y)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        p = list(y)
        rng.shuffle(p)
        hits += abs(spearman(x, np.array(p))) >= abs(o)
    return o, (hits + 1) / (n + 1)


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC.relative_to(ROOT)} -- run A27 first")
        return 1
    d = json.loads(SRC.read_text())
    ok = [r for r in d["rows"] if r.get("ok")]
    print(f"{len(ok)}/{len(d['rows'])} forwards ok; chance = {d['chance']:.3f}\n")

    # --- P1 -------------------------------------------------------------------
    print("P1 CAPABILITY GATE (3-way accuracy, i.e. argmax over {A,B,C} at final unroll):")
    best_fmt, passed = None, False
    for fmt in sorted({r["fmt"] for r in ok}):
        v1 = [r for r in ok if r["fmt"] == fmt and r["m"] == 1]
        if not v1:
            continue
        golds = collections.Counter(r["gold"] for r in v1)
        base = max(golds.values()) / len(v1)
        k = sum(r["choice_correct_final"] for r in v1)
        p = binom_p(k, len(v1), base)
        print(f"  {fmt:6s} m=1: {k}/{len(v1)} = {k / len(v1):.3f}  "
              f"majority baseline {base:.3f}  binomial p = {p:.4f}  "
              f"{'PASS' if p < 0.05 else 'fail'}")
        if p < 0.05:
            passed, best_fmt = True, fmt
    print("\n  full accuracy table:")
    for fmt in sorted({r["fmt"] for r in ok}):
        for m in sorted({r["m"] for r in ok}):
            v = [r for r in ok if r["fmt"] == fmt and r["m"] == m]
            if not v:
                continue
            print(f"    {fmt:6s} m={m}  n={len(v):3d}  "
                  f"3-way-final {np.mean([r['choice_correct_final'] for r in v]):.3f}  "
                  f"3-way-any {np.mean([r['choice_correct_any'] for r in v]):.3f}  "
                  f"vocab-acc {np.mean([r['correct'] for r in v]):.3f}  "
                  f"median best_depth {np.median([r['best_depth'] for r in v]):.1f}")

    nt = sorted({r["n_tokens"] for r in ok if r["fmt"] == "bare"})
    print(f"\n  token-count gate (bare): {nt} -> "
          f"{'CONSTANT' if len(nt) == 1 else 'VARIES -- difficulty is confounded with length'}")

    if not passed:
        print("\n  P1 FAILS in both formats. Huginn does not state-track above its own")
        print("  majority baseline at the EASIEST level, so no geometry is read and H2")
        print("  cannot be tested on this axis. That is the result: the one difficulty")
        print("  axis theory says recurrence is needed for is out of this model's reach.")
        print("  It also bounds what A5 would show -- A5 is strictly harder.")
        return 0

    # --- P2 / P3 --------------------------------------------------------------
    v = [r for r in ok if r["fmt"] == best_fmt]
    acc = {m: np.mean([r["choice_correct_final"] for r in v if r["m"] == m])
           for m in sorted({r["m"] for r in v})}
    live = [m for m, a in acc.items() if 0.05 < a < 0.95]
    print(f"\nP2 DYNAMIC RANGE: {len(live)} live levels {live} -> "
          f"{'OK' if len(live) >= 2 else 'INSUFFICIENT, H2 not computed'}")
    if len(live) < 2:
        return 0

    cells = collections.defaultdict(list)
    for r in v:
        if r["m"] > 0:                       # m = 0 excluded a priori (see P3)
            cells[(r["m"], r["item"])].append(r)
    M = np.array([k[0] for k in cells])
    D = np.array([float(np.median([x["best_depth"] for x in c])) for c in cells.values()])
    rho, p = perm_p(M, D)
    print(f"\nP3 H2 at the (m, item) unit, m=0 excluded, n = {len(M)} cells:")
    print(f"  Spearman(m, best_depth) = {rho:+.4f}   permutation p = {p:.4f}")
    nd = [spearman(M, np.random.permutation(D)) for _ in range(300)]
    print(f"  null sd {np.std(nd):.3f} -> resolves |rho| >= {1.96 * np.std(nd):.3f}")

    # --- P5 -------------------------------------------------------------------
    lt = [r for r in v if r["last_touches"]]
    ln = [r for r in v if not r["last_touches"]]
    if lt and ln:
        print(f"\nP5 SHORTCUT PROBE -- is accuracy explained by the LAST swap instead of m?")
        print(f"  last swap touches the ball : {np.mean([r['choice_correct_final'] for r in lt]):.3f} (n={len(lt)})")
        print(f"  last swap does not         : {np.mean([r['choice_correct_final'] for r in ln]):.3f} (n={len(ln)})")
        print(f"  (a large gap here with a flat m-curve means pattern-matching, not tracking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
