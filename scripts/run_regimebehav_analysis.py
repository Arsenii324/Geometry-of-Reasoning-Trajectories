"""A26's analysis: does crossing the regime boundary change what the model does?

Implements the predictions pre-registered in `scratch/ds_regimebehav/job.py`, written
before the run returned. The detection floor (P4) is the reason this file exists: D132
reported the same null with no floor, which makes it unrefuted rather than bounded, and a
null without a bound is not a result.

  P1  no-op identity: t = 0 must reproduce the unpatched rank curve EXACTLY (h_0 seeded).
      VOIDS the run on failure.
  P2  each unit must contain exactly one crossing of R > 0.6677; others dropped, counted.
  P3  PRIMARY: paired within-unit comparison of the last sub-threshold point against the
      first supra-threshold point, on best_depth, best_rank and correctness.
  P4  DETECTION FLOOR: plant effects of known size into the observed pairs and report the
      fraction this design catches. This is what converts "no difference" into "no
      difference larger than X".
  P5  readout positive control: does the gold rank move at all across the full sweep? If
      not, P3's null is uninterpretable.

Run:  uv run python scripts/run_regimebehav_analysis.py
"""

from __future__ import annotations

import json
import pathlib
import random

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch/ds_regimebehav/regimebehav.json"
THRESHOLD = 0.6677
N_PERM = 20000
SEED = 0


def wilcoxon_p(diffs, rng, n_perm=N_PERM):
    """Exact-in-spirit sign-flip permutation on the paired differences.

    Sign-flipping is the right null here: under H0 the two sides of the boundary are
    exchangeable within a unit, so each pair's difference is equally likely to carry
    either sign. `exact_floor` applies -- with n pairs the smallest attainable p is
    2^-n, and a Monte-Carlo p below that is an artefact, so it is reported.
    """
    d = np.asarray([x for x in diffs if x == x], dtype=float)
    d = d[d != 0]
    if len(d) < 2:
        return float("nan"), 0, float("nan")
    obs = float(np.mean(d))
    hits = 0
    for _ in range(n_perm):
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in d])
        hits += abs(float(np.mean(d * s))) >= abs(obs)
    return (hits + 1) / (n_perm + 1), len(d), 2.0 ** (-len(d))


def units(rows):
    """Group into (chord, sequence) units and locate the single boundary crossing."""
    by = {}
    for r in rows:
        by.setdefault((r["rot_w"], r["set_w"], r["seq_id"]), []).append(r)
    out, dropped = [], []
    for k, v in sorted(by.items()):
        v.sort(key=lambda r: r["t"])
        loc = [r for r in v if 0.0 < r["t"] < 1.0]        # the local window only
        flags = [r["rotating"] for r in loc]
        cross = [i for i in range(len(flags) - 1) if flags[i] != flags[i + 1]]
        if len(cross) != 1:
            dropped.append((k, len(cross)))
            continue
        i = cross[0]
        lo, hi = loc[i], loc[i + 1]
        below, above = (lo, hi) if not lo["rotating"] else (hi, lo)
        out.append({"key": k, "below": below, "above": above, "all": v})
    return out, dropped


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC.relative_to(ROOT)} -- run A26 first")
        return 1
    d = json.loads(SRC.read_text())
    rows = [r for r in d["rows"] if r.get("ok")]
    print(f"{len(rows)} orbits, {len(d['dropped'])} dropped by the token gate\n")

    # ---- P1 -------------------------------------------------------------------
    print("P1 INSTRUMENT NULL -- the t=0 patch is a no-op and must reproduce the "
          "unpatched RANK CURVE exactly:")
    bad = 0
    for r in [r for r in rows if r["t"] == 0.0]:
        same = r["rank_curve"] == r["base_ranks"]
        bad += not same
        if not same:
            print(f"  MISMATCH {r['rot_w']}<-{r['set_w']} seq{r['seq_id']}: "
                  f"best_depth {r['best_depth']} vs {r['base_best_depth']}, "
                  f"best_rank {r['best_rank']} vs {r['base_best_rank']}")
    n0 = sum(1 for r in rows if r["t"] == 0.0)
    print(f"  {n0 - bad}/{n0} identical")
    if bad:
        print("\n  VOID: the readout path does not reproduce under a no-op patch.")
        return 1
    print("  -> the patch touches `e` and nothing else, readout included\n")

    # ---- P5 -------------------------------------------------------------------
    spans = [max(r["best_rank"] for r in u["all"]) - min(r["best_rank"] for r in u["all"])
             for u in units(rows)[0]] or [0]
    fspan = [max(r["final_rank"] for r in u["all"]) - min(r["final_rank"] for r in u["all"])
             for u in units(rows)[0]] or [0]
    print("P5 READOUT POSITIVE CONTROL -- does the gold rank move across the full sweep?")
    print(f"  best_rank span within a unit : median {np.median(spans):.1f}, max {max(spans)}")
    print(f"  final_rank span within a unit: median {np.median(fspan):.1f}, max {max(fspan)}")
    if np.median(spans) == 0 and np.median(fspan) == 0:
        print("  -> the readout does not respond to `e` at all here; P3 is UNINTERPRETABLE")
        return 1
    print("  -> the readout does respond to `e`, so a null at the boundary is meaningful\n")

    us, dropped = units(rows)
    print(f"P2 CROSSING GATE: {len(us)} units with exactly one crossing; "
          f"{len(dropped)} dropped {[f'{k[0]}<-{k[1]}s{k[2]}:{n}' for k, n in dropped][:6]}\n")
    if len(us) < 8:
        print("  too few units to test; stopping")
        return 1

    # ---- P3 -------------------------------------------------------------------
    rng = random.Random(SEED)
    print("P3 PRIMARY -- across the boundary, two grid points 0.05 apart in `e`:")
    results = {}
    for name, f in (("best_depth", lambda r: r["best_depth"]),
                    ("best_rank (log10)", lambda r: np.log10(r["best_rank"])),
                    ("correct", lambda r: float(r["correct"]))):
        diffs = [f(u["above"]) - f(u["below"]) for u in us]
        p, n, floor = wilcoxon_p(diffs, rng)
        results[name] = (diffs, p, n)
        print(f"  {name:18s} rotating - settling = {np.mean(diffs):+.3f} "
              f"(median {np.median(diffs):+.3f})   p = {p:.4f}   "
              f"n = {n} non-tied pairs   exact floor {floor:.2e}")

    # ---- P4 -------------------------------------------------------------------
    print(f"\nP4 DETECTION FLOOR -- what this design would have caught (n = {len(us)} units):")
    print("  planted effect      | detected at alpha = 0.05")
    for name in ("best_depth", "best_rank (log10)"):
        diffs, _, _ = results[name]
        sd = float(np.std(diffs)) or 1.0
        print(f"  --- {name}, observed sd of the paired difference = {sd:.3f}")
        for mult in (0.25, 0.5, 1.0, 2.0):
            hits = 0
            trials = 300
            for s in range(trials):
                r2 = random.Random(1000 + s)
                planted = [x + mult * sd for x in diffs]
                p, _, _ = wilcoxon_p(planted, r2, n_perm=400)
                hits += p < 0.05
            print(f"      +{mult:>4.2f} sd ({mult * sd:+.2f} units)   "
                  f"{hits / trials * 100:5.1f}%")
    print("\n  A null is only as strong as this table. D132 reported the same comparison"
          "\n  with no such table, which is why it is unrefuted rather than bounded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
