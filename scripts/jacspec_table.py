"""Re-derive A53's answer from banked JSON. Zero GPU, cannot go stale.

Same contract as `scripts/arc_table.py`: every number the ledger will quote for A53 is computed
here from `scratch/ds_jacspec/jacspec.json`, so a claim can be re-checked without a GPU slot and
cannot drift from the data. The kernel prints a summary while it runs; this is the version that
gets read.

THE QUESTION. The project has two rotation lines that have never been compared:

  * the JACOBIAN line -- the recurrent map's spectrum is complex, implied rotation period 2.6-6.0
    unrolls, measured exactly on three prompts (D31 magnitudes, D55 arguments);
  * the REGIME line -- one instruction noun flips the trajectory between settling and a damped
    **period-6** rotation, at fixed token count, causally controlled by one embedding row.

If they are one phenomenon, prompts labelled rotating carry an oscillatory eigenvalue implying a
period near 6 and labelled settling prompts do not. If both arms look alike, the regime difference
is not in the leading local dynamics.

WHAT THIS PRINTS THAT THE KERNEL DOES NOT: the full period distribution per noun rather than per
arm, a rank-sum test between arms rather than a difference of medians, and the separation stated
against the within-arm spread -- because a difference of medians with no spread beside it is the
kind of number this project has had to withdraw before.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "scratch" / "ds_jacspec" / "jacspec.json"
ROT_NOUNS = ("symbol", "symptom")
SET_NOUNS = ("element", "token")


def rank_sum_p(a, b):
    """Two-sided Mann-Whitney via normal approximation. Small n here, so report U and n too."""
    if not a or not b:
        return None, None
    obs = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    while i < len(obs):
        j = i
        while j + 1 < len(obs) and obs[j + 1][0] == obs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    r_a = sum(ranks[k] for k, (_, g) in enumerate(obs) if g == 0)
    n1, n2 = len(a), len(b)
    u = r_a - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return u, None
    z = (u - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return u, p


def describe(vals):
    if not vals:
        return "n=0"
    return (f"n={len(vals):2d}  median {st.median(vals):6.3f}  "
            f"IQR {st.quantiles(vals, n=4)[0]:.3f}-{st.quantiles(vals, n=4)[2]:.3f}"
            if len(vals) >= 4 else f"n={len(vals):2d}  median {st.median(vals):6.3f}")


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"no banked result at {path} -- run A53 first (scratch/ds_jacspec/)")
        return 1
    d = json.loads(path.read_text())
    rows = [r for r in d["rows"] if r.get("ok")]
    bad = [r for r in d["rows"] if not r.get("ok")]
    print(f"{len(rows)} usable spectra, {len(bad)} failed, warmup={d.get('warmup')} unrolls, "
          f"{d.get('elapsed_s', 0):.0f}s")
    for b in bad[:4]:
        print(f"   FAILED {b.get('label')}/{b.get('item')}: {b.get('why')}")
    if not rows:
        return 1

    print("\n--- P1 GATE: contraction, against the range four methods have measured ---")
    rho = [r["rho"] for r in rows]
    med = st.median(rho)
    print(f"  median rho {med:.4f}   min {min(rho):.4f}  max {max(rho):.4f}")
    print(f"  D31 measured 0.7935 / 0.8042 / 0.8083 on three prompts (mean 0.8020)")
    print(f"  gate: {'PASS' if 0.75 <= med <= 0.90 else 'FAIL -- nothing below may be read'}")

    print("\n--- P3: how broadly is the leading eigenvalue complex? ---")
    cx = sum(1 for r in rows if not r.get("lead_real", True))
    print(f"  complex in {cx}/{len(rows)} prompts   (D31: 3/3, which is what this widens)")
    nosc = [r.get("n_osc", 0) for r in rows]
    print(f"  oscillatory modes among top {rows[0].get('n_eigs')}: median {st.median(nosc):.1f}, "
          f"range {min(nosc)}-{max(nosc)}")

    print("\n--- P2 PRIMARY: implied rotation period, by regime label ---")
    per = {}
    for arm, nouns in (("rotating", ROT_NOUNS), ("settling", SET_NOUNS)):
        per[arm] = [r["period_unrolls"] for r in rows
                    if r.get("regime") == arm and r.get("period_unrolls")]
        print(f"  {arm:9s} {describe(per[arm])}")
        for n in nouns:
            v = [r["period_unrolls"] for r in rows
                 if r.get("label") == n and r.get("period_unrolls")]
            if v:
                print(f"     {n:9s} " + " ".join(f"{x:.2f}" for x in sorted(v)))
    unl = [r["period_unrolls"] for r in rows
           if r.get("regime") == "unlabelled" and r.get("period_unrolls")]
    print(f"  {'census':9s} {describe(unl)}")

    if per["rotating"] and per["settling"]:
        u, p = rank_sum_p(per["rotating"], per["settling"])
        sep = abs(st.median(per["rotating"]) - st.median(per["settling"]))
        spread = st.pstdev(per["rotating"] + per["settling"])
        print(f"\n  separation of medians {sep:.3f} unrolls, against a pooled sd of "
              f"{spread:.3f}  ->  {sep / spread if spread else float('nan'):.2f} sd")
        print(f"  Mann-Whitney U={u:.1f}  p={p:.4f}" if p is not None else f"  U={u}")
        n6 = {a: sum(1 for x in per[a] if 5.0 <= x <= 7.0) for a in per}
        print(f"  periods in 5-7 unrolls: rotating {n6['rotating']}/{len(per['rotating'])}, "
              f"settling {n6['settling']}/{len(per['settling'])}")
        print("\n  READING, registered before the run:")
        print("    (a) same phenomenon  -- rotating clusters near 6, settling does not;")
        print("    (b) separate lines   -- both arms alike, so the regime difference is not in")
        print("        the leading local dynamics, which bounds what the regime statistic reads.")

    print("\n--- P4: how long does the rotating mode survive? ---")
    tt = [r["turns_to_1pct"] for r in rows if r.get("turns_to_1pct")]
    if tt:
        print(f"  turns to 1% amplitude: {describe(tt)}   (D55 got ~4 on three prompts)")
        print("  this bounds any scheme that hopes to carry information around the loop by")
        print("  rotation: the carrier decays, and this is how many turns it lasts.")

    print("\n--- P5 FLOOR: does the contraction rate itself track the label? ---")
    for arm in ("rotating", "settling", "unlabelled"):
        v = [r["rho"] for r in rows if r.get("regime") == arm]
        if v:
            print(f"  {arm:11s} {describe(v)}")
    a = [r["rho"] for r in rows if r.get("regime") == "rotating"]
    b = [r["rho"] for r in rows if r.get("regime") == "settling"]
    if a and b:
        u, p = rank_sum_p(a, b)
        print(f"  rotating vs settling: U={u:.1f}" + (f"  p={p:.4f}" if p is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
