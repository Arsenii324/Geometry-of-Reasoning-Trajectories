"""Does "loops relay and stabilise rather than build" hold beyond the one task it was measured on?

RESULT, WRITTEN AFTER RUNNING -- READ THIS BEFORE THE PRE-REGISTRATION BELOW.

  It does not answer the question, and the reason is the interesting part. This kernel reads
  `rank_curve`, which the census defines as the rank of the gold's FIRST token in the model's own
  logits at the IMMEDIATE next-token position (`scratch/kaggle_census/body.py:229-244`). On a model
  that writes prose, that measures whether the model is about to say "The" -- not whether it has
  the answer. Across 336 items in `scratch/kaggle_battery/` there are only 23 distinct top-1 tokens
  at unroll 1, and one of them holds top-1 on a median 88% of ALL unrolls; the decoded strings are
  in `scratch/kaggle_discourse/out/geometry-discourse.log` and they are "The", "To", "Number",
  "There". D69 had already shown one format instruction moves `echo_digit` from 0% to 83% at r=64.

  So P2's "climb" is the model briefly leaving a prose frame near unroll 4 and P3's "drop" is it
  returning. See D203, which records what this kernel does and does not license. The numbers it
  prints are correct as a description of the bare-prompt readout trajectory and must not be read as
  evidence about where computation happens. `scripts/frame_occupancy.py` is the follow-up that
  measures the frame directly.

WHY. `docs/REPORT_LOOPED_TRANSFORMERS.md` §1 makes its most ambitious claim on the strength of two
rows measured on a SINGLE counting task: the answer is linearly decodable from the state after one
loop and gets no more decodable, while what the remaining loops do is rotate it into a stable form
and carry it to the position the readout reads. If that generalises it reframes the saturation
question; if it is one task's quirk it should not be the headline. Nobody has checked, and the
check is free -- `scratch/kaggle_census/out/census.json` holds 4868 draws across 21 families with
a 48-loop rank curve each.

WHAT THIS CAN AND CANNOT TEST -- read before quoting it.
  * It CANNOT test the decodability half. That needs per-loop states plus a probe, and the banked
    state grids (`ds_pertoken`) hold one trajectory per family, which cannot fit a probe.
  * It CAN test the half that matters for the framing: the **readout lag**. If information is
    present early but the head does not surface it until much later, the gap between "best rank so
    far" and "rank right now" is exactly the transport phase, and its size across 21 families is
    the number the report should be quoting instead of one task's.

PRE-REGISTERED PREDICTIONS, written before running:

  P1  GATE. Reproduce two published numbers from this same bank, or the read is wrong:
      median best_depth = 4, and the gold's final rank worse than its best in 17 of 21 families.

  P2  THE CLIMB. If loops mostly transport rather than build, the answer should already be
      *findable* early: median rank at loop 1 should be small relative to the vocabulary (65k),
      and the improvement from loop 1 to the best rank should be modest in log terms. If instead
      rank at loop 1 is near-random, loops are doing something more than transport and §1
      overstates.

  P3  THE DISPLACEMENT. After the best loop, rank should worsen and then hold -- the report claims
      convergence removes information. Measured as best -> final, per family.

  P4  SPECIFICITY. All three should vary by family. If every family behaves identically the
      statistic is measuring the protocol, not the model (this project has been caught by that
      exact failure once already, D28).

  P5  FLOOR. Restricted to draws the model actually gets right at some depth, since a rank curve
      for an item it cannot do says nothing about relaying an answer it never had.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANK = ROOT / "scratch" / "kaggle_census" / "out" / "census.json"


def load():
    d = json.loads(BANK.read_text())
    return [r for r in d["rows"] if r.get("ok") and r.get("rank_curve")]


def main() -> int:
    if not BANK.exists():
        print(f"bank not found: {BANK}")
        return 1
    rows = load()
    fams = sorted({r["family"] for r in rows})
    print(f"{len(rows)} usable draws over {len(fams)} families, "
          f"{len(rows[0]['rank_curve'])} loops each")

    # ---------------------------------------------------------------- P1 gate
    print("\nP1  GATE -- reproduce two published numbers from this bank")
    med_bd = st.median([r["best_depth"] for r in rows])
    worse = 0
    for f in fams:
        v = [r for r in rows if r["family"] == f]
        mb = st.median([min(r["rank_curve"]) for r in v])
        mf = st.median([r["rank_curve"][-1] for r in v])
        worse += mf > mb
    print(f"    median best_depth = {med_bd:.0f}   (published: 4)")
    print(f"    families whose median final rank is worse than best: {worse}/{len(fams)}"
          f"   (published: 17/21)")
    ok1 = (abs(med_bd - 4) < 0.5) and (worse >= 15)
    print(f"    {'PASS' if ok1 else 'FAIL -- the read is wrong, nothing below counts'}")
    if not ok1:
        return 1

    # ---------------------------------------------------------------- P5 floor
    solved = [r for r in rows if min(r["rank_curve"]) == 1]
    print(f"\nP5  FLOOR -- restrict to draws solved at some depth: "
          f"{len(solved)}/{len(rows)} ({len(solved)/len(rows):.1%})")

    def phase(v):
        r1 = [x["rank_curve"][0] for x in v]
        rb = [min(x["rank_curve"]) for x in v]
        rf = [x["rank_curve"][-1] for x in v]
        bd = [x["best_depth"] for x in v]
        climb = [math.log10(max(a, 1) / max(b, 1)) for a, b in zip(r1, rb)]
        drop = [math.log10(max(c, 1) / max(b, 1)) for c, b in zip(rf, rb)]
        return r1, rb, rf, bd, climb, drop

    print("\nP2/P3  the three phases, per family (solved draws only)")
    print(f"    {'family':<14} {'n':>4} {'rank@1':>8} {'best':>6} {'final':>7} "
          f"{'bestloop':>8} {'climb':>7} {'drop':>6}")
    print(f"    {'':<14} {'':>4} {'median':>8} {'':>6} {'':>7} {'':>8} "
          f"{'log10':>7} {'log10':>6}")
    per_fam = {}
    for f in fams:
        v = [r for r in solved if r["family"] == f]
        if len(v) < 5:
            continue
        r1, rb, rf, bd, climb, drop = phase(v)
        per_fam[f] = (st.median(climb), st.median(drop), st.median(r1))
        print(f"    {f:<14} {len(v):>4} {st.median(r1):>8.0f} {st.median(rb):>6.0f} "
              f"{st.median(rf):>7.0f} {st.median(bd):>8.0f} "
              f"{st.median(climb):>7.2f} {st.median(drop):>6.2f}")

    r1, rb, rf, bd, climb, drop = phase(solved)
    print(f"\n    {'ALL':<14} {len(solved):>4} {st.median(r1):>8.0f} {st.median(rb):>6.0f} "
          f"{st.median(rf):>7.0f} {st.median(bd):>8.0f} "
          f"{st.median(climb):>7.2f} {st.median(drop):>6.2f}")

    print("\nP2  THE CLIMB -- is the answer already findable at loop 1?")
    print(f"    median rank at loop 1: {st.median(r1):.0f} of ~65000 vocabulary")
    print(f"    fraction of solved draws already at rank 1 after ONE loop: "
          f"{sum(1 for x in r1 if x == 1)/len(r1):.1%}")
    print(f"    fraction already in the top 10 after one loop: "
          f"{sum(1 for x in r1 if x <= 10)/len(r1):.1%}")
    print(f"    median climb loop1 -> best: {st.median(climb):.2f} log10 "
          f"({10**st.median(climb):.1f}x)")

    print("\nP3  THE DISPLACEMENT -- does it degrade after the best loop?")
    print(f"    median drop best -> final: {st.median(drop):.2f} log10 "
          f"({10**st.median(drop):.1f}x worse)")
    print(f"    solved draws still at rank 1 at the last loop: "
          f"{sum(1 for x in rf if x == 1)/len(rf):.1%}")

    print("\nP4  SPECIFICITY -- does any of this vary by family?")
    cl = [v[0] for v in per_fam.values()]
    dr = [v[1] for v in per_fam.values()]
    print(f"    climb across families: {min(cl):.2f} to {max(cl):.2f} log10 "
          f"(sd {st.pstdev(cl):.2f})")
    print(f"    drop  across families: {min(dr):.2f} to {max(dr):.2f} log10 "
          f"(sd {st.pstdev(dr):.2f})")
    ok4 = st.pstdev(cl) > 0.05 or st.pstdev(dr) > 0.05
    print(f"    {'PASS -- families differ, so this is not a protocol artefact' if ok4 else 'FAIL -- identical across families, suspect the protocol'}")

    print("\nREADING")
    print("    'loops relay rather than build' predicts a SMALL climb and a real drop.")
    print("    A large climb means loops are improving availability, not just transporting it,")
    print("    and the report's section 1 would be overstating a one-task result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
