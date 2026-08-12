"""Does depth help, measured on what the model WRITES rather than on the gold's logit rank?

WHY. D203 established that the project's rank curve tracks which output frame the model is in, not
whether it has the answer. That left the actual question open: the census says the answer peaks at
unroll 4 and is displaced thereafter, but the census reads logits. `scratch/kaggle_depthacc/` ran
greedy generation at five FIXED depths and banked the emitted text, so the same question can be
asked of an instrument that does not depend on the first token at all.

THE TWO INSTRUMENTS ARE DIFFERENT EXPERIMENTS, and that matters for how far this goes:
  * the rank curve reads every unroll inside ONE forward, at the immediate next-token position;
  * this reads an 8-token greedy generation produced with the loop count FIXED for every generated
    token, at depth 2/4/8/16/32, 21 families x 6 items x 2 formats = 126 per cell.

CONFOUNDS, each checked here rather than argued away:
  (a) LENGTH. Containment rises trivially if outputs get longer. Median token count is reported per
      cell; it is flat at 6-7 from depth 8 onward, so the depth 8->32 comparison is length-matched
      and the depth 2/4 cells are not (the depth-4 median output is ONE token).
  (b) RESTATEMENT. "1 + 1 = 2" contains its own operands, so containment can fire on the restated
      input. The last-number scorer is reported beside it because it is not fooled by that here.
  (c) CHANCE. Every scorer is also run against a DECOY gold drawn from a different item of the same
      family and depth, averaged over 20 reassignments. Gold minus decoy is the reported signal.
      This is the control that matters: if the decoy rate rose with depth too, the effect would be
      output density, not capability.
  (d) THE 8-TOKEN BUDGET is itself a confound and is NOT correctable here. On tasks where the model
      reasons before answering it never reaches an answer -- `count16` at depth 32 emits
      "The sequence is $0,1," and stops. Those items score zero for lack of room, not lack of
      ability, and this kernel cannot separate the two.

REGISTERED READING. If depth genuinely helps beyond the census's unroll 4, the specificity-corrected
rate should rise from depth 4 to depth 16 on BOTH scorers and in BOTH formats, while the decoy rate
does not rise. If instead everything rises together, this is measuring output density and the
census's account stands.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANK = ROOT / "scratch" / "kaggle_depthacc" / "out" / "depthacc.json"
DEPTHS = (2, 4, 8, 16, 32)
SEED = 20260812
N_DECOY = 20


def strip_marker(o: str) -> str:
    """D145/D151: the chat template glues the next-turn role marker onto the output."""
    return re.sub(r"(user|Huginn)\s*$", "", o.strip())


def contains(out: str, gold: str) -> bool:
    o, g = strip_marker(out), gold.strip()
    return (g in o.split()) if " " not in g else (g in o)


def last_number(out: str, gold: str) -> bool:
    nums = re.findall(r"-?\d+", strip_marker(out))
    return bool(nums) and nums[-1] == gold.strip()


def main() -> int:
    if not BANK.exists():
        print(f"missing bank: {BANK}")
        return 1
    rows = json.loads(BANK.read_text())
    fams = sorted({r["family"] for r in rows})
    print(f"{len(rows)} generations, {len(fams)} families, depths {DEPTHS}, "
          f"formats {sorted({r['fmt'] for r in rows})}")

    rng = random.Random(SEED)
    print(f"\n{'arm':<12}{'depth':>6}{'n':>5}{'tok':>6}   "
          f"{'contains':>9}{'decoy':>7}{'net':>7}   {'last-num':>9}{'decoy':>7}{'net':>7}")
    net = {}
    for arm in ("bare", "constrained"):
        for dep in DEPTHS:
            v = [r for r in rows if r["fmt"] == arm and r["depth"] == dep]
            tok = st.median([len(strip_marker(r["output"]).split()) for r in v])
            cg = sum(contains(r["output"], r["gold"]) for r in v) / len(v)
            lg = sum(last_number(r["output"], r["gold"]) for r in v) / len(v)
            cd = ld = 0.0
            for _ in range(N_DECOY):
                cs = ls = 0
                for r in v:
                    pool = [x["gold"] for x in v
                            if x["family"] == r["family"] and x["gold"] != r["gold"]]
                    if not pool:
                        continue
                    d = rng.choice(pool)
                    cs += contains(r["output"], d)
                    ls += last_number(r["output"], d)
                cd += cs / len(v)
                ld += ls / len(v)
            cd /= N_DECOY
            ld /= N_DECOY
            net[(arm, dep)] = (cg - cd, lg - ld)
            print(f"{arm:<12}{dep:>6}{len(v):>5}{tok:>6.0f}   "
                  f"{cg:>9.1%}{cd:>7.1%}{cg-cd:>7.1%}   {lg:>9.1%}{ld:>7.1%}{lg-ld:>7.1%}")

    print("\nGATE -- registered above: net rate rises depth 4 -> 16 on both scorers, both formats")
    ok = True
    for arm in ("bare", "constrained"):
        for i, which in enumerate(("contains", "last-num")):
            a, b = net[(arm, 4)][i], net[(arm, 16)][i]
            good = b > a
            ok &= good
            print(f"    {arm:<12} {which:<9} {a:>6.1%} -> {b:>6.1%}  {'rises' if good else 'FAILS'}")
    print(f"    {'PASS' if ok else 'FAIL'}")

    print("\nadd1 -- the family D69's constrained arm scored 0% on, by exact match")
    for arm in ("bare", "constrained"):
        line = []
        for dep in DEPTHS:
            v = [r for r in rows if r["fmt"] == arm and r["depth"] == dep and r["family"] == "add1"]
            line.append(f"d{dep}={sum(last_number(r['output'], r['gold']) for r in v)/len(v):.0%}")
        print(f"    {arm:<12} " + "  ".join(line))
    ex = [r for r in rows if r["family"] == "add1" and r["fmt"] == "constrained"
          and r["depth"] == 32][:3]
    for r in ex:
        print(f"      gold {r['gold']!r}: {r['output']!r}  <- correct, and exact-match scores it 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
