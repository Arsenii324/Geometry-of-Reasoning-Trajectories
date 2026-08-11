"""Re-locate A44's answer position WITHOUT using the gold, so wrong answers count too.

THE DEFECT THIS FIXES, WHICH THE SUPERVISOR CAUGHT BEFORE THE RUN LANDED. A44 locates the
answer position by searching the generation for the GOLD -- first an exact token match, then a
word-boundary containment scan. Both branches look for the right answer, so `ans_pos` is -1
whenever the model is wrong, and every downstream comparison is silently conditioned on
correctness.

The failure case is concrete: if the model writes `2 + 2 = 5` and the gold is `4`, we search for
`4`, find nothing, and lose the position where the model actually committed to an answer. The
geometry of a confident wrong answer -- arguably the more interesting object -- is exactly what
gets dropped.

WHY NO RE-RUN IS NEEDED. A44 banks `per_pos` with **every** generated position's emitted token
alongside its full geometry (`rank_curve` over all unrolls, `best_depth`, `rot`, `resid`). That
is enough to locate the answer by any rule, offline. So this uses the model's OWN committed
answer instead of the gold -- `last_number`, which D179 measured at accuracy 0.240 against a
false-positive rate of 0.024 (lift +0.217), and which returns a VALUE rather than a yes/no.
**Correctness becomes a covariate rather than a filter.**

WHAT IT ENABLES THAT A44'S OWN ANALYSIS CANNOT:

  * the answer position for items the model got WRONG, which is most of them on hard families
  * a correct-vs-incorrect split at the answer position -- does a wrong answer look different?
  * a check on A44's own locator: where both rules fire, do they agree?

    uv run python scripts/answerpos_analysis.py [path-to-answerpos.json]
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def locate_own(per_pos):
    """Position at which the model emitted the LAST number it will emit -- its own answer.

    Scans the emitted tokens, accumulates the running text, and returns the position whose
    token completes the final number in the generation. Falls back to the last non-stop
    position when the generation contains no digits at all.
    """
    text, spans = "", []
    for s in per_pos:
        start = len(text)
        text += s["tok"]
        spans.append((start, len(text), s["pos"]))
    # KNOWN FAILURE, MEASURED: `last_number` is structurally wrong wherever the answer is
    # stated BEFORE a distractor. On A44's `compare` family it mislabels 5 of 5 correct items
    # -- "83 is larger than 23." with gold 83 is right, and the last number is the loser.
    # Using it to define correctness there manufactured a depth-correctness effect
    # (p = 0.0359) that vanishes under a locator-independent label (p = 0.2901). See D184(3).
    nums = list(NUM.finditer(text))
    if not nums:
        return -1, None
    last = nums[-1]
    for a, b, pos in spans:
        if a < last.end() <= b:
            return pos, last.group(0)
    return spans[-1][2], last.group(0)


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "scratch/ds_answerpos/answerpos.json"
    if not path.exists():
        print(f"not found: {path}\n(A44 has not landed yet -- this script is ready for it)")
        return 1
    d = json.loads(path.read_text())
    rows = [r for r in d["rows"] if r.get("ok") and r.get("per_pos")]
    print(f"{len(rows)} generations banked\n")

    recs = []
    for r in rows:
        pos, val = locate_own(r["per_pos"])
        if pos < 0:
            continue
        correct = val is not None and val.strip() == str(r["gold"]).strip()
        recs.append({"family": r["family"], "gold": str(r["gold"]), "own": val,
                     "own_pos": pos, "gold_pos": r["ans_pos"], "correct": correct,
                     "p0": r["per_pos"][0], "pa": r["per_pos"][pos],
                     "others": [s for s in r["per_pos"] if s["pos"] not in (0, pos)],
                     "gen": r["gen"]})

    print(f"located the model's OWN answer in {len(recs)}/{len(rows)} generations "
          f"(A44's gold-based locator found {sum(1 for r in rows if r['ans_pos'] >= 0)})")
    n_wrong = sum(1 for r in recs if not r["correct"])
    print(f"  of those, {n_wrong} are WRONG -- invisible to the gold-based locator\n")

    agree = [r for r in recs if r["gold_pos"] >= 0]
    if agree:
        same = sum(1 for r in agree if r["own_pos"] == r["gold_pos"])
        print(f"LOCATOR CROSS-CHECK: where both fire (n={len(agree)}), they agree on "
              f"{same} ({same / len(agree):.2f})\n")

    print("=== RAW: what the model committed to, right and wrong ===")
    for r in ([x for x in recs if not x["correct"]][:6] +
              [x for x in recs if x["correct"]][:4]):
        tag = "OK " if r["correct"] else "WRONG"
        print(f"  [{tag}] {r['family']:12s} gold={r['gold']!r:6s} own={r['own']!r:6s} "
              f"pos={r['own_pos']:2d}  {r['gen']!r}")

    def block(label, get):
        vals = [v for r in recs for v in get(r)]
        if not vals:
            return
        print(f"  {label:24s} n={len(vals):4d}  best_depth {st.mean(v['best_depth'] for v in vals):5.2f}"
              f"   rot {st.mean(v['rot'] for v in vals):.3f}"
              f"   resid {st.mean(v['resid'] for v in vals):.4f}")

    print("\n=== GEOMETRY BY POSITION TYPE (all items, right and wrong) ===")
    block("position 0", lambda r: [r["p0"]])
    block("own-answer position", lambda r: [r["pa"]])
    block("other positions", lambda r: r["others"])

    print("\n=== THE SPLIT A44 COULD NOT MAKE: correct vs wrong, at the answer position ===")
    for want, label in ((True, "correct"), (False, "WRONG")):
        sub = [r for r in recs if r["correct"] is want]
        if not sub:
            continue
        print(f"  {label:8s} n={len(sub):3d}  best_depth "
              f"{st.mean(r['pa']['best_depth'] for r in sub):5.2f}"
              f"   rot {st.mean(r['pa']['rot'] for r in sub):.3f}"
              f"   resid {st.mean(r['pa']['resid'] for r in sub):.4f}"
              f"   own_pos {st.mean(r['own_pos'] for r in sub):5.2f}")
    cw = [r["pa"]["best_depth"] for r in recs if r["correct"]]
    ww = [r["pa"]["best_depth"] for r in recs if not r["correct"]]
    if len(cw) > 4 and len(ww) > 4:
        try:
            from scipy.stats import mannwhitneyu
            print(f"  Mann-Whitney on best_depth at the answer position: "
                  f"p = {mannwhitneyu(cw, ww).pvalue:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (mannwhitneyu unavailable: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
