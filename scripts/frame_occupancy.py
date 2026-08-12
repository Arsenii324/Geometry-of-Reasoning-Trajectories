"""What is actually on top when the gold is not? Re-derives every number in D203. Zero GPU.

WHY THIS EXISTS. `scripts/relay_phases.py` read the census rank curve as if it measured whether the
model has the answer at unroll t. It does not: the census ranks the gold's FIRST token in the raw
next-token logits at the immediate next position, and this model writes prose. The supervisor asked
whether the curve could be reading the task's formatting rather than the model. It is. This kernel
is the check, and it is the version of the analysis that survives -- D151(3) records what happens
when a claim's analysis lives in a heredoc that is then lost.

THE THREE CONFOUNDS, each measured here, each sufficient on its own:

  (a) TOKEN TYPE. 21% of census rows have multi-token golds, ranked by their first token. A word is
      not a plausible thing to say first, so word-golds start deep for reasons that have nothing to
      do with what the model knows.

  (b) OUTPUT FRAME. If the curve tracks frame rather than content, then across items with DIFFERENT
      golds the top-1 token should be the SAME FEW IDS early and late, and item-specific only in the
      brief window where the gold wins. That is a tokeniser-free test and it is the decisive one.

  (c) NO UNTRAINED CONTROL. The census has no null. `kaggle_battery` banked an untrained arm; if it
      reproduces the early-diversity-then-collapse shape, the shape is what iterating a contraction
      does to a readout, not something training built.

The decoded strings for the ids this prints are in `scratch/kaggle_discourse/out/geometry-discourse.log`
(top1@r64 column, bare arm): "The" on 24/24 add1, "The"/"Number" on echo_digit, "To" on 23/24
count16, "The" on 24/24 rot13_word. No local tokeniser is needed and none is assumed.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
CENSUS = ROOT / "scratch" / "kaggle_census" / "out" / "census.json"
BATTERY = ROOT / "scratch" / "kaggle_battery" / "out" / "battery.json"
COUNTING = ("count16", "count4", "count_mod3")


def gold_type(g: str) -> str:
    g = g.strip()
    if g.lstrip("-").isdigit():
        return "digit" if len(g.lstrip("-")) == 1 else "multidigit"
    if len(g) == 1 and g.isalpha():
        return "letter"
    return "word"


def census_part() -> None:
    rows = [r for r in json.loads(CENSUS.read_text())["rows"]
            if r.get("ok") and r.get("rank_curve")]
    solved = [r for r in rows if min(r["rank_curve"]) == 1]
    mt = sum(1 for r in rows if r.get("multi_token_gold"))
    print(f"CENSUS  {len(rows)} draws, {len({r['family'] for r in rows})} families, "
          f"{len(rows[0]['rank_curve'])} unrolls")
    print(f"  solved at some depth: {len(solved)} ({len(solved)/len(rows):.1%})")
    print(f"  multi-token golds, ranked by FIRST token only: {mt} ({mt/len(rows):.1%})")

    print("\n(a) TOKEN TYPE -- median gold rank after ONE unroll, solved draws")
    by = collections.defaultdict(list)
    for r in solved:
        by[gold_type(r["gold"])].append(r)
    print(f"    {'gold type':<12} {'n':>5} {'rank@1':>8} {'climb':>7}   families")
    for t, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        climb = [math.log10(max(x["rank_curve"][0], 1)) for x in v]
        print(f"    {t:<12} {len(v):>5} {st.median([x['rank_curve'][0] for x in v]):>8.0f} "
              f"{st.median(climb):>7.2f}   {','.join(sorted({x['family'] for x in v}))[:52]}")
    print("    -> a word gold starts ~25x deeper than a digit gold. That is the tokeniser and the")
    print("       discourse frame, not the model knowing word answers less well.")

    print("\n    the split that misled the first version of D203, kept because it is real:")
    for lbl, sel in (("counting families (what D37 measured)", lambda r: r["family"] in COUNTING),
                     ("the other eighteen", lambda r: r["family"] not in COUNTING)):
        v = [r for r in solved if sel(r)]
        climb = [math.log10(max(r["rank_curve"][0], 1)) for r in v]
        print(f"      {lbl:<38} n={len(v):>5}  rank@1 {st.median([r['rank_curve'][0] for r in v]):>4.0f}"
              f"  climb {st.median(climb):.2f} log10 ({10**st.median(climb):.0f}x)")
    print("      -- but (b) below shows the climb is frame exit, so this does NOT show loops build.")


def battery_part() -> None:
    d = json.loads(BATTERY.read_text())
    print(f"\nBATTERY  vocab {d['vocab']}, {d['max_r']} unrolls, arms {list(d['arms'])}")
    for arm in d["arms"]:
        A = d["arms"][arm]
        recs = [it for items in A.values() for it in items if it.get("top1")]
        n = len(recs)
        print(f"\n(b/c) {arm.upper()} -- {n} items over {len(A)} families")
        print(f"      {'when':<18} {'distinct top-1':>15} {'most common id':>16} {'share':>7}")
        for label, idx in (("unroll 1", 0), ("unroll 4", 3), ("unroll 16", 15),
                           ("final unroll", -1)):
            c = collections.Counter(it["top1"][idx] for it in recs)
            tid, cnt = c.most_common(1)[0]
            print(f"      {label:<18} {len(c):>10}/{n:<4} {tid:>16} {cnt/n:>6.0%}")
        c = collections.Counter()
        for it in recs:
            c[it["top1"][it["rank"].index(min(it["rank"]))]] += 1
        tid, cnt = c.most_common(1)[0]
        print(f"      {'own best unroll':<18} {len(c):>10}/{n:<4} {tid:>16} {cnt/n:>6.0%}")
        dom = collections.Counter(it["top1"][-1] for it in recs).most_common(1)[0][0]
        frac = [sum(1 for t in it["top1"] if t == dom) / len(it["top1"]) for it in recs]
        print(f"      dominant final id {dom} holds top-1 on a median {st.median(frac):.0%} "
              f"of all unrolls")


def main() -> int:
    for p in (CENSUS, BATTERY):
        if not p.exists():
            print(f"missing bank: {p}")
            return 1
    census_part()
    battery_part()
    print("""
READING (D203)
  The trained arm concentrates on a handful of top-1 tokens early and late and diversifies only
  around the unroll where the gold wins. The dominant token decodes as "The". So the census curve's
  climb is the model leaving a prose frame and its drop is the model returning to one -- neither is
  evidence about where the answer is computed.

  The untrained arm collapses to a SINGLE token on every item while still showing diversity at its
  own best unroll, so the curve's shape does not require training to produce.

  D69 settles it causally on the one family where the model is competent: bare `echo_digit` is 0%
  at r=64 and 83% under "Reply with only the answer", same model, same depth.

  CONSEQUENCE. Any per-loop "does it have the answer yet" diagnostic read off raw next-token logits
  measures output formatting on a model free to write prose. Use a probe on the state, constrain the
  format, or read at the position where the answer is due.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
