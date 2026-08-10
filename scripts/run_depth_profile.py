"""Does depth make Huginn's answer better, and is `correct` a property of the model?

Reads two banks already on disk and writes `results/depth_profile.csv`. No GPU.
OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

WHAT PROVOKED THIS. D90 recorded that h_0 "decides the answer": across ten forwards
of one prompt differing only in the unseeded initial latent, gold's rank moved by up
to 6x. That reading is wrong, and the manifest it was drawn from already contained
the refutation. `best_rank` -- a MINIMUM OVER UNROLLS -- moves across h_0. The rank
at the LAST unroll does not: it is one integer, identical across all ten draws, in
seven of the eight prompts. So h_0 is forgotten by the time the recurrence converges,
exactly as a contraction requires, and what h_0 actually controls is the depth of a
TRANSIENT EXCURSION. See `depth_profile.md` for what that costs the project.

THE THREE QUESTIONS, in the order the script answers them.

(1) IS `correct` REALISABLE? This project labels an item correct when
    `min_r rank_r == 1`. That is an oracle: it credits the model for any unroll at
    which gold happened to top the distribution, including unrolls the model would
    never stop at. The honest comparison is against the best a FIXED depth attains,
    which is the only thing a user can choose.

(2) DOES ACCURACY DECAY WITH DEPTH, and if so where does the decay stop? Tested
    family-paired, because the 504 orbits are 21 clusters of 24 and a pooled
    binomial over them would count within-family repetition as independent evidence.
    The unit is the family; the test is an exact sign test; r = 5 is fixed a priori
    from the pooled median-rank curve and is NOT re-selected per family.

(3) IS THE ANSWER LOST, OR RELOCATED? Rank is measured at ONE position -- the token
    the answer would start at. An answer emitted as "The number 4 is repeated
    exactly." is correct and scores zero there. The generation bank settles it, and
    it needs a chance control: outputs grow ~1.45x longer with depth, so containment
    of a short gold string rises for free. The control is containment of a DECOY --
    the gold of a different item in the SAME family, so token type, length and
    register are matched and only correctness differs.

WHAT THIS SCRIPT DOES NOT SHOW. It cannot locate WHERE in the generated text the
answer moved to, because the rank bank measures one position only. The mechanism
proposed on first reading -- "depth prepends a conversational preamble and pushes
gold out of first place" -- IS TESTED HERE (`preamble_split`) AND DOES NOT HOLD:
at depth 32 the median rank is 7.0 when the generation opens with a preamble word
and 7.5 when it does not. The word-list proxy is too coarse to be the mechanism.
`scratch/kaggle_answerpos/` is the run that settles it by teacher-forcing the
model's own continuation and ranking gold at every position.

Run: uv run python -m scripts.run_depth_profile
"""

from __future__ import annotations

import collections
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

GEOMCAP = os.path.join("scratch", "kaggle_geomcap", "out", "manifest.json")
DEPTHACC = os.path.join("scratch", "kaggle_depthacc", "out", "depthacc.json")
DEPTHRANK = os.path.join("scratch", "kaggle_depthacc", "out", "depthrank.json")
OUT = os.path.join("results", "depth_profile.csv")

# Fixed a priori from the pooled median-rank curve, before any family-level test.
PEAK_R = 5
# Probe depths reported against the converged tail. r=64 is the reference.
PROBE_R = (2, 4, 5, 6, 8, 12, 16, 24, 32, 48)
# Openers that mark a composed sentence rather than a blurted answer. Deliberately
# generous: this is a proxy under test, not a definition, and it FAILS (see above).
PREAMBLE = frozenset("the to answer sure here we let this in a i if first since "
                     "given based so now step".split())


def _curves(rows: list[dict], block: str = "main") -> dict[str, np.ndarray]:
    """{family: [n_items, n_unrolls] gold-rank curves} for one bank block."""
    fam: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r.get("block") == block and r.get("ok"):
            fam[r["family"]].append(r["rank_curve"])
    return {f: np.asarray(c, dtype=float) for f, c in sorted(fam.items())}


def oracle_gap(curves: dict[str, np.ndarray]) -> dict:
    """The project's `correct` label against the best any FIXED depth attains.

    `correct = (min_r rank_r == 1)` is a minimum over unrolls, so it can only
    exceed every fixed-depth accuracy. The gap is the part of the project's
    capability axis that no choice of depth realises.
    """
    allc = np.concatenate(list(curves.values()), axis=0)
    per_r = (allc == 1).mean(axis=0)
    best_r = int(np.argmax(per_r)) + 1
    return {
        "n_orbits": int(allc.shape[0]),
        "oracle_acc": float((allc.min(axis=1) == 1).mean()),
        "best_fixed_acc": float(per_r.max()),
        "best_fixed_r": best_r,
        "final_acc": float(per_r[-1]),
    }


def paired_depth_test(curves: dict[str, np.ndarray], r: int,
                      ref: int | None = None) -> dict:
    """Exact sign test of accuracy at unroll `r` against the converged tail.

    The unit is the FAMILY, not the orbit. 504 orbits are 21 clusters of 24 items
    drawn from one generator each; pooling them would treat within-family
    repetition as independent evidence and shrink the p-value by roughly sqrt(24).
    Ties are dropped, which is what an exact sign test requires.
    """
    names = list(curves)
    ref = ref if ref is not None else curves[names[0]].shape[1]
    a = np.array([(curves[f][:, r - 1] == 1).mean() for f in names])
    b = np.array([(curves[f][:, ref - 1] == 1).mean() for f in names])
    d = a - b
    nz = d[d != 0]
    p = (float(stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue)
         if len(nz) else 1.0)
    return {"r": r, "acc": float(a.mean()), "acc_ref": float(b.mean()),
            "better": int((d > 0).sum()), "tied": int((d == 0).sum()),
            "worse": int((d < 0).sum()), "n_fam": len(names), "p_sign": p}


def h0_forgotten(rows: list[dict]) -> pd.DataFrame:
    """Across unseeded replicates of ONE prompt: does the LAST rank vary?

    This is the D90 correction. `best_rank` is a minimum over unrolls and moves;
    the converged rank is a single state's readout and, if the recurrence forgets
    h_0, cannot move. Reporting both side by side is the whole point -- either
    column alone is the misleading one.
    """
    grp: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        if r.get("block") == "rep" and r.get("ok"):
            grp[(r["family"], r["item"])].append(r)
    out = []
    for (fam, item), v in grp.items():
        fin = [x["rank_curve"][-1] for x in v]
        best = [x["best_rank"] for x in v]
        out.append({
            "family": fam, "item": item, "n_draws": len(v),
            "final_rank_distinct": len(set(fin)),
            "final_rank_spread": int(max(fin) - min(fin)),
            "best_rank_distinct": len(set(best)),
            "best_rank_ratio": float(max(best) / max(1, min(best))),
        })
    return pd.DataFrame(out)


# Chat-template role markers that leak into the decoded string with NO separator:
# `echo_digit` at depth 2 emits "4Huginn\n\n", which is a correct blurt followed by
# the next turn's header. Left in place they defeat the word-boundary match on
# exactly the low-depth cells where the model blurts, which is the direction that
# would MANUFACTURE the reported rise in containment. Measured before fixing: 10 /
# 12 / 6 / 14 / 12 cells at depths 2 / 4 / 8 / 16 / 32 -- real but not monotone in
# depth, so it was never the trend, and it is removed anyway.
_ROLE = re.compile(r"(?:Huginn|user|assistant|system)\b")


def _contains(text: str, needle: str) -> bool:
    """Gold present as a standalone token, so "4" does not match "42" or "user4".

    Boundary-matched rather than substring-matched: the generations are chatty and
    full of digits, and a substring test would score "The answer is 42." as
    containing both 4 and 2. Note this deliberately still refuses "10:3" for gold
    "1" -- that is a different number, not a delimiter-free answer.
    """
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
                     _ROLE.sub(" ", text)) is not None


def containment_with_decoy(acc_rows: list[dict], fmt: str = "bare") -> pd.DataFrame:
    """Is the answer relocated rather than lost -- against a length-matched control.

    Outputs lengthen with depth, so a short gold string is contained more often for
    free. The decoy is the gold of a DIFFERENT item in the SAME family: same token
    type, same length distribution, same register, wrong answer. A rise in gold
    containment means something only if decoy containment does not rise with it.
    """
    by_fam: dict[str, set] = collections.defaultdict(set)
    for r in acc_rows:
        by_fam[r["family"]].add(r["gold"])
    out = []
    for d in sorted({r["depth"] for r in acc_rows}):
        s = [r for r in acc_rows if r["depth"] == d and r["fmt"] == fmt]
        if not s:
            continue
        gold = float(np.mean([_contains(r["output"], r["gold"]) for r in s]))
        dec = []
        for r in s:
            alt = [g for g in by_fam[r["family"]] if g != r["gold"]]
            if alt:
                dec.append(float(np.mean([_contains(r["output"], g) for g in alt])))
        decoy = float(np.mean(dec)) if dec else float("nan")
        # Floor the denominator at the resolution of the measurement rather than
        # reporting NaN when no decoy is ever matched. A decoy rate of exactly 0
        # over n cells is not evidence of infinite specificity, it is evidence that
        # the rate is below 1/n; dividing by that floor says so and keeps the
        # column readable and monotone instead of dropping out at the best depth.
        res = 1.0 / max(1, len(s))
        out.append({"fmt": fmt, "depth": d, "n": len(s), "contains_gold": gold,
                    "contains_decoy": decoy,
                    "specificity": gold / max(decoy, res) if np.isfinite(decoy)
                    else float("nan"),
                    "decoy_at_floor": bool(np.isfinite(decoy) and decoy < res),
                    "out_len": float(np.mean([len(r["output"]) for r in s]))})
    return pd.DataFrame(out)


def preamble_split(acc_rows: list[dict], rank_rows: list[dict]) -> pd.DataFrame:
    """The refuted mechanism, kept because a reader will propose it.

    If depth pushed gold out of first place by prepending conversational framing,
    then at a fixed depth the generations that open with framing should show a much
    worse gold rank than those that do not. They do not.
    """
    def opens_with_preamble(t: str) -> bool:
        m = re.match(r"\s*([A-Za-z]+)", t)
        return bool(m) and m.group(1).lower() in PREAMBLE

    def key(r: dict) -> tuple:
        return (r["family"], r["item"], r["fmt"], r["depth"])

    rmap = {key(r): r for r in rank_rows if r.get("ok")}
    out = []
    for d in sorted({r["depth"] for r in acc_rows}):
        pre, non = [], []
        for r in acc_rows:
            if r["depth"] != d or key(r) not in rmap:
                continue
            (pre if opens_with_preamble(r["output"]) else non).append(
                rmap[key(r)]["rank"])
        out.append({
            "depth": d, "n_preamble": len(pre), "n_plain": len(non),
            "preamble_rate": len(pre) / max(1, len(pre) + len(non)),
            "med_rank_preamble": float(np.median(pre)) if pre else float("nan"),
            "med_rank_plain": float(np.median(non)) if non else float("nan")})
    return pd.DataFrame(out)


def main() -> int:
    geo = json.load(open(GEOMCAP))
    acc = json.load(open(DEPTHACC))
    rnk = json.load(open(DEPTHRANK))
    curves = _curves(geo)
    n_unroll = next(iter(curves.values())).shape[1]

    og = oracle_gap(curves)
    print("=== (1) is `correct` realisable? ===")
    print(f"  orbits                       {og['n_orbits']}")
    print(f"  oracle  min_r rank==1        {og['oracle_acc']*100:5.1f}%   <- the "
          "project's `correct`")
    print(f"  best FIXED depth (r={og['best_fixed_r']})       "
          f"{og['best_fixed_acc']*100:5.1f}%")
    print(f"  converged  r={n_unroll}              {og['final_acc']*100:5.1f}%")
    print(f"  unrealisable share of `correct`  "
          f"{(1 - og['best_fixed_acc']/og['oracle_acc'])*100:.0f}%")

    print(f"\n=== (2) accuracy vs depth, family-paired against r={n_unroll} ===")
    tests = [paired_depth_test(curves, r) for r in PROBE_R]
    for t in tests:
        print(f"  r={t['r']:2d}  acc={t['acc']*100:5.1f}%  "
              f"better/tied/worse={t['better']}/{t['tied']}/{t['worse']}  "
              f"p={t['p_sign']:.4f}")
    peak = next(t for t in tests if t["r"] == PEAK_R)
    print(f"  a priori peak r={PEAK_R}: {peak['acc']*100:.1f}% vs "
          f"{peak['acc_ref']*100:.1f}%, sign p={peak['p_sign']:.2e}")

    print("\n=== (2b) h_0: forgotten at convergence, decisive in the transient ===")
    h0 = h0_forgotten(geo)
    if not h0.empty:
        print(f"  prompts with a SINGLE converged rank across draws: "
              f"{int((h0.final_rank_distinct == 1).sum())}/{len(h0)}")
        print(f"  prompts whose best_rank moves across draws:        "
              f"{int((h0.best_rank_distinct > 1).sum())}/{len(h0)}"
              f"   (max ratio {h0.best_rank_ratio.max():.1f}x)")

    print("\n=== (3) lost or relocated? gold vs length-matched decoy ===")
    cont = pd.concat([containment_with_decoy(acc, f) for f in ("bare",
                                                               "constrained")])
    for _, r in cont.iterrows():
        print(f"  {r.fmt:12s} d={int(r.depth):3d}  gold={r.contains_gold*100:5.1f}%"
              f"  decoy={r.contains_decoy*100:5.1f}%"
              f"  specificity={r.specificity:5.2f}x  len={r.out_len:5.1f}")

    print("\n=== (3b) the preamble mechanism -- REFUTED, kept on the record ===")
    ps = preamble_split(acc, rnk)
    for _, r in ps.iterrows():
        print(f"  d={int(r.depth):3d}  preamble={r.preamble_rate*100:5.1f}%  "
              f"med rank | preamble={r.med_rank_preamble:6.1f}  "
              f"| plain={r.med_rank_plain:6.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = [
        pd.DataFrame(tests).assign(arm="depth_sign_test"),
        cont.assign(arm="containment"),
        ps.assign(arm="preamble_split"),
        h0.assign(arm="h0_replicates"),
        pd.DataFrame([og]).assign(arm="oracle_gap"),
    ]
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
