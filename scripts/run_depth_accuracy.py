"""Does the model SAY what its rank curve says it knows, and does depth help or hurt?

Reads `scratch/kaggle_depthacc/out/{depthacc,depthrank}.json` and writes
`results/depth_accuracy.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented
2026-08-09.

THE GAP THIS CLOSES IS ONE THE PROJECT NAMED ITSELF. D68(5): "That generation
accuracy would be higher at r=4 is NOT established -- this measures teacher-forced
RANK at the FIRST answer position, not a decoded string, and a model that says 'The
number is 7' is not ignorant." D68, D69 and D75 all rest on rank. Rank is not
capability, and the difference is the first thing a reader will ask about.

SCORING IS REDONE HERE, NOT TRUSTED FROM THE KERNEL. The kernel banks every decoded
string precisely so the scorer can change without another GPU hour -- and it should
be able to, because "exact match" is a judgement call that D60's 85-point swing
showed can dominate a result.

WHAT MAKES THE DEPTH COMPARISON PAIRED. Accuracy at r=4 and r=32 comes from the SAME
items in the SAME run, so each family is its own control and the test is a sign test
over families rather than a comparison of two independent proportions. That matters
here specifically: D78 established h_0 is unseeded, so two runs of one item differ,
and an unpaired cross-run comparison would carry ~8 points of noise for free.

Run: uv run python -m scripts.run_depth_accuracy
"""

from __future__ import annotations

import json
import os
import re
from math import comb

import numpy as np
import pandas as pd

BANK = os.path.join("scratch", "kaggle_depthacc", "out")
OUT = os.path.join("results", "depth_accuracy.csv")
STOPWORDS = ("the", "a", "an", "is", "are")


def normalise(text: str) -> str:
    """Same normalisation as the kernel's, re-implemented here so scoring is local."""
    t = str(text).strip().lower()
    for ch in ".,!?;:'\"()[]":
        t = t.replace(ch, " ")
    return " ".join(w for w in t.split() if w not in STOPWORDS)


# D145: Huginn's chat template glues the NEXT TURN's role header onto the answer with
# no separator -- 224 of the 1260 banked generations end in `user`, e.g. '4user\n\n'
# where the gold is '4', 'Clouduser\n\n' where it is 'cloud'. Left in place the marker
# is part of the token and the whole-word match below rejects a correct answer; D86's
# headline "exact match hits 0.0% by r=8" was that rejection, not the model. Separated
# here rather than deleted, and BEFORE normalisation so the case-sensitive marker still
# matches. Same rule and same word list as `scripts/run_depth_profile.py::_contains`.
_ROLE = re.compile(r"(?:Huginn|user|assistant|system)\b")


def score(pred: str, gold: str) -> tuple[bool, bool]:
    """(exact, contains) after normalisation, whole-word for single-token golds.

    Separating the marker can only ADD matches, never remove one, so it cannot
    manufacture an exact match out of prose: '4user' -> '4' scores exact, while
    'Letter: nuser' -> 'letter n' still fails exact for gold 'n' and passes
    containment only. Verified on the bank: 3 of 1260 generations have any non-blank
    text after the marker, so substituting and truncating give identical tables.
    """
    p, g = normalise(_ROLE.sub(" ", str(pred))), normalise(gold)
    contains = (g in p.split()) if " " not in g else (g in p)
    return p == g, bool(contains)


def sign_test(deltas: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided sign test on paired differences; ties dropped and reported."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    up = int((d > 0).sum())
    k = max(up, n - up)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2.0**n)
    return up, n, p


def load(path: str = BANK) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(os.path.join(path, "depthacc.json"), encoding="utf-8") as fh:
        gen = pd.DataFrame(json.load(fh))
    rank_path = os.path.join(path, "depthrank.json")
    rk = pd.DataFrame(json.load(open(rank_path, encoding="utf-8"))) \
        if os.path.exists(rank_path) else pd.DataFrame()
    rescored = gen["output"].combine(gen["gold"], score)
    gen["exact"] = [a for a, _ in rescored]
    gen["contains"] = [b for _, b in rescored]
    gen["empty"] = gen["output"].astype(str).str.strip() == ""
    return gen, rk


def depth_profile(gen: pd.DataFrame, fmt: str, metric: str = "exact") -> pd.DataFrame:
    sub = gen[gen["fmt"] == fmt]
    return (sub.groupby(["depth", "family"])[metric].mean()
            .reset_index().pivot(index="family", columns="depth", values=metric))


def paired_depth_test(gen: pd.DataFrame, fmt: str, a: int, b: int,
                      metric: str = "exact") -> dict:
    """Family-level accuracy at depth a vs depth b, paired, exact sign test."""
    piv = depth_profile(gen, fmt, metric)
    if a not in piv.columns or b not in piv.columns:
        return {"usable": False, "why": f"depth {a} or {b} missing"}
    piv = piv[[a, b]].dropna()
    up, n, p = sign_test((piv[b] - piv[a]).to_numpy())
    return {"usable": True, "fmt": fmt, "metric": metric, "from": a, "to": b,
            "n_families": int(len(piv)), "acc_from": float(piv[a].mean()),
            "acc_to": float(piv[b].mean()), "n_rose": up, "n_moved": n, "p": p,
            "direction": "deeper is better" if up > n - up else "deeper is worse"}


def rank_vs_generation(gen: pd.DataFrame, rk: pd.DataFrame) -> pd.DataFrame:
    """P3: does gold at rank 1 predict a correct decoded string, item by item?

    The join is on (family, item, fmt, depth), so both sides come from the same
    weights at the same depth in the same run. If rank 1 does NOT predict a correct
    string, every rank-based claim in D68/D69/D75 concerns a quantity that does not
    reach the output, and this table says so in one number.
    """
    if rk.empty:
        return pd.DataFrame()
    j = gen.merge(rk[rk["ok"]], on=["family", "item", "fmt", "depth"], how="inner")
    rows = []
    for (fmt, depth), s in j.groupby(["fmt", "depth"]):
        top1 = s["rank"] == 1
        for metric in ("exact", "contains"):
            hit = s[metric].astype(bool)
            a = int((top1 & hit).sum())
            b = int((top1 & ~hit).sum())
            c = int((~top1 & hit).sum())
            d = int((~top1 & ~hit).sum())
            den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
            rows.append({
                "fmt": fmt, "depth": depth, "metric": metric, "n": len(s),
                "n_top1": int(top1.sum()),
                "p_hit_given_top1": (a / (a + b)) if a + b else float("nan"),
                "p_hit_given_not_top1": (c / (c + d)) if c + d else float("nan"),
                "phi": ((a * d - b * c) / den) if den > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def report(gen: pd.DataFrame, rk: pd.DataFrame) -> str:
    lines = ["", f"  {len(gen)} generations, {gen['family'].nunique()} families, "
                 f"depths {sorted(gen['depth'].unique())}, "
                 f"formats {sorted(gen['fmt'].unique())}"]
    n_empty = int(gen["empty"].sum())
    lines.append(f"  {n_empty} empty outputs"
                 + ("" if n_empty == 0 else
                    "  <- D62's symptom; check the generation gate before reading on"))

    lines += ["", f"  {'fmt':>12} {'depth':>6} {'exact':>8} {'contains':>9} "
                  f"{'gap':>7}   <- gap = prose wrapped around a correct answer"]
    for (fmt, depth), s in gen.groupby(["fmt", "depth"]):
        e, c = s["exact"].mean(), s["contains"].mean()
        lines.append(f"  {fmt:>12} {depth:>6} {e:>8.1%} {c:>9.1%} {c - e:>7.1%}")

    lines += ["", "  P2 -- DOES DEPTH HELP OR HURT? Paired over families, exact sign test."]
    for fmt in sorted(gen["fmt"].unique()):
        for metric in ("exact", "contains"):
            t = paired_depth_test(gen, fmt, 4, 32, metric)
            if not t.get("usable"):
                lines.append(f"    {fmt:>12} {metric:>9}: {t['why']}")
                continue
            lines.append(
                f"    {fmt:>12} {metric:>9}: r=4 {t['acc_from']:.1%} -> r=32 "
                f"{t['acc_to']:.1%}, rose in {t['n_rose']}/{t['n_moved']} families "
                f"that moved (of {t['n_families']}), p = {t['p']:.4f}  "
                f"{t['direction']}")

    rv = rank_vs_generation(gen, rk)
    if len(rv):
        lines += ["", "  P3 -- DOES RANK 1 PREDICT A CORRECT STRING?",
                  f"    {'fmt':>12} {'depth':>6} {'metric':>9} {'n top1':>7} "
                  f"{'P(hit|top1)':>12} {'P(hit|not)':>11} {'phi':>7}"]
        for _, r in rv.iterrows():
            lines.append(f"    {r['fmt']:>12} {int(r['depth']):>6} {r['metric']:>9} "
                         f"{int(r['n_top1']):>7} {r['p_hit_given_top1']:>12.1%} "
                         f"{r['p_hit_given_not_top1']:>11.1%} {r['phi']:>+7.3f}")
        weak = rv[(rv["metric"] == "exact") & (rv["phi"] < 0.3)]
        if len(weak) > len(rv[rv["metric"] == "exact"]) / 2:
            lines += [
                "    Rank 1 does NOT reliably predict a correct decoded string, so",
                "    D68/D69/D75's rank-based numbers describe what is AVAILABLE to",
                "    the readout, not what the model emits. That is the distinction",
                "    D68(5) flagged and could not test.",
            ]

    lines += ["", "  P4 -- FORMAT INTERACTION"]
    for metric in ("exact", "contains"):
        piv = gen.groupby(["fmt", "depth"])[metric].mean().unstack("fmt")
        if piv.shape[1] == 2:
            b, c = piv.columns[0], piv.columns[1]
            lines.append(f"    {metric:>9}: " + "  ".join(
                f"r={int(d)} {piv.loc[d, c] - piv.loc[d, b]:+.1%}" for d in piv.index))
            lines.append(f"    {'':>9}  (constrained minus bare; D69 measured "
                         f"+83 points on echo_digit at r=64)")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "depthacc.json")):
        print(f"no results in {BANK} -- pull the geometry-depthacc output first.")
        return
    gen, rk = load()
    save_table(OUT, gen, kind="depth_accuracy", source=BANK)
    print(f"saved {OUT} ({len(gen)} generations)")
    print(report(gen, rk))


if __name__ == "__main__":
    main()
