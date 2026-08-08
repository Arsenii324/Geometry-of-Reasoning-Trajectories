"""Analysis for the 21-family capability screen (`geometry-battery`).

Reads `scratch/kaggle_battery/out/battery.json` and writes `results/battery.csv`.
No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

WHAT THE SCREEN IS FOR. Three blocked things, and the analysis is written to answer
each of them explicitly rather than to produce a table someone then reads for a
story:

  * Is the model's competence confined to near-trivial retrieval? `acc@1` at the
    BEST depth per family answers it, and D68 is why it must be best-depth rather
    than r=64: accuracy is non-monotone in depth, `echo_digit` peaking at r=4 and
    reading 0% by r=8, and every accuracy kernel before D68 ran at a fixed depth
    past that transition.
  * Is there a cell where success and failure COEXIST at the same gold value?
    That is the ingredient B6 needs and D72 showed is missing: correctness there
    was a deterministic function of the answer, so permuting the label broke the
    prompt-geometry link at the same time. `b6_usable` reports it directly.
  * Is the Caesar family dead for reasons that are not the readout? `caesar1_letter`
    is the easiest cipher cell constructible; if that is at chance, B1 closes.

THE CONTROL IS THE UNTRAINED ARM. A decode that is broken cannot put one arm at
rank 3 and the other at chance, which is what made D68 credible. `chance_ok` checks
it per family, and a family whose untrained arm is NOT near chance is reported as
uninterpretable rather than quietly averaged in.

RANK IS ON THE FIRST GOLD TOKEN, so for the six multi-token families `acc@1` is a
necessary and not sufficient condition on the full answer. `logp_gold` sums over
every gold token and is carried through for exactly that reason.

Run: uv run python -m scripts.run_battery
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RAW = os.path.join("scratch", "kaggle_battery", "out", "battery.json")
OUT = os.path.join("results", "battery.csv")

# A family is a candidate for B6 when success and failure coexist AND some gold
# value carries both classes. The band is deliberately wide: at n=16 an observed
# 20-80% is consistent with a wide range of true rates, and the point is to find
# candidates to measure, not to estimate a rate.
B6_BAND = (0.20, 0.80)


def load(path: str = RAW) -> pd.DataFrame:
    """One row per (arm, family, item), with the depth curve kept as a list."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = []
    for arm, fams in blob.get("arms", {}).items():
        for fam, items in fams.items():
            for i, r in enumerate(items):
                rank = r["rank"]
                rows.append({
                    "arm": arm, "family": fam, "item": i,
                    "gold": r.get("gold"), "prompt": r.get("prompt"),
                    "n_gold_tok": r.get("n_gold_tok"),
                    "rank_best": int(min(rank)), "rank_final": int(rank[-1]),
                    "argmin_depth": int(np.argmin(rank)) + 1,
                    "correct_best": bool(min(rank) == 1),
                    "correct_final": bool(rank[-1] == 1),
                    "logp_gold_final": float(r["logp_gold"][-1]),
                    "rank_curve": rank,
                })
    df = pd.DataFrame(rows)
    df.attrs["chance"] = blob.get("chance", 32768)
    return df


def per_family(df: pd.DataFrame, chance: int | None = None) -> pd.DataFrame:
    """One row per family: capability, the untrained control, and B6 usability."""
    chance = chance or df.attrs.get("chance", 32768)
    out = []
    for fam, sub in df.groupby("family"):
        tr = sub[sub["arm"] == "trained"]
        un = sub[sub["arm"] == "untrained"]
        acc = float(tr["correct_best"].mean()) if len(tr) else float("nan")
        un_med = float(np.median(un["rank_final"])) if len(un) else float("nan")
        # "Near chance" is generous on purpose: the claim being checked is that the
        # untrained arm carries no answer information, and an order of magnitude is
        # the resolution that supports it. D68's gaps were 3-4 orders.
        chance_ok = bool(np.isfinite(un_med) and chance / 10 <= un_med <= chance * 10)
        row = {
            "family": fam, "n": int(len(tr)),
            "acc1_best_depth": acc,
            "acc1_final_depth": float(tr["correct_final"].mean()) if len(tr) else np.nan,
            "median_rank_best": float(np.median(tr["rank_best"])) if len(tr) else np.nan,
            "median_depth_of_best": float(np.median(tr["argmin_depth"])) if len(tr) else np.nan,
            "untrained_median_rank_final": un_med,
            "chance_ok": chance_ok,
            "log10_gap": (float(np.log10(un_med) - np.log10(max(np.median(tr["rank_best"]), 1)))
                          if len(tr) and np.isfinite(un_med) else np.nan),
        }
        row.update(b6_usability(tr))
        out.append(row)
    return pd.DataFrame(out).sort_values("acc1_best_depth", ascending=False)


def b6_usability(tr: pd.DataFrame) -> dict:
    """Does this family give B6 a within-gold-value contrast?

    D72 is the whole reason this exists. There, `add1`'s correct and incorrect gold
    sets had ZERO overlap, so correctness was a function of the answer value and the
    label-permutation null could not separate "geometry tracks correctness" from
    "geometry tracks the prompt". A family is usable only if some gold value carries
    BOTH classes, and the useful number is how many items sit in such values.
    """
    if not len(tr) or tr["gold"].isna().all():
        return {"b6_usable": False, "b6_n_in_mixed_gold": 0, "b6_n_mixed_values": 0}
    ok = set(tr.loc[tr["correct_best"], "gold"])
    bad = set(tr.loc[~tr["correct_best"], "gold"])
    mixed = ok & bad
    n_in = int(tr["gold"].isin(mixed).sum())
    acc = float(tr["correct_best"].mean())
    return {
        "b6_usable": bool(mixed and B6_BAND[0] <= acc <= B6_BAND[1]),
        "b6_n_mixed_values": int(len(mixed)),
        "b6_n_in_mixed_gold": n_in,
    }


def report(fam: pd.DataFrame) -> str:
    lines = ["", f"{'family':>16} {'acc@1':>7} {'depth':>6} {'med rank':>9} "
                 f"{'untr rank':>10} {'gap':>6} {'ctrl':>5} {'B6':>4}"]
    for _, r in fam.iterrows():
        lines.append(
            f"{r['family']:>16} {r['acc1_best_depth']:>6.0%} "
            f"{r['median_depth_of_best']:>6.0f} {r['median_rank_best']:>9.0f} "
            f"{r['untrained_median_rank_final']:>10.0f} {r['log10_gap']:>+6.2f} "
            f"{'ok' if r['chance_ok'] else 'BAD':>5} "
            f"{('yes' if r['b6_usable'] else '-'):>4}")
    can = fam[fam["acc1_best_depth"] >= 0.5]
    band = fam[(fam["acc1_best_depth"] >= B6_BAND[0]) & (fam["acc1_best_depth"] <= B6_BAND[1])]
    usable = fam[fam["b6_usable"]]
    bad = fam[~fam["chance_ok"]]
    lines += [
        "",
        f"  P2  families with trained acc@1 >= 50%: {len(can)} "
        f"({', '.join(can['family']) if len(can) else 'NONE'})",
        f"  P3  families in the {B6_BAND[0]:.0%}-{B6_BAND[1]:.0%} band: {len(band)} "
        f"({', '.join(band['family']) if len(band) else 'NONE'})",
        f"      of those, B6-usable (some gold value carries BOTH classes): {len(usable)} "
        f"({', '.join(usable['family']) if len(usable) else 'NONE -- B6 stays blocked'})",
        f"  P1  untrained arm off chance in: {len(bad)} families "
        f"({', '.join(bad['family']) if len(bad) else 'none -- control holds'})",
    ]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(RAW):
        print(f"{RAW} not present -- pull the geometry-battery output first.")
        return
    df = load()
    if df.empty:
        print("no records in the raw JSON.")
        return
    fam = per_family(df)
    save_table(OUT, df.drop(columns=["rank_curve", "prompt"]), kind="battery", source=RAW)
    print(f"saved {OUT}  ({len(df)} rows, {df['family'].nunique()} families, "
          f"{df['arm'].nunique()} arms)")
    print(report(fam))


if __name__ == "__main__":
    main()
