"""Experiment B6 — does trajectory geometry differ between correct and incorrect?

Reads the banked states from `scratch/kaggle_b6bank/out/` and writes
`results/b6.csv`. No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE FOUNDING HYPOTHESIS' STRONGEST AVAILABLE TEST. Every geometric measurement in
this project pooled correct and incorrect trajectories, and since accuracy read ~0
almost everywhere, the geometry on record describes failure. This is the first
comparison of a trajectory that got the answer right against one that got it wrong.

THE DESIGN IS THE WHOLE POINT, because the first attempt was void. D72: correctness
there was very nearly a deterministic function of the ANSWER VALUE, so permuting the
label destroyed the correctness-geometry link and the prompt-geometry link together.
Here the null permutes WITHIN a gold value, so prompt content is held fixed by
construction and only the label moves. Families where no gold value carries both
classes are DROPPED, with the count reported -- that is P1, and it is a gate rather
than a preference.

WHAT IS MEASURED, given D74 and D76. D74 showed the orbit spans ~13 effective
dimensions, so planar quantities (winding and all nine of its variants,
settle/loop/drift) cannot describe it. D76 showed the consecutive-step cosine is
exactly cos(phi) for a linear map -- a rotation estimator needing no window, plane,
centre or surrogate -- and that it separates trained from untrained completely. So
the metrics here are: effective dimensionality, the step cosine, the contraction
rate, and settling time. No planar statistic appears.

A CAVEAT THE DATA ITSELF FORCED (D78). `initialize_state` draws h_0 from an
unseeded RNG, so correctness labels carry run-to-run noise of ~8 accuracy points.
That does NOT invalidate this analysis -- labels and geometry come from the SAME
forward, so they are internally consistent -- but it does mean the label is a
property of this run, and a replication would relabel some items.

Run: uv run python -m scripts.run_b6
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from traj_geom.analysis.stratified import (
    attainable,
    combinatorial_floor,
    stratified_permutation_p,
)
from traj_geom.metrics.dimension import effective_dimension
from traj_geom.metrics.dmd import pre_floor_window

BANK = os.path.join("scratch", "kaggle_b6bank", "out")
OUT = os.path.join("results", "b6.csv")
METRICS = ("pr", "cos_consecutive", "contraction", "settle")
MIN_ITEMS_IN_MIXED = 6          # P1
N_PERM = 20000


def contraction_and_settle(traj: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Geometric decay rate of the step norm, and the unroll it reaches 1% of peak."""
    d = np.linalg.norm(np.diff(traj, axis=0), axis=1)[lo:hi]
    if len(d) < 4:
        return float("nan"), float("nan")
    y = np.log(np.clip(d, 1e-30, None))
    slope = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
    thresh = d.max() * 0.01
    below = np.flatnonzero(d < thresh)
    return float(np.exp(slope)), float(below[0] if len(below) else len(d))


def collect(path: str = BANK) -> pd.DataFrame:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        recs = [r for r in json.load(fh) if r.get("ok")]
    rows = []
    for r in recs:
        traj = np.load(os.path.join(path, r["tag"] + ".npy")).astype(np.float64)
        lo, hi = pre_floor_window(traj)
        dim = effective_dimension(traj, lo=lo, hi=hi, n_null=20)
        if not dim.get("ok"):
            continue
        rho, settle = contraction_and_settle(traj, lo, hi)
        rows.append({
            "tag": r["tag"], "family": r["family"], "item": r["item"],
            "gold": r["gold"], "correct": bool(r["correct"]),
            "best_rank": r["best_rank"], "best_depth": r["best_depth"],
            "n_tokens": r["n_tokens"], "window": hi - lo,
            "pr": dim["pr"], "cos_consecutive": dim["cos_consecutive"],
            "contraction": rho, "settle": settle,
        })
    return pd.DataFrame(rows)


def gate_p1(sub: pd.DataFrame) -> dict:
    """Is a within-gold-value contrast constructible at all? (D72's lesson.)"""
    ok = set(sub.loc[sub["correct"], "gold"])
    bad = set(sub.loc[~sub["correct"], "gold"])
    mixed = ok & bad
    n_in = int(sub["gold"].isin(mixed).sum())
    m = sub[sub["gold"].isin(mixed)]
    floor, n_arr = combinatorial_floor(m["correct"].to_numpy(), m["gold"].to_numpy()) \
        if len(m) else (1.0, 1)
    return {"n_mixed_values": len(mixed), "n_in_mixed": n_in,
            "comb_floor": floor, "n_arrangements": n_arr,
            "usable": bool(n_in >= MIN_ITEMS_IN_MIXED)}


def analyse(df: pd.DataFrame, n_perm: int = N_PERM) -> pd.DataFrame:
    fams = sorted(df["family"].unique())
    alpha = 0.05 / (len(METRICS) * max(1, len(fams)))   # Bonferroni over every cell
    ok_floor, floor = attainable(alpha, n_perm)
    rows = []
    for fam in fams:
        sub = df[df["family"] == fam]
        g = gate_p1(sub)
        for metric in METRICS:
            base = {"family": fam, "metric": metric, "n": len(sub),
                    "acc": float(sub["correct"].mean()), "alpha": alpha,
                    "p_floor": floor, "floor_ok": ok_floor, **g}
            if not g["usable"]:
                rows.append({**base, "usable": False,
                             "why": "P1: too few items in gold values carrying both classes"})
                continue
            # P1b: the DESIGN's own floor, which the sampling floor cannot see.
            # Measured on nth_item: attainable() passes at 5e-05 while the design
            # admits 300 arrangements, and simulated power is 0.00 at d=2.0 sd.
            if g["comb_floor"] >= alpha:
                rows.append({**base, "usable": False,
                             "why": (f"P1b: only {g['n_arrangements']} within-stratum "
                                     f"arrangements exist, so the smallest reachable p "
                                     f"is {g['comb_floor']:.2e} >= alpha -- rejection is "
                                     f"impossible whatever the data say")})
                continue
            res = stratified_permutation_p(sub[metric].to_numpy(), sub["correct"].to_numpy(),
                                           sub["gold"].to_numpy(), n_perm=n_perm)
            rows.append({**base, **{k: v for k, v in res.items() if k != "usable"},
                         "usable": res["usable"],
                         "sig": bool(res.get("usable") and res.get("p", 1) < alpha)})
    return pd.DataFrame(rows)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print(f"no banked states in {BANK} -- pull the geometry-b6-bank output first.")
        return
    df = collect()
    res = analyse(df)
    save_table(OUT, df, kind="b6", source=BANK)
    print(f"saved {OUT}  ({len(df)} orbits, {df['family'].nunique()} families)\n")
    a = res["alpha"].iloc[0]
    reach = "reachable" if res["floor_ok"].iloc[0] else "NOT REACHABLE"
    print(f"  Bonferroni alpha = {a:.5f} over {res['metric'].nunique()} metrics x "
          f"{res['family'].nunique()} families; sampling floor = "
          f"{res['p_floor'].iloc[0]:.2e} ({reach})")
    print()
    print(f"  {'family':>12} {'acc':>5} {'mixed':>6} {'n_in':>5} {'metric':>16} "
          f"{'diff':>10} {'p':>9} {'n_used':>7}")
    for _, r in res.iterrows():
        if not r["usable"]:
            print(f"  {r['family']:>12} {r['acc']:>5.0%} {r['n_mixed_values']:>6} "
                  f"{r['n_in_mixed']:>5} {r['metric']:>16} {'--':>10} {'--':>9} "
                  f"   DROPPED: {r['why'].split(':')[0]}")
            continue
        print(f"  {r['family']:>12} {r['acc']:>5.0%} {r['n_mixed_values']:>6} "
              f"{r['n_in_mixed']:>5} {r['metric']:>16} {r['obs']:>+10.4f} "
              f"{r['p']:>9.4f} {int(r['n_used']):>7}" + ("  SIG" if r["sig"] else ""))
    live = res[res["usable"]]
    print(f"\n  {int(live['sig'].sum())} of {len(live)} testable cells significant at "
          f"alpha={a:.5f}")
    if len(live) and not live["sig"].any():
        print("  P3: correct and incorrect trajectories do not differ on ANY measured")
        print("      geometric statistic at matched answer value -- the falsifiable")
        print("      negative that pooled measurement could never produce.")


if __name__ == "__main__":
    main()
