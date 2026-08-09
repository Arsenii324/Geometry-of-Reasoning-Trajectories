"""Is one unroll a linear map, or does D52 rest on a linearisation that does not hold?

Reads every banked trained orbit and writes `results/linearity.csv`. No GPU.
OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE QUESTION D81 LEFT OPEN, AND WHY IT IS NOT OPTIONAL. A surrogate built from a
prompt's measured eigenvalues reproduces the orbit's dimensional collapse but not its
rotation rate. Two explanations: Arnoldi returned modes the orbit does not follow, or
the dynamics are not linear at the recorded scales. The second would undercut D52 --
rho 0.7048 -> 0.8577 across fourteen weight-sets, this project's best-replicated
result -- along with every Jacobian quantity and D76's cos(phi) identity. So it is
tested directly, with no spectrum involved: for a linear map the STEPS satisfy
delta_{t+1} = A delta_t exactly, the fixed point cancelling in the difference.

THE SCORE THAT MATTERS IS RELATIVE, NOT ABSOLUTE, and getting that wrong is how this
analysis nearly reported the opposite conclusion. A rank-r fit can only predict what
its own r-dimensional basis can express, so `in_basis` -- the fraction of held-out
step energy lying inside the fitted subspace -- is a hard ceiling on `r2_linear`. The
first version of this run reported r2_linear = 0.397 at the default rank and read it
as evidence against linearity; at rank 2 the same orbits score 0.88, because the
default rank was the binding constraint rather than the model. What the operator
should be judged on is `r2_linear / in_basis`: of what the subspace CAN express, how
much does the linear operator actually get.

WHAT A LOW `in_basis` MEANS, WHICH IS INTERESTING IN ITS OWN RIGHT. It says later
steps enter directions the earlier steps never spanned -- a Krylov subspace that
keeps growing, i.e. a high-dimensional spectrum with no gap. That is precisely what
D74(4) found from the other side: DMD's snapshot singular spectrum had NO gap
(1.00, 0.79, 0.68, 0.57, 0.49, ...) and its leading modulus wandered 0.46-0.85 with
truncation rank, failing its stability gate on 80/80 orbits. Gapless is not
nonlinear.

Run: uv run python -m scripts.run_linearity
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from traj_geom.metrics.dmd import pre_floor_window
from traj_geom.metrics.linearity import linear_predictability

OUT = os.path.join("results", "linearity.csv")
RANKS = (2, 4, 8, 16, 32)
SOURCES = (
    ("ds_bank", os.path.join("scratch", "ds_bank", "out"), "_last.npy"),
    ("b6bank", os.path.join("scratch", "kaggle_b6bank", "out"), ".npy"),
    ("geomcap", os.path.join("scratch", "kaggle_geomcap", "out"), ".npy"),
)


def collect(sources=SOURCES, ranks: tuple[int, ...] = RANKS) -> pd.DataFrame:
    rows = []
    for bank, path, suf in sources:
        mp = os.path.join(path, "manifest.json")
        if not os.path.exists(mp):
            continue
        with open(mp, encoding="utf-8") as fh:
            recs = [r for r in json.load(fh) if r.get("ok")]
        for r in recs:
            if r.get("arm") not in (None, "trained"):
                continue
            f = os.path.join(path, r["tag"] + suf)
            if not os.path.exists(f):
                continue
            traj = np.load(f).astype(np.float64)
            lo, hi = pre_floor_window(traj)
            for rk in ranks:
                o = linear_predictability(traj, lo=lo, hi=hi, rank=rk)
                if not o.get("ok"):
                    continue
                rows.append({
                    "bank": bank, "tag": r["tag"],
                    "family": r.get("family") or r.get("task", "?"),
                    "window": hi - lo, "rank": o["rank"],
                    "rank_requested": o["rank_requested"],
                    "rank_clamped": bool(o["rank"] < o["rank_requested"]),
                    "r2_linear": o["r2_linear"], "r2_scalar": o["r2_scalar"],
                    "r2_scalar_in_basis": o["r2_scalar_in_basis"],
                    "r2_persistence": o["r2_persistence"],
                    "in_basis": o["in_basis"], "rho": o["rho"],
                    "n_train": o["n_train"], "n_test": o["n_test"],
                    "achieved": (o["r2_linear"] / o["in_basis"]
                                 if o["in_basis"] > 0.02 else float("nan")),
                })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    lines = [
        "",
        f"  {df['tag'].nunique()} trained orbits from {df['bank'].nunique()} banks",
        "",
        "  ABSOLUTE SCORES depend on how much basis the fit is allowed and on which",
        "  stretch of the orbit is held out, so read the last column.",
        "",
        f"  {'bank':>9} {'rank':>5} {'n':>4} {'window':>7} {'in_basis':>9} "
        f"{'r2_linear':>10} {'r2_scalar':>10} {'achieved':>9}",
    ]
    for (bk, rk), s in df.groupby(["bank", "rank_requested"]):
        lines.append(
            f"  {bk:>9} {rk:>5} {len(s):>4} {s['window'].median():>7.0f} "
            f"{s['in_basis'].median():>9.3f} {s['r2_linear'].median():>10.3f} "
            f"{s['r2_scalar'].median():>10.3f} {s['achieved'].median():>9.3f}")

    lines += ["", "  ACHIEVED = r2_linear / in_basis: of what the fitted subspace CAN",
              "  express, how much the linear operator actually predicts."]
    ach = df.dropna(subset=["achieved"])
    if len(ach):
        lines.append(f"    median {ach['achieved'].median():.3f}, "
                     f"10th percentile {ach['achieved'].quantile(0.1):.3f}, "
                     f"min {ach['achieved'].min():.3f} over {len(ach)} "
                     f"(orbit, rank) cells")
        hi = float((ach["achieved"] > 0.8).mean())
        lines.append(f"    {hi:.0%} of cells above 0.80")

    lines += ["", "  WHERE in_basis IS LOW, LATER STEPS LEAVE THE EARLIER SPAN --",
              "  a growing Krylov subspace, i.e. a gapless spectrum, not nonlinearity.",
              "  D74(4) found the same from the other side: DMD's snapshot spectrum",
              "  had no gap and failed its stability gate on 80/80 orbits."]
    by_win = df[df["rank"] == 8].groupby("bank")[["window", "in_basis"]].median()
    for bk, r in by_win.iterrows():
        lines.append(f"    {bk:>9}: window {r['window']:.0f} unrolls -> in_basis "
                     f"{r['in_basis']:.3f} at rank 8")

    lines += ["", "  DOES A GENERAL OPERATOR BEAT A SINGLE CONTRACTION RATE?",
              "  Both models confined to the SAME fitted subspace, so this is not a",
              "  comparison between a projected model and an unprojected one."]
    for rk, s in df.groupby("rank_requested"):
        n_better = int((s["r2_linear"] > s["r2_scalar_in_basis"]).sum())
        lines.append(f"    rank {rk:>2}: linear beats scalar in {n_better}/{len(s)} "
                     f"cells, median margin "
                     f"{(s['r2_linear'] - s['r2_scalar_in_basis']).median():+.3f}")
    n_clamp = int(df["rank_clamped"].sum())
    if n_clamp:
        lines.append(f"    ({n_clamp}/{len(df)} cells had the requested rank clamped "
                     f"to the training span; grouped by REQUESTED rank above)")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    df = collect()
    if df.empty:
        print("no banked trained orbits found")
        return
    save_table(OUT, df, kind="linearity", ranks=list(RANKS))
    print(f"saved {OUT} ({len(df)} cells over {df['tag'].nunique()} orbits)")
    print(report(df))


if __name__ == "__main__":
    main()
