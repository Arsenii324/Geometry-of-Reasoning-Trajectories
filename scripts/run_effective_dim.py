"""Experiment — how many directions does the latent orbit actually move in?

Reads the banked `results/trajectories/*.npy` and writes `results/effective_dim.csv`.
No GPU. OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE QUESTION, AND WHY IT IS OVERDUE. Every trajectory statistic in this project
assumes the orbit is low-dimensional: `winding_of` projects to a 2-D PCA plane, all
nine variants W1-W9 project or measure angle about a point, `classify_shape`'s
settle/loop/drift taxonomy is a vocabulary for planar curves, and the eigenplane
test projects onto the plane of a single eigenpair. None of them ever checked that
the object is 2-D. If it is not, their failures are explained rather than merely
recorded -- a projection that discards most of the active directions cannot be
rescued by choosing a better centre or a better window, which is exactly what
W2-W9 tried.

WHAT IS MEASURED. The step DIRECTIONS (unit-normalised, so the geometric decay
envelope is removed) over the pre-floor window, scored by participation ratio
against an isotropic null simulated at the same sample count. Normalisation is not
cosmetic: raw step differences give PR 5.34 while isotropic vectors rescaled to the
same norms give 5.74, so essentially all of the apparent low-dimensionality of the
un-normalised object is the envelope.

TWO ROBUSTNESS CHECKS ARE BUILT IN, both aimed at D28's failure mode -- a statistic
that reports the recording budget rather than the model:
  * the pre-floor window is found from the data, not fixed;
  * the same quantity is reported at num_steps 64 and 128, and the summary prints
    both. Winding's sign FLIPPED between those budgets; if this does too, it is the
    same class of artefact and must not be used.

Run: uv run python -m scripts.run_effective_dim
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from traj_geom.metrics.dimension import effective_dimension
from traj_geom.metrics.dmd import pre_floor_window

TRAJ = os.path.join("results", "trajectories")
OUT = os.path.join("results", "effective_dim.csv")
FLOOR_FRACS = (1.5, 2.0, 3.0)     # window sensitivity, reported not hidden


def collect(pattern: str = "*.npy", floor_frac: float = 2.0) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(TRAJ, pattern))):
        name = os.path.basename(path)
        traj = np.load(path).astype(np.float64)
        lo, hi = pre_floor_window(traj, floor_frac=floor_frac)
        out = effective_dimension(traj, lo=lo, hi=hi, n_null=20)
        if not out.get("ok"):
            continue
        ns = re.search(r"ns(\d+)", name)
        n_ops = re.search(r"_n(\d+)_", name)
        rows.append({
            "file": name,
            "family": name.split("_n")[0],
            "num_steps": int(ns.group(1)) if ns else None,
            "n_ops": int(n_ops.group(1)) if n_ops else None,
            "floor_frac": floor_frac,
            "window_lo": lo, "window_hi": hi,
            **{k: out[k] for k in ("n_steps", "ambient", "pr", "null_mean",
                                   "null_sd", "z", "ratio", "cos_consecutive")},
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    lines = [f"\n{len(df)} orbits, step directions over the pre-floor window", ""]
    g = (df.groupby(["family", "num_steps"])[
        ["n_steps", "pr", "null_mean", "ratio", "cos_consecutive"]].median().round(3))
    lines.append(g.to_string())
    lines += [
        "",
        f"  effective dimensions : median {df['pr'].median():.2f} "
        f"against an isotropic null of {df['null_mean'].median():.2f} "
        f"(ratio {df['ratio'].median():.3f})",
        f"  below the null       : {int((df['z'] < -3).sum())}/{len(df)} orbits at z < -3",
        f"  consecutive-step cos : median {df['cos_consecutive'].median():+.4f} "
        f"(isotropic would be 0, so there IS structure -- it is just not planar)",
    ]
    by_ns = df.groupby("num_steps")["ratio"].median()
    if len(by_ns) > 1:
        spread = float(by_ns.max() - by_ns.min())
        verdict = ("NOT budget-governed the way winding was" if spread < 0.25
                   else "CAUTION: budget-sensitive, treat as suspect")
        lines += [
            "",
            "  BUDGET CHECK (D28's failure mode: winding's sign flipped between "
            "these two):",
            "    " + "  ".join(f"ns={int(k)}: ratio {v:.3f}" for k, v in by_ns.items()),
            f"    same sign at both budgets, spread {spread:.3f} -- {verdict}",
        ]
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not glob.glob(os.path.join(TRAJ, "*.npy")):
        print(f"no trajectories in {TRAJ} -- they are gitignored; restore them first.")
        return
    df = collect()
    save_table(OUT, df, kind="effective_dim", source=TRAJ, floor_frac=2.0)
    print(f"saved {OUT}")
    print(report(df))

    print("\n  window sensitivity (the window is chosen from the data, so its "
          "threshold must not decide the answer):")
    for ff in FLOOR_FRACS:
        sub = collect(floor_frac=ff)
        n_sig = int((sub["z"] < -3).sum())
        print(f"    floor_frac={ff}: n={len(sub)}  "
              f"median window {sub['n_steps'].median():.0f}  "
              f"ratio {sub['ratio'].median():.3f}  z<-3 in {n_sig}/{len(sub)}")


if __name__ == "__main__":
    main()
