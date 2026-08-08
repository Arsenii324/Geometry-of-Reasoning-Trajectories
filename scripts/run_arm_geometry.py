"""Experiment — what does TRAINING do to the shape of the orbit?

Reads the banked states from `scratch/ds_bank/out/` (trained and untrained arms on
identical prompts) and writes `results/arm_geometry.csv`. No GPU.
OWNER: Data+Analysis. STATUS: implemented 2026-08-09.

THE COMPARISON THIS PROJECT NEVER HAD. D52 established that training changes the
model's DYNAMICS -- the contraction rate rho goes 0.7048 -> 0.8577 with complete
separation across 14 weight-sets. Nothing ever asked what training does to the
SHAPE of the path. Every geometric measurement on record is trained-only, including
D74's, which is why D74 could not say whether ~13 effective dimensions was a
property of learning or of the architecture.

WINDOW LENGTH IS THE CONFOUND AND IT IS NOT OPTIONAL TO CONTROL IT. The two arms do
not converge at the same rate, so their pre-floor windows differ by a factor of ~2.4
(trained median 104 unrolls, untrained 43). Participation ratio grows with the
number of samples for any near-isotropic set, so comparing the arms at their own
natural windows compares sample counts as much as geometry. Worse, `ratio` (PR over
its isotropic null) is rank-collinear with window length -- D74(6) records
`partial_spearman` REFUSING to correct for it at rho = -1.000. So everything here is
reported at a MATCHED window, and swept across window lengths rather than fixed at
one, because the sweep is itself the finding.

Run: uv run python -m scripts.run_arm_geometry
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from traj_geom.metrics.dimension import effective_dimension
from traj_geom.metrics.dmd import pre_floor_window

BANK = os.path.join("scratch", "ds_bank", "out")
OUT = os.path.join("results", "arm_geometry.csv")
WINDOWS = (10, 15, 20, 30, 41)      # 41 = the smallest pre-floor window across all orbits


def load_manifest(path: str = BANK) -> list[dict]:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
        return [r for r in json.load(fh) if r.get("ok")]


def collect(path: str = BANK, position: str = "last") -> pd.DataFrame:
    """Per-orbit geometry at every matched window in WINDOWS."""
    rows = []
    for r in load_manifest(path):
        traj = np.load(os.path.join(path, f"{r['tag']}_{position}.npy")).astype(np.float64)
        lo, hi = pre_floor_window(traj)
        for k in WINDOWS:
            if lo + k > hi:
                continue
            out = effective_dimension(traj, lo=lo, hi=lo + k, n_null=20)
            if not out.get("ok"):
                continue
            rows.append({
                "tag": r["tag"], "arm": r["arm"], "task": r["task"],
                "n_ops": r["n_ops"], "seed": r["seed"], "position": position,
                "window_k": k, "natural_window": hi - lo,
                **{c: out[c] for c in ("pr", "null_mean", "z", "ratio",
                                       "cos_consecutive")},
            })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, metric: str, k: int) -> dict:
    sub = df[df["window_k"] == k]
    a = sub[sub["arm"] == "trained"][metric].to_numpy()
    b = sub[sub["arm"] == "untrained"][metric].to_numpy()
    if len(a) < 3 or len(b) < 3:
        return {"k": k, "metric": metric, "usable": False}
    p = float(mannwhitneyu(a, b)[1])
    return {"k": k, "metric": metric, "usable": True, "trained": float(np.median(a)),
            "untrained": float(np.median(b)), "p": p, "n_trained": len(a),
            "n_untrained": len(b),
            "separated": bool(a.min() > b.max() or b.min() > a.max())}


def report(df: pd.DataFrame) -> str:
    nat = df.drop_duplicates("tag").groupby("arm")["natural_window"].median()
    lines = [
        "",
        "  natural pre-floor window (a result in itself -- trained orbits stay above",
        "  the arithmetic floor far longer, which is what D52's slower contraction predicts):",
        "    " + "  ".join(f"{a}: {v:.0f} unrolls" for a, v in nat.items()),
        "",
        f"  {'K':>4} {'trained':>9} {'untrained':>10} {'MWU p':>10} {'disjoint?':>10}   metric",
    ]
    for metric in ("pr", "cos_consecutive"):
        for k in WINDOWS:
            c = compare(df, metric, k)
            if not c.get("usable"):
                continue
            lines.append(f"  {k:>4} {c['trained']:>9.3f} {c['untrained']:>10.3f} "
                         f"{c['p']:>10.2e} {str(c['separated']):>10}   {metric}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    from traj_geom.provenance import save_table

    if not os.path.exists(os.path.join(BANK, "manifest.json")):
        print(f"no banked states in {BANK} -- they are gitignored; pull the "
              f"geom-bank job output first.")
        return
    df = pd.concat([collect(position=p) for p in ("last", "mid")], ignore_index=True)
    save_table(OUT, df, kind="arm_geometry", source=BANK, windows=list(WINDOWS))
    print(f"saved {OUT}  ({df['tag'].nunique()} orbits x {len(WINDOWS)} windows "
          f"x 2 positions)")
    print("\n=== answer-predicting position ===")
    print(report(df[df["position"] == "last"]))
    print("=== mid-prompt position (D74's 'answer token only' limit) ===")
    print(report(df[df["position"] == "mid"]))


if __name__ == "__main__":
    main()
