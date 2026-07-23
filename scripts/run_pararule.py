"""Experiment B — winding / steps-to-settle vs PARARULE reasoning depth (H2 probe).

Reproduces results/pararule.csv. Cached: if the CSV exists, only the analysis is
printed (no GPU needed). Result: everything settles, winding~depth is null.
OWNER: Data+Analysis.

Run: python -m scripts.run_pararule
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.analysis.correlate import fmt_by_level, partial_spearman, spearman
from traj_geom.data.loaders import enrich, load_pararule
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.gate import classify_shape

N_PER_DEPTH = 10


def compute() -> pd.DataFrame:
    """Extract a trajectory per PARARULE example and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    n_failed = 0
    for depth in (2, 3, 4, 5):
        for row in tqdm(list(load_pararule(depth, n=N_PER_DEPTH)), desc=f"depth {depth}"):
            try:
                row = enrich(row, tok)
                tr = extract_trajectory(model, tok, row["prompt"], num_steps=64, seed=0)
                row["winding"] = winding_of(tr, 4)
                row["shape"] = classify_shape(tr)
                row["steps_settle"] = steps_to_settle(tr)
                rows.append(row)
            except Exception as e:  # noqa: BLE001 -- one bad example must not lose
                # the rest of this sweep's already-completed rows.
                n_failed += 1
                print(f"depth={depth}: skipping an example after error: {e!r}")
            else:
                save_partial(rows, "pararule.csv")
    if n_failed:
        print(f"run_pararule: {n_failed} examples failed and were skipped.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report winding/steps vs depth and the length-controlled partial correlation."""
    df = cached("pararule.csv", compute)
    print(df.groupby("depth")[["seq_len", "winding", "steps_settle"]].mean().round(3))
    print(df.groupby("depth")["shape"].value_counts())
    # Canonical: per-level Spearman (rho on depth group-means) + N.
    print("winding~depth [per-level]:", fmt_by_level(df, "depth", "winding"))
    print("steps~depth   [per-level]:", fmt_by_level(df, "depth", "steps_settle"))
    # Secondary (per-row):
    print("winding~depth [per-row]:", spearman(df["winding"], df["depth"]))
    print(
        "winding~depth [per-row, partial|L]:",
        partial_spearman(df["winding"], df["depth"], df["seq_len"]),
    )


if __name__ == "__main__":
    main()
