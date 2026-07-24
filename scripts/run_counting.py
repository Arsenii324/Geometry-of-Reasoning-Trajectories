"""Experiment C — |winding| / steps-to-settle vs count length n_ops (H3).

Reproduces results/counting.csv. All trajectories settle, but effective compute
(steps-to-settle) and |winding| rise with the count length. OWNER: Data+Analysis.
STATUS: implemented, verified-live. `seq_len` is a deterministic function of
    `n_ops` for this task (rank-corr=1.0) -- the length-partial control below
    is mathematically degenerate, guarded (raises) by `partial_spearman`, see
    claims_ledger.md D10. Do not read a length-controlled number from this
    script as a real result.

Run: python -m scripts.run_counting
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.correlate import fmt_by_level, partial_spearman, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.gate import classify_shape
from traj_geom.shapes.synthetic import make_counting_task

N_OPS = (2, 4, 8, 16, 24, 32)
N_SEEDS = 5


def compute() -> pd.DataFrame:
    """Extract a trajectory per counting task and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="n_ops"):
        for s in range(N_SEEDS):
            task = make_counting_task(n_ops, seed=s)
            tr = extract_trajectory(model, tok, task["prompt"], num_steps=64, seed=0)
            rows.append(
                {
                    "n_ops": n_ops,
                    "seq_len": int(tok(task["prompt"], return_tensors="pt").input_ids.shape[1]),
                    "winding": abs(winding_of(tr, burn=4)),  # PCA sign is arbitrary -> |winding|
                    "shape": classify_shape(tr),
                    "steps_settle": steps_to_settle(tr),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report |winding|/steps vs count length and the length-controlled partial."""
    cdf = cached("counting.csv", compute)
    print(cdf.groupby("n_ops")[["seq_len", "winding", "steps_settle"]].mean().round(2))
    # Canonical: per-level Spearman + N.
    print("|winding|~n_ops [per-level]:", fmt_by_level(cdf, "n_ops", "winding"))
    print("steps~n_ops    [per-level]:", fmt_by_level(cdf, "n_ops", "steps_settle"))
    # Secondary (per-row):
    print("|winding|~n_ops [per-row]:", spearman(cdf["n_ops"], cdf["winding"]))
    print("steps~n_ops    [per-row]:", spearman(cdf["n_ops"], cdf["steps_settle"]))
    try:
        print(
            "|winding|~n_ops [per-row, partial|L]:",
            partial_spearman(cdf["winding"], cdf["n_ops"], cdf["seq_len"]),
        )
    except ValueError as e:
        print(f"|winding|~n_ops [per-row, partial|L]: skipped -- {e}")


if __name__ == "__main__":
    main()
