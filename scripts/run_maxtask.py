"""Experiment — running-maximum task: does the state-holding signature generalise?

Reproduces results/maxtask.csv. Unlike counting/switch, steps-to-settle does NOT
rise with length here (rho negative) — the effect is not universal. OWNER: Data+Analysis.

Run: uv run python -m scripts.run_maxtask
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.correlate import fmt_by_level, partial_spearman, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_max_task

N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 8


def compute() -> pd.DataFrame:
    """Extract a trajectory per max-task and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="maxtask"):
        for s in range(N_SEEDS):
            t = make_max_task(n_ops, seed=s)
            tr = extract_trajectory(model, tok, t["prompt"], num_steps=64, seed=0)
            rows.append(
                {
                    "n_ops": n_ops,
                    "seq_len": int(tok(t["prompt"], return_tensors="pt").input_ids.shape[1]),
                    "steps_settle": steps_to_settle(tr),
                    "winding": abs(winding_of(tr, 4)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report steps-to-settle vs length for the running-maximum task, and
    the length-controlled partial.

    NOTE: for this task's fixed template, seq_len turns out to be a
    deterministic, rank-identical function of n_ops (every digit costs one
    token, so length varies only with n_ops, never independently of it —
    unlike counting.py's operation wording). The partial correlation below
    is still reported for consistency with run_counting.py's pattern, but
    it cannot itself distinguish a real depth effect from a length effect
    here: residualizing against a confounder that's rank-identical to
    n_ops mechanically drives the result toward zero regardless of which
    explanation is true. In particular, the significant winding~n_ops
    result reported elsewhere for this task should NOT be read as
    "confirmed independent of length" just because this check ran without
    error — this check is inconclusive by construction for this template.
    """
    mx = cached("maxtask.csv", compute)
    # Canonical: per-level Spearman + N.
    print("maxtask | steps~n_ops   [per-level]:", fmt_by_level(mx, "n_ops", "steps_settle"))
    print("maxtask | winding~n_ops [per-level]:", fmt_by_level(mx, "n_ops", "winding"))
    # Secondary (per-row):
    print("maxtask | steps~n_ops   [per-row]:", spearman(mx["n_ops"], mx["steps_settle"]))
    print("maxtask | winding~n_ops [per-row]:", spearman(mx["n_ops"], mx["winding"]))
    print(
        "maxtask | steps~n_ops   [per-row, partial|L]:",
        partial_spearman(mx["steps_settle"], mx["n_ops"], mx["seq_len"]),
    )
    print(
        "maxtask | winding~n_ops [per-row, partial|L]:",
        partial_spearman(mx["winding"], mx["n_ops"], mx["seq_len"]),
    )


if __name__ == "__main__":
    main()
