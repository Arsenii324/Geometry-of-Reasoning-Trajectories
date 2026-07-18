"""Experiment F — parity switch task: generalisation beyond counting (H3).

Reproduces results/switch.csv. A light-on/off parity task also needs state-holding;
steps-to-settle rises with n_ops just as for counting, showing the effect is not
specific to arithmetic. OWNER: Data+Analysis.

Run: python -m scripts.run_switch
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model
from traj_geom.analysis.correlate import fmt_by_level, partial_spearman, spearman
from traj_geom.metrics.dynamics import steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_switch_task

N_OPS = (4, 8, 16, 24, 32, 48)
N_SEEDS = 10


def compute() -> pd.DataFrame:
    """Extract a trajectory per switch task and score its geometry."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    for n_ops in tqdm(N_OPS, desc="switch"):
        for s in range(N_SEEDS):
            t = make_switch_task(n_ops, seed=s)
            tr = extract_trajectory(model, tok, t["prompt"], num_steps=64, seed=0)
            rows.append(
                {
                    "n_ops": n_ops,
                    "seq_len": int(tok(t["prompt"], return_tensors="pt").input_ids.shape[1]),
                    "winding": abs(winding_of(tr, burn=4)),
                    "steps_settle": steps_to_settle(tr),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Report winding/steps vs parity length and the length-controlled partial.

    NOTE: for this task's fixed template, seq_len turns out to be a
    deterministic, rank-identical function of n_ops (every "Flip."/"Wait."
    token costs the same, so length varies only with n_ops, never
    independently of it — unlike counting.py's operation wording). The
    partial correlation below is still reported for consistency with
    run_counting.py's pattern, but it cannot itself distinguish a real
    depth effect from a length effect here: residualizing against a
    confounder that's rank-identical to n_ops mechanically drives the
    result toward zero regardless of which explanation is true. Treat it
    as "this check is inconclusive by construction for this template," not
    as evidence the effect is or isn't real.
    """
    sw = cached("switch.csv", compute)
    # Canonical: per-level Spearman + N.
    print("switch | steps~n_ops   [per-level]:", fmt_by_level(sw, "n_ops", "steps_settle"))
    print("switch | winding~n_ops [per-level]:", fmt_by_level(sw, "n_ops", "winding"))
    # Secondary (per-row):
    print("switch | steps~n_ops   [per-row]:", spearman(sw["n_ops"], sw["steps_settle"]))
    print("switch | winding~n_ops [per-row]:", spearman(sw["n_ops"], sw["winding"]))
    print(
        "switch | steps~n_ops   [per-row, partial|L]:",
        partial_spearman(sw["steps_settle"], sw["n_ops"], sw["seq_len"]),
    )
    print(
        "switch | winding~n_ops [per-row, partial|L]:",
        partial_spearman(sw["winding"], sw["n_ops"], sw["seq_len"]),
    )


if __name__ == "__main__":
    main()
