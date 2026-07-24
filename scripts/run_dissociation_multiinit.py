"""Experiment — multi-init robustness of the track/local dissociation.

Reproduces results/dissoc_multiinit.csv (10 lengths x 6 task-seeds x 5 init-seeds).
Checks whether the track steps-to-settle trend survives across initialisation seeds
(it is only ~1 step and unstable). Cached: analysis-only without GPU. OWNER: Data+Analysis.
STATUS: implemented, real data (dissoc_multiinit.csv, 600 rows). Reports
    per-level rho only (no p-value) -- see project_plan.md §2.7 for why this
    same data (`path_independence`, its own `lyap` column) is also a candidate
    validity-null test for the geometry itself, not yet run as one.

Run: uv run python -m scripts.run_dissociation_multiinit
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import cached, load_model, save_partial
from traj_geom.analysis.correlate import fmt_by_level, spearman_by_level
from traj_geom.metrics.convergence import path_independence
from traj_geom.metrics.dynamics import contraction_rate, steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.synthetic import make_variants

N_OPS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
N_TASK_SEEDS = 6
N_INIT_SEEDS = 5


def compute() -> pd.DataFrame:
    """Extract trajectories across task- and init-seeds; score geometry + lyap."""
    from traj_geom.extraction.hook import extract_trajectory

    model, tok = load_model()
    rows = []
    n_failed = 0
    for n_ops in tqdm(N_OPS, desc="multi-init"):
        for task_s in range(N_TASK_SEEDS):
            v = make_variants(n_ops, seed=task_s)
            for kind in ("track", "local"):
                try:
                    seq_len = int(tok(v[kind], return_tensors="pt").input_ids.shape[1])
                    trajs = [
                        extract_trajectory(model, tok, v[kind], num_steps=64, seed=i)
                        for i in range(N_INIT_SEEDS)
                    ]
                    lam = path_independence(trajs[0], trajs[1])  # gap between two inits
                    for init_s, tr in enumerate(trajs):
                        rows.append(
                            {
                                "n_ops": n_ops,
                                "kind": kind,
                                "task_seed": task_s,
                                "init_seed": init_s,
                                "seq_len": seq_len,
                                "winding": abs(winding_of(tr, burn=4)),
                                "steps_settle": steps_to_settle(tr),
                                "contraction": contraction_rate(tr),
                                "lyap": lam,
                            }
                        )
                except Exception as e:  # noqa: BLE001 -- one bad (n_ops, seed, kind)
                    # must not lose the rest of this multi-hour, 5x-init-per-point sweep.
                    n_failed += 1
                    print(
                        f"n_ops={n_ops} task_s={task_s} kind={kind}: "
                        f"skipping after error: {e!r}"
                    )
                else:
                    save_partial(rows, "dissoc_multiinit.csv")
    if n_failed:
        print(f"run_dissociation_multiinit: {n_failed} (n_ops, seed, kind) points failed.")
    return pd.DataFrame(rows)


def main() -> None:
    """Report per-init rho, level rho, within-init spread and median lyapunov."""
    dm = cached("dissoc_multiinit.csv", compute)
    for kind in ("track", "local"):
        s = dm[dm.kind == kind]
        # Canonical per-level rho, computed separately within each init seed.
        # The spread across inits IS the finding (the trend is unstable), not noise.
        per_init = [
            round(spearman_by_level(g, "n_ops", "steps_settle")[0], 3)
            for _, g in s.groupby("init_seed")
        ]
        within = round(s.groupby(["n_ops", "task_seed"])["steps_settle"].std().mean(), 2)
        pooled = fmt_by_level(s, "n_ops", "steps_settle")
        print(f"{kind}: steps~n_ops [per-level] pooled = {pooled}")
        print(f"   per-init-seed per-level rho = {per_init}")
        print(f"   within-init std of steps = {within}")
        print(f"   lyap median: {round(s.lyap.median(), 4)}\n")


if __name__ == "__main__":
    main()
