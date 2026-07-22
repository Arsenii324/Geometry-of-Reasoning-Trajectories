"""Smoke test for the count_ones/projection synthetic tasks and normed acceleration.

OWNER: Data+Analysis
STATUS: implemented (2026-07-19) — quick sanity pass, not a scored experiment
    (small N_OPS/N_SEEDS, no cached() wrapper, overwrites its CSV every run).
TASK: run count_ones and projection through the extraction hook and confirm
    winding/steps_settle/normed_acceleration all compute without error.
I/O: -> results/smoke_new_tasks.csv (task, n_ops, seq_len, winding, shape,
    steps_settle, mean_normed_accel).

Run: uv run python -m scripts.run_smoke_new_tasks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts._common import load_model
from traj_geom.extraction.hook import extract_trajectory
from traj_geom.metrics.dynamics import normed_acceleration, steps_to_settle
from traj_geom.metrics.winding import winding_of
from traj_geom.shapes.gate import classify_shape
from traj_geom.shapes.synthetic import make_count_ones_task, make_projection_task

N_OPS = (2, 4)
N_SEEDS = 2


def main() -> None:
    """Run the smoke sweep and print/save per-(task, n_ops) means."""
    model, tok = load_model()
    rows = []

    tasks_to_test = [
        ("count_ones", make_count_ones_task),
        ("projection", make_projection_task),
    ]

    for task_name, task_fn in tasks_to_test:
        print(f"Running task {task_name}")
        for n_ops in tqdm(N_OPS, desc="n_ops"):
            for s in range(N_SEEDS):
                task = task_fn(n_ops, seed=s)
                # 32 steps (not the 64 default) to keep the smoke test cheap.
                tr = extract_trajectory(model, tok, task["prompt"], num_steps=32, seed=0)

                accel = normed_acceleration(tr)
                mean_accel = float(np.mean(accel)) if len(accel) > 0 else np.nan

                rows.append(
                    {
                        "task": task_name,
                        "n_ops": n_ops,
                        "seq_len": int(tok(task["prompt"], return_tensors="pt").input_ids.shape[1]),
                        "winding": abs(winding_of(tr, burn=4)),
                        "shape": classify_shape(tr),
                        "steps_settle": steps_to_settle(tr),
                        "mean_normed_accel": mean_accel,
                    }
                )

    df = pd.DataFrame(rows)
    print("\nResults:")
    cols = ["seq_len", "winding", "steps_settle", "mean_normed_accel"]
    print(df.groupby(["task", "n_ops"])[cols].mean().round(3))

    df.to_csv("results/smoke_new_tasks.csv", index=False)
    print("Saved to results/smoke_new_tasks.csv")


if __name__ == "__main__":
    main()
