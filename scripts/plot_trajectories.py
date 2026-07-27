"""PCA plots of per-seed trajectories, shared basis per (task, n_ops).

OWNER: Data+Analysis
STATUS: implemented (2026-07-19).
TASK: for each (task, n_ops) slice, load every seed's saved trajectory
    (results/trajectories/*.npy), fit one shared 2D PCA, and plot all seeds
    in that common basis so their paths are visually comparable.
I/O: results/full_synthetic_experiments.csv + results/trajectories/*.npy
    -> figures/pca_{task}_ops{n_ops}.png.

Run: uv run python -m scripts.plot_trajectories
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RESULTS_DIR = "results"
TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_task_pca(df, task_name, target_n_ops=24, num_steps=None):
    """Fit a single PCA across all seeds for a specific task and n_ops, then plot.

    Args:
        df: full_synthetic_experiments.csv, must have task/n_ops/seed columns.
        task_name: task column value to filter on.
        target_n_ops: n_ops column value to filter on.
        num_steps: if set, only load the trajectory saved with this exact
            num_steps (ns{num_steps} in the filename). If None (default) and
            more than one ns* file exists for a given seed — e.g. leftover
            runs at both ns64 and ns128 — raises rather than silently picking
            one; pass num_steps explicitly to disambiguate.
    """
    task_df = df[(df["task"] == task_name) & (df["n_ops"] == target_n_ops)]
    if task_df.empty:
        print(f"No data for {task_name} with n_ops={target_n_ops}")
        return

    all_trajs = []
    labels = []

    for _, row in task_df.iterrows():
        seed = row["seed"]
        ns_part = f"ns{num_steps}" if num_steps is not None else "ns*"
        pattern = os.path.join(TRAJ_DIR, f"{task_name}_n{target_n_ops}_{ns_part}_init{seed}.npy")
        matches = sorted(glob.glob(pattern))

        if not matches:
            print(f"Missing file for seed {seed}: {pattern}")
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous trajectory files for seed {seed}: {matches}. "
                "Pass num_steps= to plot_task_pca to pick one instead of guessing."
            )

        traj = np.load(matches[0])  # shape: [T, d_model]
        all_trajs.append(traj)
        labels.append(seed)

    if not all_trajs:
        return

    combined = np.vstack(all_trajs)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(combined)

    plt.figure(figsize=(10, 8))

    for traj, seed in zip(all_trajs, labels, strict=True):
        proj = pca.transform(traj)
        plt.plot(proj[:, 0], proj[:, 1], alpha=0.5, label=f"Seed {seed}")
        plt.scatter(proj[0, 0], proj[0, 1], marker="o", c="green")
        plt.scatter(proj[-1, 0], proj[-1, 1], marker="x", c="red", s=100)

    plt.title(f"PCA of {task_name} trajectories (n_ops={target_n_ops})\nGreen=Start, Red=End")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(FIG_DIR, f"pca_{task_name}_ops{target_n_ops}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    """Plot count_ones/projection PCA trajectories at n_ops in (24, 32)."""
    csv_path = os.path.join(RESULTS_DIR, "full_synthetic_experiments.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for task in ["count_ones", "projection"]:
        plot_task_pca(df, task, target_n_ops=24)
        plot_task_pca(df, task, target_n_ops=32)


if __name__ == "__main__":
    main()
