"""Plots for the MVP: winding-vs-depth scatter and 2D PCA trajectory.

OWNER: Data+Analysis
STATUS: stub — implement me.
TASK: Produce the two figures the "done" criterion needs — a scatter of
    winding number against depth, and a 2D PCA path plot for inspection.
I/O: arrays / trajectory -> matplotlib Figure (saved or returned).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def scatter_winding_vs_depth(
    w: np.ndarray, d: np.ndarray, save_path: str | None = None
) -> Any:
    """Scatter winding number against reasoning depth.

    Args:
        w: Winding numbers, shape [N].
        d: Reasoning depths, shape [N].
        save_path: If given, write the figure there.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d, w, alpha=0.7)
    ax.set_xlabel("Reasoning Depth (n_ops)")
    ax.set_ylabel("Winding Number")
    ax.set_title("Winding Number vs. Reasoning Depth")
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_pca_trajectory(points_2d: np.ndarray, save_path: str | None = None) -> Any:
    """Plot a single 2D PCA-projected trajectory path.

    Args:
        points_2d: Projected path of shape [T, 2].
        save_path: If given, write the figure there.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the path as a line
    ax.plot(points_2d[:, 0], points_2d[:, 1], "-", alpha=0.6, color="blue", label="Trajectory")

    # Scatter points with color mapping to time/step
    c = np.arange(len(points_2d))
    scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=c, cmap="viridis", zorder=5)

    # Mark start and end explicitly
    ax.plot(points_2d[0, 0], points_2d[0, 1], "go", markersize=10, label="Start")
    ax.plot(points_2d[-1, 0], points_2d[-1, 1], "ro", markersize=10, label="End")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA Trajectory")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Unroll Step")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
