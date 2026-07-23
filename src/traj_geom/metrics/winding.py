"""Winding number of a 2D path, and the trajectory-level winding helper.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, winding_number / winding_of).
    winding_null_test/matched_random_walk added 2026-07-23.
TASK: signed turns of a 2D path about a center, plus winding_of that PCA-projects
    a full trajectory to 2D (after a burn-in) and winds it, plus a null-model
    check for whether an observed winding is more than chance would produce.
I/O: winding_number(pts [T,2], center?) -> float ; winding_of(traj [T,H], burn) -> float ;
    winding_null_test(traj [T,H], burn) -> dict.

NOTE: the PCA-projection sign is arbitrary, so experiments compare |winding|.
    h_0 is a random-init outlier -> drop the first `burn` states before winding.

NOTE on winding_null_test: pca_to_2d refits a fresh 2-component PCA on EACH
    trajectory's own points. PCA always finds *some* plane maximizing
    variance, so nonzero apparent winding can emerge from a high-dimensional
    random walk by chance alone, once projected -- this had never been
    checked against a null model anywhere in this project before 2026-07-23.
    winding_null_test answers that directly: is the observed |winding|
    bigger than a matched-random-walk surrogate (same per-step displacement
    sizes, randomized directions) would typically produce.
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.projection import pca_to_2d


def winding_number(points_2d: np.ndarray, center: np.ndarray | None = None) -> float:
    """Compute the signed winding number of a 2D path about a center.

    Args:
        points_2d: Ordered path of shape [T, 2].
        center: Point to wind about; defaults to the centroid of ``points_2d``.

    Returns:
        Signed number of full turns (e.g. ~+1.0 for one counter-clockwise loop).
    """
    pts = np.asarray(points_2d, dtype=float)
    c = pts.mean(0) if center is None else np.asarray(center, dtype=float)
    v = pts - c
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap into (-pi, pi]
    return float(d.sum() / (2 * np.pi))


def winding_of(traj: np.ndarray, burn: int = 4) -> float:
    """PCA-project a trajectory to 2D (after a burn-in) and return its winding.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        burn: Number of initial states to drop (h_0 init arc is an outlier).

    Returns:
        The winding number of the burned-in, PCA-projected path.
    """
    return winding_number(pca_to_2d(traj[burn:]))


def matched_random_walk(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A surrogate with the same per-step displacement SIZES as ``traj``, but
    each step's direction randomized uniformly on the hypersphere.

    Preserves how far each step moved; destroys any real directional or
    rotational correlation between consecutive steps.

    Args:
        traj: Real trajectory to match, shape [T, hidden_dim].
        rng: Caller-supplied generator, so callers control reproducibility.

    Returns:
        Surrogate trajectory of the same shape [T, hidden_dim].
    """
    traj = np.asarray(traj, dtype=float)
    t, dim = traj.shape
    step_norms = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    directions = rng.normal(size=(t - 1, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    surrogate = np.empty_like(traj)
    surrogate[0] = traj[0]
    surrogate[1:] = traj[0] + np.cumsum(directions * step_norms[:, None], axis=0)
    return surrogate


def winding_null_test(
    traj: np.ndarray, burn: int = 4, n_surrogates: int = 200, seed: int = 0
) -> dict:
    """Compare a trajectory's |winding| against a matched-random-walk null.

    Is the observed |winding| bigger than what a random walk with the SAME
    per-step displacement sizes (but random directions) typically produces,
    once both go through the same per-trajectory PCA projection? See this
    module's NOTE on why that check matters here specifically.

    Args:
        traj: Real trajectory, shape [T, hidden_dim].
        burn: Passed through to winding_of (drop the h_0-outlier arc).
        n_surrogates: Number of matched random-walk surrogates to sample.
        seed: Seed for the surrogate generator (reproducible).

    Returns:
        Dict with ``observed`` (the real |winding|), ``null_mean``,
        ``null_std``, and ``p_value`` (fraction of surrogates whose
        |winding| >= observed — small means the real path winds more than
        chance predicts).
    """
    rng = np.random.default_rng(seed)
    observed = abs(winding_of(traj, burn=burn))
    null_vals = np.array(
        [abs(winding_of(matched_random_walk(traj, rng), burn=burn)) for _ in range(n_surrogates)]
    )
    return {
        "observed": observed,
        "null_mean": float(null_vals.mean()),
        "null_std": float(null_vals.std()),
        "p_value": float(np.mean(null_vals >= observed)),
    }
