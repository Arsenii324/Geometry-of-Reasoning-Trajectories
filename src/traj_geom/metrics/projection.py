"""Project a high-dimensional trajectory into 2D for winding analysis.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/00_smoke_extract.ipynb, cell-4).
TASK: Reduce states [T, hidden_dim] to [T, 2] with sklearn PCA (2 components),
    preserving step order so the 2D path can be wound.
I/O: states [T, hidden_dim] -> points_2d [T, 2].
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_to_2d(states: np.ndarray) -> np.ndarray:
    """Project a trajectory to its first two principal components.

    Args:
        states: Trajectory array of shape [T, hidden_dim].

    Returns:
        The projected path of shape [T, 2], in original step order.

    Note:
        ``svd_solver="full"`` is deliberate. With the default ``"auto"``,
        sklearn picks *randomized* SVD for a [T, 5280] array (max dim > 500 and
        n_components < 0.8 * min(shape)), and with ``random_state=None`` that
        makes every winding number nondeterministic run to run. This is
        directly visible in `results/full_synthetic_experiments.csv`, where
        three bit-identical trajectories recorded three different winding
        values (5.588866578 / 5.588866390 / 5.588865752). The observed spread
        is small (<= 7.4e-7 relative, and the PC2/PC3 eigenvalue gap is >= 0.33
        on all 140 banked trajectories, so the plane never flips), but exact
        reproducibility is cheap here: full SVD on this shape costs ~45 ms and
        is exact rather than approximate. See docs/rigor_audit.md section 9.
    """
    return PCA(n_components=2, svd_solver="full").fit_transform(np.asarray(states))
