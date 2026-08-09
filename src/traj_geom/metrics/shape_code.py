"""The orbit's shape as a feature vector, with position deliberately removed.

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

THE OBJECTION THIS EXISTS TO ANSWER. D79 and D80 test four hand-picked statistics --
effective dimensionality, step cosine, contraction rate, settling time -- and find
none of them tracks the computation. The obvious reply is that a representation
living in a direction none of those functionals is sensitive to would be invisible.
The way to answer it is to stop picking statistics and instead ask whether ANY
function of the shape can decode the task, by handing a classifier the shape itself
and cross-validating.

WHAT "SHAPE" HAS TO MEAN HERE, AND WHY THE OBVIOUS ENCODING IS USELESS. The states
themselves are a function of the prompt -- Huginn re-injects the prompt embeddings
at every unroll through the adapter on [h, e] (D70(4)), so the token bag is present
in every state by architecture, and D73 measured random weights decoding a count
from them at R2 = 0.99999. A classifier fed raw states would therefore decode the
task family trivially and prove nothing about geometry.

So the feature is the GRAM MATRIX OF UNIT STEP DIRECTIONS: the pairwise cosines
between the normalised steps over a fixed window. That object is invariant to
translating the orbit, to rotating it, and to scaling it -- exactly the
transformations that carry "where the trajectory is" while preserving "what shape it
traces" -- and it is a COMPLETE invariant up to those, so nothing about the shape is
discarded by construction. If the task is decodable from it, the shape encodes the
task; if not, and the same classifier decodes the task easily from raw states, then
the information is in the position and not in the path.
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.dimension import step_directions


def gram_code(traj: np.ndarray, lo: int = 0, m: int = 20) -> np.ndarray:
    """Upper triangle of the m x m cosine matrix between unit step directions.

    Args:
        traj: ``[T, d]`` states.
        lo: first step index of the window.
        m: number of consecutive steps; the code has ``m (m - 1) / 2`` entries.

    Returns the flattened strict upper triangle, or an empty array when the
    trajectory is too short. Fixed ``m`` across orbits is required, not merely
    tidy: the code's LENGTH would otherwise vary with the window, and a classifier
    cannot be given ragged features -- and D80 shows the window is the single
    largest determinant of every shape statistic, so it must be held constant.
    """
    u = step_directions(np.asarray(traj, dtype=np.float64), lo=lo, hi=lo + m)
    if len(u) < m:
        return np.empty(0)
    g = u @ u.T
    iu = np.triu_indices(m, k=1)
    return g[iu]


def position_code(traj: np.ndarray, lo: int = 0, m: int = 20,
                  n_proj: int = 190, seed: int = 0) -> np.ndarray:
    """A random projection of the raw states -- the POSITIVE CONTROL feature.

    Same dimensionality as `gram_code` at the default settings, so the comparison
    is between two feature sets of equal size and not between a rich one and a
    poor one. The states carry the prompt by architecture, so a classifier that
    cannot decode the task from THIS has a problem with the classifier rather than
    with the geometry -- which is precisely what makes a null on `gram_code`
    readable.
    """
    x = np.asarray(traj, dtype=np.float64)[lo:lo + m]
    if len(x) < m:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    # Round UP then truncate, so the control has EXACTLY `n_proj` features rather
    # than `m * (n_proj // m)`. At the defaults that is the difference between 190
    # and 180, and a control with 5% fewer features than the arm it validates is a
    # gift to the arm.
    p = rng.normal(size=(x.shape[1], max(1, -(-n_proj // m))))
    p /= np.linalg.norm(p, axis=0, keepdims=True)
    return (x @ p).ravel()[:n_proj]


def is_rotation_invariant(traj: np.ndarray, lo: int = 0, m: int = 20,
                          seed: int = 0) -> float:
    """Max absolute change in `gram_code` under a random rotation of the whole orbit.

    Not a test helper -- a property the feature must have for the claim to mean
    anything, checkable on any real orbit. If rotating the trajectory changed the
    code, the code would be encoding the basis, and a "shape decodes the task"
    result could be a statement about where the model happens to put things.
    """
    x = np.asarray(traj, dtype=np.float64)
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(x.shape[1], x.shape[1])))[0]
    a, b = gram_code(x, lo, m), gram_code(x @ q, lo, m)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.max(np.abs(a - b)))
