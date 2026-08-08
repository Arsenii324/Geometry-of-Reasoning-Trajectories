"""How many directions does the trajectory actually move in?

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

WHY THIS QUESTION, AND WHY IT WAS NEVER ASKED. Every trajectory statistic this
project has built assumes the orbit is LOW-DIMENSIONAL. `winding_of` projects to a
2-D PCA plane. All nine variants W1-W9 project, or measure angle about a point, or
both. `classify_shape`'s settle/loop/drift taxonomy is a vocabulary for planar
curves. The eigenplane test projects onto the real plane of one complex eigenpair.
Every one of those is a statement about a 2-D object, and none of them checked that
the object is 2-D.

THE ENVELOPE HAS TO COME OUT FIRST, or the answer is trivially "low". Step norms
decay geometrically (~0.87 per unroll), so the first few steps carry nearly all the
energy and ANY set of directions looks low-dimensional under a
participation-ratio-of-vectors measure. Measured: raw step differences give PR 5.34,
and isotropic random vectors rescaled to the SAME norms give 5.74 -- i.e. essentially
all of that apparent low dimensionality is the decay envelope, not alignment.
Normalising each step to unit length removes it and asks the question that was meant.

THE NULL IS ISOTROPIC DIRECTIONS AT MATCHED SAMPLE COUNT. With m ~ 22 samples in
d = 5280, isotropic unit vectors have participation ratio ~ m, not ~ d, because the
sample count bounds the rank. So the null must be simulated at the same m rather
than assumed to be d; comparing against 5280 would make any real orbit look
dramatically low-dimensional and would be meaningless.
"""

from __future__ import annotations

import numpy as np


def participation_ratio(x: np.ndarray) -> float:
    """Effective number of directions spanned by the rows of ``x``.

    ``(sum s_i^2)^2 / sum s_i^4`` on the singular values: 1 if every row is
    parallel, ``rank`` if the spectrum is flat. This is the same functional D48
    used for state dimensionality, applied here to step DIRECTIONS.
    """
    s = np.linalg.svd(np.asarray(x, dtype=np.float64), compute_uv=False) ** 2
    tot = float(s.sum())
    return float(tot**2 / float((s**2).sum())) if tot > 0 else 0.0


def step_directions(traj: np.ndarray, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Unit-normalised consecutive step directions over ``traj[lo:hi]``.

    Normalising is the point: it removes the geometric decay envelope, which
    otherwise dominates any dimensionality measure (see the module docstring).
    """
    d = np.diff(np.asarray(traj, dtype=np.float64), axis=0)
    d = d[lo:hi]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    keep = (n > 0).ravel()
    return d[keep] / n[keep]


def effective_dimension(traj: np.ndarray, lo: int = 0, hi: int | None = None,
                        n_null: int = 20, seed: int = 0) -> dict:
    """Effective dimensionality of the step directions, against an isotropic null.

    Returns ``pr`` (real), ``null_mean``/``null_sd`` (isotropic at the same sample
    count and ambient dimension), ``z``, ``ratio`` and ``cos_consecutive``. The
    z-score is what makes a single orbit interpretable; the ratio is what makes a
    population of them comparable.
    """
    u = step_directions(traj, lo, hi)
    m, d = u.shape
    if m < 4:
        return {"ok": False, "why": f"only {m} steps in window", "n_steps": m}
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        r = rng.normal(size=(m, d))
        nulls.append(participation_ratio(r / np.linalg.norm(r, axis=1, keepdims=True)))
    nulls = np.asarray(nulls)
    pr = participation_ratio(u)
    sd = float(nulls.std(ddof=1))
    return {
        "ok": True, "n_steps": m, "ambient": d, "pr": pr,
        "null_mean": float(nulls.mean()), "null_sd": sd,
        "z": float((pr - nulls.mean()) / sd) if sd > 0 else float("nan"),
        "ratio": float(pr / nulls.mean()) if nulls.mean() > 0 else float("nan"),
        "cos_consecutive": float(np.mean(np.sum(u[:-1] * u[1:], axis=1))),
    }
