"""Spearman and length-controlled partial-Spearman correlations.

OWNER: Data+Analysis
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, spearman / partial_spearman).
TASK: rank correlation of a metric vs a task variable, plus a partial correlation
    that regresses out a confounder z (prompt length L) on ranks.
I/O: spearman(x, y) -> (rho, p) ; partial_spearman(x, y, z) -> (rho, p).

NOTE: partial correlation is done on ranks (rankdata) with a linear residualisation
    against z — scipy only, no pingouin needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Critical |rho| for Spearman at p<0.05 (two-tailed) by number of levels N.
# N=4 is intentionally absent: exact permutation enumeration over all 24
# possible rank orderings shows the smallest achievable two-tailed p-value,
# even at a perfect correlation, is 2/24≈0.083 — never below 0.05, for any
# sample. No threshold is reachable at N=4, so it falls through to the same
# "n/a" path already used for N>10, rather than implying 1.000 is a bar that
# could ever be met.
_SPEARMAN_CRIT_P05: dict[int, float] = {
    5: 1.000,
    6: 0.886,
    7: 0.786,
    8: 0.738,
    9: 0.700,
    10: 0.648,
}


def spearman_by_level(
    df: pd.DataFrame, xcol: str, ycol: str
) -> tuple[float, int, float | None]:
    """Per-level Spearman: correlate ``ycol`` group-means against the ``xcol`` levels.

    This is the CANONICAL statistic for all length/depth correlations in this project:
    collapse repeated measurements to one mean per level, then rank-correlate the N
    levels. It avoids the inflated N (and false significance) of the per-row rho.

    Args:
        df: Long-form results with one row per measurement.
        xcol: The level column (e.g. ``"n_ops"`` or ``"depth"``).
        ycol: The metric column (e.g. ``"steps_settle"`` or ``"winding"``).

    Returns:
        ``(rho, n_levels, crit_p05)`` — the per-level Spearman rho, the number of
        levels N, and the |rho| needed for p<0.05 at that N (None if N is outside
        the tabulated 4–10 range). Significant iff ``abs(rho) >= crit_p05``.
    """
    gm = df.groupby(xcol)[ycol].mean()
    rho = float(spearmanr(gm.index.to_numpy(), gm.to_numpy())[0])
    n = int(len(gm))
    return rho, n, _SPEARMAN_CRIT_P05.get(n)


def fmt_by_level(df: Any, xcol: str, ycol: str) -> str:
    """One-line 'rho=.. (N=.., crit=.., sig)' string for the per-level correlation."""
    rho, n, crit = spearman_by_level(df, xcol, ycol)
    if crit is None:
        return f"rho={rho:+.3f} (N={n}, crit=n/a)"
    sig = "sig" if abs(rho) >= crit else "n.s."
    return f"rho={rho:+.3f} (N={n}, crit={crit:.3f}, {sig})"


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation.

    Args:
        x: First variable, shape [N].
        y: Second variable, shape [N].

    Returns:
        ``(rho, p_value)``.
    """
    r = spearmanr(x, y)
    return float(r[0]), float(r[1])


def partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial Spearman correlation of x and y, controlling for z.

    Ranks x, y, z; linearly residualises the x- and y-ranks against the z-rank;
    then correlates the residuals.

    Args:
        x: First variable, shape [N].
        y: Second variable, shape [N].
        z: Confounder to control for (e.g. prompt length L), shape [N].

    Returns:
        ``(rho, p_value)`` of the z-controlled correlation.
    """
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    rx = xr - np.polyval(np.polyfit(zr, xr, 1), zr)
    ry = yr - np.polyval(np.polyfit(zr, yr, 1), zr)
    r = spearmanr(rx, ry)
    return float(r[0]), float(r[1])
