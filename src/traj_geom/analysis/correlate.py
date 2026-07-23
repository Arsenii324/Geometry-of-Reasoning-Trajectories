"""Spearman and length-controlled partial-Spearman correlations.

OWNER: Data+Analysis
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, spearman / partial_spearman).
    benjamini_hochberg added 2026-07-23.
TASK: rank correlation of a metric vs a task variable, plus a partial correlation
    that regresses out a confounder z (prompt length L) on ranks, plus a
    multiple-comparisons correction for when many such tests are run together.
I/O: spearman(x, y) -> (rho, p) ; partial_spearman(x, y, z) -> (rho, p) ;
    benjamini_hochberg(p_values) -> (q_values, significant).

NOTE: partial correlation is done on ranks (rankdata) with a linear residualisation
    against z — scipy only, no pingouin needed.

NOTE on benjamini_hochberg: this project runs many correlation tests across
    scripts/tasks/metrics/length-scales with no multiple-comparisons
    correction applied anywhere as of 2026-07-23 -- a real gap, flagged but
    not yet closed. This function is the tool, provided uncorrected so far
    because deciding what counts as one "family" of tests (per-script?
    per-experiment? project-wide, for the paper's actual claims?) is an
    interpretive call belonging to whoever writes those claims up, not
    something to impose unilaterally by retrofitting every script's output.
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
    x: np.ndarray, y: np.ndarray, z: np.ndarray, collinearity_thresh: float = 0.999
) -> tuple[float, float]:
    """Partial Spearman correlation of x and y, controlling for z.

    Ranks x, y, z; linearly residualises the x- and y-ranks against the z-rank;
    then correlates the residuals.

    GUARD (added 2026-07-23, see claims_ledger.md D10): if x or y is
    (near-)perfectly rank-collinear with z -- true of y=n_ops against
    z=seq_len for every synthetic task in this project except
    make_three_scale_task -- residualising that variable against z leaves
    ~0 real variance in its residual. The correlation below would then be
    computed almost entirely from polyfit's floating-point rounding error,
    not signal: not a crash, just an ordinary-looking float driven by
    noise. Raises instead of silently returning that. Callers that expect
    this on known-degenerate data (run_counting.py, run_switch.py,
    run_maxtask.py, called as partial_spearman(metric, n_ops, seq_len))
    must catch it.

    Args:
        x: First variable, shape [N].
        y: Second variable, shape [N].
        z: Confounder to control for (e.g. prompt length L), shape [N].
        collinearity_thresh: raise if |Spearman(x, z)| or |Spearman(y, z)|
            exceeds this.

    Returns:
        ``(rho, p_value)`` of the z-controlled correlation.

    Raises:
        ValueError: if x or y is too rank-collinear with z for
            residualisation to leave any real variance to correlate.
    """
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)

    rho_xz = float(spearmanr(xr, zr)[0])
    rho_yz = float(spearmanr(yr, zr)[0])
    culprit, rho_bad = ("x", rho_xz) if abs(rho_xz) >= abs(rho_yz) else ("y", rho_yz)
    if abs(rho_bad) > collinearity_thresh:
        raise ValueError(
            f"partial_spearman: {culprit} is rank-collinear with z "
            f"(rho={rho_bad:.6f}, threshold={collinearity_thresh}) -- "
            f"residualising {culprit} against z would leave no real variance "
            "to correlate; any rho returned would be floating-point noise, "
            "not signal. This confounder cannot be partialled out with this "
            "data (see claims_ledger.md D10)."
        )

    rx = xr - np.polyval(np.polyfit(zr, xr, 1), zr)
    ry = yr - np.polyval(np.polyfit(zr, yr, 1), zr)
    r = spearmanr(rx, ry)
    return float(r[0]), float(r[1])


def benjamini_hochberg(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for a family of hypothesis tests.

    Standard step-up procedure: sort ascending, find the largest rank k with
    p_(k) <= (k/m)*alpha, reject all i<=k. Adjusted p-values (q-values) are
    the smallest FDR at which each hypothesis would be rejected, computed as
    a monotone (enforced by cumulative min from the largest p-value down)
    running min of p_(i)*m/i.

    Args:
        p_values: Raw p-values from one family of tests, shape [N]. Choosing
            what counts as one "family" (all tests in one script? one
            experiment? every correlation in the paper?) is the caller's
            call, not this function's.
        alpha: Target false discovery rate.

    Returns:
        ``(q_values, significant)`` — adjusted p-values in the original
        input order, and a boolean array of which are significant at
        ``alpha`` after correction (``q <= alpha``).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]  # enforce monotonicity
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty(m)
    q[order] = q_ranked
    return q, q <= alpha
