"""Spearman and length-controlled partial-Spearman correlations.

OWNER: Data+Analysis
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, spearman / partial_spearman).
    benjamini_hochberg added 2026-07-23. multivariate_rank_control added
    2026-07-24 (see claims_ledger.md D11 -- this is the function that
    produced D11's verified multivariate correction, folded in from an
    inline ad hoc script into real, tested code).
TASK: rank correlation of a metric vs a task variable, plus a partial correlation
    that regresses out a confounder z (prompt length L) on ranks, plus a
    multiple-comparisons correction for when many such tests are run together,
    plus a multivariate rank control for when several confounders vary
    simultaneously and must be controlled for jointly, not one at a time.
I/O: spearman(x, y) -> (rho, p) ; partial_spearman(x, y, z) -> (rho, p) ;
    multivariate_rank_control(y, predictors) -> {name: (beta, p)} ;
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
from scipy.stats import t as _student_t

# Critical |rho| for Spearman at p<0.05 (two-tailed) by number of levels N.
# N=4 is intentionally absent: exact permutation enumeration over all 24
# possible rank orderings shows the smallest achievable two-tailed p-value,
# even at a perfect correlation, is 2/24≈0.083 — never below 0.05, for any
# sample. No threshold is reachable at N=4, so it falls through to the same
# "n/a" path already used for N>10, rather than implying 1.000 is a bar that
# could ever be met.
#
# GOTCHA (found 2026-07-24, see claims_ledger.md D15): this table is exact
# permutation enumeration over N DISTINCT ranks -- it assumes no ties among
# the N level-means. Real level-means CAN tie (e.g. three_scale_modk.csv's
# modulus=2 group: two different active_len levels averaged to the exact
# same steps_settle). When ties are present, |rho| >= this table's
# threshold is neither necessary nor sufficient for p<0.05 -- scipy's own
# p-value (which does account for ties) can disagree with this table right
# at the boundary (a real case: N=7, |rho|=0.775 < crit=0.786 reads "n.s."
# here, but scipy.stats.spearmanr's own p for that exact data was 0.041,
# below 0.05). If `ycol` group-means might tie, prefer the p-value
# `spearmanr` itself returns over a table lookup, or note the tie
# explicitly when reporting a table-based verdict.
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
    x: np.ndarray, y: np.ndarray, z: np.ndarray, collinearity_thresh: float = 0.95
) -> tuple[float, float]:
    """Partial Spearman correlation of x and y, controlling for z.

    Ranks x, y, z; linearly residualises the x- and y-ranks against the z-rank;
    then correlates the residuals.

    GUARD (added 2026-07-23, see claims_ledger.md D10; threshold tightened
    2026-07-24 per project_plan.md §0.3): if x or y is (near-)perfectly
    rank-collinear with z -- true of y=n_ops against z=seq_len for every
    synthetic task in this project except make_three_scale_task --
    residualising that variable against z leaves ~0 real variance in its
    residual. The correlation below would then be computed almost entirely
    from polyfit's floating-point rounding error, not signal: not a crash,
    just an ordinary-looking float driven by noise. Raises instead of
    silently returning that. Callers that expect this on known-degenerate
    data (run_counting.py, run_switch.py, run_maxtask.py, called as
    partial_spearman(metric, n_ops, seq_len)) must catch it.

    The threshold was originally 0.999, which only catches exact-or-
    floating-point-noise collinearity (rho=1.0). It silently passed
    dissociation.csv's real n_ops-vs-seq_len collinearity (rho=0.9895 --
    seq_len still varies a little at fixed n_ops via the `kind` field, so
    it's not exactly 1.0) even though only ~2% of x's rank variance would
    survive residualisation there too -- a result any downstream reader
    would trust as an ordinary partial correlation. No script currently
    calls partial_spearman on dissociation data, but the guard should not
    depend on that staying true. 0.95 catches this case while leaving
    pararule.csv's real, non-degenerate depth-vs-seq_len collinearity
    (rho=0.816) untouched.

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


def multivariate_rank_control(
    y: np.ndarray,
    predictors: dict[str, np.ndarray],
    condition_number_thresh: float = 1e10,
) -> dict[str, tuple[float, float]]:
    """Multiple rank-OLS regression of y on several simultaneous confounders.

    ``partial_spearman`` controls for one z at a time. When several candidate
    confounders vary simultaneously (e.g. three_scale's active_len /
    neutral_len / irrelevant_len), controlling for them one at a time can
    misattribute a joint effect to whichever variable happens to be
    partialled out. See claims_ledger.md D11: a single-confounder partial
    correlation on the composite ``seq_len`` reported active_len flipping to
    rho=+0.318 -- an artifact of partialling out the composite instead of
    the three individual length scales together. Controlling for all three
    simultaneously (this function) does not reproduce that flip.

    Ranks y and every predictor (Spearman is Pearson-on-ranks), fits one OLS
    with an intercept and all predictors at once, and returns each
    predictor's coefficient and its two-sided p-value from a t-test on that
    coefficient -- the rank-based analogue of a multiple regression's
    partial-effect test.

    Args:
        y: Response variable, shape [N].
        predictors: name -> array of shape [N], one entry per simultaneous
            confounder/predictor to control for jointly.
        condition_number_thresh: raise if the ranked design matrix's
            condition number exceeds this -- near-collinear predictors make
            individual betas numerically meaningless (the multivariate
            analogue of ``partial_spearman``'s single-z collinearity guard).

    Returns:
        Dict mapping each predictor name to ``(beta, p_value)``, in the same
        order as ``predictors``.

    Raises:
        ValueError: if the ranked design matrix is (near-)singular, or if
            there are not enough observations for the requested number of
            predictors.
    """
    names = list(predictors)
    y_rank = rankdata(y).astype(float)
    x_rank = np.column_stack([rankdata(predictors[name]).astype(float) for name in names])
    n, k = x_rank.shape

    design = np.column_stack([np.ones(n), x_rank])
    dof = n - (k + 1)
    if dof <= 0:
        raise ValueError(
            f"multivariate_rank_control: {n} observations, {k} predictors + "
            "intercept -- zero or negative degrees of freedom, cannot fit."
        )

    cond = float(np.linalg.cond(design))
    if cond > condition_number_thresh:
        raise ValueError(
            f"multivariate_rank_control: ranked design matrix is near-singular "
            f"(condition number={cond:.3e}, threshold={condition_number_thresh}) -- "
            "predictors are too rank-collinear with each other for individual "
            "betas to be numerically meaningful."
        )

    beta, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    resid = y_rank - design @ beta
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stats = beta / se
    p_values = 2.0 * _student_t.sf(np.abs(t_stats), dof)

    return {name: (float(beta[i + 1]), float(p_values[i + 1])) for i, name in enumerate(names)}


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
