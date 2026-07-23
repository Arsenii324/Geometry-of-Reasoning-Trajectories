"""Validation of the per-level Spearman helper and the partial_spearman guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traj_geom.analysis.correlate import benjamini_hochberg, partial_spearman, spearman_by_level


def _levels_df(levels: tuple[int, ...], slope: float) -> pd.DataFrame:
    """Several noisy rows per level whose group-means are strictly monotone in x."""
    rows = [{"x": lvl, "y": slope * lvl + s} for lvl in levels for s in range(4)]
    return pd.DataFrame(rows)


def test_spearman_by_level_monotone_increasing() -> None:
    """Strictly increasing group-means give rho=+1.0, the right N and crit."""
    df = _levels_df((2, 4, 6, 8, 10, 12), slope=2.0)
    rho, n, crit = spearman_by_level(df, "x", "y")
    assert rho == pytest.approx(1.0)
    assert n == 6
    assert crit == 0.886


def test_spearman_by_level_monotone_decreasing() -> None:
    """Strictly decreasing group-means give rho=-1.0 with N counted correctly."""
    df = _levels_df((1, 2, 3, 4, 5), slope=-3.0)
    rho, n, crit = spearman_by_level(df, "x", "y")
    assert rho == pytest.approx(-1.0)
    assert n == 5
    assert crit == 1.000


def test_partial_spearman_raises_on_rank_collinear_confounder() -> None:
    """z a deterministic (rank-identical) function of x must raise, not silently
    return a floating-point-noise-driven number.

    Reproduces the exact shape of the degenerate case documented in
    claims_ledger.md D10: every synthetic task's seq_len is a strict function
    of n_ops, so x and z share identical tie-groups (multiple seeds per
    n_ops level, same seq_len within a level).
    """
    n_ops = np.repeat([2, 4, 8, 16, 24, 32], 5)  # 5 seeds per level
    seq_len = n_ops * 3 + 10  # deterministic function of n_ops, same shape as real data
    y = np.random.default_rng(0).normal(size=n_ops.shape)
    with pytest.raises(ValueError, match="rank-collinear"):
        partial_spearman(y, n_ops, seq_len)


def test_partial_spearman_works_for_genuinely_independent_confounder() -> None:
    """z independent of x (three_scale-style) must NOT raise, and should
    return an ordinary, well-defined result.
    """
    rng = np.random.default_rng(0)
    x = np.repeat([1, 3, 5, 7, 9], 6)
    z = rng.integers(0, 10, size=x.shape)  # independent of x, not a function of it
    y = 2.0 * x + rng.normal(scale=0.1, size=x.shape)
    rho, p = partial_spearman(y, x, z)
    assert rho == pytest.approx(1.0, abs=0.05)
    assert 0.0 <= p <= 1.0


def test_benjamini_hochberg_canonical_example() -> None:
    """Hand-verifiable worked example (the standard BH illustration): of 10
    p-values, only the smallest clears the step-up threshold at alpha=0.05.

    Thresholds are (i/10)*0.05 for i=1..10: [.005,.01,.015,.02,.025,.03,
    .035,.04,.045,.05]. Only p_(1)=0.005 <= 0.005 holds; every later i fails
    its own threshold, so the step-up procedure stops at k=1.
    """
    p = np.array([0.005, 0.011, 0.02, 0.04, 0.13, 0.25, 0.5, 0.7, 0.9, 0.99])
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert sig.tolist() == [True, False, False, False, False, False, False, False, False, False]
    assert q[0] == pytest.approx(0.05)  # q_(1) = p_(1) * 10 / 1
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)  # q must be monotone in sorted-p order


def test_benjamini_hochberg_all_significant_boundary() -> None:
    """p_(i) = (i/m)*alpha exactly for every i -> every q collapses to alpha."""
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert q == pytest.approx([0.05] * 5)
    assert sig.all()


def test_benjamini_hochberg_none_significant() -> None:
    """All p-values far above alpha -> nothing survives correction."""
    p = np.array([0.2, 0.4, 0.6, 0.8])
    q, sig = benjamini_hochberg(p, alpha=0.05)
    assert not sig.any()
