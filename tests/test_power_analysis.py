"""Fast checks on the power analysis (docs/power_and_preregistration.md,
claims_ledger.md-adjacent honesty gate). The full Monte Carlo grid
(scripts/run_power_analysis.py's main()) takes ~60s -- these tests use a
much smaller trial count/grid so the suite stays fast, and pin the
loop-rate arithmetic (which is exact, not simulated) precisely.
"""

from __future__ import annotations

import pytest
from scripts.run_power_analysis import (
    BLAYNEY_RATE_BASELINE,
    BLAYNEY_RATE_OPTIMISTIC,
    LOOP_BAR,
    _current_pool_size,
    _loops_observed_outside_forceloop,
    loop_rate_power_table,
    n_needed_for_bar,
    spearman_power_curve,
)


def test_current_pool_size_matches_verified_count() -> None:
    """Regression-pins the real winding/shape-classified pool size against
    the cached CSVs (docs/power_and_preregistration.md cites 1,294).
    """
    assert _current_pool_size() == 1294


def test_loops_observed_outside_forceloop_is_zero() -> None:
    """Every real, full-compute-budget extraction in this project settles --
    the only loops ever observed come from the artificially starved
    forceloop.csv sweep, excluded here.
    """
    assert _loops_observed_outside_forceloop() == 0


def test_n_needed_for_bar_exact_arithmetic() -> None:
    """N = LOOP_BAR / rate, exactly -- no simulation involved here."""
    needed = n_needed_for_bar()
    baseline_row = needed[needed["rate_value"] == BLAYNEY_RATE_BASELINE].iloc[0]
    optimistic_row = needed[needed["rate_value"] == BLAYNEY_RATE_OPTIMISTIC].iloc[0]
    assert baseline_row["n_needed_for_5_loops"] == pytest.approx(LOOP_BAR / BLAYNEY_RATE_BASELINE)
    assert baseline_row["n_needed_for_5_loops"] == pytest.approx(25000.0)
    assert optimistic_row["n_needed_for_5_loops"] == pytest.approx(3571.4, abs=0.1)


def test_loop_rate_power_table_current_pool_does_not_clear_baseline_bar() -> None:
    """The project's real current pool (1,294) does not clear the >=5-loop
    bar at Blayney's baseline rate -- consistent with observing zero loops.
    """
    table = loop_rate_power_table()
    current = table[
        table["scenario"].str.contains("current answer-token pool")
        & table["rate"].str.contains("baseline")
    ].iloc[0]
    assert current["n_draws"] == 1294
    assert current["expected_loops"] == pytest.approx(1294 * BLAYNEY_RATE_BASELINE)
    assert not current["clears_bar_of_5"]


def test_spearman_power_increases_with_effect_size_and_sample_size() -> None:
    """Sanity check on the Monte Carlo mechanics (small grid, few trials for
    speed): power must be monotone-ish in both N and true rho, and a strong
    effect at a moderate N must clear a low bar reliably.
    """
    curve = spearman_power_curve(
        n_levels=(6, 20), true_rhos=(0.2, 0.9), n_trials=1000, seed=0
    )
    power = {
        (int(row.n_levels), row.rho_true): row.power for row in curve.itertuples()
    }
    # Higher true rho must give higher power at fixed N.
    assert power[(6, 0.9)] > power[(6, 0.2)]
    assert power[(20, 0.9)] > power[(20, 0.2)]
    # Higher N must give higher power at fixed (large) rho.
    assert power[(20, 0.9)] > power[(6, 0.9)]
    # A strong effect at N=20 should be detected most of the time.
    assert power[(20, 0.9)] > 0.9
    # A weak effect at N=6 should rarely be detected.
    assert power[(6, 0.2)] < 0.3
