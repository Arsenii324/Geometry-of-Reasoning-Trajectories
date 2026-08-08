"""Known-answer tests for the Jacobian-free rho estimator.

rho is this project's central quantity: the one thing training demonstrably
changes (D52, 0.7048 -> 0.8577 with complete separation over 14 weight-sets) and
the only one it has managed to steer (D59). `observable_convergence` is the
estimator that gets it from a convergence curve instead of a Jacobian, and it was
62% covered -- the saturation guard, `estimate_rho_from_directions` and
`predict_curve` had no test.

Everything below builds a curve with a KNOWN rho and checks the estimator returns
it. That matters more here than usual: UNDERSTANDING.md section 5 records an
unexplained 0.058 disagreement between two rho estimators (two-orbit 0.887 vs
Arnoldi ~0.80), and all three candidate explanations were tested and refuted. A
known-answer test cannot resolve that, but it does establish that THIS estimator
is unbiased on data that obeys the law exactly -- so the discrepancy is not
coming from an arithmetic error here.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.observable_convergence import (
    estimate_rho_from_curve,
    estimate_rho_from_directions,
    fit_curve,
    predict_curve,
)


def quadratic_gap(rho: float, n: int = 40, amp: float = 1.0, floor: float = 0.0) -> tuple:
    """A curve obeying the law exactly: gap(r) = floor + amp * rho**(2r)."""
    r = np.arange(n, dtype=float)
    return r, floor + amp * rho ** (2.0 * r)


# --- the estimator recovers what it was given ------------------------------


@pytest.mark.parametrize("rho", [0.60, 0.7048, 0.8577, 0.95])
def test_estimate_rho_from_curve_recovers_a_known_quadratic_rate(rho: float) -> None:
    """The two rho values in the parametrisation are D52's untrained and trained means."""
    r, gap = quadratic_gap(rho, n=30)
    got, r2 = estimate_rho_from_curve(r, gap, floor=0.0, quadratic=True)
    assert got == pytest.approx(rho, rel=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("rho", [0.60, 0.8577])
def test_estimate_rho_from_curve_handles_a_linear_observable(rho: float) -> None:
    """`quadratic=False` is for an observable linear in the state error."""
    r = np.arange(30, dtype=float)
    got, r2 = estimate_rho_from_curve(r, rho**r, floor=0.0, quadratic=False)
    assert got == pytest.approx(rho, rel=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_the_quadratic_flag_is_not_cosmetic() -> None:
    """Reading a quadratic observable as linear squares the answer; pinned so the
    two cannot be silently swapped."""
    r, gap = quadratic_gap(0.8, n=30)
    quad, _ = estimate_rho_from_curve(r, gap, floor=0.0, quadratic=True)
    lin, _ = estimate_rho_from_curve(r, gap, floor=0.0, quadratic=False)
    assert lin == pytest.approx(quad**2, rel=1e-9)


def test_a_nonzero_floor_is_estimated_when_not_supplied() -> None:
    """floor=None takes the mean of the last three points, per fit_curve."""
    rho, floor = 0.75, 0.02
    r, gap = quadratic_gap(rho, n=40, floor=floor)      # saturates well before r=40
    got, r2 = estimate_rho_from_curve(r, gap, floor=None, quadratic=True)
    assert got == pytest.approx(rho, rel=1e-3)
    assert r2 > 0.999


# --- the guard that refuses to fit a saturated curve -----------------------


def test_fit_curve_refuses_a_fully_saturated_curve() -> None:
    """A flat curve has no identifiable slope; it must raise, not return a number.

    This is the failure mode the module exists to avoid: D30 showed bf16 rounding
    makes Huginn look convergent ~4.6x too early, so a curve that has hit its
    arithmetic floor is exactly what this estimator will be handed in practice.
    """
    r = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="saturated"):
        fit_curve(r, np.full(20, 0.05))


def test_fit_curve_refuses_when_too_few_points_clear_the_floor() -> None:
    """Only two points above min_gap is not a fit, however clean they look."""
    r = np.arange(10, dtype=float)
    gap = np.concatenate([[1.0, 0.5], np.full(8, 1e-9)])
    with pytest.raises(ValueError, match="only 2 points|saturated"):
        fit_curve(r, gap, floor=0.0)


# --- prediction, not description -------------------------------------------


def test_predict_curve_round_trips_through_the_estimator() -> None:
    """predict_curve exists to check a curve against an INDEPENDENT rho.
    Feeding its output back must return the rho it was given."""
    r = np.arange(30, dtype=float)
    curve = predict_curve(r, rho=0.83, gap_at_first=2.5, floor=0.0, quadratic=True)
    got, r2 = estimate_rho_from_curve(r, curve, floor=0.0, quadratic=True)
    assert got == pytest.approx(0.83, rel=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_predict_curve_is_anchored_at_the_first_unroll() -> None:
    r = np.arange(5, 25, dtype=float)                   # deliberately not starting at 0
    curve = predict_curve(r, rho=0.9, gap_at_first=3.0, floor=0.1)
    assert curve[0] == pytest.approx(0.1 + 3.0)
    assert np.all(np.diff(curve) < 0), "a contracting prediction must decrease"


# --- directions ------------------------------------------------------------


def test_estimate_rho_from_directions_recovers_a_known_angular_decay() -> None:
    """Directions closing on a limit at rate rho give 1-cos ~ rho**(2r).

    Built from an exact rotation toward `ref` with angle theta_r = theta0 *
    rho**r, so the answer is known by construction. Tolerance is loose-ish
    because 1-cos = 1-(1+theta^2)^-1/2 is only quadratic to leading order.
    """
    rho, theta0, n, dim = 0.85, 0.05, 25, 8
    ref = np.zeros(dim)
    ref[0] = 1.0
    perp = np.zeros(dim)
    perp[1] = 1.0
    r = np.arange(n, dtype=float)
    dirs = np.array([ref + theta0 * rho**k * perp for k in r])
    got, r2 = estimate_rho_from_directions(r, dirs, reference=ref)
    assert got == pytest.approx(rho, rel=2e-2), f"recovered {got:.4f}, expected {rho}"
    assert r2 > 0.99


def test_estimate_rho_from_directions_drops_the_degenerate_final_point() -> None:
    """With reference=None the last row IS the reference, so its 1-cos is 0 and
    log(0) would poison the fit. The docstring says it is dropped; assert it."""
    rho, theta0, n, dim = 0.8, 0.05, 20, 6
    ref = np.zeros(dim)
    ref[0] = 1.0
    perp = np.zeros(dim)
    perp[1] = 1.0
    r = np.arange(n, dtype=float)
    dirs = np.array([ref + theta0 * rho**k * perp for k in r])
    dirs[-1] = ref                                       # exact limit in the last row
    got, r2 = estimate_rho_from_directions(r, dirs)
    assert np.isfinite(got) and np.isfinite(r2), "the degenerate row leaked into the fit"
    assert got == pytest.approx(rho, rel=5e-2)
