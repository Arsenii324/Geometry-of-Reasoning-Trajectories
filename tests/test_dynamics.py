"""Known-answer tests for the step-size metrics, on analytically solvable paths.

WHY THIS FILE EXISTS. `metrics/dynamics.py` carries `steps_to_settle`, whose
`steps_settle` column appears in nearly every results CSV and behind ledger rows
B3/B3c. Until now it was exercised ONLY by `test_audit_findings.py`, against the
raw `trajectories/*.npy` -- which are gitignored. In a fresh clone those tests
skip, so the module had NO test at all: coverage measured 29%, and the four
functions below were entirely uncovered.

Everything here is checked against a path whose answer is known by construction:
a geometric decay with `step_norms[k] == rho**k` exactly. That makes the expected
value a closed form rather than a number copied from a previous run, so these
tests cannot drift with the data.

THE POINT OF test_steps_to_settle_matches_its_own_documented_rho_formula: the
docstring of `steps_to_settle` states rigor_audit section 3's retraction -- that
the metric measures the CONTRACTION RATE and not "effective compute", because the
threshold is a fraction of the initial transient, giving t* = log(frac)/log(rho)
with no task information in it. That claim was prose only. It is now executable.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.dynamics import (
    contraction_rate,
    normed_acceleration,
    step_norms,
    steps_to_settle,
    steps_to_settle_by_acceleration,
)

RHO = 0.8407  # the rate quoted in the steps_to_settle docstring


def geometric_path(rho: float = RHO, n: int = 80, dim: int = 6, seed: int = 0) -> np.ndarray:
    """Path whose k-th step has norm exactly ``rho**k``.

    Steps share one direction, so the geometry is trivial and every quantity
    below has a closed form. `dim` and the direction are varied in the
    task-independence test to show the metrics ignore them.
    """
    u = np.zeros(dim)
    u[seed % dim] = 1.0
    steps = np.array([rho**k * u for k in range(n - 1)])
    return np.vstack([np.zeros(dim), np.cumsum(steps, axis=0)])


def floor_safe_n(rho: float, min_step: float = 1e-3) -> int:
    """Longest path whose SMALLEST step still sits far above 1e-9.

    `normed_acceleration` divides by ``||d_k|| + ||d_{k-1}|| + eps`` with
    eps = 1e-9. Once the step norms approach that guard it dominates the
    denominator and drags the ratio down -- 9% low at rho=0.6 by k=37, against
    1.6e-9 at rho=0.97. That is the guard working as designed, not a defect, but
    a closed-form assertion has to stay above it. The bias itself is pinned
    separately in test_normed_acceleration_is_dragged_down_by_its_own_eps_guard.

    The residual bias is eps / (2 * min_step), so min_step=1e-3 leaves ~5e-7 --
    comfortably inside a 1e-5 tolerance. min_step=1e-6 leaves ~5e-4 and makes a
    tight closed-form assertion fail for EVERY rho, which is how this constant
    was chosen: by measuring the bias, not by loosening the tolerance until the
    test passed.
    """
    return int(np.log(min_step) / np.log(rho)) + 2


# --- step_norms ------------------------------------------------------------


def test_step_norms_recovers_the_constructed_norms_exactly() -> None:
    traj = geometric_path(n=30)
    got = step_norms(traj)
    assert got.shape == (29,)
    np.testing.assert_allclose(got, [RHO**k for k in range(29)], rtol=1e-10)


# --- steps_to_settle -------------------------------------------------------


@pytest.mark.parametrize("rho", [0.70, 0.8407, 0.95])
@pytest.mark.parametrize("frac", [0.1, 0.01])
def test_steps_to_settle_matches_its_own_documented_rho_formula(rho: float, frac: float) -> None:
    """t* = log(frac)/log(rho), exactly as the docstring claims.

    The threshold is `frac` of the MAXIMUM step, and on a decaying path the
    maximum is the first step. So the crossing index is the smallest k with
    rho**k < frac -- a function of rho and frac ALONE.
    """
    got = steps_to_settle(geometric_path(rho=rho, n=400), frac=frac)
    predicted = np.log(frac) / np.log(rho)
    assert abs(got - predicted) <= 1.0, f"rho={rho} frac={frac}: got {got}, formula {predicted:.2f}"


def test_steps_to_settle_carries_no_task_information() -> None:
    """rigor_audit section 3, executable: identical rho, wildly different paths.

    Dimension, direction and overall scale all change; the contraction rate does
    not. If this metric tracked anything task-like, these could not all agree.
    Any future claim that `steps_settle` tracks difficulty has to survive this.
    """
    got = {
        steps_to_settle(geometric_path(rho=RHO, n=200, dim=d, seed=s) * scale)
        for d, s, scale in [(3, 0, 1.0), (64, 5, 1.0), (8, 2, 1e3), (128, 7, 1e-3)]
    }
    assert len(got) == 1, f"contraction-rate metric varied with task-irrelevant shape: {got}"


def test_steps_to_settle_returns_full_length_when_the_path_never_settles() -> None:
    """A constant-velocity path never drops below frac of its max."""
    traj = np.cumsum(np.ones((40, 4)), axis=0)
    assert steps_to_settle(traj) == len(step_norms(traj))


# --- contraction_rate ------------------------------------------------------


@pytest.mark.parametrize("rho", [0.5, 0.8407, 0.99])
def test_contraction_rate_recovers_log_rho(rho: float) -> None:
    """mean(diff(log ||step||)) is log(rho) exactly for a geometric decay."""
    got = contraction_rate(geometric_path(rho=rho, n=60))
    assert got == pytest.approx(np.log(rho), abs=1e-9)
    assert got < 0, "a contracting path must report a negative rate"


def test_contraction_rate_is_nan_when_too_few_steps_survive_the_floor() -> None:
    """Fewer than 3 steps above 1e-9 is not enough to fit; NaN, not a number."""
    assert np.isnan(contraction_rate(np.zeros((5, 3))))
    assert np.isnan(contraction_rate(geometric_path(n=3)))


# --- normed_acceleration ---------------------------------------------------


def test_normed_acceleration_is_zero_under_constant_velocity() -> None:
    traj = np.cumsum(np.ones((20, 5)), axis=0)
    np.testing.assert_allclose(normed_acceleration(traj), 0.0, atol=1e-9)


def test_normed_acceleration_is_one_under_full_reversal() -> None:
    """Alternating steps give ||2d|| / (||d||+||d||) = 1, the metric's ceiling."""
    d = np.array([1.0, 0.0, 0.0])
    traj = np.vstack([np.zeros(3), np.cumsum([d * (-1) ** k for k in range(20)], axis=0)])
    np.testing.assert_allclose(normed_acceleration(traj), 1.0, atol=1e-6)


@pytest.mark.parametrize("rho", [0.6, 0.8407, 0.97])
def test_normed_acceleration_on_a_geometric_decay_is_the_closed_form(rho: float) -> None:
    """Collinear geometric steps give a CONSTANT (1-rho)/(1+rho) at every k.

    Length is capped by `floor_safe_n` so every step stays far above the eps
    guard; the behaviour past that point is asserted in the next test.
    """
    got = normed_acceleration(geometric_path(rho=rho, n=floor_safe_n(rho)))
    np.testing.assert_allclose(got, (1 - rho) / (1 + rho), rtol=1e-5)


def test_normed_acceleration_is_dragged_down_by_its_own_eps_guard() -> None:
    """Below ~1e-9 steps, the metric under-reports. Pinned, because it is silent.

    `normed_acceleration` adds eps=1e-9 to the denominator to avoid dividing by
    zero. Once ||step|| is itself of order 1e-9 that term stops being negligible
    and the ratio is biased toward 0 -- so a converged path reports LESS
    curvature than it has, with no warning. This is the same arithmetic-floor
    trap as D30 (bf16 rounding made Huginn look convergent ~4.6x too early), and
    anything reading this metric deep into a settled tail is reading the guard.
    """
    rho = 0.6
    exact = (1 - rho) / (1 + rho)
    a = normed_acceleration(geometric_path(rho=rho, n=40))   # decays to ~4e-9
    assert a[0] == pytest.approx(exact, rel=1e-9), "the head must still be exact"
    assert a[-1] < exact * 0.95, (
        f"expected the tail to be dragged below the closed form, got {a[-1]:.6f} vs {exact:.6f}"
    )
    assert np.all(np.diff(a) <= 1e-12), "the bias must be monotone as steps shrink"


# --- steps_to_settle_by_acceleration ---------------------------------------


def test_acceleration_settle_returns_length_for_paths_too_short_to_differentiate() -> None:
    for n in (1, 2):
        traj = np.zeros((n, 4))
        assert steps_to_settle_by_acceleration(traj) == n


def test_the_two_settle_definitions_are_not_interchangeable() -> None:
    """The docstring warns against conflating them; this pins that they differ.

    On a geometric decay above the arithmetic floor the first-order metric
    settles at log(frac)/log(rho), while the second-order one NEVER fires -- the
    normed acceleration is constant, so it cannot fall to a fraction of its own
    maximum. Two metrics, same path, opposite verdicts.
    """
    traj = geometric_path(rho=RHO, n=floor_safe_n(RHO))
    assert steps_to_settle(traj) < 20
    assert steps_to_settle_by_acceleration(traj) == len(traj)


def test_acceleration_settle_fires_spuriously_once_the_path_hits_the_eps_floor() -> None:
    """Extend the SAME path past the floor and the verdict flips to 'settled'.

    Nothing about the dynamics changed -- the decay rate is identical -- but the
    steps are now small enough that the eps guard collapses the acceleration,
    which then does cross the threshold. So this metric's answer depends on how
    LONG the recording is, which is D28's failure mode (winding was governed by
    the recording budget). Recorded here so it is a known property rather than a
    future surprise.
    """
    safe = geometric_path(rho=RHO, n=floor_safe_n(RHO))
    past = geometric_path(rho=RHO, n=200)
    assert steps_to_settle_by_acceleration(safe) == len(safe), "must not fire above the floor"
    assert steps_to_settle_by_acceleration(past) < len(past), "must fire once below it"
