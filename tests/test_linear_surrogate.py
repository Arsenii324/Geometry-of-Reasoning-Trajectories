"""The surrogate must reproduce cases whose answer is known analytically.

If it cannot recover cos(phi) from a single rotating pair, then a mismatch against
a real orbit would say nothing about the model -- it would say the generator is
wrong. Every test here has a closed-form target.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.dimension import participation_ratio, step_directions
from traj_geom.metrics.linear_surrogate import (
    spectrum_tail,
    surrogate_orbit,
    surrogate_statistics,
)


def _cos_consecutive(traj, lo=0, hi=None):
    u = step_directions(traj, lo=lo, hi=hi)
    return float(np.mean(np.sum(u[:-1] * u[1:], axis=1)))


@pytest.mark.parametrize("phi", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("rho", [0.6, 0.95])
def test_one_rotating_pair_gives_exactly_cos_phi(phi, rho) -> None:
    """D76's identity, which is the whole reason the step cosine is trusted.

    For A = rho R(phi) on an invariant plane, consecutive step directions meet at
    exactly phi regardless of rho. If the surrogate did not reproduce this, it
    would not be generating the map it claims to.
    """
    eigs = np.array([rho * np.exp(1j * phi), rho * np.exp(-1j * phi)])
    traj = surrogate_orbit(eigs, n_steps=40, dim=30, seed=0)
    assert _cos_consecutive(traj, hi=20) == pytest.approx(np.cos(phi), abs=1e-6)


def test_a_conjugate_pair_spans_two_dimensions_not_four() -> None:
    """Arnoldi lists both members of a pair; consuming them twice would double the
    surrogate's dimensionality and inflate every participation ratio built on it."""
    eigs = np.array([0.9 * np.exp(1j), 0.9 * np.exp(-1j)])
    traj = surrogate_orbit(eigs, n_steps=40, dim=50, seed=1)
    assert np.linalg.matrix_rank(traj - traj.mean(0), tol=1e-8) == 2


def test_a_real_eigenvalue_gives_parallel_steps() -> None:
    """A pure contraction has cos = +1 exactly, for ANY rate.

    This is the alternative reading D76(1) rules out -- "the untrained arm just
    converges faster" cannot produce a cosine below 1 -- so the surrogate has to
    honour it or it cannot be used to argue about that claim.
    """
    traj = surrogate_orbit(np.array([0.8 + 0j]), n_steps=40, dim=20, seed=2)
    assert _cos_consecutive(traj, hi=25) == pytest.approx(1.0, abs=1e-6)


def _pr_of(n_pairs, seed, dim=400, n_step=60, hi=30):
    eigs = np.concatenate([[0.9 * np.exp(1j * p), 0.9 * np.exp(-1j * p)]
                           for p in np.linspace(0.3, 2.5, n_pairs)])
    t = surrogate_orbit(eigs, n_steps=n_step, dim=dim, seed=seed)
    return participation_ratio(step_directions(t, hi=hi))


def test_more_modes_raise_the_participation_ratio() -> None:
    """Effective dimension must respond to the spectrum, monotonically."""
    curve = [np.mean([_pr_of(n, s) for s in range(8)]) for n in (1, 2, 4, 8, 16, 32)]
    assert curve == sorted(curve), curve
    assert curve[0] == pytest.approx(2.0, abs=0.05), "one pair spans exactly a plane"


def test_participation_ratio_lags_far_behind_the_mode_count() -> None:
    """MEASURED, and it corrects an assumption I made before measuring it.

    PR is ENERGY-weighted, and the surrogate's per-mode amplitudes are random, so a
    mixture of 2k equal-modulus real dimensions does NOT give PR = 2k. Measured at
    30 step directions: 4 dims -> 2.2, 8 -> 3.7, 16 -> 5.4, 32 -> 7.4, 64 -> 11.4,
    128 -> 12.8. My first draft of this test asserted PR > 8 at 16 dimensions and
    failed at 5.4.

    It matters for reading D74. "The orbit spans ~13 effective dimensions" does NOT
    mean ~13 modes are active; under random excitation it implies substantially
    more, so 13 is a floor on the operator's active spectrum rather than a count of
    it.
    """
    assert np.mean([_pr_of(8, s) for s in range(8)]) < 8.0
    assert np.mean([_pr_of(64, s) for s in range(4)]) > 11.0


def test_the_contraction_rate_comes_back() -> None:
    """The step-norm decay of the surrogate must equal the dominant modulus."""
    eigs = np.array([0.87 * np.exp(1j * 0.6), 0.87 * np.exp(-1j * 0.6)])
    out = surrogate_statistics(eigs, n_steps=60, dim=100, n_draw=16, hi=30)
    assert out["ok"]
    assert out["contraction"] == pytest.approx(0.87, abs=0.02)


def test_statistics_report_a_spread_that_the_initial_condition_creates(
) -> None:
    """The band must be non-degenerate, and it must come from the START.

    With the spectrum fixed, the only thing varying across draws is where the orbit
    began -- which is exactly what D78 says the real model redraws every forward. A
    zero-width band would mean the surrogate cannot express that variation and no
    real orbit could ever be said to land inside it.
    """
    eigs = np.concatenate([[0.9 * np.exp(1j * p), 0.9 * np.exp(-1j * p)]
                           for p in (0.4, 1.1, 1.9, 2.4)])
    out = surrogate_statistics(eigs, n_steps=60, dim=300, n_draw=40, hi=30)
    assert out["ok"] and out["n_draw"] == 40
    assert out["pr_hi"] > out["pr_lo"]
    assert out["cos_hi"] > out["cos_lo"]


def test_spectrum_tail_extends_below_the_measured_modes() -> None:
    """The truncation control: added modes must be SUBDOMINANT, never new leaders."""
    eigs = np.array([0.9 + 0.1j, 0.9 - 0.1j, 0.6 + 0.2j, 0.6 - 0.2j])
    wide = spectrum_tail(eigs, n_extra=6, decay=0.85, seed=0)
    assert len(wide) == len(eigs) + 6
    assert np.allclose(wide[:4], eigs)
    assert np.abs(wide[4:]).max() < np.abs(eigs).min()


def test_truncation_shows_up_where_it_should() -> None:
    """Adding subdominant modes must raise effective dimension without moving the
    contraction rate -- otherwise the sensitivity check would confound the two."""
    eigs = np.concatenate([[0.9 * np.exp(1j * p), 0.9 * np.exp(-1j * p)]
                           for p in (0.5, 1.2, 2.0)])
    a = surrogate_statistics(eigs, n_steps=60, dim=300, n_draw=24, hi=25)
    b = surrogate_statistics(spectrum_tail(eigs, 8, seed=1), n_steps=60, dim=300,
                             n_draw=24, hi=25)
    assert b["pr"] > a["pr"]
    assert b["contraction"] == pytest.approx(a["contraction"], abs=0.05)


def test_an_empty_spectrum_refuses_rather_than_returning_zeros() -> None:
    out = surrogate_statistics(np.array([], dtype=complex), n_steps=40, dim=20)
    assert out["ok"] is False
