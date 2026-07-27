"""Tests for the on-manifold surrogate.

A surrogate is only as good as the invariants it preserves, so these tests
check those invariants directly rather than any downstream statistic. They run
at several dimensions because the construction lives in the (n-1)-dimensional
complement of the pole and a dimension-dependent bug would otherwise only
appear at n=5280, where it is expensive to notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.surrogate import manifold_matched_surrogate, spherical_decomposition


def _sphere_traj(dim: int, n_steps: int = 80, rho: float = 0.85,
                 radius: float = 76.37, seed: int = 0) -> np.ndarray:
    """A rotating, contracting path confined exactly to a sphere."""
    r = np.random.default_rng(seed)
    pole = r.normal(size=dim)
    pole /= np.linalg.norm(pole)
    q = r.normal(size=dim)
    q -= q @ pole * pole
    q /= np.linalg.norm(q)
    phi = 1.2 * rho ** np.arange(n_steps)
    out = []
    for t in range(n_steps):
        w = r.normal(size=dim)
        w -= w @ pole * pole
        w -= w @ q * q
        w /= np.linalg.norm(w)
        qq = np.cos(0.3 * t) * q + np.sin(0.3 * t) * w
        qq /= np.linalg.norm(qq)
        out.append(radius * (np.cos(phi[t]) * pole + np.sin(phi[t]) * qq))
    return np.array(out)


@pytest.mark.parametrize("dim", [8, 64, 512])
def test_surrogate_stays_on_the_sphere(dim: int) -> None:
    """The invariant the old null violated: it diffused 76.37 -> 154.76."""
    a = _sphere_traj(dim)
    s = manifold_matched_surrogate(a, np.random.default_rng(0))
    na, ns = np.linalg.norm(a, axis=1), np.linalg.norm(s, axis=1)
    assert np.abs(na - ns).max() < 1e-6 * na.mean()


@pytest.mark.parametrize("dim", [8, 64, 512])
def test_surrogate_preserves_the_radial_profile(dim: int) -> None:
    """Radial motion generates angle, so it must be held fixed, not randomised."""
    a = _sphere_traj(dim)
    s = manifold_matched_surrogate(a, np.random.default_rng(1))
    _, phi_a, _, _ = spherical_decomposition(a)
    _, phi_s, _, _ = spherical_decomposition(s)
    assert np.abs(phi_a - phi_s).max() < 1e-4


@pytest.mark.parametrize("dim", [8, 64, 512])
def test_surrogate_preserves_consecutive_angular_steps(dim: int) -> None:
    """Step magnitudes follow from this plus the radial profile."""
    a = _sphere_traj(dim)
    s = manifold_matched_surrogate(a, np.random.default_rng(2))

    def ang_steps(x):
        u = x / np.linalg.norm(x, axis=1, keepdims=True)
        return np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1))

    assert np.abs(ang_steps(a) - ang_steps(s)).max() < 1e-6


def test_surrogate_preserves_step_magnitudes() -> None:
    """The one property the OLD null already had; must not be lost."""
    a = _sphere_traj(256)
    s = manifold_matched_surrogate(a, np.random.default_rng(3))
    da = np.linalg.norm(np.diff(a, axis=0), axis=1)
    ds = np.linalg.norm(np.diff(s, axis=0), axis=1)
    assert np.abs(da - ds).max() / da.mean() < 1e-5


def test_surrogate_is_random() -> None:
    """Two draws must differ, or the 'null distribution' has zero width."""
    a = _sphere_traj(256)
    s1 = manifold_matched_surrogate(a, np.random.default_rng(4))
    s2 = manifold_matched_surrogate(a, np.random.default_rng(5))
    assert np.linalg.norm(s1 - s2, axis=1).mean() > 1e-3 * np.linalg.norm(a, axis=1).mean()


def test_surrogate_is_reproducible_given_a_seed() -> None:
    a = _sphere_traj(64)
    s1 = manifold_matched_surrogate(a, np.random.default_rng(7))
    s2 = manifold_matched_surrogate(a, np.random.default_rng(7))
    assert np.array_equal(s1, s2)


def test_decomposition_pole_is_on_the_manifold() -> None:
    """Unlike the Euclidean centroid, which sits 2.8% inside the sphere."""
    a = _sphere_traj(128)
    pole, _, _, radius = spherical_decomposition(a)
    assert np.linalg.norm(pole) == pytest.approx(1.0, abs=1e-9)
    mean_norm = float(np.linalg.norm(a, axis=1).mean())
    assert np.linalg.norm(radius * pole) == pytest.approx(mean_norm, rel=1e-6)


def test_surrogate_destroys_rotational_coherence() -> None:
    """The property under test MUST be destroyed, or the null is not a null.

    The input is a coherently rotating path; its surrogate should not inherit
    that coherence. Measured by the mean cosine between consecutive azimuth
    directions after removing the (preserved) constrained component.
    """
    a = _sphere_traj(256, rho=0.999)          # near-constant radius: pure rotation
    s = manifold_matched_surrogate(a, np.random.default_rng(11))
    _, _, qa, _ = spherical_decomposition(a)
    _, _, qs, _ = spherical_decomposition(s)
    # third-neighbour azimuth alignment: coherent rotation keeps this high
    coh_a = float(np.mean((qa[:-3] * qa[3:]).sum(1)))
    coh_s = float(np.mean((qs[:-3] * qs[3:]).sum(1)))
    assert coh_s < coh_a, f"surrogate coherence {coh_s} not below real {coh_a}"
