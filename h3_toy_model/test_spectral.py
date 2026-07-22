"""Validate spectral_norm_at_point against known-answer linear and nonlinear
maps before trusting it on the TRM model. Run with:
    .venv/bin/python -m pytest src/test_spectral.py -v
"""

from __future__ import annotations

import torch

from spectral import spectral_norm_at_point


def test_identity_map_has_unit_norm() -> None:
    """d(identity)/dx = I everywhere; top singular value is exactly 1."""
    x = torch.randn(16)
    sigma = spectral_norm_at_point(lambda h: h, x, n_iter=20)
    assert abs(sigma - 1.0) < 1e-4


def test_scaling_map_matches_scale_factor() -> None:
    """f(x) = c*x has Jacobian c*I; top singular value is |c|."""
    x = torch.randn(16)
    for c in (0.3, 2.0, 5.0):
        sigma = spectral_norm_at_point(lambda h, c=c: c * h, x, n_iter=30)
        assert abs(sigma - abs(c)) < 1e-3, f"c={c}: got {sigma}"


def test_linear_map_matches_true_top_singular_value() -> None:
    """f(x) = Ax has Jacobian A everywhere; compare against torch.linalg.svd."""
    torch.manual_seed(0)
    dim = 32
    A = torch.randn(dim, dim)
    true_sigma_max = torch.linalg.svdvals(A)[0].item()

    x = torch.randn(dim)
    est = spectral_norm_at_point(lambda h: A @ h, x, n_iter=60)

    rel_err = abs(est - true_sigma_max) / true_sigma_max
    assert rel_err < 0.01, f"true={true_sigma_max:.4f} est={est:.4f} rel_err={rel_err:.4f}"


def test_nonlinear_map_matches_finite_difference_estimate() -> None:
    """For a nonlinear map, cross-check against a brute-force finite-difference
    estimate of the operator norm: max over many random directions of
    ||f(x+eps*v)-f(x)|| / eps, which should undershoot (or match) the power
    -iteration estimate, never wildly exceed it.
    """
    torch.manual_seed(1)
    dim = 24
    W1 = torch.randn(dim, dim) * 0.5
    W2 = torch.randn(dim, dim) * 0.5

    def f(h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(W1 @ h) + W2 @ (h * h)

    x = torch.randn(dim) * 0.3
    est = spectral_norm_at_point(f, x, n_iter=80)

    eps = 1e-4
    fx = f(x)
    best_fd = 0.0
    g = torch.Generator().manual_seed(2)
    for _ in range(500):
        v = torch.randn(dim, generator=g)
        v = v / v.norm()
        fd = (f(x + eps * v) - fx).norm().item() / eps
        best_fd = max(best_fd, fd)

    # Random directions rarely hit the true top singular direction exactly,
    # so the finite-difference max-over-samples should be <= the power
    # -iteration estimate, with some slack for both numerical error.
    assert best_fd <= est * 1.15, f"finite-diff max {best_fd:.4f} exceeds power-iter estimate {est:.4f}"
    assert best_fd >= est * 0.5, f"finite-diff max {best_fd:.4f} suspiciously far below estimate {est:.4f}"


def test_early_stopping_does_not_break_convergence() -> None:
    """A tight tol should still converge close to the true value, just in fewer steps."""
    torch.manual_seed(3)
    dim = 20
    A = torch.randn(dim, dim)
    true_sigma_max = torch.linalg.svdvals(A)[0].item()
    x = torch.randn(dim)
    est = spectral_norm_at_point(lambda h: A @ h, x, n_iter=200, tol=1e-8)
    assert abs(est - true_sigma_max) / true_sigma_max < 0.01
