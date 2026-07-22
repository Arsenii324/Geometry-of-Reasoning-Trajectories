"""Power-iteration estimator of the local Jacobian operator norm (top singular
value) of a map h -> f(h), for empirically testing the Contraction Bottleneck
Theorem's premise (rho(d_h R) < 1) on a model we can actually run (CPU-only;
no GPU/Huginn access on this machine).

STATUS: new 2026-07-18, self-tested against known-answer linear maps in
test_spectral.py before being trusted on the TRM model in h3_validation.py.

Why operator norm (top singular value), not spectral radius (top eigenvalue
magnitude): the Contraction Bottleneck proof's Lipschitz condition
||R(x)-R(y)|| <= c||x-y|| is a statement about the operator norm of the local
Jacobian, sup_v ||Jv||/||v||, not its eigenvalues. For a non-normal Jacobian
(the generic case for a learned nonlinear map) the two differ, sometimes by a
lot, so estimating eigenvalues (as a naive single-direction power iteration
on J alone would) is the wrong quantity here.

The right classical tool is power iteration on J^T J (equivalently, alternate
one JVP and one VJP through the same function): its dominant eigenvalue is
sigma_max(J)^2. This is the same method spectral normalization (Miyato et al.,
2018) uses for weight matrices; here it's applied to the *local* Jacobian of a
nonlinear map at a specific point instead of a fixed linear layer.
"""

from __future__ import annotations

from typing import Callable

import torch


def spectral_norm_at_point(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_iter: int = 30,
    tol: float = 1e-6,
    generator: torch.Generator | None = None,
) -> float:
    """Estimate the operator norm (top singular value) of d(func)/dx at x.

    Args:
        func: A function taking and returning a single tensor of shape [dim]
            (unbatched -- call once per point of interest).
        x: The point at which to linearize func, shape [dim]. Does not need
            requires_grad set; this function handles that internally.
        n_iter: Maximum power-iteration steps.
        tol: Stop early if the singular-value estimate changes by less than
            this between iterations.
        generator: Optional torch.Generator for a reproducible random start
            vector.

    Returns:
        The estimated top singular value (a nonnegative float). This is a
        lower bound on the true value that becomes tight as n_iter grows,
        provided the top singular value is non-degenerate (no exact tie for
        first place) -- standard power-iteration behavior.
    """
    x = x.detach().clone().requires_grad_(False)
    dim = x.shape[-1]
    v = torch.randn(dim, generator=generator) if generator is not None else torch.randn(dim)
    v = v / v.norm()

    prev_sigma = None
    for _ in range(n_iter):
        _, jv = torch.autograd.functional.jvp(func, x, v, create_graph=False)
        _, jtjv = torch.autograd.functional.vjp(func, x, v=jv, create_graph=False)
        norm = jtjv.norm()
        if norm < 1e-12:
            return 0.0
        v = jtjv / norm
        sigma = norm.sqrt().item()  # ||J^T J v|| ~ sigma^2 ||v|| after convergence -> sigma = sqrt(norm)
        if prev_sigma is not None and abs(sigma - prev_sigma) < tol:
            prev_sigma = sigma
            break
        prev_sigma = sigma

    # Final, more accurate estimate via a direct Rayleigh quotient at the converged v.
    _, jv = torch.autograd.functional.jvp(func, x, v, create_graph=False)
    return float(jv.norm().item() / (v.norm().item() + 1e-12))
