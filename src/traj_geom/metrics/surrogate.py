"""On-manifold surrogate trajectories for null-testing rotation statistics.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25, replacing the null used by every winding test
    in this project (docs/rigor_audit.md section 8).
TASK: generate random trajectories that match the real ones in every respect
    EXCEPT the rotational degrees of freedom, so that a winding excess can be
    attributed to rotation rather than to something else the null failed to
    reproduce.

WHY THE EXISTING NULL IS NOT USABLE
  `metrics.winding.matched_random_walk` preserves per-step displacement
  magnitudes and randomises directions isotropically in ambient R^n. Two
  structures survive in the real data that it destroys:

  (1) THE MANIFOLD. Every SandwichBlock ends in an RMSNorm, so the recurrent
      state satisfies ||h_t|| = 76.37 +- 0.007 (0.0073% variation). The
      surrogate's norm diffuses to 154.76 -- a 5.86% deviation, over twice the
      sphere radius. Comparing the two therefore partly measures on- versus
      off-manifold, not rotation.

  (2) THE RADIAL PROFILE. The real path converges toward a fixed point; the
      surrogate wanders. This matters more than it looks, because RADIAL
      MOTION GENERATES ANGLE: a path falling inward past a center sweeps
      angle for purely radial reasons. That is precisely the mechanism that
      pins the project's winding statistic near 0.62 for every trajectory. A
      null that does not reproduce the convergence cannot separate "rotates"
      from "converges".

THE CONSTRUCTION
  Work intrinsically on the sphere, in coordinates centred on the converged
  state. Let ``p`` be the unit vector toward the fixed point. Any state is

      h_t = R * ( p cos(phi_t) + q_t sin(phi_t) ),    q_t unit, q_t . p = 0

  where ``phi_t`` is the geodesic angle from the fixed point and ``q_t`` is an
  azimuth direction in the (n-1)-dimensional orthogonal complement of p.
  This splits the trajectory exactly into a RADIAL part (the phi_t sequence)
  and a ROTATIONAL part (the q_t sequence).

  The surrogate keeps ``phi_t`` exactly -- so the convergence profile, the
  step magnitudes, the two-regime structure and the noise floor are all
  reproduced -- and re-randomises ``q_t``. Consecutive azimuths are not free:
  preserving the step size fixes their inner product through the spherical law
  of cosines,

      cos(psi_t) = cos(phi_t)cos(phi_t+1) + sin(phi_t)sin(phi_t+1) (q_t . q_t+1)

  so ``q_t . q_{t+1}`` is determined, and the only freedom left is the
  direction of the component orthogonal to q_t -- which is exactly the
  rotational degree of freedom under test. The surrogate samples it uniformly.

  Net effect: geodesic distance to the fixed point, every consecutive angular
  step, and the norm constraint are ALL preserved exactly (to floating point),
  and nothing else is.

WHAT THIS NULL DOES AND DOES NOT TEST
  It tests: given this convergence profile and these step sizes, is the
  angular motion more coherent than chance?  That is H2's question.

  It does NOT control for tangent-space ANISOTROPY. The recurrent Jacobian has
  structure, so the real q_t may concentrate in a low-dimensional subspace;
  this null draws them isotropically, and concentration alone could register
  as excess. A stricter null preserving the empirical covariance of {q_t}
  (phase randomisation, in the manner of IAAFT) would separate "rotates" from
  "moves in a preferred subspace". Not implemented; stated so the limitation
  is not silently inherited.

  Also worth noting: the strongest available null needs no synthesis at all.
  The 140 banked real trajectories already lie on the manifold with every
  structure intact, so a permutation test across prompts tests "does winding
  vary with difficulty" assumption-free. Synthetic surrogates are needed only
  for "is there rotation at all", which permutation cannot address.

I/O: manifold_matched_surrogate(traj, rng) -> ndarray of the same shape.
"""

from __future__ import annotations

import numpy as np

__all__ = ["manifold_matched_surrogate", "spherical_decomposition"]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def spherical_decomposition(
    traj: np.ndarray, n_tail: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Split a trajectory into its radial and rotational parts.

    Args:
        traj: Trajectory of shape [T, hidden_dim], assumed to lie (near) a
            sphere about the origin.
        n_tail: Number of final states averaged to estimate the fixed point.

    Returns:
        ``(pole, phi, q, radius)`` -- the unit vector toward the converged
        state, the geodesic angle of each state from it (shape [T]), the unit
        azimuth directions orthogonal to ``pole`` (shape [T, hidden_dim]), and
        the mean state norm.
    """
    a = np.asarray(traj, dtype=np.float64)
    radius = float(np.linalg.norm(a, axis=1).mean())
    pole = _unit(a[-n_tail:].mean(0))

    proj = a @ pole
    phi = np.arccos(np.clip(proj / (np.linalg.norm(a, axis=1) + 1e-12), -1.0, 1.0))

    perp = a - np.outer(proj, pole)
    norms = np.linalg.norm(perp, axis=1, keepdims=True)
    q = perp / np.where(norms > 0, norms, 1.0)
    return pole, phi, q, radius


def manifold_matched_surrogate(
    traj: np.ndarray,
    rng: np.random.Generator,
    n_tail: int = 20,
    subspace_dim: int | None = None,
) -> np.ndarray:
    """A random trajectory matching ``traj`` in norm, radial profile and step sizes.

    See the module docstring for the construction and for what it controls.

    Args:
        traj: Real trajectory of shape [T, hidden_dim].
        rng: Random generator.
        n_tail: Number of final states averaged to locate the fixed point.
        subspace_dim: If given, confine the surrogate's rotational freedom to a
            random subspace of this dimension. This exists to CONTROL FOR
            ANISOTROPY: winding is measured after a 2-D projection, so a
            trajectory whose motion is confined to few dimensions projects more
            coherently and accumulates more angle for reasons unrelated to
            rotation. Sweeping this parameter measures how much of any observed
            winding excess could be dimensionality rather than rotation. Leave
            ``None`` for the unconstrained (isotropic) null.

    Returns:
        Surrogate trajectory of the same shape. Its state norms, its geodesic
        distances to the fixed point, and its consecutive angular steps all
        match the input; its rotational degrees of freedom are random.
    """
    a = np.asarray(traj, dtype=np.float64)
    pole, phi, q_real, radius = spherical_decomposition(a, n_tail=n_tail)
    n_steps, dim = a.shape

    basis: np.ndarray | None = None
    if subspace_dim is not None:
        # a random orthonormal basis of the requested dimension, orthogonal to
        # the pole (the pole direction is radial, not rotational)
        g = rng.normal(size=(dim, min(subspace_dim, dim - 1)))
        g -= np.outer(pole, pole @ g)
        basis, _ = np.linalg.qr(g)

    # consecutive angular steps of the REAL path, to be preserved
    unit_states = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    cos_psi = np.clip((unit_states[:-1] * unit_states[1:]).sum(1), -1.0, 1.0)

    def _draw(v: np.ndarray | None) -> np.ndarray:
        """Random unit vector orthogonal to ``pole`` (and to ``v`` if given).

        Drawn from ``basis``'s span when a subspace is requested, else from the
        full ambient space.
        """
        for _ in range(8):
            w = (
                basis @ rng.normal(size=basis.shape[1])
                if basis is not None
                else rng.normal(size=dim)
            )
            w -= (w @ pole) * pole
            if v is not None:
                w -= (w @ v) * v
            nw = np.linalg.norm(w)
            if nw > 1e-9:
                return w / nw
        fallback = basis[:, 0] if basis is not None else np.eye(dim)[0]
        return _unit(fallback - (fallback @ pole) * pole)

    q = np.empty_like(q_real)
    q[0] = _draw(None)

    for t in range(n_steps - 1):
        s0, s1 = np.sin(phi[t]), np.sin(phi[t + 1])
        if s0 < 1e-9 or s1 < 1e-9:
            # a state sitting on the pole has no defined azimuth; carry it over
            q[t + 1] = q[t]
            continue
        # spherical law of cosines fixes the azimuth inner product exactly
        c = (cos_psi[t] - np.cos(phi[t]) * np.cos(phi[t + 1])) / (s0 * s1)
        c = float(np.clip(c, -1.0, 1.0))
        w = _draw(q[t])
        q[t + 1] = _unit(c * q[t] + np.sqrt(max(0.0, 1.0 - c * c)) * w)

    return radius * (np.cos(phi)[:, None] * pole[None, :] + np.sin(phi)[:, None] * q)
