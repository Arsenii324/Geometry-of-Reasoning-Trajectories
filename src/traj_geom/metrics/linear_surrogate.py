"""Can a linear contraction with the measured spectrum reproduce the real orbit?

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

THE HYPOTHESIS THIS MAKES TESTABLE. Near a fixed point h*, one unroll of Huginn's
weight-tied core block is approximately a linear map A, and the orbit is

    h_t - h* ~ sum_i c_i lambda_i^t v_i

with the eigenpairs (lambda_i, v_i) set by the WEIGHTS and the coefficients c_i set
by where the orbit started. D78 established that h_0 is `randn_like(input_embeds)`
from an unseeded generator, so those coefficients are random. If the decomposition
holds, every shape statistic this project measures is a functional of the spectrum
plus initial-condition noise, and NONE of them can carry information about the
computation beyond what the spectrum carries.

THAT IS A SUFFICIENCY CLAIM, WHICH IS WHY IT IS WORTH MORE THAN ANOTHER NULL. A
surrogate built from nothing but the measured eigenvalues and a random start knows
no prompt, no task and no answer. If its geometry matches the real orbit's, the real
orbit's geometry demonstrably contains nothing else.

WHY A SINGLE EIGENPLANE WAS THE WRONG TEST, AND THIS IS NOT. `geom-eigenplane`
projected the orbit onto the LEADING eigenpair's plane and asked whether it rotates
at `arg(lambda_1)` radians per unroll. It rotates at 0.03-1.27 times that rate
(median ~0.24, D74(4)). That is not evidence against linearity -- it is what a
mixture of ~13 comparable modes looks like when you watch one of them. This module
keeps every mode Arnoldi returned and asks about statistics that are defined on the
whole path rather than on one plane.

THE TRUNCATION IS REAL AND IS REPORTED, NOT HIDDEN. Arnoldi returns the dominant
`n_eigs` eigenvalues of a 5280-dimensional operator, and D74 measured ~13 active
directions, so a surrogate built from 8 eigenvalues is missing modes by
construction. `spectrum_tail` extends a measured spectrum with a geometric
continuation so the sensitivity to that truncation can be measured instead of
assumed.
"""

from __future__ import annotations

import numpy as np


def surrogate_orbit(eigs: np.ndarray, n_steps: int, dim: int, seed: int = 0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """A real orbit of a linear map with exactly this spectrum, from a random start.

    Args:
        eigs: complex eigenvalues. Conjugate pairs are consumed as pairs and give a
            rotating 2-plane; real eigenvalues give a decaying line. A pair listed
            twice (as Arnoldi returns it) contributes ONE plane, not two.
        n_steps: unrolls to generate.
        dim: ambient dimension; the invariant subspaces are placed in a random
            orthonormal frame, because the real eigenvectors are not available here
            and no statistic used downstream depends on their orientation.
        seed / rng: the random start. This is the ONLY input carrying no weight
            information, and it stands in for h_0.

    Returns ``[n_steps, dim]``, centred so that h* is the origin.

    The frame is orthonormal on purpose. Real eigenvectors of a non-normal operator
    are NOT orthogonal, and that non-normality is exactly what makes transient
    growth possible -- so this surrogate is the NORMAL approximation to the map, and
    a mismatch against the real orbit is evidence of non-normality rather than of
    nonlinearity. Stated here because the two are easy to confuse.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    lam = np.asarray(eigs, dtype=np.complex128)
    frame = np.linalg.qr(rng.normal(size=(dim, min(dim, 2 * len(lam)))))[0]
    out = np.zeros((n_steps, dim), dtype=np.float64)
    t = np.arange(n_steps, dtype=np.float64)
    col, used = 0, np.zeros(len(lam), dtype=bool)
    for i, z in enumerate(lam):
        if used[i]:
            continue
        used[i] = True
        r, phi = float(abs(z)), float(np.angle(z))
        if abs(z.imag) < 1e-12:                    # real mode: a decaying line
            if col >= frame.shape[1]:
                break
            out += np.outer(rng.normal() * r**t, frame[:, col])
            col += 1
            continue
        # consume the conjugate partner so the pair yields ONE rotating plane
        for j in range(i + 1, len(lam)):
            if not used[j] and abs(lam[j] - np.conj(z)) < 1e-9 * max(1.0, abs(z)):
                used[j] = True
                break
        if col + 1 >= frame.shape[1]:
            break
        a, psi = rng.normal(), rng.uniform(0, 2 * np.pi)
        env = a * r**t
        out += np.outer(env * np.cos(phi * t + psi), frame[:, col])
        out += np.outer(env * np.sin(phi * t + psi), frame[:, col + 1])
        col += 2
    return out


def spectrum_tail(eigs: np.ndarray, n_extra: int, decay: float = 0.85,
                  seed: int = 0) -> np.ndarray:
    """Extend a truncated spectrum with a geometric continuation.

    Arnoldi returns the dominant modes only, so a surrogate built from them alone is
    missing whatever sits below the cut. This appends `n_extra` modes whose moduli
    continue the measured tail geometrically and whose arguments are drawn uniformly,
    which is the least-informative extension consistent with the part that was
    measured. Its purpose is to show whether a conclusion SURVIVES the truncation,
    not to claim these modes exist.
    """
    lam = np.asarray(eigs, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    base = float(np.min(np.abs(lam))) if len(lam) else 0.5
    out = []
    for k in range(1, n_extra + 1):
        r = base * decay**k
        phi = rng.uniform(0, np.pi)
        out += [r * np.exp(1j * phi), r * np.exp(-1j * phi)]
    return np.concatenate([lam, np.array(out[:n_extra], dtype=np.complex128)])


def surrogate_statistics(eigs: np.ndarray, n_steps: int, dim: int, n_draw: int = 32,
                         seed: int = 0, lo: int = 0,
                         hi: int | None = None) -> dict:
    """Distribution of the shape statistics over independent random starts.

    The spread across draws IS the prediction's uncertainty: with the spectrum held
    fixed, everything that moves is the initial condition, which is precisely the
    quantity D78 says the real model also redraws on every forward. So a real orbit
    landing inside this spread is landing where the linear picture says it should.
    """
    from traj_geom.metrics.dimension import participation_ratio, step_directions

    cos_v, pr_v, rho_v = [], [], []
    for k in range(n_draw):
        traj = surrogate_orbit(eigs, n_steps, dim, seed=seed + k)
        u = step_directions(traj, lo=lo, hi=hi)
        if len(u) < 3:
            continue
        cos_v.append(float(np.mean(np.sum(u[:-1] * u[1:], axis=1))))
        pr_v.append(participation_ratio(u))
        d = np.linalg.norm(np.diff(traj, axis=0), axis=1)[lo:hi]
        good = d > 0
        if good.sum() >= 4:
            y = np.log(d[good])
            rho_v.append(float(np.exp(np.polyfit(
                np.arange(len(y), dtype=float), y, 1)[0])))
    if not cos_v:
        return {"ok": False, "why": "surrogate produced too few usable steps"}

    def band(v):
        return (float(np.median(v)), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

    cos_m, cos_lo, cos_hi = band(cos_v)
    pr_m, pr_lo, pr_hi = band(pr_v)
    rho_m, rho_lo, rho_hi = band(rho_v) if rho_v else (float("nan"),) * 3
    return {"ok": True, "n_draw": len(cos_v),
            "cos_consecutive": cos_m, "cos_lo": cos_lo, "cos_hi": cos_hi,
            "pr": pr_m, "pr_lo": pr_lo, "pr_hi": pr_hi,
            "contraction": rho_m, "contraction_lo": rho_lo,
            "contraction_hi": rho_hi}
