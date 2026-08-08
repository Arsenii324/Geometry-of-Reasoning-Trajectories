"""Dynamic mode decomposition of a latent trajectory: the operator, from the orbit.

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

WHY THIS EXISTS. This project has two ways of getting the recurrent map's spectrum
and they disagree by an amount nobody could explain (UNDERSTANDING section 5, "the
0.058 estimator gap"): Arnoldi on Jacobian-vector products gives |lambda| ~ 0.80
(D31, D55), two-orbit separation gives rho ~ 0.887 (D52, D59). Three candidate
explanations were tested and all three were REFUTED, so the gap stood open.

DMD is a third, independent route, and it is the one that asks the question the
other two do not: **what operator does the ORBIT actually obey?** Arnoldi
linearises at the fixed point and returns asymptotic eigenvalues. Two-orbit
separation measures how fast two trajectories converge. DMD fits the best linear
operator to the observed sequence over the window the model actually runs in.

THE FIXED POINT DROPS OUT, WHICH REMOVES A KNOWN TRAP. If h_{t+1} - h* = A(h_t - h*)
then consecutive DIFFERENCES obey the same map exactly:

    delta_t = h_{t+1} - h_t   =>   delta_{t+1} = A delta_t

so fitting on differences needs no estimate of h* at all. That matters here because
`metrics/regime.py` records that measuring a single trajectory's approach to its own
endpoint is near-tautological for any convergent sequence -- a mistake this project
already made once. Differences avoid taking h_end as a stand-in for h*.

WHAT IT CANNOT DO. DMD recovers only the modes the orbit EXCITES. A mode the
trajectory never explores is invisible to it, exactly as a mode Arnoldi's Krylov
subspace never reaches is invisible to Arnoldi. So DMD and Arnoldi disagreeing does
not by itself say which is wrong; it says the orbit is governed by something other
than what Arnoldi returned, which is the scientifically relevant statement for a
model that is run for 32-64 unrolls rather than to convergence.
"""

from __future__ import annotations

import numpy as np


def dmd_eigenvalues(
    traj: np.ndarray, rank: int | None = None, max_rank: int = 24,
    on_differences: bool = True,
) -> dict:
    """Leading DMD eigenvalues of the map that generated ``traj``.

    RANK IS NOT CHOSEN BY ENERGY, and that is the load-bearing design decision.
    An earlier draft kept the smallest rank capturing 99.9% of the squared
    singular-value mass. On a deliberately non-normal test operator
    ``[[0.8, 20], [0, 0.8]]`` that rule selects rank 1 -- the second singular value
    carries 0.008% of the energy -- and returns **rho = 0.62 against a true 0.80**,
    with no sign anything is wrong. Non-normality is exactly what concentrates
    energy into one direction while the dynamics still needs two, and non-normality
    is the live hypothesis for the discrepancy this estimator exists to
    investigate. An energy rule would therefore have failed hardest on the case
    that matters. ``tests/test_dmd.py`` pins that number so the rule cannot come
    back.

    The default keeps every numerically meaningful direction instead, and
    `dmd_rank_sweep` is the companion that makes the remaining rank-sensitivity
    visible rather than hidden in a default.

    Args:
        traj: ``[T, d]`` states in order.
        rank: SVD truncation rank. ``None`` keeps all directions above a relative
            tolerance, capped at ``max_rank``.
        max_rank: Hard cap. At d >> T the trailing directions are arithmetic noise
            and keeping them lets the fitted operator interpolate.
        on_differences: Fit on consecutive differences (default) so the fixed point
            cancels. Set False only for tests on already-centred series.

    Returns:
        dict with ``eigs`` (complex, sorted by decreasing modulus), ``rho`` (leading
        modulus), ``arg`` (argument of the leading OSCILLATORY mode, or None if the
        spectrum is entirely real), ``rank``, ``n_snapshots`` and ``sv_energy``.
    """
    x = np.asarray(traj, dtype=np.float64)
    if on_differences:
        x = np.diff(x, axis=0)
    if len(x) < 3:
        raise ValueError(f"need >= 3 snapshots, got {len(x)}")
    a, b = x[:-1].T, x[1:].T                      # [d, m] each

    u, s, vt = np.linalg.svd(a, full_matrices=False)
    numerical = int(np.sum(s > s[0] * 1e-10))
    rank = numerical if rank is None else int(rank)
    rank = int(max(1, min(rank, max_rank, len(s), numerical)))

    ur, sr, vr = u[:, :rank], s[:rank], vt[:rank].T
    # A_tilde is A projected onto the leading left-singular subspace; its
    # eigenvalues are DMD eigenvalues of the full operator restricted there.
    a_tilde = ur.T @ b @ vr @ np.diag(1.0 / sr)
    vals = np.linalg.eigvals(a_tilde)
    vals = vals[np.argsort(-np.abs(vals))]
    osc = [z for z in vals if abs(z.imag) > 1e-12]
    return {
        "eigs": vals,
        "rho": float(abs(vals[0])),
        "arg": (float(abs(np.angle(osc[0]))) if osc else None),
        "rho_osc": (float(abs(osc[0])) if osc else None),
        "rank": rank,
        "n_snapshots": int(a.shape[1]),
        "sv_energy": float((s[:rank] ** 2).sum() / max(float((s**2).sum()), 1e-300)),
    }


def dmd_rank_sweep(traj: np.ndarray, ranks=(2, 3, 4, 6, 8, 12, 16),
                   tol: float = 0.02) -> dict:
    """Leading modulus as a function of truncation rank, plus a stability gate.

    DMD's answer depends on where the SVD is truncated, and no default is right for
    every orbit. Rather than pick one and hope, report the whole curve and gate on
    whether it is flat: if the leading modulus moves by less than ``tol`` across the
    upper half of the sweep, the estimate is a property of the data; if it drifts,
    it is a property of the truncation and must be reported as such.

    This is the same discipline as `attainable` and `has_dynamic_range` in the
    kernel library -- ask whether the measurement CAN mean what it appears to,
    before reading it.

    Returns ``{"ranks", "rho", "arg", "stable", "spread"}``.
    """
    rs, rhos, args = [], [], []
    for r in ranks:
        try:
            out = dmd_eigenvalues(traj, rank=r)
        except ValueError:
            continue
        if out["rank"] != r:           # capped out; no new information beyond here
            continue
        rs.append(r)
        rhos.append(out["rho"])
        args.append(out["arg"])
    if not rs:
        return {"ranks": [], "rho": [], "arg": [], "stable": False, "spread": None}
    upper = rhos[len(rhos) // 2:]
    spread = float(max(upper) - min(upper)) if upper else float("inf")
    return {"ranks": rs, "rho": rhos, "arg": args,
            "stable": bool(spread < tol), "spread": spread}


def pre_floor_window(traj: np.ndarray, floor_frac: float = 2.0,
                     tail: int = 10) -> tuple[int, int]:
    """Indices ``[a, b]`` of the regime above the arithmetic noise floor.

    Huginn's states are stored bf16-exact (D30), so once the step size reaches the
    rounding floor every statistic computed over that stretch describes the
    arithmetic and not the model -- which is exactly how winding came to flip sign
    with the recording budget (D28). The floor is estimated from the tail of the
    step-norm sequence rather than assumed.
    """
    d = np.linalg.norm(np.diff(np.asarray(traj, dtype=np.float64), axis=0), axis=1)
    if len(d) < tail + 2:
        return 0, len(d)
    # MAX of the tail, not the median: step norms in the floor regime are
    # chi-distributed, so their max runs 2-3x their median and a median-based
    # threshold lets isolated noise steps read as signal. Measured: a synthetic
    # floor appended to a clean spiral extended the window 12 steps past the end
    # of the signal under the median rule.
    floor = float(np.max(d[-tail:]))
    keep = np.where(d > floor_frac * floor)[0]
    if len(keep) < 3:
        return 0, len(d)
    return int(keep[0]), int(keep[-1]) + 1
