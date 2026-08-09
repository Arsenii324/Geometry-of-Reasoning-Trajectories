"""Is one unroll a linear map on the states the orbit actually visits?

OWNER: Data+Analysis
STATUS: implemented 2026-08-09.

WHY IT HAS TO BE ASKED. D52 -- this project's best-replicated result, rho going
0.7048 -> 0.8577 across fourteen weight-sets -- is a statement about a linearised
map. So is every Jacobian quantity, and so is D76's identity that the
consecutive-step cosine equals cos(phi). D81 then found that a linear surrogate
built from the measured spectrum reproduces the orbit's dimensional collapse but not
its rotation rate, leaving two possibilities: Arnoldi returned modes the orbit does
not follow, or the dynamics are not linear at the scales recorded. The second would
undercut D52 and everything downstream of it, so it cannot be left as a remark.

THE TEST IS HELD-OUT PREDICTION, WHICH IS THE ONLY VERSION THAT MEANS ANYTHING. For
a linear map the STEPS satisfy delta_{t+1} = A delta_t exactly -- no fixed point
needed, since it cancels in the difference. With ~90 steps in 5280 dimensions, a
least-squares A fits the training steps perfectly by construction, so in-sample
error is guaranteed zero and proves nothing whatever. Everything here is scored on
steps the fit never saw.

THREE MODELS, DELIBERATELY NESTED, because "linear" is not one hypothesis:
    scalar      delta_{t+1} = rho delta_t          (1 parameter: pure contraction)
    linear      delta_{t+1} = A delta_t            (r^2 parameters in a rank-r basis)
    persistence delta_{t+1} = delta_t              (0 parameters, the floor)
If `scalar` already explains the test steps, the orbit is a pure contraction and
rotation is a small correction. If `linear` beats it substantially, rotation is real
and structured. If NEITHER predicts held-out steps, the map is not linear here.
"""

from __future__ import annotations

import numpy as np


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of the held-out step ENERGY explained; 0 for predicting nothing.

    Normalised by the energy of the target rather than by its variance about a mean:
    the target is a displacement, its natural null is the zero vector, and
    subtracting a mean would credit a model for the average step direction.
    """
    num = float(((truth - pred) ** 2).sum())
    den = float((truth**2).sum())
    return float(1.0 - num / den) if den > 0 else float("nan")


def linear_predictability(traj: np.ndarray, lo: int = 0, hi: int | None = None,
                          train_frac: float = 0.6, rank: int | None = None,
                          normalise: bool = True) -> dict:
    """Held-out one-step prediction of the step sequence, three models.

    Args:
        traj: ``[T, d]`` states.
        lo, hi: the window to use, normally the pre-floor one.
        train_frac: leading fraction of the window used to fit.
        rank: basis size for the linear model; defaults to the numerical rank of
            the training steps, capped so the fit stays overdetermined.
        normalise: scale every step to unit norm before fitting. ON BY DEFAULT and
            it matters: step norms decay geometrically by ~0.87 per unroll, so a
            raw least-squares fit is dominated by the first few steps and its
            held-out score is dominated by the last few, which are ~1000x smaller.
            Unnormalised, `r2_linear` would mostly report the decay envelope. With
            normalisation the question is about DIRECTION, which is what every
            shape statistic in this project measures.

    Returns ``r2_linear``, ``r2_scalar``, ``r2_persistence``, the fitted ``rho``,
    the ``rank`` used, and ``n_train`` / ``n_test``.
    """
    x = np.asarray(traj, dtype=np.float64)
    d = np.diff(x, axis=0)[lo:hi]
    n = np.linalg.norm(d, axis=1)
    keep = n > 0
    d, n = d[keep], n[keep]
    if len(d) < 12:
        return {"ok": False, "why": f"only {len(d)} usable steps"}
    u = d / n[:, None] if normalise else d

    a, b = u[:-1], u[1:]                       # predict b from a
    n_tr = int(len(a) * train_frac)
    if n_tr < 6 or len(a) - n_tr < 4:
        return {"ok": False, "why": f"split leaves {n_tr}/{len(a) - n_tr}"}
    a_tr, b_tr, a_te, b_te = a[:n_tr], b[:n_tr], a[n_tr:], b[n_tr:]

    # Work in the training steps' own basis: the ambient 5280 dimensions are
    # irrelevant to a map the orbit only ever exercises on a ~n_tr-dimensional
    # subspace, and fitting in the full space would be rank-deficient by definition.
    q, s, _ = np.linalg.svd(a_tr.T, full_matrices=False)
    r = rank if rank is not None else int(
        min((s > s[0] * 1e-8).sum(), max(2, n_tr // 3)))
    # A requested rank above what the training steps actually span is silently
    # unavailable, not an error: `q` has only min(n_tr, d) columns, and asking for
    # more produced a shape mismatch rather than a clear refusal.
    r = int(min(r, q.shape[1]))
    q = q[:, :r]
    at, bt = a_tr @ q, b_tr @ q
    # Ridge, not a bare pseudo-inverse: with r comparable to n_tr the normal
    # equations are near-singular and an unregularised solve returns an operator
    # that predicts the training steps and nothing else.
    lam = 1e-6 * float(np.trace(at.T @ at)) / max(1, r)
    mat = np.linalg.solve(at.T @ at + lam * np.eye(r), at.T @ bt)

    pred_lin = (a_te @ q) @ mat @ q.T
    rho = float((a_tr * b_tr).sum() / max(1e-30, (a_tr * a_tr).sum()))
    return {"ok": True,
            "r2_linear": _r2(b_te, pred_lin),
            "r2_scalar": _r2(b_te, rho * a_te),
            # The like-for-like baseline. `r2_scalar` above is NOT confined to the
            # fitted subspace, so at small r it can beat `r2_linear` simply by being
            # allowed to point anywhere -- which says the projection costs something,
            # not that a scalar describes the map better. Comparing the two
            # SUBSPACE-RESTRICTED models isolates the question actually being asked:
            # does a general operator beat a single contraction rate?
            "r2_scalar_in_basis": _r2(b_te, rho * (a_te @ q) @ q.T),
            "r2_persistence": _r2(b_te, a_te),
            "rho": rho, "rank": int(r),
            "rank_requested": int(rank) if rank is not None else int(r),
            "n_train": int(n_tr),
            "n_test": int(len(a_te)), "normalised": bool(normalise),
            "in_basis": _r2(b_te, (b_te @ q) @ q.T)}


def rank_sweep(traj: np.ndarray, lo: int = 0, hi: int | None = None,
               ranks: tuple[int, ...] = (2, 4, 8, 16, 32)) -> list[dict]:
    """Held-out score against basis size -- the shape of the curve is the finding.

    A linear map exercised over r directions should improve up to r and then flatten.
    A score that keeps climbing with rank, or peaks and then FALLS, says the extra
    directions are being fitted to noise, which is what D74(4) already observed for
    DMD: the snapshot spectrum had no gap and the leading modulus wandered 0.46-0.85
    with the truncation rank, failing its stability gate on 80/80 orbits.
    """
    out = []
    for r in ranks:
        res = linear_predictability(traj, lo=lo, hi=hi, rank=r)
        if res.get("ok"):
            out.append({"rank": r, **{k: res[k] for k in
                                      ("r2_linear", "r2_scalar", "in_basis")}})
    return out
