"""Observable convergence in contracting iterative models: law, estimator, tests.

OWNER: Data+Analysis
STATUS: derived and validated 2026-07-28.

THE LAW
    Let an iterative model have state h_r converging to a fixed point h* at
    contraction rate rho, so ||h_r - h*|| ~ C rho^r. Let O be any observable
    that is smooth at h*. Then

        |O(h_r) - O(h*)|  <=  ||grad O|| * C rho^r  +  O(rho^{2r})              (1)

    so EVERY smooth observable converges to its limiting value at rate rho.
    For observables that are QUADRATIC in the state error -- which includes the
    two that matter in practice -- the gap closes at rho^{2r}:

        feature DIRECTION  v_r = g(h_r)/||g(h_r)||
            the angle to v_inf is eps_r ~ rho^r, and since
            1 - cos(eps) ~ eps^2/2,
                1 - |cos(v_r, v_inf)|  ~  C1 rho^{2r}                           (2)

        probe PERFORMANCE  R^2_r of any fixed decoder
            prediction error is linear in the state error, R^2 gap is its
            square,
                R^2_inf - R^2_r  ~  C2 rho^{2r}                                 (3)

    Both give a straight line in log against r with slope 2 ln rho, so rho is
    recoverable from either curve WITHOUT touching the operator:

        rho_hat = exp( slope / 2 )                                              (4)

WHY THIS MATTERS, AND WHAT IT INVALIDATES
    A feature direction extracted from an iterative model at unroll r is not
    the model's feature direction; it is a biased estimate whose bias decays as
    rho^r. On the model measured here that bias is severe at small r --
    |cos(v_1, v_inf)| = 0.45, so a direction read off after one unroll is only
    45% aligned with the converged one.

    The sharper consequence is methodological. A feature whose decodability
    RISES with depth is routinely read as the model "building" or "developing"
    that feature. Equation (3) says a rising curve is the DEFAULT for any fixed
    quantity already present in the state, because the observable is still
    converging. Rising decodability is therefore NOT evidence of computation
    over depth. Distinguishing the two requires checking whether the rate
    departs from rho -- which is exactly what this module's estimator provides.

    Empirically that distinction is not academic: on Huginn the per-position
    counting register's decodability is FLAT in r (rho_obs +0.006, p=0.958)
    while its measured DIRECTION rotates at exactly rho, and the answer-token
    readout curve rises at exactly rho. Same model, same unrolls: one quantity
    genuinely does not change, and two observables of it converge at the
    contraction rate. Reading either curve as "the feature develops" would have
    been wrong.

VALIDATION, AND A CORRECTION FROM REAL DATA (2026-08-04)
    On synthetic contracting systems with a planted feature direction and a
    KNOWN rho, `estimate_rho_from_curve` recovers rho to ~1.3% mean relative
    error over rho in 0.70..0.90 (see tests). It degrades as rho -> 1, because
    a fixed unroll budget no longer reaches convergence; `fit_quality` exposes
    that rather than hiding it.

    THAT SYNTHETIC FIGURE DOES NOT TRANSFER. Huginn's rho was later measured
    DIRECTLY on the operator, by two-orbit convergence and by step-norm decay
    (claims_ledger D44). Against those measurements this estimator is off by
    3-7%, not 1.3%, and the error is SYSTEMATIC IN DIRECTION:

        untrained model    inferred 0.6618   direct 0.7150    7.4% LOW
        trained model      inferred 0.9155   direct 0.8866    3.3% HIGH

    It is biased OUTWARD at both ends, so it exaggerates any contrast it is used
    to measure -- here by 1.5x (inferred gap 0.254 against a direct gap 0.172).
    Use it to establish the SIGN and rough scale of a difference in contraction
    rate; do not use it for a number that a conclusion has to bear weight on.
    A claim that flips under a 3-7% change in rho is not safe on this estimator,
    which is exactly how claims_ledger D43 came to be retracted.

    On Huginn, four independent observable-based estimates agree:
        feature rotation, running count       0.868
        feature rotation, nesting depth       0.861
        readout curve, run 1                  0.905
        readout curve, run 2                  0.910
    against two direct measurements of the same quantity obtained by unrelated
    methods: two-orbit convergence 0.85-0.90 (claims_ledger D24(7)) and Arnoldi
    on Jacobian-vector products 0.79-0.81 (D31, measured on the diagonal block
    J_TT, which D31 notes is not the operator the orbit obeys).

SCOPE AND LIMITS, STATED
    * Requires an attracting fixed point. Huginn contracts (rho < 1, verified
      two ways); a non-contracting or chaotic iterate is out of scope.
    * Requires O smooth at h*. A thresholded or argmax observable is not, and
      equation (1) says nothing about it.
    * (2) and (3) assume the state error is the ONLY r-dependence. If the
      quantity itself changes with r -- genuine computation -- the curve
      departs from rho^{2r}, which is the intended signal, not a failure.
    * Finite arithmetic puts a floor under any observable gap; `fit_curve`
      takes an explicit floor argument because ignoring it biases the slope.

I/O: pure functions on curves; no model or GPU needed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fit_curve",
    "estimate_rho_from_curve",
    "estimate_rho_from_directions",
    "predict_curve",
]


def fit_curve(
    unrolls: np.ndarray, gap: np.ndarray, floor: float | None = None, min_gap: float = 1e-4
) -> tuple[float, float, float]:
    """Fit ``log(gap - floor) = slope * r + intercept``.

    Args:
        unrolls: Unroll counts r, strictly positive.
        gap: The observable's distance from its limit at each r (non-negative).
        floor: Saturation level to subtract. If None, estimated as the mean of
            the last three points -- ignoring a floor biases the slope toward
            zero, which would understate rho.
        min_gap: Points with ``gap - floor`` below this are dropped as
            unresolvable.

    Returns:
        ``(slope, intercept, r_squared)``. ``r_squared`` is of the log-linear
        fit and is the diagnostic for whether the law applies at all.

    Raises:
        ValueError: if fewer than three usable points remain.
    """
    r = np.asarray(unrolls, dtype=float)
    g = np.asarray(gap, dtype=float)
    if floor is None:
        floor = float(np.mean(g[-3:])) if len(g) >= 3 else 0.0
    y = g - floor
    keep = y > min_gap
    if keep.sum() < 3:
        raise ValueError(
            f"only {int(keep.sum())} points above the floor; the curve has "
            "saturated and the slope cannot be identified"
        )
    slope, intercept = np.polyfit(r[keep], np.log(y[keep]), 1)
    pred = slope * r[keep] + intercept
    obs = np.log(y[keep])
    denom = ((obs - obs.mean()) ** 2).sum()
    r2 = float(1 - ((obs - pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    return float(slope), float(intercept), r2


def estimate_rho_from_curve(
    unrolls: np.ndarray, gap: np.ndarray, floor: float | None = None, quadratic: bool = True
) -> tuple[float, float]:
    """Recover the contraction rate from an observable's convergence curve.

    This is the practical payload: rho without the Jacobian, from a curve
    anyone doing interpretability on an iterative model already has.

    Args:
        unrolls: Unroll counts.
        gap: Distance from the limit at each unroll.
        floor: Saturation level; see `fit_curve`.
        quadratic: True if the observable is quadratic in the state error --
            which holds for ``1 - cos`` between directions and for an R^2 gap,
            the two cases in the module docstring. False for an observable
            linear in the error.

    Returns:
        ``(rho_hat, fit_r_squared)``. Treat a fit R^2 below ~0.9 as the law not
        applying rather than as a noisy estimate.
    """
    slope, _, r2 = fit_curve(unrolls, gap, floor=floor)
    return float(np.exp(slope / (2.0 if quadratic else 1.0))), r2


def estimate_rho_from_directions(
    unrolls: np.ndarray, directions: np.ndarray, reference: np.ndarray | None = None
) -> tuple[float, float]:
    """Recover rho from a sequence of feature directions estimated at each unroll.

    Args:
        unrolls: Unroll counts, one per direction.
        directions: Array [n_unrolls, dim]; each row need not be normalised.
        reference: The limiting direction. Defaults to the last row, which
            makes the final point degenerate -- it is dropped automatically.

    Returns:
        ``(rho_hat, fit_r_squared)``.
    """
    v = np.asarray(directions, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    ref = v[-1] if reference is None else np.asarray(reference, dtype=float)
    ref = ref / np.linalg.norm(ref)
    cos = np.abs(v @ ref)
    r = np.asarray(unrolls, dtype=float)
    keep = cos < 1 - 1e-9
    return estimate_rho_from_curve(r[keep], 1.0 - cos[keep], floor=0.0, quadratic=True)


def predict_curve(unrolls: np.ndarray, rho: float, gap_at_first: float,
                  floor: float = 0.0, quadratic: bool = True) -> np.ndarray:
    """The curve the law predicts, given rho measured some OTHER way.

    Used to check a measured curve against an independent rho rather than
    fitting rho to the curve itself -- the difference between a prediction and
    a description.
    """
    r = np.asarray(unrolls, dtype=float)
    power = 2.0 if quadratic else 1.0
    return floor + gap_at_first * rho ** (power * (r - r[0]))
