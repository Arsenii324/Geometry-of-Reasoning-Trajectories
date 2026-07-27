"""Separate a trajectory's signal regime from its arithmetic-noise regime.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25, in response to docs/rigor_audit.md sections
    1, 2, 3 and 6, which found the same error in four different metrics.
TASK: given a trajectory, decide which steps carry model dynamics and which
    carry only floating-point rounding, and provide floor-aware replacements
    for the statistics that were averaging over both.

WHY THIS EXISTS
  Huginn is run in bfloat16 (`extraction/model.py` defaults to
  `torch.bfloat16`; verified in the saved data -- every value in
  `results/trajectories/*.npy` is exactly bf16-representable, bit-identical
  under a bfloat16 round trip). bf16 carries 8 mantissa bits, so relative
  precision is 2^-8 ~ 3.9e-3 rather than float32's 1.2e-7.

  A trajectory therefore has two regimes:
    * t < ~15   the state converges; step norms fall by ~70x. Real dynamics.
    * t > ~15   the state sits at a floor set by rounding noise under
                contraction. The stationary RMS is eta/sqrt(1-rho^2) -- isotropic
                noise adds in QUADRATURE (an earlier version of this docstring
                used eta/(1-rho), which sums aligned displacements and is wrong;
                simulation at rho=0.871 gives 0.171 against the correct formula's
                0.175). That predicts 0.175 against ~0.91 observed, a factor of
                5, so the floor is set by the SLOWEST-contracting modes, not the
                aggregate rate. The reading "this is noise" therefore rests on
                ISOTROPY, not magnitude: tail PCA top-2 explains only 8.8-10.7%
                with participation ratio 18-19/20, where genuine 2-D dynamics
                would give ~2.

  Metrics that average over "the whole path" or "the second half" are then
  dominated by arithmetic. Three concrete casualties, all reproduced in the
  audit:
    * `convergence.consecutive_step_cosine` averages the second half and
      reports -0.276. Restricted to the converging regime it is +0.084 --
      the path glides; it does not zig-zag. (White-noise increments have
      lag-1 autocorrelation exactly -0.5, which is what the second half is
      approaching.)
    * `convergence.path_independence` fits a log-slope through the floor and
      understates contraction by ~0.084/step (x0.9244 vs x0.8407 true).
    * `dynamics.steps_to_settle` thresholds at 10% of the MAXIMUM step, which
      is always the h_0 transient, making it a function of rho alone:
      log(0.1)/log(0.8407) = 13.27 predicted vs 13.84 observed.

THE CHOICE MADE HERE, AND WHY
  The floor is located EMPIRICALLY (median step norm over the final quarter),
  not from the bf16 arithmetic prediction. The prediction is exposed
  separately by `predicted_arithmetic_floor` for cross-checking, but it is not
  used to make decisions, because it needs rho -- which is what we are trying
  to measure -- and because it assumes rounding is the only noise source. The
  empirical floor makes no such assumption and degrades gracefully if a run is
  done in float32, or if a genuine (non-noise) limit set turns out to exist.

  These are ADDITIONS, not edits to the existing metrics. The existing
  functions stay bit-compatible so every cached CSV remains reproducible; the
  audit documents their bias instead of silently changing what past results
  mean.

I/O: all functions take traj [T, hidden] (or two such) and return floats/ints.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "predicted_arithmetic_floor",
    "empirical_floor",
    "converging_regime",
    "contraction_from_pair",
    "step_cosine_converging",
    "classify_shape_regime",
    "h1_persistence_signal",
]

_BF16_MANTISSA_BITS = 8


def predicted_arithmetic_floor(traj: np.ndarray, mantissa_bits: int = _BF16_MANTISSA_BITS) -> float:
    """Norm of a single rounding-error vector, predicted from the dtype alone.

    For a state of norm R in ``n`` dimensions the per-coordinate RMS is
    ``R/sqrt(n)``; one ulp at that magnitude is ``rms * 2**-mantissa_bits``;
    rounding error is uniform on +-ulp/2, so its RMS is ``ulp/sqrt(12)``, and
    the error vector's norm is ``sqrt(n)`` times that.

    This is the PER-STEP injection, not the equilibrium floor. In a system
    contracting at rate rho the accumulated floor is this divided by
    ``(1 - rho)``.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        mantissa_bits: Mantissa bits of the compute dtype (8 for bfloat16,
            23 for float32).

    Returns:
        Predicted norm of one rounding-noise vector.
    """
    a = np.asarray(traj, dtype=np.float64)
    n = a.shape[1]
    rms = float(np.linalg.norm(a, axis=1).mean()) / np.sqrt(n)
    ulp = rms * 2.0**-mantissa_bits
    return float(np.sqrt(n) * ulp / np.sqrt(12.0))


def empirical_floor(traj: np.ndarray, tail_frac: float = 0.25) -> float:
    """Median step norm over the final ``tail_frac`` of the path.

    Median rather than mean: it is insensitive to a few large steps if the
    path has not fully converged, so a not-yet-settled trajectory yields a
    large floor and is correctly reported as having little or no noise regime
    rather than a spuriously clean one.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        tail_frac: Fraction of the path treated as the tail.

    Returns:
        The empirical noise-floor step size, or 0.0 if the path is too short.
    """
    s = np.linalg.norm(np.diff(np.asarray(traj, dtype=np.float64), axis=0), axis=1)
    if len(s) < 4:
        return 0.0
    k = max(1, int(len(s) * tail_frac))
    return float(np.median(s[-k:]))


def converging_regime(traj: np.ndarray, k: float = 3.0, tail_frac: float = 0.25) -> tuple[int, int]:
    """Index range ``[0, end)`` of steps whose motion exceeds the noise floor.

    ``end`` is the first step at which the step norm drops below ``k`` times
    the empirical floor and stays there. Requiring it to STAY below matters: a
    single lucky small step early on must not truncate the regime.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        k: Multiple of the floor a step must exceed to count as signal.
        tail_frac: Passed through to `empirical_floor`.

    Returns:
        ``(0, end)``. ``end`` is the number of *steps* (so state indices
        ``0..end`` inclusive are in the regime). ``end`` is at least 2 so that
        downstream fits always have something to work with, and never exceeds
        the number of steps available.

    Note:
        If the path shows no dynamic range -- its largest step is not itself
        ``k`` times the tail floor -- then early and late motion are not
        separable and there is no identifiable noise regime. In that case the
        WHOLE path is returned as signal. Claiming a floor here would be
        unfounded: a path with uniform step sizes (a steady drift, or a run in
        float32 where rounding never becomes visible) has no noise regime to
        exclude, and truncating it to a stub would silently discard real data.
    """
    s = np.linalg.norm(np.diff(np.asarray(traj, dtype=np.float64), axis=0), axis=1)
    if len(s) < 4:
        return (0, len(s))
    floor = empirical_floor(traj, tail_frac)
    if floor <= 0.0 or s.max() <= k * floor:
        return (0, len(s))
    above = s > k * floor
    # last index that is above the floor; everything after is noise
    idx = np.where(above)[0]
    end = int(idx[-1]) + 1 if len(idx) else 2
    return (0, int(np.clip(end, 2, len(s))))


def contraction_from_pair(
    traj_a: np.ndarray, traj_b: np.ndarray, k: float = 3.0
) -> tuple[float, float]:
    """Per-step contraction of the MAP, from two orbits of the same prompt.

    This is the quantity H3 is about: whether nearby states are pulled
    together by the recurrent map. It is NOT the same as how fast one
    trajectory approaches its own endpoint, which is near-tautological for any
    convergent sequence and was measured by mistake earlier in this project.

    Requires ``traj_a`` and ``traj_b`` to come from the SAME prompt with
    DIFFERENT ``h_0``. Note that `results/trajectories/*.npy` does NOT satisfy
    this -- its ``init{N}`` field is the task seed, so those files differ by
    prompt (see docs/rigor_audit.md section 5). The top-level
    `trajectories/` directory with its `manifest.csv` does satisfy it.

    Args:
        traj_a: First trajectory of shape [T, hidden_dim].
        traj_b: Second trajectory, same shape, same prompt, different h_0.
        k: Multiple of the gap floor above which a point is used in the fit.

    Returns:
        ``(rho, floor)`` -- the per-step multiplier (``exp`` of the fitted
        log-slope; < 1 means contracting) and the residual gap the two orbits
        settle to. ``rho`` is NaN if fewer than 4 points sit above the floor.
    """
    a = np.asarray(traj_a, dtype=np.float64)
    b = np.asarray(traj_b, dtype=np.float64)
    d = np.linalg.norm(a - b, axis=1)
    if len(d) < 5:
        return (float("nan"), float("nan"))
    tail = max(1, len(d) // 4)
    floor = float(np.median(d[-tail:]))
    mask = d > k * floor
    if mask.sum() < 4:
        return (float("nan"), floor)
    # Fit only the leading contiguous run above the floor: once the gap has
    # reached the floor, later excursions above it are noise fluctuations and
    # must not re-enter the fit.
    end = int(np.argmax(~mask)) if (~mask).any() else len(d)
    end = max(end, 4)
    t = np.arange(end)
    slope = np.polyfit(t, np.log(d[:end]), 1)[0]
    return (float(np.exp(slope)), floor)


def classify_shape_regime(
    traj: np.ndarray,
    k: float = 3.0,
    loop_frac: float = 0.10,
    min_lag: int = 3,
    settled_margin: float = 0.9,
) -> str:
    """Settle / loop / drift, decided against the noise floor rather than the max.

    `shapes.gate.classify_shape` is DEGENERATE on real Huginn trajectories: it
    asks whether the last step is below 10% of the LARGEST step, but the
    largest step is the h_0 transient (~72) and the last is the bfloat16 floor
    (~1). Measured over all 140 banked trajectories the ratio never exceeds
    0.0188 -- five times under the threshold -- so it returns "settle"
    unconditionally and "loop"/"drift" are unreachable. Its fallback branch is
    no better: the noise ball supplies far-pair distances ~0.012 of the cloud
    diameter, well under `return_frac`, so any non-settling path would be
    called "loop" for the same structural reason.

    This version asks a question that can actually come out either way: did
    the path reach its noise floor within the compute budget?

      * settle -- it did, with room to spare (converging regime ends before
        ``settled_margin`` of the budget). The state converged.
      * loop   -- it did not, and within the still-moving portion it comes
        back near a point it visited more than ``min_lag`` steps earlier.
      * drift  -- it did not, and it never returns.

    The loop test is run on the SIGNAL portion only, so the noise ball cannot
    manufacture the small distances that made the original test vacuous, and
    it uses the CHORD-TO-ARC ratio rather than a distance normalised by the
    cloud diameter::

        min over |i-j| > min_lag of   ||x_i - x_j||  /  (arc length i..j)

    Normalising by the diameter is length-dependent and wrong: the smallest
    far-pair distance scales like ``min_lag * spacing``, which shrinks as T
    grows, so a long enough path trips the threshold whatever its shape. A
    straight 90-point line reads "loop" under that rule -- confirmed, for both
    this function's first version and `shapes.gate.classify_shape`. Chord/arc
    is immune: it is exactly 1.0 for a straight line, tends to 0 for a closed
    orbit, and does not depend on how finely the path is sampled.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        k: Floor multiple defining signal, passed to `converging_regime`.
        loop_frac: Chord-to-arc ratio below which the path counts as having
            returned. The default 0.10 sits between a diffusive path (which
            reaches ~1/sqrt(L), about 0.11 at L=90) and a genuine orbit
            (which reaches ~0), so ordinary wandering is not called a loop.
        min_lag: Minimum index separation for a pair to count as a return
            (adjacent states are always close; that is not a loop).
        settled_margin: Fraction of the budget by which the regime must end
            for the path to count as settled.

    Returns:
        ``"settle"``, ``"loop"`` or ``"drift"``.
    """
    from scipy.spatial.distance import pdist, squareform

    a = np.asarray(traj, dtype=np.float64)
    n_steps = max(len(a) - 1, 1)
    _, end = converging_regime(a, k=k)

    if end < settled_margin * n_steps:
        return "settle"

    sig = a[: end + 1]
    if len(sig) < min_lag + 2:
        return "drift"

    chord = squareform(pdist(sig))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sig, axis=0), axis=1))])
    arc_between = np.abs(np.subtract.outer(arc, arc))

    m = len(sig)
    mask = (np.abs(np.subtract.outer(range(m), range(m))) > min_lag) & (arc_between > 0)
    if not mask.any():
        return "drift"
    return "loop" if (chord[mask] / arc_between[mask]).min() < loop_frac else "drift"


def h1_persistence_signal(traj: np.ndarray, k: float = 3.0) -> tuple[np.ndarray, float]:
    """H1 persistence of the SIGNAL portion, normalised by its own diameter.

    `metrics.homology.h1_persistence` runs Vietoris-Rips on the whole
    trajectory and divides the longest bar by the whole cloud's diameter. On
    real trajectories 84-85% of the points sit in the arithmetic-noise ball
    (diameter ~9-15) while the diameter is set by the transient (~93). The
    resulting score is therefore approximately
    (noise-ball scale)/(transient extent) -- which is the right order for the
    project's reported max persistence of 0.009, and is not a topological
    property of the computation.

    Restricting to the converging regime removes the blob and makes the
    normalisation refer to the same points the features come from.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        k: Floor multiple defining signal, passed to `converging_regime`.

    Returns:
        ``(dgm1, max_persist_norm)``, same contract as `h1_persistence`.

    Raises:
        ImportError: if the ``tda`` extra (ripser) is not installed.
    """
    from traj_geom.metrics.homology import h1_persistence

    a = np.asarray(traj, dtype=np.float64)
    _, end = converging_regime(a, k=k)
    return h1_persistence(a[: end + 1])


def step_cosine_converging(traj: np.ndarray, k: float = 3.0) -> float:
    """Mean adjacent-step cosine, restricted to the converging regime.

    The floor-unaware counterpart, `convergence.consecutive_step_cosine`,
    averages over the second half of the path. For a path that settles at
    t ~ 14 out of 128 that window is entirely arithmetic noise, whose
    increments are anti-correlated for a trivial reason (white-noise
    increments have lag-1 autocorrelation exactly -0.5) and have nothing to do
    with the model.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        k: Passed through to `converging_regime`.

    Returns:
        Mean cosine between consecutive steps inside the converging regime,
        or NaN if fewer than two such steps exist. Positive means the path
        glides; negative means it genuinely zig-zags.
    """
    a = np.asarray(traj, dtype=np.float64)
    _, end = converging_regime(a, k=k)
    dd = np.diff(a[: end + 1], axis=0)
    if len(dd) < 2:
        return float("nan")
    nrm = np.linalg.norm(dd, axis=1)
    good = nrm > 0
    dd, nrm = dd[good], nrm[good]
    if len(dd) < 2:
        return float("nan")
    cos = (dd[:-1] * dd[1:]).sum(1) / (nrm[:-1] * nrm[1:])
    return float(np.mean(np.clip(cos, -1.0, 1.0)))
