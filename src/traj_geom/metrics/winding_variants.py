"""Alternative winding/rotation estimators, for adjudicating the metric itself.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25, in response to claims_ledger.md D22 -- the
    baseline `winding_of` fails a matched-random-walk null test (82% of real
    trajectories wind LESS than chance) with a diagnosed mechanism: it
    measures angle about the trajectory's own centroid, and a settling path
    collapses 18x toward that centroid, so the collapse itself sweeps ~0.6
    turns regardless of content. These variants each attack a different part
    of that failure so the choice of metric can be decided by evidence
    rather than by inheritance.
TASK: given one trajectory [T, hidden], return several rotation estimates
    that differ in (a) what point rotation is measured about, (b) whether the
    radial collapse is removed first, (c) whether a projection is used at all.
I/O: winding_variants(traj) -> dict[str, float]  (all non-negative magnitudes)

THE VARIANTS AND WHAT EACH TESTS
  W1_centroid            the CURRENT metric: 2-D PCA, angle about the point
                         cloud's centroid. Baseline to beat.
  W2_fixedpoint          same projection, but angle about the trajectory's
                         final state (its settled fixed point). For a
                         converging spiral this is the geometrically correct
                         center; the centroid is not.
  W3_sphere              remove the radial collapse BEFORE choosing the
                         projection: recentre on the fixed point, normalise
                         every state to unit norm in the FULL space, then fit
                         PCA. This stops PC1 from being spent on the decay
                         direction.
  W4_turn2d_signed       center-free: the signed rotation of the step-direction
                         itself in 2-D. Consistent orbiting accumulates; random
                         turning cancels. Sensitive to net rotation, not radius.
  W5_turnfull_unsigned   center-free AND projection-free: total unsigned
                         turning between consecutive step vectors in the full
                         5280-d space. Measures total curvature. NOTE it is
                         expected to sit far below a random-walk null (random
                         directions turn ~90 deg per step), so a low z here is
                         uninformative about orbiting -- included as a control.
  W6_pc23                same as W1 but on PC2-PC3, discarding the dominant
                         (drift/decay) component that PC1 is spent on.

  --- added 2026-07-25, closing a blind spot in every variant above ---
  Every W1/W2/W4/W6 above is a NET (signed) quantity: |sum of dtheta|. A path
  that winds +5 turns, contracts, then winds -5 turns the other way sums to
  ~0 and is reported as "no rotation" despite rotating ten times. W7/W8/W9
  close that gap and remain correct as ESTIMATORS -- see tests.

  !! THE MOTIVATING EVIDENCE DID NOT SURVIVE AUDIT (2026-07-25, same day) !!
  These three were introduced because results/convergence.csv reports a mean
  adjacent-step cosine of -0.276, read as "the real paths zig-zag inward, so
  direction reversal is real in this data". docs/rigor_audit.md section 2
  shows that reading is wrong. `consecutive_step_cosine` averages over the
  SECOND HALF of the path; paths settle at t ~ 14, so for num_steps=128 that
  window lies entirely below the bfloat16 rounding floor. Split by regime:

      converging regime  t = 0..14    mean cos = +0.084
      post-floor regime  t = 64..127  mean cos = -0.331   <- the -0.276 figure

  White-noise increments have lag-1 autocorrelation exactly -0.5, which is
  what the second half is approaching. In the regime where the model is
  actually computing the cosine is POSITIVE: the path glides, it does not
  zig-zag. Use `metrics.regime.step_cosine_converging` for the honest number.

  So: keep these variants (a net measure really is blind to reversal, and the
  synthetic tests pin that), but do NOT cite Huginn's adjacent-step cosine as
  evidence that reversal occurs here. Whether it occurs is currently unknown.
  W7_absturn_fixedpt     TOTAL ABSOLUTE angular variation about the fixed
                         point: sum |dtheta|, not |sum dtheta|. Counts
                         rotation regardless of direction, so +5 then -5
                         reads as 10, not 0.
  W8_absturn_centroid    same, about the centroid (unsigned counterpart of W1).
  W9_windowed_maxnet     max |net winding| over sliding windows: finds
                         LOCALLY coherent rotation that cancels globally.
                         Window is fixed at 1/4 of the (post-burn) path.

INTERPRETATION RULE (fixed in advance). A variant is BETTER only if its
    observed value exceeds ITS OWN matched-random-walk null by more than the
    baseline does -- larger raw magnitude means nothing, since every variant
    has a different scale. Selecting a variant by raw size would be metric
    shopping. See scripts/run_winding_variants_null.py, which applies the
    identical surrogate to every variant.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["winding_variants", "VARIANT_NAMES"]

VARIANT_NAMES = (
    "W1_centroid",
    "W2_fixedpoint",
    "W3_sphere",
    "W4_turn2d_signed",
    "W5_turnfull_unsigned",
    "W6_pc23",
    "W7_absturn_fixedpt",
    "W8_absturn_centroid",
    "W9_windowed_maxnet",
)


def _dtheta(xy: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Per-step angular increments about a center, unwrapped to (-pi, pi]."""
    v = xy - center
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    return (d + np.pi) % (2 * np.pi) - np.pi


def _wind_about(xy: np.ndarray, center: np.ndarray) -> float:
    """|NET turns| about a center. Cancels on direction reversal -- see the
    module docstring's note on W7/W8."""
    return float(abs(_dtheta(xy, center).sum() / (2 * np.pi)))


def _absturn_about(xy: np.ndarray, center: np.ndarray) -> float:
    """TOTAL turns about a center, direction-agnostic: sum|dtheta| / 2pi."""
    return float(np.abs(_dtheta(xy, center)).sum() / (2 * np.pi))


def _windowed_maxnet(xy: np.ndarray, center: np.ndarray, frac: float = 0.25) -> float:
    """Largest |net winding| in any sliding window of ``frac`` of the path.

    Detects locally coherent rotation that global net measures cancel away.
    """
    d = _dtheta(xy, center)
    w = max(3, int(len(d) * frac))
    if len(d) <= w:
        return float(abs(d.sum() / (2 * np.pi)))
    c = np.concatenate([[0.0], np.cumsum(d)])
    sums = c[w:] - c[:-w]
    return float(np.abs(sums).max() / (2 * np.pi))


def winding_variants(traj: np.ndarray, burn: int = 4) -> dict[str, float]:
    """Compute every rotation estimate for one trajectory.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        burn: Initial states to drop (the random-h_0 transient), matching
            `winding_of`'s own default so W1 reproduces the baseline exactly.

    Returns:
        Dict keyed by ``VARIANT_NAMES``; every value is a non-negative
        magnitude, but the variants are on DIFFERENT scales and must only be
        compared against their own nulls, never against each other.
    """
    a = np.asarray(traj, dtype=np.float64)[burn:]
    out: dict[str, float] = {}

    pca3 = PCA(n_components=3, svd_solver="full").fit(a)
    xyz = pca3.transform(a)
    xy = xyz[:, :2]

    out["W1_centroid"] = _wind_about(xy, xy.mean(0))
    out["W2_fixedpoint"] = _wind_about(xy, xy[-1])
    out["W6_pc23"] = _wind_about(xyz[:, 1:3], xyz[:, 1:3].mean(0))

    # W3: kill the radial collapse before the projection is chosen.
    r = a - a[-1]
    n = np.linalg.norm(r, axis=1, keepdims=True)
    n[n == 0] = 1.0
    sphere = PCA(n_components=2, svd_solver="full").fit_transform(r / n)
    out["W3_sphere"] = _wind_about(sphere, sphere.mean(0))

    # W4: signed rotation of the 2-D step direction (no center at all).
    step2 = np.diff(xy, axis=0)
    ang = np.arctan2(step2[:, 1], step2[:, 0])
    dd = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
    out["W4_turn2d_signed"] = float(abs(dd.sum() / (2 * np.pi)))

    # W5: total unsigned turning in the full space (no center, no projection).
    step_full = np.diff(a, axis=0)
    nn = np.linalg.norm(step_full, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    u = step_full / nn
    cos = np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0)
    out["W5_turnfull_unsigned"] = float(np.arccos(cos).sum() / (2 * np.pi))

    # Direction-agnostic and windowed measures (see module docstring).
    out["W7_absturn_fixedpt"] = _absturn_about(xy, xy[-1])
    out["W8_absturn_centroid"] = _absturn_about(xy, xy.mean(0))
    out["W9_windowed_maxnet"] = _windowed_maxnet(xy, xy[-1])

    return out
