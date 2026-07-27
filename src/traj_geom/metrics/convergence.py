"""Objective convergence metrics for a latent trajectory (settle vs loop/drift).

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp.ipynb).
TASK: model-free, sign-robust descriptors of how a path approaches (or fails to
    approach) a fixed point — the diagnostics behind the "everything settles" result.
I/O: traj [T, dim] -> float; path_independence takes two trajectories.

`step_norms` is reused from metrics.dynamics (single source of truth).
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.dynamics import step_norms


def convergence_rate(traj: np.ndarray) -> float:
    """Log-slope of ``||h_t - h_final||`` over t. Kept for reproducibility.

    !! TWO CAVEATS (docs/rigor_audit.md sections 6 and 7).

    (1) It fits a single slope over the WHOLE path. The distance decays
    exponentially and then plateaus at the bfloat16 floor, and the plateau
    drags the fit toward zero -- measured bias +0.084/step (x0.9244
    whole-range against x0.8407 on the converging region alone).

    (2) More fundamentally, distance to a trajectory's own final state is
    close to tautological: any convergent sequence approaches its own limit.
    It is NOT the contractivity of the recurrent map, which is what H3 is
    about. That requires two orbits of the SAME prompt with DIFFERENT h_0 --
    see `path_independence`, or `metrics.regime.contraction_from_pair` for the
    floor-aware version.

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        The fitted log-slope, or NaN if too few points remain above the floor.
    """
    hf = traj[-1]
    d = np.linalg.norm(traj[:-1] - hf, axis=1)
    d = d[d > 1e-9]
    return float(np.polyfit(np.arange(len(d)), np.log(d), 1)[0]) if len(d) > 3 else np.nan


def consecutive_step_cosine(traj: np.ndarray) -> float:
    """Mean cosine of adjacent steps (2nd half). Kept for reproducibility.

    Previously attributed to "Pappone et al. / Movahedi et al." for the reading
    that a negative mean cosine means the path zig-zags inward rather than
    gliding. !! BOTH ATTRIBUTIONS FAIL (docs/rigor_audit.md section 20):
    Pappone et al. (arXiv:2509.23314) studies a GPT-2-scale model rather than
    Huginn and reports a consecutive-step cosine settling at +0.5 to +0.65 --
    POSITIVE, the opposite sign to the -0.276 attributed to it here, and
    consistent with this module's own converging-regime value of +0.084.
    "Movahedi et al." could not be located at all and should be treated as
    unverified until someone produces the reference. Do not repeat either
    citation in paper-facing text without checking it.

    !! THAT READING DOES NOT HOLD FOR THIS DATA (docs/rigor_audit.md section 2).
    Averaging the SECOND HALF is only meaningful if the path is still moving
    there. Huginn's paths settle at t ~ 14, so for num_steps=128 the second
    half lies entirely below the bfloat16 rounding floor, and this function
    reports a property of arithmetic rather than of the model::

        converging regime  t = 0..14    mean cos = +0.084
        post-floor regime  t = 64..127  mean cos = -0.331   <- reported here

    White-noise increments have lag-1 autocorrelation exactly -0.5, which is
    what the second half is approaching. In the regime where the model is
    actually computing the cosine is POSITIVE -- the path glides.

    Left bit-compatible on purpose so cached CSVs remain reproducible. For the
    honest number use `metrics.regime.step_cosine_converging`.

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        Mean adjacent-step cosine over the second half, or NaN if too short.
    """
    dd = np.diff(traj, axis=0)
    c = [
        float(np.dot(dd[i], dd[i + 1]) / (np.linalg.norm(dd[i]) * np.linalg.norm(dd[i + 1]) + 1e-9))
        for i in range(len(dd) - 1)
    ]
    return float(np.mean(c[len(c) // 2:])) if len(c) > 2 else np.nan


def drift_to_loop_ratio(traj: np.ndarray) -> float:
    """``||h_final - h_0|| / mean step``. >>1 = a transient arc to a far point, not a loop.

    Pappone et al.'s drift-to-loop ratio (DLR).

    Args:
        traj: Trajectory of shape [T, dim].

    Returns:
        The DLR (dimensionless).
    """
    return float(np.linalg.norm(traj[-1] - traj[0]) / (step_norms(traj).mean() + 1e-9))


def path_independence(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """Log-slope of the gap between two trajectories from different inits.

    <0 = the two paths converge to a common point (path-independence, Geiping et al.).

    This is the RIGHT quantity for H3 -- unlike `convergence_rate`, it measures
    contraction of the map rather than a sequence approaching its own limit.

    !! BIASED BY THE NOISE FLOOR (docs/rigor_audit.md section 6). The fit spans
    the whole path, but the gap stops shrinking once it reaches the bfloat16
    floor (~1.4 at num_steps=64), so the plateau flattens the slope::

        whole-range fit  x0.9244        early-region fit  x0.8407

    i.e. contraction is understated by ~0.084 per step, affecting all 600
    `lyap` values in results/dissoc_multiinit.csv. Left bit-compatible for
    reproducibility; use `metrics.regime.contraction_from_pair` for the
    floor-aware estimate, which also returns the floor itself.

    !! REQUIRES GENUINE MULTI-INIT INPUT. The two trajectories must share a
    prompt and differ in h_0. `results/trajectories/*.npy` does NOT satisfy
    this -- its `init{N}` field is the TASK seed (rigor_audit section 5), so
    those files differ by prompt and this function would return a
    prompt-separation figure, not a contraction rate. The top-level
    `trajectories/` directory with `manifest.csv` does satisfy it.

    Args:
        traj_a: First trajectory of shape [T, dim].
        traj_b: Second trajectory of shape [T, dim].

    Returns:
        The fitted log-slope of ``||a_t - b_t||``, or NaN if too few points remain.
    """
    d = np.linalg.norm(traj_a - traj_b, axis=1)
    d = d[d > 1e-9]
    return float(np.polyfit(np.arange(len(d)), np.log(d), 1)[0]) if len(d) > 3 else np.nan
