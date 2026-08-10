"""Step-size dynamics of a trajectory: per-step displacement and settling time.

OWNER: Extraction+Winding
STATUS: implemented (from notebooks/01_mvp_h2.ipynb, step_norms / steps_to_settle).
TASK: measure how fast the recurrent path converges — the "effective compute"
    signal that turned out to carry the MVP result.
I/O: step_norms(traj [T,H]) -> [T-1] ; steps_to_settle(traj, frac) -> int.
"""

from __future__ import annotations

import numpy as np


def step_norms(traj: np.ndarray) -> np.ndarray:
    """Return the L2 norm of each recurrent step ``||h_{t+1} - h_t||``.

    Args:
        traj: Trajectory of shape [T, hidden_dim].

    Returns:
        Array of shape [T-1] with the per-step displacement norms.
    """
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)


def steps_to_settle(traj: np.ndarray, frac: float = 0.1) -> int:
    """First step at which the displacement drops below ``frac`` of its max.

    Canonical, first-order definition — every existing experiment's
    `steps_settle` column is computed with this. See
    `steps_to_settle_by_acceleration` for the second-order alternative.

    !! DO NOT READ THIS AS "EFFECTIVE COMPUTE" (docs/rigor_audit.md section 3).
    It was documented that way, and that reading is wrong. The threshold is
    ``frac`` of the MAXIMUM step, and the maximum is always the initial
    transient in which the model forgets its random ``h_0``. For a path
    decaying geometrically at rate rho the crossing point is therefore fixed
    by rho alone::

        t* = log(frac)/log(rho) = log(0.1)/log(0.8407) = 13.27
        observed mean over 140 banked trajectories = 13.84 (std 1.98)

    The prediction uses no task information whatsoever and matches to within
    0.6 steps. This function measures the CONTRACTION RATE, which is
    task-independent -- which is also why its variance is so small that
    correlating it against difficulty has almost no signal to find. Any claim
    of the form "steps_settle tracks difficulty" needs re-deriving.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        frac: Fraction of the max step norm below which the path is "settled".

    Returns:
        Index of the first settled step, or the number of steps if it never settles.
    """
    s = step_norms(traj)
    below = np.where(s < frac * s.max())[0]
    return int(below[0]) if len(below) else len(s)


def steps_to_settle_by_acceleration(traj: np.ndarray, frac: float = 0.1) -> int:
    """First step at which the normed acceleration drops below ``frac`` of its max.

    A second-order alternative to `steps_to_settle`: thresholds on how much the
    step *direction and size* are still changing (Pappone-style acceleration),
    rather than on step size alone. Deliberately a distinct name/column from
    `steps_to_settle` — the two measure different things and are not
    interchangeable; do not conflate them when reading a results CSV.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        frac: Fraction of the max normed acceleration below which "settled".

    Returns:
        Index of the first settled step, or the number of steps if it never settles.
    """
    if len(traj) < 3:
        return len(traj)
    a = normed_acceleration(traj)
    if len(a) == 0:
        return len(traj)
    below = np.where(a < frac * a.max())[0]
    return int(below[0] + 2) if len(below) else len(traj)


def contraction_rate(traj: np.ndarray) -> float:
    """Mean log-change of step size along the path. <0 = the path is contracting.

    Args:
        traj: Trajectory of shape [T, hidden_dim].

    Returns:
        Mean of ``diff(log(step_norms))``, or NaN if fewer than 3 steps survive the floor.
    """
    s = step_norms(traj)
    s = s[s > 1e-9]
    return float(np.mean(np.diff(np.log(s)))) if len(s) >= 3 else np.nan


def normed_acceleration(traj: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute the normed acceleration hat(a)^k for a trajectory.

    Formula: ``||delta^k - delta^{k-1}||_2 / (||delta^k||_2 + ||delta^{k-1}||_2 + eps)``
    where ``delta^k = h_k - h_{k-1}``.

    Args:
        traj: Trajectory of shape [T, hidden_dim].
        eps: Small constant to avoid division by zero.

    Returns:
        Array of shape [T-2] with the normed acceleration at each step k >= 2.
    """
    delta = np.diff(traj, axis=0)
    norm_delta = np.linalg.norm(delta, axis=1)

    delta_k = delta[1:]
    delta_k_minus_1 = delta[:-1]

    num = np.linalg.norm(delta_k - delta_k_minus_1, axis=1)
    den = norm_delta[1:] + norm_delta[:-1] + eps
    return num / den



def rotation_power(traj: np.ndarray, period: int = 6, tail: int = 24) -> float:
    """Fraction of a trajectory tail's non-DC spectral power at a given period.

    The CONTINUOUS extension of the binary "does this orbit rotate" label that D132
    and D134 used. `traj` is [T, H]; returns a value in [0, 1] -- a pure period-`period`
    orbit tends to 1, a monotone settle to 0.

    Lives here, and not in the two callers, because D137 compares kernel-side and
    analysis-side values of exactly this number: `scripts/run_rotation_continuum.py`
    computed it locally on banked orbits, and `scratch/ds_einterp/job.py` computes it
    on GPU inside the run. Two copies would let those drift, and the comparison
    between them is what D137's instrument null rests on.

    Raises when the tail cannot resolve the requested period rather than returning a
    number computed from an aliased bin: `tail` must be a whole multiple of `period`
    (so the period lands on an exact rfft bin) and must span at least two cycles.
    """
    if period <= 0 or tail <= 0:
        raise ValueError(f"period and tail must be positive, got {period}, {tail}")
    if tail % period:
        raise ValueError(
            f"tail={tail} is not a whole multiple of period={period}, so period "
            f"{period} does not land on an rfft bin and the returned power would be "
            f"split across neighbours. Choose a tail divisible by the period.")
    if tail // period < 2:
        raise ValueError(f"tail={tail} spans {tail // period} cycle(s) of period "
                         f"{period}; at least 2 are needed to distinguish a "
                         f"periodic orbit from a single excursion.")
    if traj.shape[0] < tail:
        raise ValueError(f"trajectory has {traj.shape[0]} steps, fewer than tail={tail}")
    x = np.asarray(traj[-tail:], dtype=np.float64)
    x = x - x.mean(0)
    f = np.abs(np.fft.rfft(x, axis=0)) ** 2
    total = f[1:].sum()
    return float(f[tail // period].sum() / total) if total > 0 else 0.0
