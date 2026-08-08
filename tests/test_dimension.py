"""Known-answer validation for the effective-dimension estimator.

The claim it will support -- that the orbit is NOT low-dimensional, so every
planar description of it is inadequate by construction -- is a claim about an
estimator's output, so the estimator has to be shown to return the right answer on
inputs whose dimensionality is known by construction, INCLUDING the case that
motivated it: a low-dimensional path whose step sizes decay geometrically.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.dimension import (
    effective_dimension,
    participation_ratio,
    step_directions,
)


def test_participation_ratio_endpoints() -> None:
    """1 for parallel rows, k for k orthonormal rows."""
    assert participation_ratio(np.tile([1.0, 0.0, 0.0], (8, 1))) == pytest.approx(1.0)
    assert participation_ratio(np.eye(5)) == pytest.approx(5.0)


def test_a_planar_spiral_reads_as_two_dimensional() -> None:
    """The case every winding metric assumes. It must come back as ~2, not ~m."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(300, 2)))
    t = np.arange(40)
    xy = (0.87**t)[:, None] * np.stack([np.cos(0.4 * t), np.sin(0.4 * t)], axis=1)
    out = effective_dimension(xy @ q.T)
    assert out["pr"] < 2.5, out
    assert out["z"] < -5, "a planar orbit must sit far BELOW the isotropic null"


def test_the_decay_envelope_alone_does_not_create_low_dimensionality() -> None:
    """The trap this estimator exists to avoid.

    Isotropic directions with geometrically decaying NORMS look low-dimensional to
    any un-normalised measure. Measured on the real orbits: raw differences give
    PR 5.34 and norm-matched isotropic vectors give 5.74 -- nearly all of the
    apparent structure was the envelope. After normalisation the estimator must
    report this construction as isotropic.
    """
    rng = np.random.default_rng(1)
    m, d = 30, 400
    dirs = rng.normal(size=(m, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    traj = np.cumsum(dirs * (0.87 ** np.arange(m))[:, None], axis=0)
    out = effective_dimension(traj)
    assert abs(out["z"]) < 4, f"isotropic-by-construction should not flag: {out}"
    assert out["ratio"] == pytest.approx(1.0, abs=0.15)


def test_normalisation_actually_removes_the_envelope() -> None:
    """`step_directions` must return unit vectors, or the envelope leaks back in."""
    rng = np.random.default_rng(2)
    traj = np.cumsum(rng.normal(size=(20, 50)) * (0.5 ** np.arange(20))[:, None], axis=0)
    u = step_directions(traj)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0)


def test_intermediate_dimensionality_is_recovered() -> None:
    """A 6-dimensional random walk must read near 6, not 2 and not m."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(200, 6)))
    steps = rng.normal(size=(40, 6)) @ q.T
    out = effective_dimension(np.cumsum(steps, axis=0))
    assert 4.0 < out["pr"] < 8.0, out


def test_window_selection_is_honoured() -> None:
    rng = np.random.default_rng(4)
    traj = np.cumsum(rng.normal(size=(60, 30)), axis=0)
    assert effective_dimension(traj, lo=10, hi=25)["n_steps"] == 15


def test_refuses_a_window_too_short_to_mean_anything() -> None:
    out = effective_dimension(np.cumsum(np.ones((4, 10)), axis=0))
    assert out["ok"] is False


def test_null_is_simulated_at_the_sample_count_not_the_ambient_dimension() -> None:
    """With m << d the isotropic null is ~m, not ~d.

    Comparing against d would make every real orbit look dramatically
    low-dimensional and the statistic would be meaningless.
    """
    rng = np.random.default_rng(5)
    traj = np.cumsum(rng.normal(size=(25, 5280)), axis=0)
    out = effective_dimension(traj)
    assert 18 < out["null_mean"] <= 24, out["null_mean"]
