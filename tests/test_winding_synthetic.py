"""winding_number sanity checks on synthetic paths with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.winding import matched_random_walk, winding_null_test, winding_number


def _circle(n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def _spiral_trajectory(n_turns: float = 3.0, n_steps: int = 60, dim: int = 8) -> np.ndarray:
    """A genuine multi-turn spiral in dims 0-1 of a higher-dim space, dims 2+ zero."""
    t = np.linspace(0.0, 2 * np.pi * n_turns, n_steps)
    radius = np.linspace(0.5, 2.0, n_steps)  # not a pure circle -> more trajectory-like
    traj = np.zeros((n_steps, dim))
    traj[:, 0] = radius * np.cos(t)
    traj[:, 1] = radius * np.sin(t)
    return traj


def test_winding_number_circle_is_one() -> None:
    """One full counter-clockwise circle winds ~ +1."""
    assert winding_number(_circle()) == pytest.approx(1.0, abs=0.02)


def test_winding_number_reversed_circle_is_minus_one() -> None:
    """Reversing traversal flips the winding sign to ~ -1."""
    assert winding_number(_circle()[::-1]) == pytest.approx(-1.0, abs=0.02)


def test_winding_number_two_turns() -> None:
    """A path wrapping twice winds ~ +2."""
    t = np.linspace(0.0, 4.0 * np.pi, 800, endpoint=False)
    two = np.stack([np.cos(t), np.sin(t)], axis=1)
    assert winding_number(two) == pytest.approx(2.0, abs=0.02)


def test_winding_null_test_detects_real_spiral() -> None:
    """A genuine multi-turn spiral should wind far more than a matched
    random walk with the same per-step displacement sizes -- this is the
    null-model check flagged as missing project-wide (see winding.py NOTE).
    """
    traj = _spiral_trajectory(n_turns=3.0, n_steps=60, dim=8)
    result = winding_null_test(traj, burn=0, n_surrogates=100, seed=0)
    assert result["observed"] > 2.0  # ~3 real turns, some slack from projection/burn
    assert result["observed"] > result["null_mean"] + 3 * result["null_std"]
    assert result["p_value"] < 0.05


def test_winding_null_test_random_walk_is_not_extreme() -> None:
    """A matched-random-walk surrogate of the spiral should NOT itself wind
    anywhere near as much as the real spiral -- sanity check that the null
    isn't degenerate (e.g. always rejecting regardless of input). Both
    generator and test seeds are fixed, so this is deterministic, not flaky.
    """
    rng = np.random.default_rng(1)
    traj = _spiral_trajectory(n_turns=3.0, n_steps=60, dim=8)
    random_traj = matched_random_walk(traj, rng)
    result = winding_null_test(random_traj, burn=0, n_surrogates=100, seed=2)
    assert result["observed"] < 2.0
