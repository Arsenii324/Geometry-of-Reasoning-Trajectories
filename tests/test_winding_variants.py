"""Unit tests for the alternative rotation estimators.

The load-bearing test here is `test_reversing_spiral_*`: it encodes the
blind spot that motivated W7/W8/W9 -- a path that winds +5 turns and then
-5 turns the other way is rotating ten times, but every NET (signed) metric
reports ~0.

CAVEAT ON THE MOTIVATION (docs/rigor_audit.md section 2, 2026-07-25). These
tests remain valid as ESTIMATOR tests: a net measure genuinely is blind to
reversal, and that is what is pinned below. But the claim that reversal
actually occurs in Huginn's trajectories -- previously supported by
convergence.csv's mean adjacent-step cosine of -0.276 -- did not survive
audit. That statistic is averaged over the second half of the path, which for
num_steps=128 is entirely below the bfloat16 rounding floor; restricted to the
converging regime the cosine is +0.084, i.e. the path glides. Whether Huginn's
trajectories reverse direction is currently an open question, not an
established fact.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.winding_variants import VARIANT_NAMES, winding_variants


def _reversing_spiral(turns: int = 5, n: int = 300) -> np.ndarray:
    """+turns one way, contract 10x, then -turns the other way. Total rotation
    is 2*turns; NET rotation is ~0."""
    t = np.linspace(0, turns * 2 * np.pi, n)
    r = np.exp(-t / 40)
    p1 = np.c_[r * np.cos(t), r * np.sin(t)]
    p2 = np.c_[0.1 * r[::-1] * np.cos(-t), 0.1 * r[::-1] * np.sin(-t)]
    path = np.vstack([p1, p2])
    return np.c_[path, np.zeros((len(path), 3))]  # pad to >2 dims for PCA(3)


def _clean_spiral(turns: int = 3, n: int = 200) -> np.ndarray:
    """A single-direction decaying spiral: net and total rotation agree."""
    t = np.linspace(0, turns * 2 * np.pi, n)
    r = np.exp(-t / 30)
    return np.c_[r * np.cos(t), r * np.sin(t), np.zeros(n), np.zeros(n), np.zeros(n)]


def test_all_variants_present_and_finite() -> None:
    v = winding_variants(_clean_spiral())
    assert set(v) == set(VARIANT_NAMES)
    assert all(np.isfinite(x) for x in v.values())
    assert all(x >= 0 for x in v.values())


def test_reversing_spiral_defeats_the_net_metrics() -> None:
    """The blind spot, pinned: the project's actual metric (W1) reports
    essentially zero rotation for a path that rotates 10 times.
    """
    v = winding_variants(_reversing_spiral(turns=5))
    assert v["W1_centroid"] < 0.5, "W1 should cancel on a reversing path"


def test_reversing_spiral_is_caught_by_the_absolute_metrics() -> None:
    """W7/W8 count rotation regardless of direction, so a +5/-5 path reads
    close to its true total of 10 turns.
    """
    v = winding_variants(_reversing_spiral(turns=5))
    assert v["W8_absturn_centroid"] == pytest.approx(10.0, abs=1.5)
    assert v["W7_absturn_fixedpt"] > 5.0
    # And the windowed net measure finds locally-coherent rotation.
    assert v["W9_windowed_maxnet"] > 1.0


def test_clean_spiral_net_and_absolute_agree() -> None:
    """Sanity in the other direction: with no reversal, net ~= absolute, so
    the absolute metrics are not simply inflating everything.
    """
    v = winding_variants(_clean_spiral(turns=3))
    assert v["W8_absturn_centroid"] == pytest.approx(v["W1_centroid"], rel=0.25)


def test_straight_line_has_no_rotation_under_any_variant() -> None:
    """Degenerate control: a straight path should not register turns."""
    n = 100
    line = np.c_[np.linspace(0, 1, n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)]
    v = winding_variants(line)
    for k in ("W1_centroid", "W4_turn2d_signed", "W8_absturn_centroid", "W9_windowed_maxnet"):
        assert v[k] < 0.6, f"{k} found rotation in a straight line"
