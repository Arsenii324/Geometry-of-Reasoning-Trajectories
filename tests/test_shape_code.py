"""The shape code must keep shape and discard position -- both halves are testable.

If it kept position, a "shape decodes the task" result would be the architecture's
prompt re-injection (D70(4)) read back out, which D73 measured random weights doing
at R2 = 0.99999. If it discarded shape, a null would be trivial. Both are planted.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.shape_code import (
    gram_code,
    is_rotation_invariant,
    position_code,
)

DIM = 50
M = 20


def _spiral(phi=0.4, rho=0.9, n=40, seed=0, offset=0.0, scale=1.0):
    """A planar decaying spiral, optionally translated and rescaled."""
    rng = np.random.default_rng(seed)
    b = np.linalg.qr(rng.normal(size=(DIM, 2)))[0]
    t = np.arange(n, dtype=float)
    env = scale * rho**t
    x = np.outer(env * np.cos(phi * t), b[:, 0]) + np.outer(env * np.sin(phi * t),
                                                            b[:, 1])
    return x + offset * rng.normal(size=DIM)


def test_the_code_is_rotation_invariant_to_machine_precision() -> None:
    """THE PROPERTY THE WHOLE ARGUMENT RESTS ON."""
    assert is_rotation_invariant(_spiral(seed=1), m=M) < 1e-9


def test_the_code_is_translation_invariant() -> None:
    """Moving the whole orbit must not change its shape code.

    Position is exactly what the raw states carry by architecture, so a code that
    moved with the orbit would be a position code wearing a shape label.
    """
    x = _spiral(seed=2)
    shifted = x + np.linspace(1, 2, DIM) * 37.0
    assert np.allclose(gram_code(x, m=M), gram_code(shifted, m=M), atol=1e-9)


def test_the_code_is_scale_invariant() -> None:
    a = gram_code(_spiral(seed=3, scale=1.0), m=M)
    b = gram_code(_spiral(seed=3, scale=1000.0), m=M)
    assert np.allclose(a, b, atol=1e-6)


def test_the_code_does_change_with_the_shape() -> None:
    """And it must not be invariant to everything -- a constant feature decodes
    nothing and would make any null vacuous."""
    a = gram_code(_spiral(phi=0.4, seed=4), m=M)
    b = gram_code(_spiral(phi=1.5, seed=4), m=M)
    assert np.max(np.abs(a - b)) > 0.5, "different rotation rates gave the same code"


def test_a_pure_contraction_gives_an_all_ones_code() -> None:
    """Known answer: with no rotation every step direction is identical."""
    rng = np.random.default_rng(5)
    v = rng.normal(size=DIM)
    traj = np.array([v * 0.85**t for t in range(40)])
    assert np.allclose(gram_code(traj, m=M), 1.0, atol=1e-9)


def test_the_code_length_is_fixed_and_does_not_depend_on_the_orbit() -> None:
    """A classifier cannot take ragged features, and D80 makes the window the
    single largest determinant of every shape statistic -- so it is held fixed."""
    lens = {len(gram_code(_spiral(n=n, seed=6), m=M)) for n in (25, 40, 90)}
    assert lens == {M * (M - 1) // 2}


def test_a_short_orbit_returns_empty_rather_than_a_short_code() -> None:
    assert gram_code(_spiral(n=8, seed=7), m=M).size == 0
    assert position_code(_spiral(n=8, seed=7), m=M).size == 0


def test_the_position_control_has_exactly_the_same_feature_count() -> None:
    """A control with fewer features than the arm it validates is a gift to the arm.

    The first version returned `m * (n_proj // m)` = 180 against the shape code's
    190, which would have made the positive control 5% weaker than the thing it
    exists to bound.
    """
    x = _spiral(seed=8)
    g = gram_code(x, m=M)
    assert len(position_code(x, m=M, n_proj=len(g))) == len(g)


def test_the_position_control_does_carry_position() -> None:
    """It must separate two orbits that differ only by a translation -- exactly the
    information the shape code throws away."""
    x = _spiral(seed=9)
    a = position_code(x, m=M)
    b = position_code(x + 50.0, m=M)
    assert np.max(np.abs(a - b)) > 1.0
    assert np.allclose(gram_code(x, m=M), gram_code(x + 50.0, m=M), atol=1e-9)


@pytest.mark.parametrize("m", [5, 12, 20])
def test_the_code_is_the_strict_upper_triangle(m) -> None:
    """No diagonal: the self-cosines are 1 by construction and carry nothing, and
    including them would dilute every feature-scaled classifier."""
    c = gram_code(_spiral(seed=10), m=m)
    assert len(c) == m * (m - 1) // 2
    assert np.abs(c).max() <= 1.0 + 1e-9
