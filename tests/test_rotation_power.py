"""Exact identities for `rotation_power`, the statistic D137 rests on.

These are exact rather than approximate on purpose: a pure period-p orbit puts ALL
its non-DC power in one bin, so the answer is 1.0 to floating point and any change
that splits or misplaces the bin fails immediately. Verified to fail when the bin
index is perturbed by one -- see `test_bin_offset_would_be_caught`.
"""

import numpy as np
import pytest

from traj_geom.metrics.dynamics import rotation_power


def _pure(period, n=64):
    t = np.arange(n)[:, None]
    return np.concatenate([np.cos(2 * np.pi * t / period),
                           np.sin(2 * np.pi * t / period)], axis=1)


def test_pure_period_is_exactly_one():
    assert rotation_power(_pure(6), 6, 24) == pytest.approx(1.0, abs=1e-12)


def test_wrong_period_gets_none_of_the_power():
    # a pure period-6 orbit read at period 4 (bin 6 of 24) must be ~0
    assert rotation_power(_pure(6), 4, 24) == pytest.approx(0.0, abs=1e-12)


def test_monotone_settle_is_small():
    t = np.arange(64)[:, None]
    assert rotation_power(0.8 ** t * np.ones((64, 2)), 6, 24) < 0.1


def test_bin_offset_would_be_caught():
    """The test above fails if the bin is off by one -- so it is not vacuous."""
    x = _pure(6)[-24:].astype(np.float64)
    x = x - x.mean(0)
    f = np.abs(np.fft.rfft(x, axis=0)) ** 2
    correct, offset = f[4].sum() / f[1:].sum(), f[5].sum() / f[1:].sum()
    assert correct == pytest.approx(1.0, abs=1e-12)
    assert offset < 1e-12          # an off-by-one bin returns ~0, not ~1


@pytest.mark.parametrize("period,tail", [(6, 25), (6, 6), (0, 24), (6, 0)])
def test_unresolvable_requests_raise(period, tail):
    with pytest.raises(ValueError):
        rotation_power(_pure(6), period, tail)


def test_short_trajectory_raises():
    with pytest.raises(ValueError):
        rotation_power(_pure(6, n=12), 6, 24)
