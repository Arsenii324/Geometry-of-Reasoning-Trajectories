"""Unit tests for regime separation (signal vs arithmetic noise).

These use synthetic paths with a KNOWN construction, so a failure points at the
estimator rather than at the data. The real-data counterparts -- the actual
Huginn numbers these fixes were derived from -- live in test_audit_findings.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.regime import (
    classify_shape_regime,
    contraction_from_pair,
    converging_regime,
    empirical_floor,
    predicted_arithmetic_floor,
    step_cosine_converging,
)

_DIM = 32
_T = np.linspace(0, 6 * np.pi, 90)


def _circle() -> np.ndarray:
    return np.c_[np.cos(_T), np.sin(_T), np.zeros((90, _DIM - 2))] * 10


def _line() -> np.ndarray:
    return np.c_[np.linspace(0, 50, 90), np.zeros((90, _DIM - 1))]


def _decaying_spiral() -> np.ndarray:
    e = np.exp(-_T / 12)
    return np.c_[e * np.cos(_T) * 10, e * np.sin(_T) * 10, np.zeros((90, _DIM - 2))]


def _decay_then_floor(rho: float = 0.85, n_decay: int = 20, n_floor: int = 100,
                      dim: int = 64, floor: float = 1.0, seed: int = 0) -> np.ndarray:
    """A path that contracts geometrically, then jitters isotropically forever.

    This is the shape the audit found in the real trajectories: a short
    informative transient followed by a long stretch of numerical noise.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    v /= np.linalg.norm(v)
    states = [100.0 * v]
    for t in range(1, n_decay):
        states.append(100.0 * rho**t * v)
    for _ in range(n_floor):
        j = rng.normal(size=dim)
        states.append(floor * j / np.linalg.norm(j))
    return np.array(states)


def test_regime_ends_near_the_decay_length() -> None:
    """The converging regime must stop where the geometric decay stops."""
    a = _decay_then_floor(n_decay=20, n_floor=100)
    _, end = converging_regime(a)
    assert 10 <= end <= 30, f"regime end {end} should track the 20-step decay"


def test_regime_does_not_truncate_a_path_with_no_floor() -> None:
    """A path that never reaches a noise floor keeps (almost) all its steps."""
    n = 60
    t = np.linspace(0, 1, n)
    a = np.c_[t, t**2, np.zeros((n, 3))]
    _, end = converging_regime(a)
    assert end >= n // 2, "a steadily-moving path should not be cut short"


def test_empirical_floor_matches_the_constructed_floor() -> None:
    """Jitter of radius r gives steps of order r, not orders of magnitude off."""
    a = _decay_then_floor(floor=1.0, n_floor=200)
    f = empirical_floor(a)
    assert 0.5 < f < 4.0, f"floor estimate {f} is far from the constructed 1.0"


def test_predicted_floor_scales_with_mantissa_bits() -> None:
    """float32 must predict a far smaller rounding floor than bfloat16."""
    a = _decay_then_floor()
    bf16 = predicted_arithmetic_floor(a, mantissa_bits=8)
    fp32 = predicted_arithmetic_floor(a, mantissa_bits=23)
    assert bf16 > fp32
    assert bf16 / fp32 == pytest.approx(2.0**15, rel=1e-6)


def test_cosine_sees_glide_not_the_noise_tail() -> None:
    """THE load-bearing test for finding #2.

    A path that glides smoothly inward and then sits in isotropic noise has
    POSITIVE step cosine while it is moving and negative cosine in the noise.
    Averaging the second half reports the noise; the regime-restricted
    statistic must report the glide.
    """
    a = _decay_then_floor(rho=0.85, n_decay=25, n_floor=120)
    assert step_cosine_converging(a) > 0.5, "a straight glide must read positive"


def test_cosine_still_detects_a_genuine_zigzag() -> None:
    """The fix must not simply force the answer positive."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=32)
    v /= np.linalg.norm(v)
    w = rng.normal(size=32)
    w -= w.dot(v) * v
    w /= np.linalg.norm(w)
    states, x = [np.zeros(32)], np.zeros(32)
    for t in range(40):                       # genuine alternating zig-zag, decaying
        x = x + 0.9**t * (v if t % 2 == 0 else -v) + 0.15 * 0.9**t * w
        states.append(x.copy())
    assert step_cosine_converging(np.array(states)) < 0.0


def test_contraction_from_pair_recovers_a_known_rate() -> None:
    """Two orbits whose gap decays at a known rho must return that rho."""
    rng = np.random.default_rng(2)
    rho, dim, n = 0.80, 48, 60
    base = rng.normal(size=(n, dim)) * 0.0
    d0 = rng.normal(size=dim)
    d0 /= np.linalg.norm(d0)
    a = base.copy()
    b = np.array([base[t] + 50.0 * rho**t * d0 for t in range(n)])
    got, _ = contraction_from_pair(a, b)
    assert got == pytest.approx(rho, rel=0.05)


def test_contraction_from_pair_is_not_diluted_by_a_floor() -> None:
    """The whole point of finding #6: a plateau must not flatten the estimate."""
    rng = np.random.default_rng(3)
    rho, dim = 0.80, 48
    d0 = rng.normal(size=dim)
    d0 /= np.linalg.norm(d0)
    gap = [50.0 * rho**t for t in range(25)] + [1.0] * 100   # decay, then floor
    a = np.zeros((len(gap), dim))
    b = np.array([g * d0 for g in gap])
    got, floor = contraction_from_pair(a, b)
    assert got == pytest.approx(rho, rel=0.10), "floor must be excluded from the fit"
    assert floor == pytest.approx(1.0, rel=0.2)

    naive = np.exp(np.polyfit(np.arange(len(gap)), np.log(gap), 1)[0])
    assert naive > got + 0.05, "the naive whole-range fit should be visibly biased"


def test_shape_classifier_can_return_all_three_labels() -> None:
    """The property H1's original instrument lacks.

    `shapes.gate.classify_shape` returns "settle" for all 140 real
    trajectories because it compares the last step against the LARGEST step,
    and the largest is the h_0 transient (~72) while the last is the bf16
    floor (~1) -- a ratio that never exceeds 0.0188, against a 0.1 threshold.
    An instrument with only one reachable output cannot test a three-way
    hypothesis.
    """
    labels = {
        classify_shape_regime(_circle()),
        classify_shape_regime(_line()),
        classify_shape_regime(_decaying_spiral()),
    }
    assert labels == {"loop", "drift", "settle"}


def test_closed_orbit_is_a_loop() -> None:
    assert classify_shape_regime(_circle()) == "loop"


def test_straight_line_is_drift_not_loop() -> None:
    """Regression on a bug present in BOTH the original and my first version.

    Normalising the minimum far-pair distance by the cloud diameter is
    length-dependent: that minimum scales like min_lag * spacing, which
    shrinks as T grows, so a long straight path trips the loop threshold. The
    chord-to-arc ratio is exactly 1.0 for a line at every sampling density.
    """
    assert classify_shape_regime(_line()) == "drift"


def test_random_walk_is_drift_not_loop() -> None:
    """A diffusive path reaches chord/arc ~ 1/sqrt(L) ~ 0.11 at L=90, which
    must sit ABOVE the loop threshold -- ordinary wandering is not a return."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(90, _DIM)), axis=0)
    assert classify_shape_regime(walk) == "drift"


def test_decaying_spiral_settles_rather_than_looping() -> None:
    """It rotates, but it converges; 'settle' is the honest label."""
    assert classify_shape_regime(_decaying_spiral()) == "settle"


def test_chord_to_arc_is_invariant_to_sampling_density() -> None:
    """The property that fixes the length-dependence bug."""
    for n in (40, 90, 300):
        t = np.linspace(0, 50, n)
        line = np.c_[t, np.zeros((n, _DIM - 1))]
        assert classify_shape_regime(line) == "drift", f"failed at n={n}"


def test_short_inputs_do_not_raise() -> None:
    """Degenerate shapes return NaN/edge values rather than exploding."""
    tiny = np.zeros((3, 5))
    assert empirical_floor(tiny) == 0.0
    assert np.isnan(contraction_from_pair(tiny, tiny)[0])
    assert np.isnan(step_cosine_converging(np.zeros((2, 5))))
