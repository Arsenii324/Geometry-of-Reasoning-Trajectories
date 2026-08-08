"""Known-answer validation of the DMD estimator, before it is pointed at Huginn.

The claim this estimator will be used to make -- that the orbit obeys a different
operator than Arnoldi returns -- is only worth anything if the estimator recovers a
KNOWN operator when it is handed one. So every test here plants a spectrum and
checks it comes back, including in the two regimes that matter for the real data:
d >> T, and a non-normal operator.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.metrics.dmd import dmd_eigenvalues, pre_floor_window


def _orbit(a: np.ndarray, x0: np.ndarray, t: int) -> np.ndarray:
    xs = [x0]
    for _ in range(t - 1):
        xs.append(a @ xs[-1])
    return np.stack(xs)


def _rot(rho: float, phi: float) -> np.ndarray:
    return rho * np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def test_recovers_a_planted_rotation_rate_and_decay() -> None:
    """The core known answer: a decaying spiral with rho=0.87, phi=0.4 rad/step."""
    rho, phi = 0.87, 0.4
    traj = _orbit(_rot(rho, phi), np.array([1.0, 0.3]), 60)
    out = dmd_eigenvalues(traj)
    assert out["rho"] == pytest.approx(rho, abs=1e-6)
    assert out["arg"] == pytest.approx(phi, abs=1e-6)


def test_works_when_dimension_far_exceeds_snapshots() -> None:
    """The real regime: d=5280, T~30. The operator must still come back exactly."""
    rng = np.random.default_rng(0)
    d, rho, phi = 400, 0.9, 0.25
    q, _ = np.linalg.qr(rng.normal(size=(d, 2)))     # embed the plane in d dims
    traj2 = _orbit(_rot(rho, phi), np.array([1.0, 0.0]), 40)
    traj = traj2 @ q.T
    out = dmd_eigenvalues(traj)
    assert out["rho"] == pytest.approx(rho, abs=1e-6)
    assert out["arg"] == pytest.approx(phi, abs=1e-6)
    assert out["rank"] <= 4


def test_the_fixed_point_cancels_so_no_h_star_is_needed() -> None:
    """h* drops out of consecutive differences, which is why this is fit on them.

    Taking h_end as a stand-in for h* is the near-tautology `metrics/regime.py`
    records this project already committed once. Offsetting the whole orbit by an
    arbitrary constant must not move the spectrum.
    """
    rho, phi = 0.85, 0.3
    traj = _orbit(_rot(rho, phi), np.array([1.0, 0.2]), 50)
    shifted = traj + np.array([123.0, -45.0])
    a = dmd_eigenvalues(traj)
    b = dmd_eigenvalues(shifted)
    assert b["rho"] == pytest.approx(a["rho"], abs=1e-9)
    assert b["arg"] == pytest.approx(a["arg"], abs=1e-9)


def test_recovers_a_non_normal_operator() -> None:
    """Non-normality is the live explanation for the Arnoldi/orbit disagreement,
    so the estimator has to survive it. A strongly non-normal 2x2 with equal
    eigenvalues transiently GROWS before decaying; the answer must be the
    eigenvalues, not the transient."""
    a = np.array([[0.8, 20.0], [0.0, 0.8]])
    traj = _orbit(a, np.array([1.0, 1.0]), 60)
    out = dmd_eigenvalues(traj)
    assert out["rho"] == pytest.approx(0.8, abs=1e-5)


def test_energy_based_rank_selection_would_have_been_wrong_here() -> None:
    """Pins the reason rank is NOT chosen by explained energy.

    On this operator the second singular value carries 0.008% of the squared mass,
    so a 99.9%-energy rule truncates to rank 1 and returns 0.62 against a true 0.80
    -- a 22% error, in the same direction as the open Arnoldi-vs-orbit gap, with
    nothing to indicate a problem. Non-normality is precisely what creates that
    gap between energy and dynamics, and it is the case this estimator exists for.
    """
    a = np.array([[0.8, 20.0], [0.0, 0.8]])
    traj = _orbit(a, np.array([1.0, 1.0]), 60)
    assert dmd_eigenvalues(traj, rank=1)["rho"] == pytest.approx(0.62, abs=0.01)
    assert dmd_eigenvalues(traj, rank=2)["rho"] == pytest.approx(0.80, abs=1e-6)


def test_rank_sweep_reports_stability() -> None:
    """A flat sweep means the estimate is a property of the data, not the cut."""
    from traj_geom.metrics.dmd import dmd_rank_sweep
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(60, 2)))
    traj = _orbit(_rot(0.87, 0.4), np.array([1.0, 0.0]), 50) @ q.T
    sw = dmd_rank_sweep(traj)
    assert sw["stable"], sw
    assert all(abs(r - 0.87) < 1e-6 for r in sw["rho"])


def test_two_modes_are_both_recovered_and_ordered() -> None:
    a = np.zeros((4, 4))
    a[:2, :2] = _rot(0.9, 0.2)
    a[2:, 2:] = _rot(0.5, 1.1)
    traj = _orbit(a, np.array([1.0, 0.0, 1.0, 0.0]), 60)
    out = dmd_eigenvalues(traj)
    mods = np.abs(out["eigs"])
    assert mods[0] == pytest.approx(0.9, abs=1e-6)
    assert np.isclose(mods, 0.5, atol=1e-6).any()
    assert (np.diff(mods) <= 1e-12).all(), "eigenvalues must be sorted by modulus"


def test_real_spectrum_reports_no_argument() -> None:
    """arg is undefined without a complex pair, and must be None rather than 0.0."""
    traj = _orbit(np.diag([0.9, 0.4]), np.array([1.0, 1.0]), 40)
    assert dmd_eigenvalues(traj)["arg"] is None


def test_pre_floor_window_excludes_an_arithmetic_floor() -> None:
    """D28's lesson: a statistic over a window that mixes signal with the bf16
    floor reports the window. The window must be found from the data."""
    rng = np.random.default_rng(1)
    good = _orbit(_rot(0.8, 0.3), np.array([1.0, 0.0]), 30)
    floor = 1e-7 * rng.normal(size=(40, 2))
    traj = np.vstack([good, good[-1] + floor])
    a, b = pre_floor_window(traj)
    assert a == 0
    assert 20 <= b <= 34, f"window {a}:{b} should stop near the end of the signal"


def test_rank_cap_is_honoured() -> None:
    rng = np.random.default_rng(2)
    traj = rng.normal(size=(50, 100))
    assert dmd_eigenvalues(traj, max_rank=5)["rank"] <= 5


def test_rejects_too_few_snapshots() -> None:
    with pytest.raises(ValueError):
        dmd_eigenvalues(np.zeros((2, 5)))


def test_precision_floor_is_zero_for_genuinely_fp32_data() -> None:
    """It must not shorten a window that carries real fp32 signal."""
    from traj_geom.metrics.dmd import precision_floor
    rng = np.random.default_rng(21)
    traj = np.cumsum(rng.normal(size=(40, 64)), axis=0)
    assert precision_floor(traj) == 0.0


def test_precision_floor_excludes_the_bf16_dead_regime() -> None:
    """The tail-based floor alone is NOT enough on bf16-stored data.

    Consecutive rounding errors partially cancel, so observed step norms in the
    dead regime sit BELOW the rounding scale that produced them, and a tail-based
    floor happily admits them. Measured on a real orbit: bf16 storage gave an
    85-step window when the step norm crossed the rounding scale at unroll 39, and
    a statistic over that window reports the arithmetic (D28's failure mode).
    """
    import torch

    from traj_geom.metrics.dmd import pre_floor_window, precision_floor

    rng = np.random.default_rng(22)
    d = 512
    v = rng.normal(size=d)
    v *= 76.4 / np.linalg.norm(v)                     # Huginn's state norm
    step = rng.normal(size=(120, d))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    traj = v[None, :] + np.cumsum(step * (0.8 ** np.arange(120))[:, None] * 5.0, axis=0)
    bf = torch.from_numpy(traj.astype(np.float32)).to(torch.bfloat16)
    bf16 = bf.to(torch.float32).numpy().astype(np.float64)

    assert precision_floor(bf16) > 0.0
    _, hi_bf = pre_floor_window(bf16)
    _, hi_fp = pre_floor_window(traj)
    assert hi_bf < hi_fp, (hi_bf, hi_fp)
    # and the cut must land near where the signal actually crosses the floor
    steps = np.linalg.norm(np.diff(bf16, axis=0), axis=1)
    crossing = int(np.argmax(steps < precision_floor(bf16)))
    assert abs(hi_bf - crossing) < 0.5 * max(crossing, 1), (hi_bf, crossing)

