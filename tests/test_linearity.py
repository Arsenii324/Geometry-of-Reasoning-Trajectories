"""The linearity probe must pass what is linear and fail what is not.

A held-out score is only worth reading if it can go DOWN. Half of these tests plant
nonlinear or chaotic dynamics and require the probe to report low predictability;
without them a high score on the real orbits would mean nothing, since a probe that
always says "linear" says nothing.
"""

from __future__ import annotations

import numpy as np

from traj_geom.metrics.linearity import linear_predictability, rank_sweep

DIM = 40


def _linear_orbit(n_step=80, n_dir=6, rho=0.87, seed=0, scale=70.0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(DIM, 2 * n_dir)))[0]
    t = np.arange(n_step, dtype=float)
    out = np.zeros((n_step, DIM))
    for i in range(n_dir):
        r, phi = rho * 0.93**i, 0.3 + 0.2 * i
        a = rng.normal() * scale
        out += np.outer(a * r**t * np.cos(phi * t), basis[:, 2 * i])
        out += np.outer(a * r**t * np.sin(phi * t), basis[:, 2 * i + 1])
    return out


def _pure_contraction(n_step=80, rho=0.87, seed=0, scale=70.0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM) * scale
    return np.array([v * rho**t for t in range(n_step)])


def _equal_modulus_orbit(n_step=80, n_dir=6, rho=0.87, seed=0, scale=70.0):
    """Every mode at the SAME modulus, so none dominates and rotation stays alive."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(DIM, 2 * n_dir)))[0]
    t = np.arange(n_step, dtype=float)
    out = np.zeros((n_step, DIM))
    for i in range(n_dir):
        phi = 0.4 + 0.35 * i
        a = rng.normal() * scale
        out += np.outer(a * rho**t * np.cos(phi * t), basis[:, 2 * i])
        out += np.outer(a * rho**t * np.sin(phi * t), basis[:, 2 * i + 1])
    return out


def _chaotic(n_step=80, seed=0, scale=70.0):
    """Steps whose directions are independent draws: nothing to predict."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_step, DIM))
    steps *= (scale * 0.87 ** np.arange(n_step))[:, None]
    return np.cumsum(steps, axis=0)


def test_a_linear_orbit_is_predicted_out_of_sample() -> None:
    out = linear_predictability(_linear_orbit(seed=1))
    assert out["ok"]
    assert out["r2_linear"] > 0.9, out


def test_a_chaotic_orbit_is_not() -> None:
    """THE TEST THAT GIVES THE OTHERS MEANING.

    Steps drawn independently carry no linear structure, so held-out prediction
    must fail. If it did not, the probe would be reporting its own capacity to
    interpolate rather than a property of the dynamics.
    """
    scores = [linear_predictability(_chaotic(seed=s))["r2_linear"] for s in range(6)]
    assert max(scores) < 0.5, scores


def test_a_pure_contraction_is_explained_by_the_scalar_model() -> None:
    """One parameter should suffice when there is no rotation, and the full
    operator must not do dramatically better -- that would mean it is fitting
    noise."""
    out = linear_predictability(_pure_contraction(seed=2))
    assert out["r2_scalar"] > 0.99, out
    assert out["r2_linear"] > 0.9


def test_rotation_is_where_the_full_operator_earns_its_parameters() -> None:
    """With several EQUAL-modulus rotating modes the scalar model must fall behind.

    Equal moduli is the point, and it corrects an expectation I had before
    measuring. On a spectrum whose moduli decay (0.87 x 0.93^i), the orbit collapses
    onto its leading mode within the window and consecutive steps become nearly
    parallel, so the one-parameter scalar model already scores 0.91 and the full
    operator can only add 0.09. That is not a defect -- it is D80's depth law
    showing up in the fit -- but it means the linear-vs-scalar gap is only large
    where several modes stay comparably alive.
    """
    equal = _linear_orbit(n_dir=6, seed=3, rho=0.87)
    out = linear_predictability(equal)
    assert out["r2_linear"] > out["r2_scalar"], out

    flat = _equal_modulus_orbit(seed=3)
    out2 = linear_predictability(flat)
    assert out2["r2_linear"] > out2["r2_scalar"] + 0.2, out2


def test_in_sample_fit_is_not_what_is_reported() -> None:
    """With 5280 dimensions and ~90 steps a least-squares operator fits the
    TRAINING steps exactly, so an in-sample score would read 1.0 for chaotic data
    too. Verified by giving the probe noise and requiring a low score anyway."""
    out = linear_predictability(_chaotic(seed=7))
    assert out["ok"]
    assert out["n_test"] >= 4
    assert out["r2_linear"] < 0.5


def test_normalisation_is_on_by_default_and_changes_the_question() -> None:
    """Unnormalised, the score is dominated by the decay envelope.

    Step norms fall ~0.87 per unroll, so the first training step is ~1000x the last
    test step; a raw fit would be scored almost entirely on magnitude. The default
    asks about DIRECTION, which is what every shape statistic here measures.
    """
    traj = _chaotic(seed=11)
    assert linear_predictability(traj)["normalised"] is True
    raw = linear_predictability(traj, normalise=False)
    norm = linear_predictability(traj, normalise=True)
    assert raw["r2_linear"] != norm["r2_linear"]


def test_rank_sweep_saturates_for_a_genuinely_low_rank_map() -> None:
    """Improvement should stop once the basis covers the active modes."""
    sweep = rank_sweep(_linear_orbit(n_dir=3, seed=5), ranks=(2, 4, 8, 16))
    assert len(sweep) >= 3
    best = max(s["r2_linear"] for s in sweep)
    at8 = [s for s in sweep if s["rank"] == 8]
    assert at8 and at8[0]["r2_linear"] > best - 0.15, sweep


def test_refuses_a_window_too_short_to_split() -> None:
    out = linear_predictability(np.random.default_rng(0).normal(size=(8, DIM)))
    assert out["ok"] is False


def test_in_basis_reports_the_ceiling_the_projection_imposes() -> None:
    """`in_basis` is how much of the held-out step the fitted subspace can express.

    Without it a low `r2_linear` is ambiguous: the operator may be wrong, or the
    test steps may simply live outside the directions the training steps spanned.
    Only the first is evidence about linearity.
    """
    out = linear_predictability(_linear_orbit(seed=9))
    assert 0.0 <= out["in_basis"] <= 1.0 + 1e-9
    assert out["r2_linear"] <= out["in_basis"] + 0.05, (
        "prediction cannot beat the subspace it is confined to")


def test_the_scalar_baseline_is_confined_to_the_same_subspace() -> None:
    """THE UNFAIR COMPARISON I SHIPPED FIRST, and what it would have concluded.

    `r2_scalar` predicts rho * delta_t in the FULL space, while `r2_linear` is
    confined to an r-dimensional basis. At small r the scalar model then wins
    trivially -- measured on the real orbits, it beat the rank-2 linear fit in
    119/152 cells -- which says the projection costs something, not that a single
    contraction rate describes the map better. `r2_scalar_in_basis` applies the
    same projection to both, and on that comparison the general operator wins
    148/152.
    """
    out = linear_predictability(_equal_modulus_orbit(seed=4), rank=2)
    assert out["ok"]
    assert out["r2_scalar_in_basis"] <= out["in_basis"] + 1e-9, (
        "the projected baseline cannot beat the subspace it is confined to")
    assert out["r2_linear"] > out["r2_scalar_in_basis"], out


def test_a_clamped_rank_is_reported_not_silently_substituted() -> None:
    """Requesting more basis than the training steps span must be visible.

    An unreported clamp turns a rank sweep into several groups labelled by ranks
    that were never used, which is how the first version of the real-data report
    grew spurious rank-17/19/20 rows.
    """
    out = linear_predictability(_linear_orbit(n_step=40, seed=6), rank=500)
    assert out["ok"]
    assert out["rank_requested"] == 500
    assert out["rank"] < 500
