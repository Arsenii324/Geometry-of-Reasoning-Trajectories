"""Guards D41, D42 and D43 — the trained-vs-untrained contraction results.

Every number here is read from the kernel outputs in `scratch/kaggle_*/out/`,
never transcribed, so prose and data cannot drift apart (the failure mode that
already bit this project once: claims_ledger D21, error 1).

Two of these tests exist to encode a RETRACTION, in the same spirit as
`test_ratio_symmetry_is_an_algebraic_identity`:

  * `test_accumulator_model_does_not_fit` pins the mechanism I proposed in
    D41(4) and had to withdraw. Without it, "re-injection makes the untrained
    net a coherent accumulator" is an attractive story that reads as though it
    were measured.
  * `test_r1_gap_is_below_the_level_ceiling` pins the scoping error in the first
    version of D41: the 3.4x gap at r=1 is a difference at COARSE resolution and
    is not evidence either model has the count after one unroll.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED = os.path.join(ROOT, "scratch", "kaggle_readout", "out", "readout_vs_depth.json")
UNTRAINED = os.path.join(ROOT, "scratch", "kaggle_untrained_depth", "out",
                         "readout_vs_depth.json")

M = 64
N_SEEDS = 44
P_ONES = (0.2, 0.35, 0.5, 0.65, 0.8)


def _curve(path: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(os.path.dirname(os.path.dirname(path)))} output absent")
    rows = json.load(open(path, encoding="utf-8"))
    return (np.array([r["unroll"] for r in rows], float),
            np.array([r["r2"] for r in rows], float), rows)


def _counts() -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the realised counts — the kernel's generator, reproduced exactly."""
    counts, levels = [], []
    for p in P_ONES:
        for s in range(N_SEEDS):
            rng = random.Random(s * 7919 + int(p * 1000))
            counts.append(sum(1 if rng.random() < p else 0 for _ in range(M)))
            levels.append(p)
    return np.array(counts, float), np.array(levels)


def _level_ceiling() -> float:
    """R^2 a decoder that knows only p_one could reach. D41(8)."""
    c, lv = _counts()
    within = np.mean([c[lv == p].var(ddof=1) for p in P_ONES])
    return float(1 - within / c.var())


# --- the confound control that licenses D41 -------------------------------


def test_level_only_ceiling_is_what_d41_claims() -> None:
    assert _level_ceiling() == pytest.approx(0.9237, abs=5e-4)


def test_both_models_beat_the_level_ceiling() -> None:
    """If either failed this, that model would be decoding p_one, not the count."""
    ceil = _level_ceiling()
    for path, name in ((TRAINED, "trained"), (UNTRAINED, "untrained")):
        _, r2, _ = _curve(path)
        assert r2.max() > ceil, f"{name} never resolves count within level"


def test_untrained_is_the_more_precisely_decoded_model() -> None:
    """D41's headline. If this ever flips, the whole result is gone."""
    _, tr, _ = _curve(TRAINED)
    _, un, _ = _curve(UNTRAINED)
    assert un.max() > tr.max(), (
        f"untrained {un.max():.4f} should exceed trained {tr.max():.4f}"
    )


def test_r1_gap_is_below_the_level_ceiling() -> None:
    """Encodes D41's scoping correction — see this module's docstring."""
    ceil = _level_ceiling()
    _, tr, _ = _curve(TRAINED)
    _, un, _ = _curve(UNTRAINED)
    assert tr[0] < ceil and un[0] < ceil, (
        "both r=1 values must sit BELOW the level ceiling; if one rises above it "
        "the 'coarse resolution only' caveat in D41(1) must be rewritten"
    )


def test_permutation_null_is_negative_so_the_probe_is_not_interpolating() -> None:
    """The anti-interpolation defence, D41(6). d=5280 >> n=220 makes this the
    first objection; a negative permuted null is what answers it."""
    for path, name in ((TRAINED, "trained"), (UNTRAINED, "untrained")):
        _, _, rows = _curve(path)
        nulls = [r["null_mean"] for r in rows if "null_mean" in r]
        if not nulls:
            pytest.skip(f"{name} run recorded no permutation null")
        assert max(nulls) < 0, (
            f"{name} permuted null reaches {max(nulls):+.4f}; a null at or above "
            "zero would mean the probe has capacity to interpolate"
        )


# --- D42: which mechanism fits, and which does not ------------------------


def test_accumulator_model_does_not_fit() -> None:
    """WHY D41(4)'s mechanism was retracted — see this module's docstring.

    Coherent signal against quadrature noise gives
    ``R2_r = A^2 u_r / (1 + A^2 u_r)`` with ``u_r = (1-rho^r)/(1+rho^r)``.
    Fitted to the real curves it pins rho at the upper bound and leaves large
    systematic residuals. Recorded so the story is not reintroduced as a finding.
    """
    from scipy.optimize import curve_fit

    def model(r, amp, rho):
        u = (1 - rho ** r) / (1 + rho ** r)
        return amp * amp * u / (1 + amp * amp * u)

    for path, name in ((TRAINED, "trained"), (UNTRAINED, "untrained")):
        r, y, _ = _curve(path)
        yc = np.clip(y, 1e-9, 1 - 1e-9)
        (a, rho), _ = curve_fit(model, r, yc, p0=[3.0, 0.85],
                                bounds=([1e-3, 0.05], [1e6, 0.999]), maxfev=200000)
        resid = yc - model(r, a, rho)
        assert rho > 0.99, f"{name}: accumulator fit no longer pins rho at the bound"
        assert np.abs(resid).max() > 0.04, (
            f"{name}: accumulator residuals are now small ({np.abs(resid).max():.4f}); "
            "if this fires the retraction in D41(4)/D42(1) must be revisited"
        )


def test_convergence_law_recovers_the_two_contraction_rates() -> None:
    """D42(2). The trained value must stay consistent with the three independent
    direct measurements (orbit 0.85-0.90, Arnoldi 0.79-0.81, rotation ~0.86)."""
    from traj_geom.observable_convergence import estimate_rho_from_curve

    r, tr, _ = _curve(TRAINED)
    r_u, un, _ = _curve(UNTRAINED)
    rho_t, fit_t = estimate_rho_from_curve(r, tr.max() - tr, floor=0.0)
    rho_u, fit_u = estimate_rho_from_curve(r_u, 1.0 - un, floor=0.0)

    assert 0.85 <= rho_t <= 0.95, f"trained rho drifted to {rho_t:.3f}"
    assert 0.55 <= rho_u <= 0.75, f"untrained rho drifted to {rho_u:.3f}"
    assert rho_u < rho_t - 0.15, (
        f"D42 requires the untrained operator to contract materially FASTER; "
        f"got untrained {rho_u:.3f} vs trained {rho_t:.3f}"
    )
    assert fit_t > 0.85 and fit_u > 0.90, (
        f"log-linear fit quality collapsed ({fit_t:.3f}, {fit_u:.3f}); below ~0.9 "
        "the law does not apply and the rates are not measurements"
    )


# --- D43: the bound against Geiping et al.'s published saturation ----------

# Read directly from the paper, recorded in docs/deep_research_huginn_literature.md
# lines 101-116. The LARGEST reported saturation point is what the bound must clear.
PUBLISHED_SATURATION = {"HellaSwag": 8, "ARC-C (no few-shot)": 12,
                        "ARC-C (25-50 few-shot)": 32, "GSM8K": 32}


def test_published_saturation_respects_the_contraction_bound() -> None:
    """D43(1). State convergence is necessary for accuracy saturation, so no task
    may saturate LATER than the state settles. One-sided: earlier is fine."""
    from traj_geom.observable_convergence import estimate_rho_from_curve

    r, tr, _ = _curve(TRAINED)
    rho, _ = estimate_rho_from_curve(r, tr.max() - tr, floor=0.0)
    tau = -1.0 / np.log(rho)
    bound = 3.0 * tau                       # 95% settled
    worst = max(PUBLISHED_SATURATION.values())
    assert worst <= bound + 1e-9, (
        f"a task saturates at r={worst} but the state is 95% settled by "
        f"r={bound:.1f}; D43's bound is violated and the claim fails"
    )


def test_separation_survives_every_fit_window() -> None:
    """The fit window is an analyst degree of freedom, so it must not be load-bearing.

    D42's trained all-r fit is 0.8972, marginally under the applicability bar
    `observable_convergence.py` sets for itself, and the natural fix — drop r=1,
    which is not yet in the linear regime the law assumes — moves rho. If the
    trained/untrained separation depended on that choice it would not be a result.
    """
    from traj_geom.observable_convergence import estimate_rho_from_curve

    r_t, tr, _ = _curve(TRAINED)
    r_u, un, _ = _curve(UNTRAINED)
    for lo in (1, 2):
        mt, mu = r_t >= lo, r_u >= lo
        rho_t, _ = estimate_rho_from_curve(r_t[mt], (tr.max() - tr)[mt], floor=0.0)
        rho_u, _ = estimate_rho_from_curve(r_u[mu], (1.0 - un)[mu], floor=0.0)
        assert rho_t - rho_u > 0.20, (
            f"window r>={lo}: separation collapsed to {rho_t - rho_u:+.4f}"
        )


def test_d43_bound_holds_under_every_fit_window() -> None:
    """D43 must not depend on the window either — see the test above."""
    from traj_geom.observable_convergence import estimate_rho_from_curve

    r_t, tr, _ = _curve(TRAINED)
    worst = max(PUBLISHED_SATURATION.values())
    for lo in (1, 2, 4):
        m = r_t >= lo
        if ((tr.max() - tr)[m] > 1e-4).sum() < 3:
            continue
        rho, _ = estimate_rho_from_curve(r_t[m], (tr.max() - tr)[m], floor=0.0)
        bound = 3.0 * (-1.0 / np.log(rho))
        assert worst <= bound, (
            f"window r>={lo}: task saturates at r={worst} but state settles by "
            f"r={bound:.1f}"
        )


def test_untrained_rho_rests_on_few_points() -> None:
    """Documents a real limitation rather than leaving it to be discovered.

    The untrained curve saturates so fast that only ~4 unrolls sit above the
    floor, so its rho is fitted on r in {1,2,4,8}. That is enough for a slope but
    it is not a lot, and the queued operator-level sweep is what replaces it.
    """
    _, un, _ = _curve(UNTRAINED)
    n_above = int(((1.0 - un) > 1e-4).sum())
    assert n_above <= 6, (
        f"untrained curve now has {n_above} points above the floor; if this grew, "
        "the 'rests on 4 points' caveat in D42(2) is stale and should be relaxed"
    )
