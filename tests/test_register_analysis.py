"""Tests for the register analysis — the project's strongest results (D34, D36).

Two kinds here. Synthetic tests establish that the estimators recover a planted
register and reject a planted non-register. Real-data tests pin the published
numbers so they cannot drift.

The most important test is `test_ratio_symmetry_is_an_algebraic_identity`: it
encodes WHY the first symmetry test was retracted, so nobody (including me)
reintroduces it as a measurement.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scripts.run_register_analysis import (
    increment_direction,
    pair_displacement,
    register_correlation,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "register_analysis.csv")
DIM = 64


def _planted(n: int = 40, seed: int = 0, step: float = 1.0, noise: float = 0.05):
    """A synthetic sequence carrying a real register along a known direction."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM)
    v /= np.linalg.norm(v)
    bits = rng.integers(0, 2, n)
    value = np.cumsum(np.where(bits == 1, 1, -1)).astype(float)
    states = np.stack([value[i] * step * v + noise * rng.normal(size=DIM) for i in range(n)])
    return states, bits.astype(bool), value, v


# --- the estimators recover a planted register ----------------------------


def test_increment_direction_recovers_the_planted_axis() -> None:
    states, inc, _, v = _planted(noise=0.02)
    got, _ = increment_direction(states, inc)
    assert abs(float(got @ v)) > 0.9, "should recover the planted direction"


def test_register_correlation_is_high_for_a_planted_register() -> None:
    states, inc, value, _ = _planted(noise=0.02)
    got, off = increment_direction(states, inc)
    r = register_correlation(states, inc, value, off, got)
    assert abs(r) > 0.5, f"planted register should be recovered, got r={r}"


def test_register_correlation_is_near_zero_when_there_is_no_register() -> None:
    """THE specificity control: states carrying only the CURRENT symbol, no memory.

    If the estimator fired here it would be measuring token identity, which is
    exactly the confound D34/D36 had to rule out.
    """
    rng = np.random.default_rng(3)
    n = 40
    v = rng.normal(size=DIM)
    v /= np.linalg.norm(v)
    bits = rng.integers(0, 2, n)
    value = np.cumsum(np.where(bits == 1, 1, -1)).astype(float)
    # state depends ONLY on the current bit -- no accumulation whatsoever
    states = np.stack([(1.0 if b else -1.0) * v + 0.05 * rng.normal(size=DIM) for b in bits])
    got, off = increment_direction(states, bits.astype(bool))
    r = register_correlation(states, bits.astype(bool), value, off, got)
    assert abs(r) < 0.3, f"no-memory states should not show a register, got r={r}"


def test_matched_pairs_return_to_start_for_a_planted_register() -> None:
    states, inc, _, _ = _planted(noise=0.01)
    got, off = increment_direction(states, inc)
    syms = np.array(["1" if b else "0" for b in inc])
    d = pair_displacement(states, syms, off, got)
    matched = np.mean([d["pair_01"], d["pair_10"]])
    same = np.mean([abs(d["pair_11"]), abs(d["pair_00"])])
    assert abs(matched) < 0.25 * same, "a matched pair must roughly cancel"
    assert d["pair_11"] > 0 > d["pair_00"], "same-symbol pairs must have opposite signs"


# --- the retracted test, encoded so it is not reintroduced -----------------


def test_ratio_symmetry_is_an_algebraic_identity() -> None:
    """WHY the first symmetry test was vacuous — see D36(5).

    Measuring displacements relative to the MEAN step and taking their ratio
    returns n_increment/n_decrement exactly, for ANY data, because the mean is a
    weighted average of the two group means. It reported 0.96875 for balanced
    strings, which is exactly 31/32, with sd 1.9e-16 across 16 random strings.
    """
    rng = np.random.default_rng(5)
    for n_inc, n_dec in ((31, 32), (10, 40), (25, 25)):
        d = rng.normal(size=(n_inc + n_dec, DIM))
        inc = np.zeros(n_inc + n_dec, dtype=bool)
        inc[:n_inc] = True
        v = d[inc].mean(0) - d[~inc].mean(0)
        v /= np.linalg.norm(v)
        base = d.mean(0) @ v
        po = (d[inc] @ v).mean() - base
        pc = (d[~inc] @ v).mean() - base
        assert -pc / po == pytest.approx(n_inc / n_dec, rel=1e-9), (
            "the ratio is n_inc/n_dec by algebra, independent of the data — "
            "it measures the symbol counts, not the model"
        )


# --- real-data regression tests -------------------------------------------


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    if not os.path.exists(CSV):
        pytest.skip("register_analysis.csv not present (run scripts.run_register_analysis)")
    return pd.read_csv(CSV)


def test_all_three_conditions_present(real: pd.DataFrame) -> None:
    assert set(real.kind) == {"a", "b_bal", "b_unbal"}


def test_offset_selection_agrees_within_each_condition(real: pd.DataFrame) -> None:
    """If strings disagreed on the offset, the ||v||-maximising search would be
    fitting per-string noise rather than finding the symbol region."""
    for kind, g in real.groupby("kind"):
        assert g.offset.nunique() == 1, f"{kind} offsets disagree: {sorted(set(g.offset))}"


def test_register_correlation_is_positive_in_every_string(real: pd.DataFrame) -> None:
    """The sign consistency that makes v a direction rather than 48 unrelated fits."""
    for kind, g in real.groupby("kind"):
        assert (g.register_r > 0).all(), f"{kind} has sign flips: {g.register_r.tolist()}"
        assert g.register_r.mean() > 0.15, f"{kind} mean r too low: {g.register_r.mean()}"


def test_task_b_same_symbol_pairs_have_opposite_signs(real: pd.DataFrame) -> None:
    for kind in ("b_bal", "b_unbal"):
        g = real[real.kind == kind]
        assert g["pair_(("].mean() > 0 > g["pair_))"].mean()


def test_task_b_matched_pairs_return_to_start(real: pd.DataFrame) -> None:
    """The Z-action property — D36(4). Not forced by the construction of v."""
    for kind in ("b_bal", "b_unbal"):
        g = real[real.kind == kind]
        matched = np.concatenate([g["pair_()"].values, g["pair_)("].values])
        same = np.mean(np.abs(np.concatenate([g["pair_(("].values, g["pair_))"].values])))
        assert abs(matched.mean()) < 0.15 * same, (
            f"{kind}: matched pairs should cancel relative to same-symbol pairs"
        )


# --- D50: the window bug, encoded so the retraction cannot be undone --------


def test_digit_region_starts_at_index_three_not_four() -> None:
    """WHY D34/D36's `register_r` was superseded — see claims_ledger D50.

    The saved `token_ids` put the 64 digit tokens at indices 3..66, but both the
    ||v||-maximising offset search and the baseline kernel's hardcode select 4.
    The margin the search decides on is only ~9% in ||v||, which was never enough
    to fix an alignment that `meta.json` recorded exactly.
    """
    import json

    meta_path = os.path.join(ROOT, "scratch", "kaggle_states", "out2", "meta.json")
    if not os.path.exists(meta_path):
        pytest.skip("saved states absent")
    meta = [m for m in json.load(open(meta_path, encoding="utf-8")) if m["kind"] == "a"]
    assert meta, "no task-a trajectories"
    for m in meta:
        ids = np.asarray(m["token_ids"])
        digits = np.where((ids == 349) | (ids == 345))[0]
        assert int(digits[0]) == 3, (
            f"{m['name']}: digits start at {digits[0]}, not 3; D50's window "
            "correction assumes 3"
        )
        assert len(digits) == 64


def test_register_r_does_not_survive_the_window_correction() -> None:
    """D50(1). If this ever fires, D34/D36 can be reinstated rather than superseded."""
    csv = os.path.join(ROOT, "results", "register_window_recheck.csv")
    if not os.path.exists(csv):
        pytest.skip("run scripts.recheck_register_window")
    df = pd.read_csv(csv)
    assert df.r_buggy_window.mean() > 0.20, "the buggy value no longer reproduces"
    assert df.r_correct_window.mean() < 0.15, (
        f"corrected register_r is now {df.r_correct_window.mean():.4f}; D50 says "
        "the correlation does not survive the window correction"
    )
    assert not (df.r_correct_window > 0).all(), (
        "corrected values are sign-consistent again; D50's key contrast with the "
        "buggy window (16/16 positive) would need restating"
    )


def test_v_is_the_current_token_contrast() -> None:
    """D50(3). v is 88x chance aligned with token identity, in the right space."""
    csv = os.path.join(ROOT, "results", "register_window_recheck.csv")
    if not os.path.exists(csv):
        pytest.skip("run scripts.recheck_register_window")
    df = pd.read_csv(csv)
    chance = float(np.sqrt(2.0 / (np.pi * 5280)))
    assert df.cos_v_residual_contrast.mean() > 20 * chance, (
        f"|cos(v, current-token contrast)| fell to "
        f"{df.cos_v_residual_contrast.mean():.4f}; D50(3) reports 0.974"
    )


def test_the_real_register_is_architectural() -> None:
    """D53. The untrained arm must keep carrying the count as well as the trained one.

    Only the PAIRED arms are comparable: the two trained measurements available
    differ by 0.065 across runs, twice the 0.032 between-arm difference, so
    run-to-run variation exceeds the effect and the unpaired number cannot be used.
    """
    import json as _json

    path = os.path.join(ROOT, "results", "register_arms.json")
    if not os.path.exists(path):
        pytest.skip("run scripts.recheck_register_window")
    arms = {a["arm"]: a for a in _json.load(open(path, encoding="utf-8"))}
    if "trained (paired)" not in arms or "untrained (paired)" not in arms:
        pytest.skip("paired arms not captured")
    t, u = arms["trained (paired)"], arms["untrained (paired)"]
    assert u["cv_r2"] > 0.5, f"untrained register collapsed to {u['cv_r2']:.4f}"
    assert u["cv_r2"] > t["cv_r2"] - 0.10, (
        f"trained {t['cv_r2']:.4f} now exceeds untrained {u['cv_r2']:.4f} by more "
        "than 0.10; D53 says the register is architectural"
    )
    for a in (t, u):
        assert a["cos_v_vs_decoder"] < 0.01, (
            "D34/D36's v is no longer orthogonal to the count direction; D50(5) "
            "and D53(2) both rest on it being so, in BOTH arms"
        )
