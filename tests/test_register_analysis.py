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
