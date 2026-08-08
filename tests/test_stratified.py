"""Validation of the stratified null, before B6's data exists.

The single most important test here is `test_refuses_the_d72_design`: if the null
returns a number when no stratum carries both classes, the whole point of the
re-run is lost and B6 fails the same way twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from traj_geom.analysis.stratified import (
    attainable,
    stratified_diff,
    stratified_permutation_p,
)


def _design(n_per_stratum=8, n_strata=3, effect=0.0, seed=0):
    """Balanced strata, half each class, with a planted within-stratum effect.

    Stratum MEANS are deliberately far apart, so a test that pools across strata
    instead of conditioning on them would be swamped by between-stratum variance.
    """
    rng = np.random.default_rng(seed)
    vals, labs, strat = [], [], []
    for s in range(n_strata):
        base = 100.0 * s                      # huge between-stratum offset
        for i in range(n_per_stratum):
            is_pos = i < n_per_stratum // 2
            vals.append(base + effect * is_pos + rng.normal(0, 1))
            labs.append(is_pos)
            strat.append(f"g{s}")
    return np.array(vals), np.array(labs), np.array(strat)


def test_refuses_the_d72_design() -> None:
    """No stratum carries both classes -> the test must REFUSE, not return a number.

    This is D72 exactly: `add1`'s correct and incorrect gold sets were disjoint, so
    correctness was a function of the answer value and no permutation could separate
    "geometry tracks correctness" from "geometry tracks the prompt".
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([True, True, False, False])
    strata = np.array(["a", "b", "c", "d"])       # every stratum is one item
    out = stratified_permutation_p(values, labels, strata, n_perm=200)
    assert out["usable"] is False
    assert "both classes" in out["why"]


def test_partially_usable_design_uses_only_the_mixed_strata() -> None:
    """A stratum with one class contributes nothing, and n_used says so."""
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    labels = np.array([True, False, True, True, False])
    strata = np.array(["a", "a", "b", "c", "c"])   # only a and c are mixed
    diff, n_used = stratified_diff(values, labels, strata)
    assert n_used == 4, "the single-class stratum b must not be counted"
    assert diff == pytest.approx(((1.0 - 3.0) * 2 + (7.0 - 9.0) * 2) / 4)


def test_finds_a_planted_within_stratum_effect() -> None:
    v, lab, s = _design(effect=3.0, seed=1)
    out = stratified_permutation_p(v, lab, s, n_perm=2000, seed=0)
    assert out["usable"]
    assert out["obs"] == pytest.approx(3.0, abs=1.0)
    assert out["p"] < 0.01, out


def test_does_not_invent_an_effect() -> None:
    v, lab, s = _design(effect=0.0, seed=2)
    assert stratified_permutation_p(v, lab, s, n_perm=2000, seed=0)["p"] > 0.05


def test_is_calibrated_under_the_null() -> None:
    """p<0.05 must fire near 5% of the time, not far above it."""
    hits = 0
    trials = 60
    for k in range(trials):
        v, lab, s = _design(effect=0.0, seed=100 + k)
        hits += stratified_permutation_p(v, lab, s, n_perm=400, seed=k)["p"] < 0.05
    assert hits / trials < 0.18, f"rejects at {hits / trials:.0%} under the null"


def test_between_stratum_offsets_do_not_leak_into_the_statistic() -> None:
    """The reason to stratify: a 100x between-stratum offset must not register.

    An unstratified comparison on this design would be dominated by which strata
    happen to hold which class -- which is the D72 confound in another costume.
    """
    v, lab, s = _design(effect=0.0, seed=3)
    diff, _ = stratified_diff(v, lab, s)
    assert abs(diff) < 2.0, f"between-stratum offset leaked: {diff}"


def test_floor_guard_matches_the_kernel_library() -> None:
    """geometry-correctness was drafted with a floor ABOVE its own alpha (D72)."""
    ok, floor = attainable(0.05 / 12, 200)
    assert not ok and floor == pytest.approx(1 / 201)
    ok2, floor2 = attainable(0.05 / 12, 20000)
    assert ok2 and floor2 < 0.05 / 12


def test_reported_p_never_undercuts_its_own_floor() -> None:
    """(hits+1)/(n+1) can never return 0, which is what makes the floor meaningful."""
    v, lab, s = _design(effect=50.0, seed=4)
    out = stratified_permutation_p(v, lab, s, n_perm=200, seed=0)
    assert out["p"] >= out["p_floor"]
