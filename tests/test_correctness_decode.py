"""Validation of the within-family correctness decode, on planted data.

The two ways this analysis could lie are both planted here:

  * it could "detect" correctness by recognising the FAMILY and betting on its base
    rate -- D84 showed family is decodable at 100%, and families run 0% to 100%
    accurate (D85), so that shortcut is worth a lot of balanced accuracy and
    nothing at all scientifically;
  * it could report a null from a design with no reach, which is D70's failure and
    the reason the power curve exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts.run_correctness_decode import (
    centre_within_family,
    decode_correctness,
    detection_power,
    report,
    usable_families,
)

NF = 6


def _bank(n_per=24, dim=40, effect=0.0, family_signal=0.0, seed=0,
          acc_by_family=None):
    """Orbits with a planted within-family effect and a separate family signal.

    `effect` shifts the CORRECT class along a random direction, in units of the
    feature sd -- the thing the analysis is meant to find. `family_signal` shifts
    every orbit of a family along its own direction, which the analysis must be
    blind to.
    """
    rng = np.random.default_rng(seed)
    acc = acc_by_family or {f"f{i}": 0.25 + 0.1 * i for i in range(NF)}
    v_eff = rng.normal(size=dim)
    v_eff /= np.linalg.norm(v_eff)
    rows = []
    for i, (fam, p) in enumerate(sorted(acc.items())):
        v_fam = rng.normal(size=dim)
        v_fam /= np.linalg.norm(v_fam)
        n_pos = int(round(p * n_per))
        for j in range(n_per):
            ok = j < n_pos
            x = rng.normal(size=dim) + family_signal * v_fam + effect * ok * v_eff
            rows.append({"bank": "b", "tag": f"{fam}_{j}", "_group": "b",
                         "family": fam, "correct": bool(ok), "n_tokens": 30 + i,
                         "shape": x, "position": x.copy()})
    return pd.DataFrame(rows)


def test_centering_removes_the_family_signal_entirely() -> None:
    """A family offset must not survive centring; otherwise the classifier can use
    it, and 'correct' becomes 'came from a family with a high base rate'."""
    df = _bank(family_signal=25.0, seed=1)
    x = centre_within_family(df, "shape")
    fams = df["family"].to_numpy()
    means = np.stack([x[fams == f].mean(axis=0) for f in np.unique(fams)])
    assert np.abs(means).max() < 1e-9, "family means survive centring"


def test_a_family_only_signal_is_not_read_as_correctness(monkeypatch) -> None:
    """THE SHORTCUT THIS DESIGN EXISTS TO BLOCK.

    Families differ hugely in base rate, so an uncentred classifier that merely
    recognises the family scores well above chance on `correct` while knowing
    nothing about correctness. With centring it must return a null.
    """
    df = _bank(effect=0.0, family_signal=40.0, seed=2)
    out = decode_correctness(df, "shape", n_perm=120, seed=0)
    assert out["usable"]
    assert out["p"] > 0.05, out
    assert out["balanced_accuracy"] < 0.62, out


def test_a_planted_within_family_effect_is_found() -> None:
    """Positive control: without it a null would be indistinguishable from a
    pipeline that returns nulls unconditionally."""
    out = decode_correctness(_bank(effect=1.5, seed=3), "shape", n_perm=120)
    assert out["usable"] and out["p"] < 0.05, out
    assert out["balanced_accuracy"] > 0.65, out


def test_the_test_is_calibrated_under_the_null() -> None:
    """Rejection rate near alpha over many seeds, NOT a null on one seed.

    The first version of this asserted p > 0.05 at seed 4 and failed at p = 0.0398,
    which is not a defect: a calibrated test at alpha = 0.05 fires on about one draw
    in twenty, and demanding otherwise tests the seed rather than the estimator.
    The same correction was needed for the ICC recovery test in
    `tests/test_reliability.py`. The real run's power curve agrees -- 5% detection
    at a planted effect of exactly zero.
    """
    hits, trials = 0, 20
    for k in range(trials):
        out = decode_correctness(_bank(effect=0.0, seed=100 + k), "shape",
                                 n_perm=60, seed=k)
        hits += out["usable"] and out["p"] < 0.05
    assert hits / trials < 0.25, f"rejects at {hits / trials:.0%} under the null"


def test_single_class_families_are_excluded_not_counted() -> None:
    """A 100%-correct family carries no within-family contrast, and counting it
    would inflate n while adding nothing -- the accounting `stratified_diff`
    applies when it drops single-class strata."""
    acc = {"pure_hi": 1.0, "pure_lo": 0.0, "mixed_a": 0.5, "mixed_b": 0.4}
    df = _bank(acc_by_family=acc, seed=5)
    keep = usable_families(df)
    assert set(keep) == {"mixed_a", "mixed_b"}
    out = decode_correctness(df, "shape", n_perm=60)
    assert out["n"] == 48 and out["n_families"] == 2


def test_the_power_curve_rises_with_the_planted_effect() -> None:
    """The floor is measured, not assumed. A curve that did not rise would mean the
    simulation is not injecting what it claims."""
    df = _bank(effect=0.0, seed=6)
    p = detection_power(df, "shape", effects=(0.0, 2.0), n_trials=6, n_perm=30)
    lo = float(p[p["effect_sd"] == 0.0]["detected"].iloc[0])
    hi = float(p[p["effect_sd"] == 2.0]["detected"].iloc[0])
    assert lo <= 0.35, f"false-positive rate {lo:.0%} at zero effect"
    assert hi > lo, p


def test_report_states_the_bound_rather_than_only_the_p_value() -> None:
    """A p-value with no reach behind it is D70's mistake in another costume."""
    df = _bank(effect=0.0, seed=7)
    res = decode_correctness(df, "shape", n_perm=60)
    power = pd.DataFrame({"effect_sd": [0.0, 0.5, 2.0], "n_trials": [5] * 3,
                          "detected": [0.0, 0.2, 1.0]})
    txt = report(res, power)
    assert "rules out a within-family shape difference of 2.00 sd" in txt


def test_report_refuses_when_nothing_was_detectable() -> None:
    """If no planted effect reaches 80%, the design cannot support a null at all --
    D79(2)'s situation, where `attainable` passed and simulated power was 0.00."""
    res = decode_correctness(_bank(effect=0.0, seed=8), "shape", n_perm=60)
    power = pd.DataFrame({"effect_sd": [0.0, 0.5, 2.0], "n_trials": [5] * 3,
                          "detected": [0.0, 0.0, 0.2]})
    txt = report(res, power)
    assert "cannot support a null about correctness at all" in txt


def test_the_permutation_is_within_family() -> None:
    """Permuting labels ACROSS families would break the family/correctness link the
    centring preserves, and the null would then be easier to beat than the data --
    inflating significance rather than controlling it."""
    import ast
    import os
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "run_correctness_decode.py"),
               encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_perm_within_family")
    body = ast.unparse(fn)
    assert "np.unique(fams)" in body and "rng.permutation(y[m])" in body


@pytest.mark.parametrize("feature", ["shape", "position"])
def test_both_features_run_through_the_same_path(feature) -> None:
    out = decode_correctness(_bank(effect=1.5, seed=9), feature, n_perm=60)
    assert out["usable"] and out["feature"] == feature
