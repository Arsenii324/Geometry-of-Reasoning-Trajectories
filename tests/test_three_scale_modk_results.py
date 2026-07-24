"""Regression-pins the verified modulus-counting sweep result
(claims_ledger.md D15) against the real cached
results/three_scale_modk.csv -- the project's first genuinely
unconfounded H2 test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from traj_geom.analysis.correlate import spearman_by_level


def test_seq_len_exactly_constant_across_the_real_sweep() -> None:
    """The actual property this task design exists to guarantee, confirmed
    on real Huginn output, not just the generator's own offline check.
    """
    df = pd.read_csv("results/three_scale_modk.csv")
    assert len(df) == 126
    assert df["seq_len"].nunique() == 1
    assert df["seq_len"].iloc[0] == 42


def test_three_scale_modk_reproduces_verified_d15_null() -> None:
    """No significant winding~active_len at either modulus tested -- a
    clean null now that the D11 prefix confound is genuinely removed.
    """
    df = pd.read_csv("results/three_scale_modk.csv")

    for k, expected_rho in ((2, 0.179), (5, -0.036)):
        sub = df[df["modulus"] == k]
        rho, n, crit = spearman_by_level(sub, "active_len", "winding")
        assert rho == pytest.approx(expected_rho, abs=0.001)
        assert n == 7
        assert crit == pytest.approx(0.786)
        assert abs(rho) < crit  # not significant
