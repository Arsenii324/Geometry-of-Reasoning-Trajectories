"""Regression-pins the verified modulus-counting sweep results
(claims_ledger.md D15) against the real cached results/three_scale_modk.csv
(N=7) and results/three_scale_modk_extended.csv (N=15, the replication) --
the project's first genuinely unconfounded H2 test, run twice at two
sample sizes.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import combine_pvalues, spearmanr

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


def test_three_scale_modk_extended_replicates_and_strengthens_d15() -> None:
    """N=15 replication: winding stays null (more decisively), steps_settle
    replicates and strengthens (both moduli individually significant now).
    """
    df = pd.read_csv("results/three_scale_modk_extended.csv")
    assert len(df) == 270
    assert df["seq_len"].nunique() == 1
    assert df["seq_len"].iloc[0] == 54

    winding_ps, steps_ps = [], []
    expected = {
        2: {"winding_rho": 0.043, "winding_p": 0.8795, "steps_rho": -0.549, "steps_p": 0.0339},
        5: {"winding_rho": -0.461, "winding_p": 0.0839, "steps_rho": -0.739, "steps_p": 0.0016},
    }
    for k, exp in expected.items():
        sub = df[df["modulus"] == k]
        gm_w = sub.groupby("active_len")["winding"].mean()
        rho_w, p_w = spearmanr(gm_w.index, gm_w.values)
        gm_s = sub.groupby("active_len")["steps_settle"].mean()
        rho_s, p_s = spearmanr(gm_s.index, gm_s.values)

        assert rho_w == pytest.approx(exp["winding_rho"], abs=0.001)
        assert p_w == pytest.approx(exp["winding_p"], abs=0.001)
        assert rho_s == pytest.approx(exp["steps_rho"], abs=0.001)
        assert p_s == pytest.approx(exp["steps_p"], abs=0.001)
        winding_ps.append(p_w)
        steps_ps.append(p_s)

    _, p_winding_combined = combine_pvalues(winding_ps, method="fisher")
    _, p_steps_combined = combine_pvalues(steps_ps, method="fisher")
    assert p_winding_combined == pytest.approx(0.266, abs=0.001)
    assert p_winding_combined > 0.05  # winding: still not significant
    assert p_steps_combined == pytest.approx(0.0006, abs=0.0001)
    assert p_steps_combined < 0.01  # steps_settle: clearly significant
