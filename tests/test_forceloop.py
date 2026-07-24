"""Reproduces the force-loop Fisher exact test (claims_ledger.md B4) against
the real cached results/forceloop.csv, so the project's cleanest positive H1
result has a real reproducing test, not just a re-derived-by-hand number.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import fisher_exact
from scripts.run_forceloop import _unsettled_fisher_test


def test_unsettled_fisher_test_on_synthetic_table() -> None:
    """A clean-cut synthetic table gives the exact expected counts and p-value."""
    fl = pd.DataFrame(
        {
            "num_steps": [16] * 16,
            "n_ops": [8] * 8 + [24] * 8,
            "shape": ["settle"] * 7 + ["loop"] + ["loop"] * 7 + ["settle"],
        }
    )
    settled_a, unsettled_a, settled_b, unsettled_b, p = _unsettled_fisher_test(
        fl, num_steps=16, n_ops_a=8, n_ops_b=24
    )
    assert (settled_a, unsettled_a, settled_b, unsettled_b) == (7, 1, 1, 7)
    _, expected_p = fisher_exact([[7, 1], [1, 7]])
    assert p == pytest.approx(expected_p)


def test_force_loop_reproduces_verified_b4_result() -> None:
    """Regression-pins the exact verified B4 numbers (claims_ledger.md) against
    the real cached forceloop.csv: 1/8 -> 7/8 unsettled at num_steps=16,
    Fisher p=0.0101, 100% settle at num_steps>=24.
    """
    fl = pd.read_csv("results/forceloop.csv")

    n_ops_levels = sorted(fl["n_ops"].unique())
    assert n_ops_levels[:2] == [8, 24]

    settled_a, unsettled_a, settled_b, unsettled_b, p = _unsettled_fisher_test(
        fl, num_steps=16, n_ops_a=n_ops_levels[0], n_ops_b=n_ops_levels[1]
    )
    assert (settled_a, unsettled_a) == (7, 1)  # n_ops=8: 1/8 unsettled
    assert (settled_b, unsettled_b) == (1, 7)  # n_ops=24: 7/8 unsettled
    assert p == pytest.approx(0.0101, abs=0.0001)

    overall = fl["shape"].value_counts()
    assert overall["settle"] == 82
    assert overall["loop"] == 13
    assert overall["drift"] == 1

    settle_ge24 = (fl.loc[fl["num_steps"] >= 24, "shape"] == "settle").mean()
    assert settle_ge24 == pytest.approx(1.0)
