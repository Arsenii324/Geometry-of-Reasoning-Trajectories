"""Regression-pins the count_ones / projection numbers cited in
claims_ledger.md D18 against the real cached
results/full_synthetic_experiments.csv.

Added 2026-07-25 after an accountability audit found this CSV had a
ledger-cited figure (count_ones mean_normed_accel~n_ops = -0.976) with no
pinning test -- the exact gap the project's own verifiability practice is
supposed to prevent.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import spearmanr


def _per_level_rho(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[float, int]:
    gm = df.groupby(xcol)[ycol].mean()
    return float(spearmanr(gm.index.to_numpy(), gm.to_numpy())[0]), int(len(gm))


def test_count_ones_accel_is_the_tightest_but_length_confounded() -> None:
    """count_ones mean_normed_accel~n_ops = -0.976 (the tightest monotone
    relationship in the project) -- but n_ops is rank-collinear with seq_len,
    so it is uninterpretable as a depth effect (D10/D18).
    """
    df = pd.read_csv("results/full_synthetic_experiments.csv")
    co = df[df["task"] == "count_ones"]

    rho, n = _per_level_rho(co, "n_ops", "mean_normed_accel")
    assert rho == pytest.approx(-0.976, abs=0.001)
    assert n == 8

    # The D10 confound that makes it uninterpretable.
    length_rho = float(spearmanr(co["n_ops"], co["seq_len"])[0])
    assert length_rho == pytest.approx(1.0, abs=1e-9)


def test_projection_accel_is_weak_and_not_significant() -> None:
    """The sibling task's accel effect is weak -- so -0.976 is specific to
    count_ones, not a generic property of the metric.
    """
    df = pd.read_csv("results/full_synthetic_experiments.csv")
    pr = df[df["task"] == "projection"]

    rho, n = _per_level_rho(pr, "n_ops", "mean_normed_accel")
    assert rho == pytest.approx(-0.190, abs=0.001)
    assert n == 8
    assert abs(rho) < 0.738  # crit |rho| at N=8; not significant


def test_both_tasks_present_with_expected_shape() -> None:
    """Guards against a silently truncated or re-generated CSV."""
    df = pd.read_csv("results/full_synthetic_experiments.csv")
    assert len(df) == 80
    assert set(df["task"].unique()) == {"count_ones", "projection"}
