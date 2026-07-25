"""Regression-pins the matched-random-walk null-test result (claims_ledger.md
D22) against the real cached results/winding_null.csv -- the finding that
observed |winding| does not beat, and mostly falls below, a step-size-matched
null on real Huginn trajectories.

Note these tests read the RESULT csv (committed), not the raw .npy paths
(untracked/local), so they run anywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from traj_geom.analysis.correlate import benjamini_hochberg


def _load() -> pd.DataFrame:
    return pd.read_csv("results/winding_null.csv")


def test_null_test_covered_all_banked_trajectories() -> None:
    """140 trajectories, both tasks, both compute budgets."""
    df = _load()
    assert len(df) == 140
    assert set(df["task"].unique()) == {"count_ones", "projection"}
    assert set(df["num_steps"].unique()) == {64, 128}


def test_observed_winding_falls_below_the_matched_null() -> None:
    """The headline: real paths wind LESS than random walks with the same
    step sizes -- the opposite direction from 'the paths orbit'.
    """
    df = _load()
    assert (df["observed"] < df["null_mean"]).mean() == pytest.approx(0.821, abs=0.01)
    assert df["z_score"].median() == pytest.approx(-5.61, abs=0.1)
    assert df["observed"].mean() == pytest.approx(0.6246, abs=0.001)
    assert df["null_mean"].mean() == pytest.approx(0.6604, abs=0.001)
    # Two-thirds of trajectories: EVERY surrogate wound at least as much.
    assert (df["p_value"] >= 0.999).mean() == pytest.approx(0.664, abs=0.01)


def test_only_a_small_minority_beats_chance_after_fdr() -> None:
    """Honest accounting: a genuine but small minority does exceed the null.
    9/140 survive BH-FDR -- barely above the 5% false-positive expectation.
    """
    df = _load()
    assert int((df["p_value"] < 0.05).sum()) == 14
    _, sig = benjamini_hochberg(df["p_value"].to_numpy())
    assert int(sig.sum()) == 9


def test_gap_widens_with_compute_budget_the_settling_mechanism() -> None:
    """Mechanism: observed winding stays flat as the budget doubles while the
    random-walk null keeps accumulating angle -- because the real path has
    settled and only jitters, so it accrues no net rotation.
    """
    df = _load()
    ns64, ns128 = df[df["num_steps"] == 64], df[df["num_steps"] == 128]

    # Observed is flat across budgets...
    assert ns64["observed"].mean() == pytest.approx(0.6242, abs=0.001)
    assert ns128["observed"].mean() == pytest.approx(0.6250, abs=0.001)
    # ...while the null rises with more steps.
    assert ns64["null_mean"].mean() == pytest.approx(0.6284, abs=0.001)
    assert ns128["null_mean"].mean() == pytest.approx(0.6845, abs=0.001)
    # So the z-gap blows open.
    assert ns64["z_score"].mean() == pytest.approx(-1.62, abs=0.05)
    assert ns128["z_score"].mean() == pytest.approx(-15.99, abs=0.05)
    assert ns128["z_score"].mean() < ns64["z_score"].mean()


def test_winding_is_near_constant_on_answer_token_sweeps() -> None:
    """The constancy that makes H2 largely untestable as operationalized:
    the best-powered sweep (modk N=15, 270 rows, 15 difficulty levels) has
    winding std = 0.005.
    """
    modk = pd.read_csv("results/three_scale_modk_extended.csv")
    assert modk["winding"].std() == pytest.approx(0.005, abs=0.001)
    assert modk["winding"].min() > 0.58
    assert modk["winding"].max() < 0.63

    # Contrast: the all-token run does vary genuinely.
    bl = pd.read_csv("results/blayney_repro.csv")
    assert bl["winding"].std() > 0.4
